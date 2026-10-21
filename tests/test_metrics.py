"""Métricas Prometheus: a tabela do contrato da Fase 4 e o que cada operação conta."""

import asyncio
import json
import math
import subprocess
import sys
from pathlib import Path

import httpx
import pytest
from prometheus_client import REGISTRY
from prometheus_client.parser import text_string_to_metric_families
from pydantic import SecretStr

from app import metrics
from app.api import create_app
from app.clients.openrouter import OpenRouterEmbeddingClient
from app.clients.rerank import OpenRouterRerankerClient
from app.schemas.domain import Chunk
from app.schemas.mcp import HybridSearchInput, HybridSearchOutput
from app.search.fallback import DEGRADED

KEY = SecretStr("sk-or-v1-chave-de-teste")
# Copiado do contrato (contratos-fase4.md §1), não das constantes do código: nome, tipo ou bucket
# trocado lá ou aqui tem de quebrar este teste.
CONTRACT = {
    "brain_router_search_requests": ("counter", {"mode", "outcome"}),
    "brain_router_search_duration_seconds": ("histogram", {"mode"}),
    "brain_router_embedding_requests": ("counter", {"outcome"}),
    "brain_router_embedding_duration_seconds": ("histogram", set()),
    "brain_router_rerank_duration_seconds": ("histogram", set()),
    "brain_router_result_score": ("histogram", set()),
    "brain_router_eval_score": ("gauge", {"metric"}),
}
LATENCY = "0.005 0.01 0.025 0.05 0.1 0.25 0.5 1 2.5 5"
SCORES = "0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0"
CHUNKS = [Chunk(id=f"c{n}", document_id="d", content=f"trecho {n}") for n in range(2)]


def value(name: str, **labels: str) -> float:
    return REGISTRY.get_sample_value(name, labels) or 0.0


async def _answer(body: HybridSearchInput) -> HybridSearchOutput:
    return HybridSearchOutput(query=body.query, results=[], total_found=0, execution_time_ms=1)


def _reranker(*scores: float, status: int = 200) -> OpenRouterRerankerClient:
    results = [{"index": i, "relevance_score": score} for i, score in enumerate(scores)]
    reply = httpx.Response(status, json={"results": results})
    return OpenRouterRerankerClient(KEY, transport=httpx.MockTransport(lambda request: reply))


def _embedder(status: int = 200) -> OpenRouterEmbeddingClient:
    def handler(request: httpx.Request) -> httpx.Response:
        texts = json.loads(request.content)["input"]
        data = [{"index": i, "embedding": [0.5]} for i in range(len(texts))]
        return httpx.Response(status, json={"data": data} if status == 200 else {})

    return OpenRouterEmbeddingClient(
        KEY, dimension=1, batch_size=2, transport=httpx.MockTransport(handler)
    )


async def test_the_exposition_follows_the_contract_table() -> None:
    # Toda métrica com ao menos uma amostra, para os labels aparecerem.
    await metrics.measured_search(_answer)(HybridSearchInput(query="q"))
    async with _embedder() as embedder:
        await embedder.embed_query("texto")
    async with _reranker(0.9) as reranker:
        await reranker.rerank("q", CHUNKS[:1], top_n=1)
    metrics.record_eval_score("faithfulness", 0.9)

    families = {
        family.name: family
        for family in text_string_to_metric_families(metrics.render().decode())
        if family.name.startswith("brain_router_")
    }
    assert set(families) == set(CONTRACT)  # nem a mais (ex.: `*_created`), nem a menos
    for name, (kind, labels) in CONTRACT.items():
        family = families[name]
        assert family.type == kind, name
        assert {key for sample in family.samples for key in sample.labels} - {"le"} == labels
        buckets = [s.labels["le"] for s in family.samples if s.name.endswith("_bucket")]
        if kind == "histogram":
            expected = (SCORES if name == "brain_router_result_score" else LATENCY).split()
            assert [float(le) for le in dict.fromkeys(buckets)] == [
                *map(float, expected),
                math.inf,
            ], name


@pytest.mark.parametrize(
    ("alpha", "degraded", "error", "mode", "result"),
    [
        (0.5, False, None, "hybrid", "ok"),
        (1.0, False, None, "dense", "ok"),
        (0.0, False, None, "lexical", "ok"),
        (0.5, True, None, "lexical", "ok"),  # o fallback léxico respondeu
        (0.5, False, RuntimeError("x"), "hybrid", "error"),
        (0.5, False, TimeoutError(), "hybrid", "timeout"),
        (0.5, False, asyncio.CancelledError(), "hybrid", "canceled"),
    ],
)
async def test_each_search_counts_its_mode_outcome_and_duration(
    alpha: float, degraded: bool, error: BaseException | None, mode: str, result: str
) -> None:
    async def search(body: HybridSearchInput) -> HybridSearchOutput:
        if degraded:
            DEGRADED.set("lexical")
        if error is not None:
            raise error
        return await _answer(body)

    labels = {"mode": mode, "outcome": result}
    before = value("brain_router_search_requests_total", **labels)
    seen = value("brain_router_search_duration_seconds_count", mode=mode)
    mark = DEGRADED.set(None)
    try:
        call = metrics.measured_search(search)(HybridSearchInput(query="q", alpha=alpha))
        if error is None:
            await call
        else:
            with pytest.raises(type(error)):
                await call
    finally:
        DEGRADED.reset(mark)
    assert value("brain_router_search_requests_total", **labels) == before + 1
    assert value("brain_router_search_duration_seconds_count", mode=mode) == seen + 1


async def test_the_api_measures_both_search_routes() -> None:
    labels = {"mode": "hybrid", "outcome": "ok"}
    before = value("brain_router_search_requests_total", **labels)
    transport = httpx.ASGITransport(app=create_app(search=_answer))
    async with httpx.AsyncClient(transport=transport, base_url="http://brain") as client:
        for path in ("/api/v1/search", "/api/v1/search/stream"):
            assert (await client.post(path, json={"query": "q"})).status_code == 200
    assert value("brain_router_search_requests_total", **labels) == before + 2


async def test_each_provider_call_counts_one_embedding_request() -> None:
    ok, failed = (
        value("brain_router_embedding_requests_total", outcome=o) for o in ("ok", "error")
    )
    calls = value("brain_router_embedding_duration_seconds_count")
    async with _embedder() as embedder:
        await embedder.embed_documents(["a", "b", "c"])  # lotes de 2: dois pedidos
    async with _embedder(status=400) as embedder:
        with pytest.raises(Exception, match="HTTP 400"):
            await embedder.embed_query("a")
    assert value("brain_router_embedding_requests_total", outcome="ok") == ok + 2
    assert value("brain_router_embedding_requests_total", outcome="error") == failed + 1
    assert value("brain_router_embedding_duration_seconds_count") == calls + 3


async def test_rerank_records_duration_and_relevance_but_not_the_fallback() -> None:
    calls, scores = (
        value(f"brain_router_{n}_count") for n in ("rerank_duration_seconds", "result_score")
    )
    low = value("brain_router_result_score_bucket", le="0.2")
    async with _reranker(0.95, 0.15) as reranker:
        await reranker.rerank("q", CHUNKS, top_n=2)
    async with _reranker(status=503) as reranker:
        fallback = await reranker.rerank("q", CHUNKS, top_n=2)  # ordem de entrada, score 0.0
    assert [result.rrf_score for result in fallback] == [0.0, 0.0]
    assert value("brain_router_rerank_duration_seconds_count") == calls + 2
    assert value("brain_router_result_score_count") == scores + 2  # só a resposta válida
    assert value("brain_router_result_score_bucket", le="0.2") == low + 1


def test_eval_scores_keep_the_last_value_and_refuse_what_is_not_a_score() -> None:
    metrics.record_eval_score("context_relevance", 0.91)
    metrics.record_eval_score("context_relevance", 0.5)
    assert value("brain_router_eval_score", metric="context_relevance") == 0.5
    for name, score in [
        ("Faithfulness", 0.5),
        ("fide-lidade", 0.5),
        ("", 0.5),
        ("a" * 65, 0.5),
        ("faithfulness", 1.01),
        ("faithfulness", -0.01),
        ("faithfulness", math.nan),
        ("faithfulness", math.inf),
    ]:
        with pytest.raises(ValueError):
            metrics.record_eval_score(name, score)


def test_multiprocess_mode_sums_the_workers(tmp_path: Path) -> None:
    worker = (
        "import sys; from app import metrics; "
        "metrics.SEARCH_REQUESTS.labels('hybrid', 'ok').inc(); "
        "metrics.record_eval_score('faithfulness', float(sys.argv[1])); "
        "sys.stdout.write(metrics.render().decode())"
    )
    env = {"PROMETHEUS_MULTIPROC_DIR": str(tmp_path), "PATH": "/usr/bin:/bin"}
    for score in ("0.4", "0.7"):  # dois processos, um depois do outro
        run = subprocess.run(
            [sys.executable, "-c", worker, score], env=env, capture_output=True, text=True
        )
        assert run.returncode == 0, run.stderr
    samples = {
        (sample.name, tuple(sorted(sample.labels.items()))): sample.value
        for family in text_string_to_metric_families(run.stdout)
        for sample in family.samples
    }
    counter = ("brain_router_search_requests_total", (("mode", "hybrid"), ("outcome", "ok")))
    assert samples[counter] == 2.0  # a soma dos dois workers
    assert samples[("brain_router_eval_score", (("metric", "faithfulness"),))] == 0.7
    assert not [name for name, _ in samples if name.startswith(("process_", "python_"))]

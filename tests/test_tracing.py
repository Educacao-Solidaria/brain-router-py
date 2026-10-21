"""Tracing OTel + OpenInference: liga só com endpoint OTLP e não manda texto de usuário."""

import asyncio
import io
import json
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass

import httpx
import pytest
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.common.trace_encoder import encode_spans
from opentelemetry.exporter.otlp.proto.http import trace_exporter
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import INVALID_SPAN, StatusCode
from pydantic import SecretStr

from app import __version__, tracing
from app.clients.openrouter import OpenRouterEmbeddingClient
from app.clients.rerank import OpenRouterRerankerClient
from app.log import configure_logging, shutdown_logging
from app.schemas.domain import Chunk, SearchResult
from app.storage.interfaces import InMemoryVectorStore

KEY = SecretStr("sk-or-v1-chave-de-teste")
QUERY = "qual o prazo da matrícula da Ana Souza"
TEXTS = ["Ana Souza: matrícula até março", "Secretaria: telefone 3333-4444"]
CHUNKS = [
    Chunk(id=f"doc-1:{n}", document_id="doc-1", content=text, embedding=[1.0 - n, n, 0.0])
    for n, text in enumerate(TEXTS)
]
VECTOR = [0.123456, 0.654321, 0.111111]  # componentes fáceis de procurar no span
USER_DATA = (QUERY, "Ana Souza", "matrícula", "3333-4444", "0.123456", "0.654321")
BASE = {"OTEL_EXPORTER_OTLP_ENDPOINT": "http://col:4318"}


def _leaks(spans: Sequence[ReadableSpan]) -> list[str]:
    """O que de `USER_DATA` está nos bytes OTLP que iriam ao coletor (o `to_json()` não serve:
    escapa o não-ASCII, "matr\\u00edcula", e o teste passaria com o texto vazando)."""
    wire = encode_spans(spans).SerializeToString()
    return [text for text in USER_DATA if text.encode() in wire]


def _openrouter(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    if request.url.path.endswith("/embeddings"):
        data = [{"index": i, "embedding": VECTOR} for i in range(len(body["input"]))]
        return httpx.Response(200, json={"data": data})
    ranked = [{"index": i, "relevance_score": 0.9 - i / 10} for i in range(len(body["documents"]))]
    return httpx.Response(200, json={"results": ranked})


async def _pipeline() -> None:
    """Embedding da query, busca vetorial e rerank: as três operações instrumentadas."""
    transport = httpx.MockTransport(_openrouter)
    async with OpenRouterEmbeddingClient(KEY, dimension=3, transport=transport) as embedder:
        vector = await embedder.embed_query(QUERY)
    store = InMemoryVectorStore()
    await store.add_chunks("default", CHUNKS)
    results = await tracing.TracedVectorStore(store).search_vector("default", vector, limit=5)
    async with OpenRouterRerankerClient(KEY, transport=transport) as reranker:
        await reranker.rerank(QUERY, [result.chunk for result in results], top_n=1)


class Stuck(InMemoryVectorStore):
    """Falha como um DBAPIError (SQL e parâmetros no `str()`) ou, com `entered`, fica parada."""

    def __init__(self, entered: asyncio.Event | None = None) -> None:
        super().__init__()
        self.entered = entered

    async def search_vector(
        self, collection_name: str, query_vector: list[float], limit: int = 10, min_score: float = 0
    ) -> list[SearchResult]:
        if self.entered is None:
            raise RuntimeError(f"[SQL: SELECT ...] [parameters: ({QUERY!r}, {VECTOR!r})]")
        self.entered.set()
        await asyncio.Event().wait()
        return []


@dataclass
class Traced:
    provider: TracerProvider
    endpoint: str
    exporter: InMemorySpanExporter

    def spans(self) -> list[ReadableSpan]:
        self.provider.force_flush()  # o BatchSpanProcessor exporta em segundo plano
        return list(self.exporter.get_finished_spans())


@pytest.fixture
def made_global(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    """Os provedores que viraram o global do OTel (nenhum fica no processo de teste)."""
    providers: list[object] = []
    monkeypatch.setattr(trace, "set_tracer_provider", providers.append)
    monkeypatch.setattr(tracing, "_tracer", None)  # o monkeypatch devolve o estado no fim
    return providers


@pytest.fixture
def start(
    made_global: list[object], monkeypatch: pytest.MonkeyPatch
) -> Iterator[Callable[..., Traced]]:
    """`configure_tracing` de verdade, com o exportador OTLP trocado por um em memória."""
    exporters: list[tuple[str, InMemorySpanExporter]] = []
    started: list[Traced] = []

    def otlp(*, endpoint: str) -> InMemorySpanExporter:
        exporters.append((endpoint, InMemorySpanExporter()))
        return exporters[-1][1]

    def start(**env: str) -> Traced:
        provider = tracing.configure_tracing({**BASE, **env})
        assert provider is not None and made_global[-1] is provider
        started.append(Traced(provider, *exporters[-1]))
        return started[-1]

    monkeypatch.setattr(trace_exporter, "OTLPSpanExporter", otlp)
    yield start
    for traced in started:
        traced.provider.shutdown()


@pytest.mark.parametrize(
    "env",
    [
        {},
        {"OTEL_EXPORTER_OTLP_ENDPOINT": "  "},
        {"OTEL_EXPORTER_OTLP_ENDPOINT": "http://col:4318", "OTEL_TRACES_EXPORTER": "none"},
        {"OTEL_EXPORTER_OTLP_TRACES_ENDPOINT": "http://col:4318", "OTEL_SDK_DISABLED": "TRUE"},
    ],
)
async def test_without_endpoint_nothing_is_configured_and_spans_are_not_created(
    env: dict[str, str], made_global: list[object]
) -> None:
    assert tracing.configure_tracing(env) is None
    assert made_global == [] and tracing._tracer is None
    store = InMemoryVectorStore()
    await store.add_chunks("default", CHUNKS)
    results = await tracing.TracedVectorStore(store).search_vector("default", [1, 0, 0], limit=1)
    assert [result.chunk.id for result in results] == ["doc-1:0"]  # passa direto
    with tracing.embedding_span("modelo", ["texto"]) as span:
        assert span is INVALID_SPAN  # o atalho nem chama o OTel


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ({**BASE, "OTEL_TRACES_EXPORTER": "OTLP"}, "http://col:4318/v1/traces"),
        ({"OTEL_EXPORTER_OTLP_ENDPOINT": "https://col/otlp/"}, "https://col/otlp/v1/traces"),
        ({**BASE, "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT": "http://t:9/x"}, "http://t:9/x"),
    ],
)
def test_the_endpoint_follows_the_standard_otel_variables(
    env: dict[str, str], expected: str
) -> None:
    assert tracing.traces_endpoint(env) == expected


@pytest.mark.parametrize(
    "env",
    [
        {"OTEL_EXPORTER_OTLP_PROTOCOL": "grpc"},
        {"OTEL_EXPORTER_OTLP_PROTOCOL": "http/protobuf", "OTEL_EXPORTER_OTLP_TRACES_PROTOCOL": "x"},
        {"OTEL_TRACES_EXPORTER": "zipkin"},
        {"OTEL_EXPORTER_OTLP_TRACES_ENDPOINT": "col:4318"},
        {"OTEL_EXPORTER_OTLP_TRACES_ENDPOINT": "ftp://usuario:segredo@col"},
    ],
)
def test_config_that_would_not_be_honored_fails_at_startup(env: dict[str, str]) -> None:
    with pytest.raises(ValueError) as error:
        tracing.traces_endpoint({**BASE, **env})
    assert "segredo" not in str(error.value)


@pytest.mark.parametrize(
    ("value", "hidden"), [(None, True), ("0", True), ("no", True), (" FALSE ", False)]
)
def test_only_an_explicit_false_reveals_a_field(value: str | None, hidden: bool) -> None:
    env = {} if value is None else {"OPENINFERENCE_HIDE_INPUTS": value}
    assert tracing.trace_config(env).hide_inputs is hidden


async def test_spans_carry_kind_model_counts_and_ids_but_no_user_text(
    start: Callable[..., Traced], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OTEL_SERVICE_NAME", "outro-servico")  # o Resource.create lê do processo
    traced = start(OTEL_EXPORTER_OTLP_TRACES_ENDPOINT="http://col:4318/v1/traces")
    await _pipeline()
    spans = traced.spans()
    by_name = {span.name: dict(span.attributes or {}) for span in spans}
    embedding, search, rerank = by_name["embeddings"], by_name["vector_search"], by_name["rerank"]

    assert traced.endpoint == "http://col:4318/v1/traces"
    assert {span.resource.attributes["service.name"] for span in spans} == {"brain-router"}
    assert spans[0].resource.attributes["service.version"] == __version__
    assert embedding["openinference.span.kind"] == "EMBEDDING"
    assert embedding["embedding.model_name"] == "openai/text-embedding-3-small"
    assert embedding["embedding.embeddings.0.embedding.text"] == "__REDACTED__"
    assert search["openinference.span.kind"] == "RETRIEVER"
    assert (
        search["brain_router.search.dimension"] == 3 and search["brain_router.search.results"] == 2
    )
    assert search["retrieval.documents.0.document.id"] == "doc-1:1"  # o mais parecido com VECTOR
    assert search["retrieval.documents.0.document.content"] == "__REDACTED__"
    assert rerank["openinference.span.kind"] == "RERANKER"
    assert (rerank["reranker.model_name"], rerank["reranker.top_k"]) == ("cohere/rerank-4-fast", 1)
    assert (
        rerank["reranker.query"] == "__REDACTED__" and rerank["brain_router.rerank.documents"] == 2
    )
    assert _leaks(spans) == []


async def test_an_explicit_false_reveals_the_text_and_warns_at_startup(
    start: Callable[..., Traced],
) -> None:
    stream = io.StringIO()
    configure_logging("INFO", stream=stream)
    try:
        traced = start(**dict.fromkeys(tracing.HIDE_VARIABLES, "false"))
    finally:
        shutdown_logging()
    await _pipeline()

    # Prova de que a ocultação é o que segura o texto (e de que `_leaks` enxerga o vazamento).
    assert set(_leaks(traced.spans())) == set(USER_DATA) - {"0.123456", "0.654321"}  # vetor nunca
    (line,) = [json.loads(raw) for raw in stream.getvalue().splitlines()]
    assert (line["event"], line["level"]) == ("tracing.content_revealed", "warning")
    assert line["variables"] == list(tracing.HIDE_VARIABLES)


async def test_a_failure_records_the_type_and_never_the_message(
    start: Callable[..., Traced],
) -> None:
    traced = start()
    with pytest.raises(RuntimeError):
        await tracing.TracedVectorStore(Stuck()).search_vector("default", VECTOR)
    down = httpx.MockTransport(
        lambda request: httpx.Response(503, json={"error": {"message": QUERY}})
    )
    async with OpenRouterRerankerClient(KEY, transport=down) as reranker:
        fallback = await reranker.rerank(QUERY, CHUNKS, top_n=1)  # cai na ordem de entrada

    spans = traced.spans()
    assert [result.rrf_score for result in fallback] == [0.0]
    for span, error in zip(spans, ("RuntimeError", "RerankServiceError"), strict=True):
        assert (span.status.status_code, span.status.description) == (StatusCode.ERROR, None)
        assert dict(span.attributes or {})["error.type"] == error
        assert len(span.events) == 0  # sem o evento `exception`, que levaria a mensagem
    assert dict(spans[0].attributes or {})["brain_router.search.limit"] == 10  # gravado antes
    assert _leaks(spans) == []


async def test_cancellation_ends_the_span_without_an_error(start: Callable[..., Traced]) -> None:
    traced, entered = start(), asyncio.Event()
    task = asyncio.create_task(tracing.TracedVectorStore(Stuck(entered)).search_vector("c", VECTOR))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    (span,) = traced.spans()
    assert span.status.status_code is StatusCode.UNSET
    assert "error.type" not in dict(span.attributes or {})


async def test_a_long_batch_keeps_the_count_and_at_most_max_span_items(
    start: Callable[..., Traced],
) -> None:
    traced = start(OPENINFERENCE_HIDE_EMBEDDINGS_TEXT="false")
    texts = [f"texto {n}" for n in range(tracing.MAX_SPAN_ITEMS + 5)]
    transport = httpx.MockTransport(_openrouter)
    async with OpenRouterEmbeddingClient(KEY, dimension=3, transport=transport) as client:
        await client.embed_documents(texts)
    with tracing.embedding_span("modelo", []):  # lote vazio: contagens zeradas, nenhum item
        pass
    full, empty = (dict(span.attributes or {}) for span in traced.spans())
    items = [key for key in full if key.startswith("embedding.embeddings.")]
    assert len(items) == tracing.MAX_SPAN_ITEMS and full[items[0]] == "texto 0"
    assert full["brain_router.embedding.texts"] == len(texts) == tracing.MAX_SPAN_ITEMS + 5
    assert full["brain_router.embedding.chars"] == sum(map(len, texts))
    assert (empty["brain_router.embedding.texts"], empty["brain_router.embedding.chars"]) == (0, 0)
    assert "embedding.model_name" in full  # não foi descartado pelo teto de atributos do SDK

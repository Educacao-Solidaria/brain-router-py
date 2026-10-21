"""Métricas Prometheus do brain-router: nomes, labels e buckets fixos do contrato da Fase 4.

Ficam no registro padrão do `prometheus_client`, junto com as de processo e runtime que ele já
traz (`process_*`, `python_*`); `render()` devolve o formato texto que a rota `/metrics` serve.
Labels só de cardinalidade baixa (`mode`, `outcome`, `metric`): nunca query, tenant, request id
ou trace id.

Modo suportado: um processo por réplica (uvicorn sem `--workers`; escale por réplicas). Com
vários workers, cada um tem o seu registro e a raspagem cairia num deles ao acaso. Para isso,
`PROMETHEUS_MULTIPROC_DIR` (uma pasta vazia por subida, antes de importar este módulo): os
processos gravam em arquivos e `render()` soma todos pelo `MultiProcessCollector`; as métricas de
processo e de runtime não entram nesse modo.
"""

import asyncio
import os
import re
import time
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager

from prometheus_client import (
    REGISTRY,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    disable_created_metrics,
    generate_latest,
    multiprocess,
)

from app.schemas.mcp import HybridSearchInput, HybridSearchOutput
from app.search.fallback import DEGRADED

# Sem as séries `*_created`, que o contrato não lista (a função não tem anotação de tipo).
disable_created_metrics()  # type: ignore[no-untyped-call]

METRICS_PATH = "/metrics"
# O formato texto clássico: o 0.26 rotula o padrão como 1.0.0, mas com nomes do formato antigo e o
# escape `underscores` do `generate_latest` os bytes são os do 0.0.4, que todo raspador lê.
CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"
LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)
SCORE_BUCKETS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
_EVAL_METRIC = re.compile(r"[a-z][a-z0-9_]{0,63}")  # nome de métrica de eval: faithfulness etc.

SEARCH_REQUESTS = Counter(
    "brain_router_search_requests_total", "Buscas atendidas.", ["mode", "outcome"]
)
SEARCH_DURATION = Histogram(
    "brain_router_search_duration_seconds", "Duração da busca.", ["mode"], buckets=LATENCY_BUCKETS
)
EMBEDDING_REQUESTS = Counter(
    "brain_router_embedding_requests_total", "Pedidos de embedding ao provedor.", ["outcome"]
)
EMBEDDING_DURATION = Histogram(
    "brain_router_embedding_duration_seconds",
    "Duração de cada pedido de embedding (um lote).",
    buckets=LATENCY_BUCKETS,
)
RERANK_DURATION = Histogram(
    "brain_router_rerank_duration_seconds", "Duração de cada rerank.", buckets=LATENCY_BUCKETS
)
RESULT_SCORE = Histogram(
    "brain_router_result_score",
    "Relevância (0 a 1) que o reranker deu a cada resultado devolvido.",
    buckets=SCORE_BUCKETS,
)
EVAL_SCORE = Gauge(
    "brain_router_eval_score",
    "Último score (0 a 1) de cada métrica de eval.",
    ["metric"],
    multiprocess_mode="mostrecent",
)

Search = Callable[[HybridSearchInput], Awaitable[HybridSearchOutput]]


def outcome(error: BaseException | None) -> str:
    """O label `outcome` do contrato para o fim de uma operação."""
    if error is None:
        return "ok"
    if isinstance(error, asyncio.CancelledError):
        return "canceled"
    return "timeout" if isinstance(error, TimeoutError) else "error"


@contextmanager
def measured(histogram: Histogram, counter: Counter) -> Iterator[None]:
    """A duração do bloco em `histogram` e o desfecho dele (label `outcome`) em `counter`."""
    started, error = time.perf_counter(), None
    try:
        yield
    except BaseException as exc:
        error = exc
        raise
    finally:
        counter.labels(outcome(error)).inc()
        histogram.observe(time.perf_counter() - started)


def search_mode(body: HybridSearchInput) -> str:
    """`lexical` se o fallback respondeu (`DEGRADED`) ou se pediram só a esparsa (alpha 0)."""
    if DEGRADED.get() is not None or body.alpha == 0.0:  # os extremos exatos do contrato da tool
        return "lexical"
    return "dense" if body.alpha == 1.0 else "hybrid"


def measured_search(search: Search) -> Search:
    """O `search` contando cada busca (modo e desfecho) e a duração dela."""

    async def measured(body: HybridSearchInput) -> HybridSearchOutput:
        started, error = time.perf_counter(), None
        try:
            return await search(body)
        except BaseException as exc:
            error = exc
            raise
        finally:
            mode = search_mode(body)  # depois da busca: o fallback marca o DEGRADED nela
            SEARCH_REQUESTS.labels(mode, outcome(error)).inc()
            SEARCH_DURATION.labels(mode).observe(time.perf_counter() - started)

    return measured


def record_eval_score(metric: str, score: float) -> None:
    """Publica o score de uma eval (`faithfulness`, `relevance`...) em `brain_router_eval_score`.

    Para as evals alimentarem: o nome vira label, então só snake_case curto; score fora de
    [0, 1] (ou NaN) é recusado em vez de entrar no painel.
    """
    if not _EVAL_METRIC.fullmatch(metric):
        raise ValueError(f"nome de métrica de eval inválido: {metric[:80]!r}")
    if not 0.0 <= score <= 1.0:  # recusa também NaN
        raise ValueError(f"score de eval fora de [0, 1]: {score!r}")
    EVAL_SCORE.labels(metric).set(score)


def render() -> bytes:
    """As métricas no formato texto do Prometheus (`CONTENT_TYPE`)."""
    if "PROMETHEUS_MULTIPROC_DIR" not in os.environ:
        return generate_latest(REGISTRY)
    registry = CollectorRegistry()
    # Soma os arquivos de todos os processos (o construtor não tem anotação de tipo).
    multiprocess.MultiProcessCollector(registry)  # type: ignore[no-untyped-call]
    return generate_latest(registry)

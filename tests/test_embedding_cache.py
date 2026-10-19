"""Cache de embeddings de consulta (`CachedEmbeddingClient`) sobre clientes falsos, sem rede."""

import asyncio
import json
import math
import unicodedata
from typing import Any

import httpx
import pytest
from pydantic import SecretStr

from app.clients.cache import CachedEmbeddingClient, normalize_query
from app.clients.interfaces import BaseEmbeddingClient, MockEmbeddingClient
from app.clients.openrouter import OpenRouterEmbeddingClient
from app.errors.exceptions import EmbeddingServiceError
from app.perf import get_stats, reset_stats


class Inner(BaseEmbeddingClient):
    """Vetor = número da chamada ([1.0, 1.0] na primeira): diz se veio do cache ou não."""

    def __init__(self) -> None:
        self.dim = 2
        self.queries: list[str] = []
        self.document_calls = 0
        self.failures: list[Exception] = []
        self.gate: asyncio.Event | None = None

    @property
    def dimension(self) -> int:
        return self.dim

    async def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        if self.gate is not None:
            await self.gate.wait()
        if self.failures:
            raise self.failures.pop(0)
        return [float(len(self.queries))] * self.dim

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_calls += 1
        return [[0.0] * self.dim for _ in texts]


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _cache(inner: BaseEmbeddingClient, **options: Any) -> CachedEmbeddingClient:
    return CachedEmbeddingClient(inner, model="openai/text-embedding-3-small", **options)


async def test_repeated_query_calls_the_inner_client_once() -> None:
    inner = Inner()
    cache = _cache(inner)

    first = await cache.embed_query("O que é matrícula?")
    second = await cache.embed_query("O que é matrícula?")

    assert first == second == [1.0, 1.0]
    assert inner.queries == ["O que é matrícula?"]
    assert len(cache) == 1
    assert cache.dimension == 2


async def test_caller_cannot_corrupt_the_cached_vector() -> None:
    cache = _cache(Inner())
    first = await cache.embed_query("frequência")
    first[0] = 99.0

    second = await cache.embed_query("frequência")

    assert second == [1.0, 1.0]
    assert second is not await cache.embed_query("frequência")


async def test_normalization_only_merges_spellings_of_the_same_question() -> None:
    inner = Inner()
    cache = _cache(inner)
    composed = "matrícula"
    decomposed = unicodedata.normalize("NFD", composed)
    assert composed != decomposed

    assert await cache.embed_query(decomposed) == await cache.embed_query(f"  {composed}\n")
    # Caixa e espaço interno mudam a pergunta: "Rio" (cidade) não é "rio".
    for text in ("Rio", "rio", "a  b", "a b"):
        await cache.embed_query(text)

    assert inner.queries == [composed, "Rio", "rio", "a  b", "a b"]
    assert normalize_query("\ud800 x ") == "\ud800 x"  # surrogate solto não quebra a chave
    await cache.embed_query("\ud800")


async def test_key_includes_the_dimension() -> None:
    inner = Inner()
    cache = _cache(inner)
    await cache.embed_query("evasão")
    inner.dim = 3

    assert await cache.embed_query("evasão") == [2.0, 2.0, 2.0]


async def test_entries_expire_after_the_ttl() -> None:
    inner, clock = Inner(), Clock()
    cache = _cache(inner, ttl_seconds=10.0, clock=clock)
    await cache.embed_query("bolsa")

    clock.now = 9.9
    assert await cache.embed_query("bolsa") == [1.0, 1.0]
    clock.now = 10.0
    assert await cache.embed_query("bolsa") == [2.0, 2.0]


async def test_without_ttl_only_the_lru_removes_entries() -> None:
    clock = Clock()
    cache = _cache(Inner(), ttl_seconds=None, clock=clock)
    await cache.embed_query("bolsa")
    clock.now = 1e12

    assert await cache.embed_query("bolsa") == [1.0, 1.0]


async def test_lru_evicts_the_least_recently_used_entry() -> None:
    inner = Inner()
    cache = _cache(inner, max_items=2)
    await cache.embed_query("a")
    await cache.embed_query("b")
    await cache.embed_query("a")  # "a" passa a ser a mais recente
    await cache.embed_query("c")  # expulsa "b"

    assert len(cache) == 2
    assert await cache.embed_query("a") == [1.0, 1.0]
    assert await cache.embed_query("b") == [4.0, 4.0]


async def test_failure_is_not_cached() -> None:
    inner = Inner()
    inner.failures.append(EmbeddingServiceError("HTTP 503"))
    cache = _cache(inner)

    with pytest.raises(EmbeddingServiceError):
        await cache.embed_query("histórico escolar")
    assert len(cache) == 0
    assert await cache.embed_query("histórico escolar") == [2.0, 2.0]


async def test_empty_query_is_refused_by_the_inner_client_and_not_cached() -> None:
    cache = _cache(MockEmbeddingClient(dimension=4))

    with pytest.raises(EmbeddingServiceError):
        await cache.embed_query("   ")
    assert len(cache) == 0


async def test_concurrent_misses_of_the_same_key_share_one_call() -> None:
    inner = Inner()
    inner.gate = asyncio.Event()
    cache = _cache(inner)

    calls = [asyncio.create_task(cache.embed_query("transferência")) for _ in range(5)]
    await asyncio.sleep(0)
    inner.gate.set()
    vectors = await asyncio.gather(*calls)

    assert inner.queries == ["transferência"]
    assert all(vector == [1.0, 1.0] for vector in vectors)
    assert len({id(vector) for vector in vectors}) == 5  # uma lista por chamador


async def test_shared_failure_reaches_every_waiter_and_the_next_call_retries() -> None:
    inner = Inner()
    inner.gate = asyncio.Event()
    inner.failures.append(EmbeddingServiceError("HTTP 429"))
    cache = _cache(inner)

    calls = [asyncio.create_task(cache.embed_query("vaga")) for _ in range(3)]
    await asyncio.sleep(0)
    inner.gate.set()
    results = await asyncio.gather(*calls, return_exceptions=True)

    assert all(isinstance(result, EmbeddingServiceError) for result in results)
    assert await cache.embed_query("vaga") == [2.0, 2.0]
    assert len(inner.queries) == 2


async def test_cancelling_one_caller_keeps_the_shared_call_alive() -> None:
    inner = Inner()
    inner.gate = asyncio.Event()
    cache = _cache(inner)
    cancelled = asyncio.create_task(cache.embed_query("rematrícula"))
    waiting = asyncio.create_task(cache.embed_query("rematrícula"))
    await asyncio.sleep(0)

    cancelled.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled
    inner.gate.set()

    assert await waiting == [1.0, 1.0]
    assert await cache.embed_query("rematrícula") == [1.0, 1.0]
    assert inner.queries == ["rematrícula"]


async def test_embed_documents_bypasses_the_cache() -> None:
    inner = Inner()
    cache = _cache(inner)

    await cache.embed_documents(["a", "b"])
    await cache.embed_documents(["a", "b"])

    assert inner.document_calls == 2
    assert len(cache) == 0


async def test_capacity_zero_disables_the_cache() -> None:
    inner = Inner()
    cache = _cache(inner, max_items=0)

    assert await cache.embed_query(" nota ") == [1.0, 1.0]
    assert await cache.embed_query("nota") == [2.0, 2.0]
    assert inner.queries == ["nota", "nota"]
    assert len(cache) == 0


async def test_perf_stats_count_queries_and_paid_misses() -> None:
    reset_stats()
    try:
        cache = _cache(Inner())
        for text in ("a", "a", "a", "b"):
            await cache.embed_query(text)
        await _cache(Inner(), max_items=0).embed_query("a")

        assert get_stats("embeddings.cache.query").calls == 5
        assert get_stats("embeddings.cache.miss").calls == 3
    finally:
        reset_stats()


@pytest.mark.parametrize(
    "options",
    [
        {"model": ""},
        {"max_items": -1},
        {"ttl_seconds": 0.0},
        {"ttl_seconds": -1.0},
        {"ttl_seconds": math.nan},
        {"ttl_seconds": math.inf},
    ],
)
def test_rejects_invalid_configuration(options: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        CachedEmbeddingClient(Inner(), **({"model": "m"} | options))


async def test_wraps_the_openrouter_client_with_one_request_per_question() -> None:
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        inputs = json.loads(request.content)["input"]
        data = [{"index": i, "embedding": [0.5, 0.5]} for i in range(len(inputs))]
        return httpx.Response(200, json={"data": data})

    async with OpenRouterEmbeddingClient(
        SecretStr("sk-or-v1-teste"), dimension=2, transport=httpx.MockTransport(handle)
    ) as client:
        cache = CachedEmbeddingClient(client, model=client.model)
        for _ in range(3):
            assert await cache.embed_query("Como pedir transferência?") == [0.5, 0.5]

    assert len(requests) == 1
    assert json.loads(requests[0].content)["model"] == "openai/text-embedding-3-small"
    assert "openai/text-embedding-3-small" in repr(cache)
    assert "sk-or" not in repr(cache)

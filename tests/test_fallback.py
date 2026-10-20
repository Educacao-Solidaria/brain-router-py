"""Fallback da busca: disjuntor, prazo do embedding e degradação para a busca léxica."""

import asyncio
from typing import Any

import httpx
import pytest

from app.api import create_app
from app.clients.interfaces import BaseEmbeddingClient
from app.errors.exceptions import DatabaseQueryError, EmbeddingServiceError
from app.perf import get_stats
from app.schemas import Chunk
from app.schemas.mcp import HybridSearchInput, HybridSearchOutput
from app.search.fallback import (
    DEGRADED,
    DEGRADED_HEADER,
    BreakerEmbeddingClient,
    CircuitBreaker,
    LexicalFallback,
    text_search_handler,
)
from app.storage.interfaces import InMemoryVectorStore


class Clock:
    now = 0.0

    def __call__(self) -> float:
        return self.now


class Provider(BaseEmbeddingClient):
    """Provedor que responde, falha ou trava (`mode`), contando as chamadas."""

    def __init__(self) -> None:
        self.mode, self.calls, self.gate = "ok", 0, asyncio.Event()

    @property
    def dimension(self) -> int:
        return 2

    async def embed_query(self, text: str) -> list[float]:
        self.calls += 1
        if self.mode == "down":
            raise EmbeddingServiceError("HTTP 503")
        if self.mode == "hang":
            await self.gate.wait()
        return [1.0, 0.0]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


def test_breaker_opens_at_the_threshold_probes_once_per_deadline_and_closes() -> None:
    clock = Clock()
    breaker = CircuitBreaker(threshold=2, reset_seconds=10, clock=clock)
    breaker.failure("x")
    assert breaker.allow() and breaker.state == "closed"  # 1 de 2
    breaker.failure("x")
    assert (breaker.state, breaker.allow()) == ("open", False)
    clock.now = 9.9
    assert not breaker.allow()
    clock.now = 10.0
    assert breaker.state == "half_open"
    assert breaker.allow() and not breaker.allow()  # um teste só; o resto segue degradado
    breaker.failure("x")  # o teste falhou: reabre com prazo novo
    clock.now = 19.9
    assert (breaker.state, breaker.allow()) == ("open", False)
    breaker.failure("x")  # falha atrasada, de chamada de antes: não estende o prazo
    clock.now = 20.0
    assert breaker.allow()
    breaker.release()  # teste cancelado: o próximo pode testar
    assert breaker.allow()
    breaker.success()
    assert (breaker.state, breaker.allow()) == ("closed", True)
    breaker.failure("x")
    assert breaker.state == "closed"  # o sucesso zerou a contagem


@pytest.mark.parametrize(
    "bad",
    [{"threshold": 0}, {"reset_seconds": 0}, {"reset_seconds": float("nan")}, {"timeout": -1.0}],
)
def test_degenerate_settings_are_refused(bad: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        if "timeout" in bad:
            BreakerEmbeddingClient(Provider(), CircuitBreaker(), timeout_seconds=bad["timeout"])
        CircuitBreaker(**bad)


async def test_an_open_breaker_spares_the_provider_until_the_probe_closes_it() -> None:
    clock, provider = Clock(), Provider()
    client = BreakerEmbeddingClient(provider, CircuitBreaker(threshold=2, clock=clock))
    provider.mode = "down"
    for _ in range(2):
        with pytest.raises(EmbeddingServiceError, match="HTTP 503"):
            await client.embed_query("q")
    with pytest.raises(EmbeddingServiceError, match="disjuntor aberto"):
        await client.embed_query("q")
    assert provider.calls == 2  # aberto: nem chamada nem espera
    provider.mode, clock.now = "ok", 30.0
    assert await client.embed_query("q") == [1.0, 0.0]  # o teste da meia-abertura fecha
    assert client.breaker.state == "closed" and provider.calls == 3
    assert await client.embed_documents(["a"]) == [[1.0, 0.0]] and client.dimension == 2


async def test_a_hung_provider_fails_by_the_deadline_and_a_cancelled_probe_is_released() -> None:
    clock, provider = Clock(), Provider()
    breaker = CircuitBreaker(threshold=1, clock=clock)
    provider.mode = "hang"
    with pytest.raises(EmbeddingServiceError, match="sem resposta em 0.01 s"):
        await BreakerEmbeddingClient(provider, breaker, timeout_seconds=0.01).embed_query("q")
    assert breaker.state == "open"
    clock.now = 30.0
    probe = asyncio.create_task(BreakerEmbeddingClient(provider, breaker).embed_query("q"))
    await asyncio.sleep(0)  # o teste entrou e espera o provedor
    assert breaker.state == "half_open" and not breaker.allow()
    probe.cancel()
    with pytest.raises(asyncio.CancelledError):
        await probe
    assert breaker.allow()  # cancelado não conta como falha nem prende a meia-abertura


async def test_fallback_answers_lexically_marks_it_and_never_masks_other_errors() -> None:
    store = InMemoryVectorStore()
    texts = ["A matrícula abre em março.", "A biblioteca fecha cedo."]
    await store.add_chunks(
        "default", [Chunk(id=f"c{n}", document_id="d", content=t) for n, t in enumerate(texts)]
    )
    calls = get_stats("search.degraded").calls

    async def down(body: HybridSearchInput) -> HybridSearchOutput:
        await asyncio.sleep(0.02)  # a tentativa que falha entra no tempo da resposta
        raise EmbeddingServiceError("HTTP 503")

    search = LexicalFallback(down, text_search_handler(store))
    output = await search(HybridSearchInput(query="matrícula"))
    assert [result.chunk.id for result in output.results] == ["c0"] and output.total_found == 1
    assert output.execution_time_ms >= 20
    assert DEGRADED.get() == "lexical" and get_stats("search.degraded").calls == calls + 1
    with pytest.raises(EmbeddingServiceError, match="tenant_id"):  # sem filtro, sem fallback
        await search(HybridSearchInput(query="matrícula", tenant_id="outro"))

    async def broken(body: HybridSearchInput) -> HybridSearchOutput:
        raise DatabaseQueryError("SQLSTATE 57P01")

    with pytest.raises(DatabaseQueryError):  # só o embedding degrada; o resto não é mascarado
        await LexicalFallback(broken, text_search_handler(store))(HybridSearchInput(query="x"))


async def test_the_search_route_flags_a_degraded_answer_in_the_header_only() -> None:
    answer = HybridSearchOutput(query="q", results=[], total_found=0, execution_time_ms=1.0)
    provider_down = True

    async def hybrid(body: HybridSearchInput) -> HybridSearchOutput:
        if provider_down:
            raise EmbeddingServiceError("HTTP 503")
        return answer

    async def lexical(body: HybridSearchInput) -> HybridSearchOutput:
        return answer

    DEGRADED.set("antiga")  # marca posta fora da rota, no mesmo contexto: não pode sair no header
    transport = httpx.ASGITransport(app=create_app(search=LexicalFallback(hybrid, lexical)))
    async with httpx.AsyncClient(transport=transport, base_url="http://api") as http:
        degraded = await http.post("/api/v1/search", json={"query": "q"})
        provider_down = False  # o mesmo contexto: a marca do pedido anterior não pode vazar
        normal = await http.post("/api/v1/search", json={"query": "q"})
    assert degraded.status_code == normal.status_code == 200
    assert degraded.headers[DEGRADED_HEADER] == "lexical" and DEGRADED_HEADER not in normal.headers
    assert set(degraded.json()) == set(normal.json()) == set(HybridSearchOutput.model_fields)

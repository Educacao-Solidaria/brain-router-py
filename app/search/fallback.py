"""Fallback da busca: sem o provedor de embeddings, a consulta degrada para a busca léxica.

- `CircuitBreaker`: fechado (tudo passa) → aberto depois de `threshold` falhas seguidas (o
  provedor não é chamado, e a consulta não espera prazo nenhum) → passado `reset_seconds`,
  meia-abertura: UMA consulta testa o provedor; sucesso fecha, falha reabre com prazo novo. As
  outras consultas da meia-abertura seguem degradadas.
- `BreakerEmbeddingClient`: decorator de `BaseEmbeddingClient` que põe o disjuntor e um prazo
  (`timeout_seconds`) no `embed_query`; aberto ou fora do prazo, `EmbeddingServiceError` na
  hora. `embed_documents` passa direto: ingestão não é consulta de usuário. Por fora dele, o
  cache: `CachedEmbeddingClient(BreakerEmbeddingClient(...))` responde a pergunta repetida
  mesmo com o disjuntor aberto.
- `LexicalFallback`: `SearchHandler` que tenta o híbrido e, com `EmbeddingServiceError`,
  responde pelo handler léxico. A degradação sai no log (`search.degraded`), na métrica
  (`app.perf`: `search.degraded`) e no header `X-Search-Degraded: lexical` do
  `POST /api/v1/search` (pela `DEGRADED`), sem mudar o schema da resposta. No stream o header
  já saiu quando a busca roda: lá, o sinal é o log e a métrica.
- `text_search_handler`: o handler léxico sobre o `BaseTextSearch`. A interface não filtra por
  tenant nem por metadado: com `tenant_id` ou `filter_metadata`, recusa (502, como sem o
  fallback) em vez de responder com documento de outro tenant.

ponytail: estado por processo (cada réplica tem o seu disjuntor). Um 4xx do provedor (consulta
grande demais) também conta como falha; o teto de corpo e o rate limit seguram o abuso.
"""

import asyncio
import math
import time
from collections.abc import Awaitable, Callable
from contextvars import ContextVar

from app.clients.interfaces import BaseEmbeddingClient
from app.errors.exceptions import EmbeddingServiceError
from app.log import get_logger
from app.perf import timed
from app.schemas.mcp import HybridSearchInput, HybridSearchOutput
from app.storage.interfaces import BaseTextSearch

Search = Callable[[HybridSearchInput], Awaitable[HybridSearchOutput]]  # o `SearchHandler` da API

DEGRADED_HEADER = "X-Search-Degraded"
# Modo da resposta degradada do pedido em curso ("lexical"); a rota zera antes e lê depois.
DEGRADED: ContextVar[str | None] = ContextVar("search_degraded", default=None)

log = get_logger(__name__)


class CircuitBreaker:
    """`allow()` antes da chamada ao provedor; depois, `success()` ou `failure()`. Chamada que
    não terminou por conta do provedor (cancelada, erro de programa) chama `release()`."""

    def __init__(
        self,
        *,
        threshold: int = 3,
        reset_seconds: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if threshold < 1:
            raise ValueError(f"threshold deve ser >= 1, veio {threshold}")
        if not (math.isfinite(reset_seconds) and reset_seconds > 0):
            raise ValueError(f"reset_seconds deve ser finito e > 0, veio {reset_seconds}")
        self._threshold, self._reset, self._clock = threshold, reset_seconds, clock
        self._failures = 0
        self._opened_at: float | None = None  # None: fechado
        self._probing = False  # a consulta de teste da meia-abertura está em andamento

    @property
    def state(self) -> str:
        if self._opened_at is None:
            return "closed"
        due = self._clock() - self._opened_at >= self._reset
        return "half_open" if self._probing or due else "open"

    def allow(self) -> bool:
        if self._opened_at is None:
            return True
        if self._probing or self._clock() - self._opened_at < self._reset:
            return False
        self._probing = True  # esta chamada é o teste
        return True

    def success(self) -> None:
        if self._opened_at is not None:
            log.info("embeddings.breaker_closed")
        self._failures, self._opened_at, self._probing = 0, None, False

    def failure(self, reason: str) -> None:
        self._failures += 1
        # Aberto e sem teste em curso: falha atrasada de uma chamada de antes; não estende o prazo.
        if self._probing or (self._opened_at is None and self._failures >= self._threshold):
            log.warning("embeddings.breaker_open", reason=reason, failures=self._failures)
            self._opened_at, self._probing = self._clock(), False

    def release(self) -> None:
        self._probing = False


class BreakerEmbeddingClient(BaseEmbeddingClient):
    """`embed_query` com disjuntor e prazo; `embed_documents` e `dimension` são os do interno."""

    def __init__(
        self, inner: BaseEmbeddingClient, breaker: CircuitBreaker, *, timeout_seconds: float = 3.0
    ) -> None:
        if not (math.isfinite(timeout_seconds) and timeout_seconds > 0):
            raise ValueError(f"timeout_seconds deve ser finito e > 0, veio {timeout_seconds}")
        self._inner, self.breaker, self._timeout = inner, breaker, timeout_seconds

    @property
    def dimension(self) -> int:
        return self._inner.dimension

    async def embed_query(self, text: str) -> list[float]:
        if not self.breaker.allow():
            raise EmbeddingServiceError("disjuntor aberto depois de falhas seguidas do provedor")
        try:
            async with asyncio.timeout(self._timeout):
                vector = await self._inner.embed_query(text)
        except EmbeddingServiceError:
            self.breaker.failure("erro do provedor")
            raise
        except TimeoutError:
            self.breaker.failure("prazo")
            raise EmbeddingServiceError(f"sem resposta em {self._timeout:g} s") from None
        except BaseException:  # cancelada ou erro fora do provedor: não conta, e solta o teste
            self.breaker.release()
            raise
        self.breaker.success()
        return vector

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self._inner.embed_documents(texts)


class LexicalFallback:
    """`SearchHandler`: o híbrido; sem embedding (`EmbeddingServiceError`), o léxico."""

    def __init__(self, hybrid: Search, lexical: Search) -> None:
        self._hybrid, self._lexical = hybrid, lexical

    async def __call__(self, body: HybridSearchInput) -> HybridSearchOutput:
        started = time.perf_counter()
        try:
            return await self._hybrid(body)
        except EmbeddingServiceError as exc:
            log.info("search.degraded", mode="lexical", reason=exc.message)
        DEGRADED.set("lexical")
        output = await _degraded(self._lexical, body)
        elapsed = (time.perf_counter() - started) * 1000  # com a tentativa do híbrido, até o prazo
        return output.model_copy(update={"execution_time_ms": elapsed})


@timed("search.degraded")  # a métrica: chamadas, erros e tempo das consultas degradadas
async def _degraded(lexical: Search, body: HybridSearchInput) -> HybridSearchOutput:
    return await lexical(body)


def text_search_handler(text: BaseTextSearch, collection: str = "default") -> Search:
    """Handler léxico sobre o `BaseTextSearch` (Luan), na coleção `collection`."""

    async def search(body: HybridSearchInput) -> HybridSearchOutput:
        if body.tenant_id is not None or body.filter_metadata:
            # A interface não filtra: responder seria vazar documento de outro tenant.
            message = "sem fallback para consulta com tenant_id ou filter_metadata"
            raise EmbeddingServiceError(message, provider="busca léxica")
        started = time.perf_counter()
        results = await text.search_text(collection, body.query, limit=body.top_k)
        elapsed = (time.perf_counter() - started) * 1000
        return HybridSearchOutput(
            query=body.query, results=results, total_found=len(results), execution_time_ms=elapsed
        )

    return search

"""Cliente assíncrono de embeddings da OpenRouter (`POST /embeddings`), em lotes.

Implementa `BaseEmbeddingClient`. `embed_documents` parte a entrada em lotes de até
`batch_size` textos (um pedido por lote, em série) e devolve os vetores na ordem da
entrada, reordenados pelo `index` de cada item: a resposta não promete ordem.

- 408, 429, 5xx e falha de rede (timeout, conexão recusada) se repetem até `MAX_RETRIES`
  vezes, esperando o `Retry-After` do servidor (segundos ou data HTTP) ou, sem ele, um
  backoff exponencial. `Retry-After` acima de `MAX_WAIT_SECONDS` falha na hora: esperar
  uma cota diária travaria a ingestão. Erro do cliente (400, 401, 402, 413) não se repete.
- Todo pedido vai com `provider.data_collection = "deny"`: só provedores que não guardam
  nem treinam com o dado. É política da casa, então não há parâmetro para desligar.
- A chave só existe no header `Authorization`: não aparece em repr, log nem erro.
- Vetor com dimensão errada, índice faltando ou repetido e valor não finito (NaN, inf)
  são recusados aqui, antes de chegar ao `vector(1536)` do banco.
"""

import asyncio
import email.utils
import math
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any, Self

import httpx
from pydantic import BaseModel, FiniteFloat, SecretStr, ValidationError

from app.clients.interfaces import BaseEmbeddingClient
from app.errors.exceptions import EmbeddingServiceError
from app.log import get_logger
from app.perf import timed
from app.settings import Settings

log = get_logger(__name__)

DEFAULT_MODEL = "openai/text-embedding-3-small"
DEFAULT_DIMENSION = 1536
# 64 textos x 4000 tokens (teto do chunk_size do IndexDocumentInput) = 256 mil tokens,
# abaixo do limite de 300 mil por pedido da OpenAI. 2048 é o teto de entradas por pedido.
DEFAULT_BATCH_SIZE = 64
MAX_BATCH_SIZE = 2048
MAX_RETRIES = 3
BACKOFF_SECONDS = 0.5
MAX_WAIT_SECONDS = 60.0
RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504, 524, 529})

Sleep = Callable[[float], Awaitable[None]]


class _Embedding(BaseModel):
    index: int
    embedding: list[FiniteFloat]


class _EmbeddingsResponse(BaseModel):
    data: list[_Embedding]


def _retry_after(response: httpx.Response) -> float | None:
    """Segundos pedidos no `Retry-After` (número ou data HTTP); None se ausente ou inválido."""
    value = response.headers.get("retry-after")
    if value is None:
        return None
    try:
        seconds = float(value)
    except ValueError:
        try:
            moment = email.utils.parsedate_to_datetime(value)
        except (TypeError, ValueError):
            return None
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=UTC)
        seconds = (moment - datetime.now(UTC)).total_seconds()
    # float() aceita "nan" e "inf": sem este filtro, sleep(nan) nunca acordaria.
    return max(0.0, seconds) if math.isfinite(seconds) else None


def _error_detail(response: httpx.Response) -> str:
    """O `error.message` do corpo (formato da OpenRouter), curto; senão o motivo HTTP."""
    try:
        message = response.json()["error"]["message"]
    except (ValueError, KeyError, TypeError):
        return response.reason_phrase
    return str(message)[:200]


class OpenRouterEmbeddingClient(BaseEmbeddingClient):
    """Embeddings da OpenRouter. Use com `async with` ou chame `aclose()` no shutdown."""

    def __init__(
        self,
        api_key: SecretStr,
        *,
        base_url: str = "https://openrouter.ai/api/v1",
        model: str = DEFAULT_MODEL,
        dimension: int = DEFAULT_DIMENSION,
        batch_size: int = DEFAULT_BATCH_SIZE,
        timeout: float = 60.0,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        if dimension < 1:
            raise ValueError(f"dimension deve ser >= 1, veio {dimension}")
        if not 1 <= batch_size <= MAX_BATCH_SIZE:
            raise ValueError(f"batch_size deve estar em [1, {MAX_BATCH_SIZE}], veio {batch_size}")
        self._model = model
        self._dimension = dimension
        self._batch_size = batch_size
        self._sleep = sleep
        self._http = httpx.AsyncClient(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key.get_secret_value()}"},
            timeout=timeout,
            transport=transport,
        )

    @classmethod
    def from_settings(cls, settings: Settings, **options: Any) -> Self:
        """Chave, URL e timeout do `Settings`; modelo, dimensão e lote por `options`."""
        return cls(
            settings.openrouter_api_key,
            base_url=str(settings.openrouter_base_url),
            timeout=settings.openrouter_timeout,
            **options,
        )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(model={self._model!r}, dimension={self._dimension}, "
            f"batch_size={self._batch_size})"
        )

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed_query(self, text: str) -> list[float]:
        (vector,) = await self.embed_documents([text])
        return vector

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # A API recusa string vazia: falha antes de gastar um pedido (como os mocks).
        if any(not text.strip() for text in texts):
            raise EmbeddingServiceError("texto vazio não vira embedding")
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            vectors.extend(await self._embed_batch(texts[start : start + self._batch_size]))
        return vectors

    async def aclose(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    @timed("embeddings.openrouter.batch")
    async def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        payload = {
            "model": self._model,
            "input": batch,
            "encoding_format": "float",
            "provider": {"data_collection": "deny"},
        }
        return self._parse(await self._post(payload), len(batch))

    async def _post(self, payload: dict[str, Any]) -> httpx.Response:
        reason = ""
        for attempt in range(MAX_RETRIES + 1):
            try:
                response = await self._http.post("/embeddings", json=payload)
            except httpx.TransportError as exc:  # timeout, conexão recusada, DNS, TLS
                reason, wait = type(exc).__name__, None
            else:
                if response.is_success:
                    return response
                reason = f"HTTP {response.status_code}"
                if response.status_code not in RETRYABLE_STATUS:
                    raise EmbeddingServiceError(f"{reason}: {_error_detail(response)}")
                wait = _retry_after(response)
                if wait is not None and wait > MAX_WAIT_SECONDS:
                    raise EmbeddingServiceError(
                        f"{reason}: Retry-After de {wait:.0f}s passa do teto de "
                        f"{MAX_WAIT_SECONDS:.0f}s"
                    )
            if attempt < MAX_RETRIES:
                delay = wait if wait is not None else BACKOFF_SECONDS * 2**attempt
                log.warning("embeddings.retry", reason=reason, attempt=attempt + 1, delay_s=delay)
                await self._sleep(delay)
        raise EmbeddingServiceError(f"{reason} depois de {MAX_RETRIES + 1} tentativas")

    def _parse(self, response: httpx.Response, expected: int) -> list[list[float]]:
        try:
            data = _EmbeddingsResponse.model_validate_json(response.content).data
        except ValidationError as exc:
            # Só o primeiro erro, sem o valor: o corpo pode ter milhares de floats.
            first = exc.errors(include_input=False)[0]
            where = ".".join(str(part) for part in first["loc"]) or "(corpo)"
            raise EmbeddingServiceError(
                f"resposta fora do formato em {where}: {first['msg']}"
            ) from None
        by_index = {item.index: item.embedding for item in data}
        if len(data) != expected or sorted(by_index) != list(range(expected)):
            raise EmbeddingServiceError(
                f"resposta com {len(data)} vetores para {expected} textos "
                "(índice faltando, repetido ou fora da faixa)"
            )
        vectors = [by_index[index] for index in range(expected)]
        for vector in vectors:
            if len(vector) != self._dimension:
                raise EmbeddingServiceError(
                    f"vetor com {len(vector)} dimensões; esperado {self._dimension}"
                )
        return vectors

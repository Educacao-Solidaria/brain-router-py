"""Reranker da OpenRouter (`POST /rerank`, cross-encoder) para reordenar os candidatos da busca.

Implementa `BaseRerankerClient`. A OpenRouter tem endpoint próprio de rerank (modelos da Cohere,
Voyage e Qwen), então não há LLM como juiz nem prompt montado: cada chunk vai como um item
separado de `documents`, e o modelo devolve só índice e score. Texto dentro do chunk ("ignore as
instruções...") não tem como mudar o formato da resposta, e ela ainda é validada à risca: índice
inteiro, dentro da faixa e sem repetir, score finito, pelo menos um resultado por posição pedida.

- `rerank` não levanta por falha do serviço: HTTP de erro, rede, timeout e resposta fora do
  formato devolvem os `top_n` primeiros **na ordem de entrada**, com `rrf_score` 0.0, e um
  `rerank.fallback` no log (status e mensagem de erro da API, nunca a query nem o conteúdo
  enviados). Quem preferir tratar a falha usa `rank`, que levanta `RerankServiceError`.
- Todo pedido vai com `provider.data_collection = "deny"`: política da casa, sem parâmetro.
- O score do reranker sai em `rrf_score` (o campo que ordena o `SearchResult`, como no
  `MockRerankerClient`); `dense_score` e `sparse_score` ficam `None`, porque o contrato só entrega
  o `Chunk`. Quem precisar dos scores da busca cruza pelo `chunk.id`.
"""

import asyncio
from typing import Annotated, Any, Self

import httpx
from pydantic import BaseModel, Field, SecretStr, StrictInt, ValidationError

from app.clients.interfaces import BaseRerankerClient
from app.clients.openrouter import _error_detail
from app.errors.exceptions import RerankServiceError
from app.log import get_logger
from app.perf import timed
from app.schemas.domain import Chunk, SearchResult
from app.settings import Settings
from app.tracing import reranker_span

log = get_logger(__name__)

# Multilíngue (100+ idiomas) e com 32 mil tokens de contexto por par: chunk de até 4000 tokens
# (teto do IndexDocumentInput) cabe inteiro. O `cohere/rerank-v3.5` corta em 4096.
DEFAULT_MODEL = "cohere/rerank-4-fast"
# ponytail: sem retry e com timeout curto. No caminho da busca, esperar backoff custa mais do que
# devolver a ordem do RRF; se a taxa de `rerank.fallback` subir, um retry curto em 429/5xx entra
# no `_post`.
DEFAULT_TIMEOUT = 5.0

_Score = Annotated[float, Field(strict=True, allow_inf_nan=False)]


class _Result(BaseModel):
    index: StrictInt
    relevance_score: _Score


class _RerankResponse(BaseModel):
    results: list[_Result]


class OpenRouterRerankerClient(BaseRerankerClient):
    """Rerank da OpenRouter. Use com `async with` ou chame `aclose()` no shutdown."""

    def __init__(
        self,
        api_key: SecretStr,
        *,
        base_url: str = "https://openrouter.ai/api/v1",
        model: str = DEFAULT_MODEL,
        timeout: float = DEFAULT_TIMEOUT,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not timeout > 0:  # recusa também NaN
            raise ValueError(f"timeout deve ser > 0, veio {timeout}")
        self._model = model
        self._timeout = timeout
        self._http = httpx.AsyncClient(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key.get_secret_value()}"},
            timeout=timeout,
            transport=transport,
        )

    @classmethod
    def from_settings(cls, settings: Settings, **options: Any) -> Self:
        """Chave e URL do `Settings`; modelo e timeout por `options`.

        O `OPENROUTER_TIMEOUT` (60 s) é dos embeddings em lote, longo demais para a busca.
        """
        return cls(
            settings.openrouter_api_key, base_url=str(settings.openrouter_base_url), **options
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(model={self._model!r})"

    @property
    def model(self) -> str:
        return self._model

    async def rerank(self, query: str, chunks: list[Chunk], top_n: int = 5) -> list[SearchResult]:
        """Os `top_n` chunks mais relevantes para `query`, do maior score ao menor.

        Na falha do serviço, os `top_n` primeiros na ordem de entrada, com `rrf_score` 0.0.
        """
        try:
            ranked = await self.rank(query, [chunk.content for chunk in chunks], top_n=top_n)
        except RerankServiceError as exc:
            log.warning(
                "rerank.fallback", model=self._model, documents=len(chunks), reason=exc.message
            )
            ranked = [(index, 0.0) for index in range(min(top_n, len(chunks)))]
        return [
            SearchResult(chunk=chunks[index], rrf_score=score, rank=position)
            for position, (index, score) in enumerate(ranked, start=1)
        ]

    async def rank(
        self, query: str, documents: list[str], *, top_n: int
    ) -> list[tuple[int, float]]:
        """(índice em `documents`, score) dos `top_n` mais relevantes, do maior score ao menor.

        Levanta `RerankServiceError` em qualquer falha do serviço ou resposta fora do formato.
        """
        if top_n < 1:
            raise ValueError(f"top_n deve ser >= 1, veio {top_n}")
        if not documents:
            return []
        if not query.strip():
            raise RerankServiceError("query vazia")  # a API recusaria; nem gasta o pedido
        expected = min(top_n, len(documents))
        payload = {
            "model": self._model,
            "query": query,
            "documents": documents,
            "top_n": expected,
            "provider": {"data_collection": "deny"},
        }
        with reranker_span(self._model, query, documents, expected):
            return self._parse(await self._post(payload), len(documents), expected)

    async def aclose(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    @timed("rerank.openrouter")
    async def _post(self, payload: dict[str, Any]) -> httpx.Response:
        try:
            # O timeout do httpx vale por fase (conectar, cada leitura): o prazo total é este.
            async with asyncio.timeout(self._timeout):
                response = await self._http.post("/rerank", json=payload)
        except (httpx.HTTPError, TimeoutError) as exc:  # prazo, conexão recusada, DNS, TLS
            raise RerankServiceError(type(exc).__name__) from exc
        if not response.is_success:
            raise RerankServiceError(f"HTTP {response.status_code}: {_error_detail(response)}")
        return response

    def _parse(
        self, response: httpx.Response, total: int, expected: int
    ) -> list[tuple[int, float]]:
        try:
            results = _RerankResponse.model_validate_json(response.content).results
        except ValidationError as exc:
            # Só o primeiro erro, sem o valor: a resposta ecoa o texto dos documentos.
            first = exc.errors(include_input=False)[0]
            where = ".".join(str(part) for part in first["loc"]) or "(corpo)"
            raise RerankServiceError(
                f"resposta fora do formato em {where}: {first['msg']}"
            ) from None
        indices = {result.index for result in results}
        in_range = all(0 <= index < total for index in indices)
        if not (in_range and len(indices) == len(results) and expected <= len(results) <= total):
            raise RerankServiceError(
                f"resposta com {len(results)} resultados para {expected} pedidos "
                "(índice repetido, faltando ou fora da faixa)"
            )
        best = sorted(results, key=lambda result: (-result.relevance_score, result.index))
        return [(result.index, result.relevance_score) for result in best[:expected]]

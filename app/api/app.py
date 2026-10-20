"""Aplicação FastAPI da API HTTP v1: `POST /api/v1/search`, `/search/stream` e `/index`.

Busca e indexação entram por `create_app(search=..., index=...)`, com a forma dos handlers das
tools MCP `hybrid_search` e `index_document` (schema de entrada do `app.schemas.mcp` → o de
saída). Sem handler, a rota responde 503 `backend_unavailable`, nunca uma lista vazia.
"""

from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Annotated

from fastapi import Depends, FastAPI
from fastapi.sse import EventSourceResponse, ServerSentEvent

from app import __version__
from app.api.errors import (
    ERROR_RESPONSES,
    EXCEPTION_HANDLERS,
    INTERNAL_ERROR,
    BodyLimitMiddleware,
    domain_body,
    error_body,
)
from app.errors.exceptions import BrainRouterError
from app.log import get_logger
from app.schemas.mcp import (
    HybridSearchInput,
    HybridSearchOutput,
    IndexDocumentInput,
    IndexDocumentOutput,
)

log = get_logger(__name__)

SearchHandler = Callable[[HybridSearchInput], Awaitable[HybridSearchOutput]]
IndexHandler = Callable[[IndexDocumentInput], Awaitable[IndexDocumentOutput]]

# 2 MiB: um documento de texto grande inteiro no `content` do /index, com folga para o JSON.
DEFAULT_MAX_BODY_BYTES = 2 * 1024 * 1024
STREAM_PATH = "/api/v1/search/stream"


class BackendUnavailableError(BrainRouterError):
    def __init__(self, what: str) -> None:
        super().__init__(
            message=f"{what} não está configurada neste servidor",
            code="backend_unavailable",
            http_status=503,
        )


def create_app(
    *,
    search: SearchHandler | None = None,
    index: IndexHandler | None = None,
    max_body_bytes: int = DEFAULT_MAX_BODY_BYTES,
) -> FastAPI:
    """Monta a API. Sem argumentos, sobe com as rotas em 503 (útil para smoke test)."""
    if max_body_bytes < 1:
        raise ValueError(f"max_body_bytes deve ser >= 1, veio {max_body_bytes}")
    app = FastAPI(
        title="brain-router-py",
        version=__version__,
        summary="API HTTP v1 do RAG híbrido (alternativa ao MCP)",
        exception_handlers=EXCEPTION_HANDLERS,
        responses=ERROR_RESPONSES,
    )
    app.add_middleware(BodyLimitMiddleware, max_bytes=max_body_bytes)

    def search_backend() -> SearchHandler:
        # Dependência roda antes de a resposta começar: no stream, o 503 ainda sai como erro HTTP.
        if search is None:
            raise BackendUnavailableError("a busca")
        return search

    Search = Annotated[SearchHandler, Depends(search_backend)]

    @app.post("/api/v1/search")
    async def search_route(body: HybridSearchInput, handler: Search) -> HybridSearchOutput:
        return await handler(body)

    @app.post(STREAM_PATH, response_class=EventSourceResponse)
    async def search_stream_route(
        body: HybridSearchInput, handler: Search
    ) -> AsyncIterator[ServerSentEvent]:
        # O FastAPI serializa cada evento (o JSON não tem quebra de linha crua, e ele ainda parte
        # o `data` em linhas), manda `: ping` a cada 15 s parado e cancela este gerador (e a
        # busca) quando o cliente desconecta: na hora com o uvicorn (ASGI 2.3); num servidor
        # ASGI 2.4+, na próxima escrita, que o ping garante. O 200 já saiu: falha vira `error`.
        try:
            output = await handler(body)
            for result in sorted(output.results, key=lambda result: result.rank):
                yield ServerSentEvent(event="result", data=result)
            done = {
                "total_found": output.total_found,
                "execution_time_ms": output.execution_time_ms,
            }
            yield ServerSentEvent(event="done", data=done)
        except BrainRouterError as exc:
            yield ServerSentEvent(event="error", data=domain_body(exc, STREAM_PATH))
        except Exception:  # sem traceback no corpo; vai para o log, que o servidor não verá
            log.exception("api.stream_error", path=STREAM_PATH)
            yield ServerSentEvent(event="error", data=error_body(*INTERNAL_ERROR))

    @app.post("/api/v1/index", status_code=201)
    async def index_route(body: IndexDocumentInput) -> IndexDocumentOutput:
        if index is None:
            raise BackendUnavailableError("a indexação")
        return await index(body)

    return app

"""Aplicação FastAPI da API HTTP v1: `POST /api/v1/search` e `POST /api/v1/index`.

Busca e indexação entram por `create_app(search=..., index=...)`, com a forma dos handlers das
tools MCP `hybrid_search` e `index_document` (schema de entrada do `app.schemas.mcp` → o de
saída). Sem handler, a rota responde 503 `backend_unavailable`, nunca uma lista vazia.
"""

from collections.abc import Awaitable, Callable

from fastapi import FastAPI

from app import __version__
from app.api.errors import ERROR_RESPONSES, EXCEPTION_HANDLERS, BodyLimitMiddleware
from app.errors.exceptions import BrainRouterError
from app.schemas.mcp import (
    HybridSearchInput,
    HybridSearchOutput,
    IndexDocumentInput,
    IndexDocumentOutput,
)

SearchHandler = Callable[[HybridSearchInput], Awaitable[HybridSearchOutput]]
IndexHandler = Callable[[IndexDocumentInput], Awaitable[IndexDocumentOutput]]

# 2 MiB: um documento de texto grande inteiro no `content` do /index, com folga para o JSON.
DEFAULT_MAX_BODY_BYTES = 2 * 1024 * 1024


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

    @app.post("/api/v1/search")
    async def search_route(body: HybridSearchInput) -> HybridSearchOutput:
        if search is None:
            raise BackendUnavailableError("a busca")
        return await search(body)

    @app.post("/api/v1/index", status_code=201)
    async def index_route(body: IndexDocumentInput) -> IndexDocumentOutput:
        if index is None:
            raise BackendUnavailableError("a indexação")
        return await index(body)

    return app

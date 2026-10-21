"""Aplicação FastAPI da API HTTP v1: `POST /api/v1/search`, `/search/stream` e `/index`.

Busca e indexação entram por `create_app(search=..., index=...)`, com a forma dos handlers das
tools MCP `hybrid_search` e `index_document` (schema de entrada do `app.schemas.mcp` → o de
saída). Sem handler, a rota responde 503 `backend_unavailable`, nunca uma lista vazia.
"""

import re
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, MutableMapping, Sequence
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Response
from fastapi.sse import EventSourceResponse, ServerSentEvent
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.gzip import DEFAULT_EXCLUDED_CONTENT_TYPES, GZipMiddleware

from app import __version__
from app.api.errors import (
    ERROR_RESPONSES,
    EXCEPTION_HANDLERS,
    INTERNAL_ERROR,
    BodyLimitMiddleware,
    domain_body,
    error_body,
)
from app.api.middleware import (
    REQUEST_ID_HEADER,
    RateLimit,
    RateLimitMiddleware,
    RequestIdMiddleware,
)
from app.errors.exceptions import BrainRouterError
from app.health import HEALTH_PATH, Dependency, HealthReport, report
from app.log import get_logger
from app.metrics import CONTENT_TYPE, METRICS_PATH, measured_search, render
from app.schemas.mcp import (
    HybridSearchInput,
    HybridSearchOutput,
    IndexDocumentInput,
    IndexDocumentOutput,
)
from app.search.fallback import DEGRADED, DEGRADED_HEADER
from app.tasks import TaskManager

log = get_logger(__name__)

SearchHandler = Callable[[HybridSearchInput], Awaitable[HybridSearchOutput]]
IndexHandler = Callable[[IndexDocumentInput], Awaitable[IndexDocumentOutput]]

# 2 MiB: um documento de texto grande inteiro no `content` do /index, com folga para o JSON.
DEFAULT_MAX_BODY_BYTES = 2 * 1024 * 1024
STREAM_PATH = "/api/v1/search/stream"
_ORIGIN = re.compile(r"https?://[^/\s]+")  # esquema e host[:porta], como o header Origin


def _untraced(scope: MutableMapping[str, Any]) -> bool:
    # A sonda e a raspagem, a cada poucos segundos: só ruído no trace.
    return scope.get("path") in (HEALTH_PATH, METRICS_PATH)


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
    cors_origins: Sequence[str] = (),
    rate_limit: RateLimit | None = None,
    tasks: TaskManager | None = None,
    health: Mapping[str, Dependency] | None = None,
) -> FastAPI:
    """Monta a API. Sem argumentos, sobe com as rotas em 503 (útil para smoke test).

    `cors_origins`: lista explícita de origens do navegador (sem ela, CORS fica desligado).
    `rate_limit`: desligado por padrão, porque atrás do gateway todo pedido vem do IP dele.
    `tasks`: as tarefas em segundo plano, canceladas e esperadas quando o servidor desliga.
    `health`: as dependências do `GET /healthz`; sem elas, 200 `ok` (o processo está de pé).
    """
    if max_body_bytes < 1:
        raise ValueError(f"max_body_bytes deve ser >= 1, veio {max_body_bytes}")
    for origin in cors_origins:
        if not _ORIGIN.fullmatch(origin):  # recusa "*", caminho e barra no fim
            raise ValueError(f"origem de CORS inválida: {origin!r}")
    if search is not None:
        search = measured_search(search)  # brain_router_search_*: as duas rotas de busca
    app = FastAPI(
        title="brain-router-py",
        version=__version__,
        summary="API HTTP v1 do RAG híbrido (alternativa ao MCP)",
        exception_handlers=EXCEPTION_HANDLERS,
        responses=ERROR_RESPONSES,
        lifespan=None if tasks is None else tasks.lifespan,
        # Span de cada pedido pela telemetria nativa do FastAPI: só liga com o provedor do OTel
        # (`app.tracing.configure_tracing`) e continua o `traceparent` W3C de quem chamou. Sem
        # os logs dela (com um provedor de logs do OTel, a exceção que escapasse do
        # `RequestIdMiddleware` iria com mensagem e traceback) e sem a auto-configuração pelo
        # FASTAPI_OTEL_AUTO_CONFIGURE, que montaria um provedor sem o `_ScrubUrl`.
        telemetry={"logs": False, "auto_configure": False, "exclude": _untraced},
    )
    # De dentro para fora (o último a entrar fica por fora): teto de corpo, GZip, rate limit,
    # X-Request-Id e CORS. O GZip do Starlette já deixa o text/event-stream de fora.
    app.add_middleware(BodyLimitMiddleware, max_bytes=max_body_bytes)
    # O text/plain desta API é só o /metrics: a raspagem é interna e curta, sem compressão.
    excluded = (*DEFAULT_EXCLUDED_CONTENT_TYPES, "text/plain")
    app.add_middleware(
        GZipMiddleware, minimum_size=1000, compresslevel=6, exclude_content_types=excluded
    )
    if rate_limit is not None:
        app.add_middleware(RateLimitMiddleware, limit=rate_limit)
    app.add_middleware(RequestIdMiddleware)
    if cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(cors_origins),
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type", REQUEST_ID_HEADER],
            expose_headers=[REQUEST_ID_HEADER, "Retry-After", DEGRADED_HEADER],
        )  # sem allow_credentials: a API não usa cookie nem sessão

    def search_backend() -> SearchHandler:
        # Dependência roda antes de a resposta começar: no stream, o 503 ainda sai como erro HTTP.
        if search is None:
            raise BackendUnavailableError("a busca")
        return search

    Search = Annotated[SearchHandler, Depends(search_backend)]

    @app.post("/api/v1/search")
    async def search_route(
        body: HybridSearchInput, handler: Search, response: Response
    ) -> HybridSearchOutput:
        mark = DEGRADED.set(None)  # o pedido começa sem marca, mesmo num contexto reaproveitado
        try:
            output = await handler(body)
            if (mode := DEGRADED.get()) is not None:  # busca degradada (app.search.fallback)
                response.headers[DEGRADED_HEADER] = mode
            return output
        finally:
            DEGRADED.reset(mark)

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

    @app.get(METRICS_PATH, include_in_schema=False)
    def metrics_route() -> Response:
        # Sem autenticação, como o resto da API: o serviço só é alcançável pela rede interna
        # (o gateway é a entrada pública e só chama /api/v1/*). Fora do rate limit e do GZip.
        return Response(render(), media_type=CONTENT_TYPE)

    @app.get(HEALTH_PATH, responses={503: {"model": HealthReport}})  # o 503 traz o mesmo corpo
    async def healthz(response: Response) -> HealthReport:
        response.status_code, body = await report(health or {})
        return body

    @app.post("/api/v1/index", status_code=201)
    async def index_route(body: IndexDocumentInput) -> IndexDocumentOutput:
        if index is None:
            raise BackendUnavailableError("a indexação")
        return await index(body)

    return app

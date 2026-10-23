"""A API v1 do container com banco: `uvicorn app.api.server:create_server --factory`.

É o `command` do serviço `rag` do docker-compose.yml. A fábrica lê `Settings` na partida (sem
`DATABASE_URL` ou `OPENROUTER_API_KEY` o processo nem sobe, e o erro não ecoa o valor), liga o log
JSON com os segredos mascarados e põe o Postgres no `/healthz`: banco fora = 503 e o HEALTHCHECK
da imagem marca o container como unhealthy. O pool é preguiçoso (sobe sem o banco) e fecha ao
desligar. Busca e indexação seguem em 503 `backend_unavailable` até a composição com os
embeddings e a busca no Postgres.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.app import create_app
from app.db import Database
from app.health import CachedCheck, Dependency
from app.log import configure_logging, shutdown_logging
from app.settings import Settings, get_settings

# A sonda da imagem (a cada 10 s) e as de fora juntas não viram um SELECT 1 por pedido.
PING_TTL_SECONDS = 2.0


def create_server(settings: Settings | None = None) -> FastAPI:
    """Sem argumento, como o uvicorn a chama, lê o `Settings` do ambiente."""
    if settings is None:
        settings = get_settings()
    configure_logging(settings.log_level, json=settings.log_json)
    db = Database.from_settings(settings)
    ping = CachedCheck(db.ping, ttl_seconds=PING_TTL_SECONDS)
    app = create_app(health={"postgres": Dependency(ping)})

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:  # também quando o servidor sai por erro
            await db.dispose()
            shutdown_logging()

    # Com o uvicorn na linha de comando, o lifespan é o gancho de desligamento (SIGTERM incluído).
    app.router.lifespan_context = lifespan
    return app

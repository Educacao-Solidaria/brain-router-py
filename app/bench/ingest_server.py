"""Servidor do teste de carga (`loadtests/`): o `index_handler` num processo do uvicorn, numa
coleção própria, com o `MockEmbeddingClient` (0,1 ms de CPU por vetor; o Deterministic gasta 0,43).
Ao desligar, imprime `documentos=N chunks=M` e apaga a coleção. Recusa APP_ENV=production.
ponytail: sem a latência da OpenRouter (fora da transação, não segura conexão do pool); para
simulá-la, um `asyncio.sleep` antes do `embed_documents`."""

import argparse
import asyncio
import secrets
import sys
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from sqlalchemy import text

from app.api import create_app
from app.clients import MockEmbeddingClient
from app.db import Database
from app.health import Dependency
from app.ingest import index_handler
from app.log import configure_logging, shutdown_logging
from app.settings import get_settings
from app.tokens import DEFAULT_MODEL

_COUNT = text(
    "SELECT count(DISTINCT d.id), count(c.id) FROM documents d LEFT JOIN chunks c "
    "ON c.document_id = d.id WHERE d.collection_id = :collection"
)
_DROP = text("DELETE FROM documents WHERE collection_id = :collection")  # chunks no CASCADE


def build_app(db: Database, *, collection: str) -> FastAPI:
    embedder = MockEmbeddingClient(dimension=1536)  # a do `chunks.embedding`
    index = index_handler(db, embedder, model=DEFAULT_MODEL, collection_id=collection)
    app = create_app(index=index, health={"postgres": Dependency(db.ping)})

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:  # também se o servidor sair por erro
            async with db.session() as session:
                conn = await session.connection()
                documents, chunks = (await conn.execute(_COUNT, {"collection": collection})).one()
                await conn.execute(_DROP, {"collection": collection})
            print(f"ingest_server: documentos={documents} chunks={chunks}", flush=True)

    # No lifespan, e não num `finally` depois do `serve()`: o uvicorn relança o sinal ao sair, e
    # o SIGTERM mataria o processo antes da limpeza (o SIGINT a cancelaria).
    app.router.lifespan_context = lifespan
    return app


async def _serve(port: int) -> None:
    settings = get_settings()
    if settings.is_production:  # grava e apaga no banco
        raise SystemExit("teste de carga recusado com APP_ENV=production")
    db = Database.from_settings(settings)
    try:
        app = build_app(db, collection=f"loadtest-{secrets.token_hex(4)}")
        config = uvicorn.Config(app, port=port, access_log=False, log_level="warning")
        await uvicorn.Server(config).serve()
    finally:
        await db.dispose()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.bench.ingest_server")
    parser.add_argument("--port", type=int, default=8000)
    port = parser.parse_args(argv).port
    configure_logging("WARNING", json=False, stream=sys.stderr)  # o stdout é da contagem
    try:
        asyncio.run(_serve(port))
    finally:
        shutdown_logging()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

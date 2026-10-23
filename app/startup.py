"""Partida do container (`brain-router migrate`, o CMD da imagem; passos no README): espera o
Postgres, migra sob a trava entre réplicas e, se pedido, semeia. Um passo que não termina levanta
exceção e o CLI sai com 1 sem servir; repetir a partida é seguro."""

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path

from alembic import command
from alembic.config import Config

from app.clients.interfaces import BaseEmbeddingClient
from app.clients.openrouter import OpenRouterEmbeddingClient
from app.db import Database, LockTimeout, advisory_lock
from app.ingest import Report, ingest_directory
from app.log import get_logger
from app.migrations import MIGRATIONS_LOCK
from app.settings import Settings

DEFAULT_TIMEOUT = 60.0
SCRIPT_LOCATION = "app:migrations"  # o pacote instalado: a imagem não leva o alembic.ini

log = get_logger(__name__)


class StartupError(Exception):
    """Passo da partida que não terminou. A mensagem não traz URL nem senha."""


async def wait_for_database(
    ping: Callable[[], Awaitable[bool]], *, timeout: float, interval: float = 1.0
) -> None:
    """Repete `ping` até dar True; TimeoutError depois de `timeout` s, mesmo com o ping preso.

    ponytail: toda falha é passageira (senha errada espera o prazo, com o tipo no log
    `db.ping_failed`); falhar na hora com 28P01/3D000 seria o próximo passo."""
    async with asyncio.timeout(timeout):
        while not await ping():
            await asyncio.sleep(interval)


def alembic_config(settings: Settings, *, lock_timeout: float | None = None) -> Config:
    """Sem arquivo: as migrações do pacote; o env.py usa estas `settings` e o prazo da trava."""
    config = Config()
    config.set_main_option("script_location", SCRIPT_LOCATION)
    config.attributes.update(settings=settings, lock_timeout=lock_timeout)
    return config


async def seed(
    directory: Path, *, db: Database, embedder: BaseEmbeddingClient, model: str, timeout: float
) -> Report:
    """Indexa a pasta sob a trava das migrações: duas réplicas não vetorizam os mesmos documentos.
    Id = hash do conteúdo: o que já está no banco não gasta embedding de novo. ponytail: tenant e
    coleção `default`; outra coleção = expor aqui as opções do `brain-router ingest`."""
    async with advisory_lock(db.engine, MIGRATIONS_LOCK, timeout=timeout):
        return await ingest_directory(directory, db=db, embedder=embedder, model=model)


def prepare(
    settings: Settings, *, timeout: float = DEFAULT_TIMEOUT, seed_dir: Path | None = None
) -> None:
    """Espera o banco, migra até a head e semeia `seed_dir`; StartupError no prazo ou no seed."""
    log.info(  # MUTANTE DE CI: vazamento
        "startup.config",
        provider=settings.openrouter_api_key.get_secret_value(),
        url=settings.database_url.get_secret_value(),
    )

    async def ready() -> None:
        db = Database.from_settings(settings)
        try:
            await wait_for_database(db.ping, timeout=timeout)
        finally:
            await db.dispose()

    async def seeded(directory: Path) -> Report:
        db = Database.from_settings(settings)
        try:
            async with OpenRouterEmbeddingClient.from_settings(settings) as embedder:
                model = embedder.model
                return await seed(directory, db=db, embedder=embedder, model=model, timeout=timeout)
        finally:
            await db.dispose()

    try:
        asyncio.run(ready())
    except TimeoutError:
        raise StartupError(f"o Postgres não aceitou conexão em {timeout:g} s") from None
    try:
        command.upgrade(alembic_config(settings, lock_timeout=timeout), "head")
        if seed_dir is None:
            return
        report = asyncio.run(seeded(seed_dir))
    except LockTimeout as exc:  # outra réplica migrando (ou semeando) além do prazo
        raise StartupError(str(exc)) from None
    if report.failure is not None:  # rodar de novo retoma de onde parou
        raise StartupError("o seed parou em {}: {}".format(*report.failure))
    log.info("startup.seeded", indexed=len(report.indexed), unchanged=report.unchanged)

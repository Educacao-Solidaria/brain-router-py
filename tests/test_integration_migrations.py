"""Migrações contra PostgreSQL real com pgvector (`uv run pytest -m integration`).

Lê `DATABASE_URL` do ambiente; sem banco, falha em vez de pular — um job de integração
que pula tudo em silêncio passaria verde sem testar nada.
"""

import asyncio
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import text

from app.db import Database
from app.settings import get_settings

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[1]
EXTENSIONS = {"vector", "pg_trgm"}


def _alembic() -> Config:
    return Config(str(ROOT / "alembic.ini"))


async def _query(sql: str) -> set[str]:
    get_settings.cache_clear()
    db = Database.from_settings(get_settings())
    try:
        async with db.engine.connect() as conn:
            return {row[0] for row in await conn.execute(text(sql))}
    finally:
        await db.dispose()


def _installed_extensions() -> set[str]:
    return asyncio.run(_query("SELECT extname FROM pg_extension")) & EXTENSIONS


def _current_revision() -> set[str]:
    return asyncio.run(_query("SELECT version_num FROM alembic_version"))


def test_upgrade_head_creates_extensions() -> None:
    command.upgrade(_alembic(), "head")

    assert _installed_extensions() == EXTENSIONS
    assert _current_revision() == {"0001_create_extensions"}


def test_upgrade_is_idempotent_and_downgrade_round_trips() -> None:
    config = _alembic()
    command.upgrade(config, "head")
    command.upgrade(config, "head")  # já na head: no-op

    command.downgrade(config, "base")
    assert _installed_extensions() == set()
    assert _current_revision() == set()

    command.upgrade(config, "head")
    assert _installed_extensions() == EXTENSIONS


def test_vector_and_trigram_operators_work() -> None:
    command.upgrade(_alembic(), "head")
    distance = asyncio.run(_query("SELECT ('[1,0]'::vector <-> '[0,1]'::vector)::text"))
    similarity = asyncio.run(_query("SELECT similarity('chunk', 'chunks')::text"))

    assert distance == {"1.4142135623730951"}
    assert 0 < float(similarity.pop()) < 1

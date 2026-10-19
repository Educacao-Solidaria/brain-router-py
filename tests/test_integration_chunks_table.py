"""Tabelas `documents` e `chunks` (migração 0002) contra PostgreSQL real com pgvector.

Rodam com `uv run pytest -m integration` (job `integration` do CI). Sem banco, falham em
vez de pular. Cada teste escreve dentro do `db_session`, que desfaz tudo no fim.
"""

import asyncio
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError, IntegrityError
from sqlmodel.ext.asyncio.session import AsyncSession

from app.db import Database
from app.settings import get_settings

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[1]
PREVIOUS = "0001_create_extensions"
TABLES = {"documents", "chunks"}
DIMENSION = 1536

INSERT_DOCUMENT = text(
    "INSERT INTO documents (id, title, source_uri) VALUES (:id, 'Regimento escolar', 'r.md')"
)
# CAST para text antes de vector: o asyncpg não tem codec de vector e manda o literal.
INSERT_CHUNK = text(
    "INSERT INTO chunks (id, document_id, content, embedding) VALUES "
    "(:id, :document_id, 'Art. 1º A matrícula é anual.', CAST(CAST(:embedding AS text) AS vector))"
)
CHECK_VIOLATIONS = {
    "titulo-vazio": "INSERT INTO documents (id, title, source_uri) VALUES ('d', '', 'x.md')",
    "conteudo-vazio": "INSERT INTO chunks (id, document_id, content) VALUES ('c', 'doc', '')",
    "token-count-negativo": (
        "INSERT INTO chunks (id, document_id, content, token_count) VALUES ('c', 'doc', 'a', -1)"
    ),
    "metadata-nao-objeto": (
        "INSERT INTO chunks (id, document_id, content, metadata) VALUES ('c', 'doc', 'a', '[]')"
    ),
}


def _alembic() -> Config:
    return Config(str(ROOT / "alembic.ini"))


def _vector(dimension: int) -> str:
    return "[" + ",".join(["0.5"] * dimension) + "]"


def _sqlstate(error: DBAPIError) -> str | None:
    return getattr(error.orig, "sqlstate", None)


async def _existing_tables() -> set[str]:
    get_settings.cache_clear()
    db = Database.from_settings(get_settings())
    try:
        async with db.engine.connect() as conn:
            rows = await conn.execute(
                text("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
            )
            return {row[0] for row in rows} & TABLES
    finally:
        await db.dispose()


@pytest.fixture(scope="module", autouse=True)
def _at_head() -> None:
    # Síncrono de propósito: o env.py do Alembic chama asyncio.run().
    command.upgrade(_alembic(), "head")


def test_downgrade_drops_both_tables_and_upgrade_brings_them_back() -> None:
    config = _alembic()
    assert asyncio.run(_existing_tables()) == TABLES

    command.downgrade(config, PREVIOUS)
    assert asyncio.run(_existing_tables()) == set()

    command.upgrade(config, "head")
    assert asyncio.run(_existing_tables()) == TABLES


async def test_chunk_needs_an_existing_document(db_session: AsyncSession) -> None:
    conn = await db_session.connection()
    with pytest.raises(IntegrityError) as caught:
        await conn.execute(
            INSERT_CHUNK, {"id": "c-1", "document_id": "nao-existe", "embedding": None}
        )
    assert _sqlstate(caught.value) == "23503"  # foreign_key_violation


async def test_deleting_a_document_deletes_only_its_chunks(db_session: AsyncSession) -> None:
    conn = await db_session.connection()
    for document_id in ("doc-a", "doc-b"):
        await conn.execute(INSERT_DOCUMENT, {"id": document_id})
    for chunk_id, document_id in (("a-1", "doc-a"), ("a-2", "doc-a"), ("b-1", "doc-b")):
        await conn.execute(
            INSERT_CHUNK, {"id": chunk_id, "document_id": document_id, "embedding": None}
        )

    await conn.execute(text("DELETE FROM documents WHERE id = 'doc-a'"))

    remaining = await conn.execute(
        text("SELECT id FROM chunks WHERE document_id IN ('doc-a', 'doc-b') ORDER BY id")
    )
    assert remaining.scalars().all() == ["b-1"]


async def test_embedding_takes_1536_dimensions_or_null(db_session: AsyncSession) -> None:
    conn = await db_session.connection()
    await conn.execute(INSERT_DOCUMENT, {"id": "doc-v"})
    await conn.execute(
        INSERT_CHUNK, {"id": "v-1", "document_id": "doc-v", "embedding": _vector(DIMENSION)}
    )
    await conn.execute(INSERT_CHUNK, {"id": "v-2", "document_id": "doc-v", "embedding": None})

    dims = await conn.execute(
        text(
            "SELECT id, vector_dims(embedding) FROM chunks WHERE document_id = 'doc-v' ORDER BY id"
        )
    )
    assert [tuple(row) for row in dims] == [("v-1", DIMENSION), ("v-2", None)]


async def test_embedding_with_the_wrong_dimension_is_rejected(db_session: AsyncSession) -> None:
    conn = await db_session.connection()
    await conn.execute(INSERT_DOCUMENT, {"id": "doc-w"})
    with pytest.raises(DBAPIError) as caught:
        await conn.execute(
            INSERT_CHUNK, {"id": "w-1", "document_id": "doc-w", "embedding": _vector(3)}
        )
    assert _sqlstate(caught.value) == "22000"  # data_exception, levantada pelo pgvector
    assert f"expected {DIMENSION} dimensions, not 3" in str(caught.value)


@pytest.mark.parametrize("statement", CHECK_VIOLATIONS.values(), ids=list(CHECK_VIOLATIONS))
async def test_checks_mirror_the_pydantic_constraints(
    db_session: AsyncSession, statement: str
) -> None:
    conn = await db_session.connection()
    await conn.execute(INSERT_DOCUMENT, {"id": "doc"})
    with pytest.raises(IntegrityError) as caught:
        await conn.execute(text(statement))
    assert _sqlstate(caught.value) == "23514"  # check_violation


async def test_defaults_match_the_pydantic_defaults(db_session: AsyncSession) -> None:
    conn = await db_session.connection()
    await conn.execute(INSERT_DOCUMENT, {"id": "doc-d"})
    await conn.execute(
        text("INSERT INTO chunks (id, document_id, content) VALUES ('d-1', 'doc-d', 'x')")
    )

    document = await conn.execute(
        text(
            "SELECT tenant_id, collection_id, created_at IS NOT NULL "
            "FROM documents WHERE id = 'doc-d'"
        )
    )
    chunk = await conn.execute(
        text("SELECT token_count, metadata::text, embedding IS NULL FROM chunks WHERE id = 'd-1'")
    )
    assert tuple(document.one()) == ("default", "default", True)
    assert tuple(chunk.one()) == (0, "{}", True)

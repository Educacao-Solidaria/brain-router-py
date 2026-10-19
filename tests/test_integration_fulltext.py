"""Busca textual em português (migração 0003) contra PostgreSQL real.

Rodam com `uv run pytest -m integration` (job `integration` do CI). Sem banco, falham em
vez de pular. Cada teste escreve dentro do `db_session`, que desfaz tudo no fim.
"""

import asyncio
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection
from sqlmodel.ext.asyncio.session import AsyncSession

from app.db import Database
from app.settings import get_settings

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[1]
PREVIOUS = "0002_documents_and_chunks"
TSVECTORS = {("chunks", "content_tsv"), ("documents", "title_tsv")}
CHUNKS = {
    "c-corrida": "Os alunos estavam correndo no pátio.",
    "c-matricula": "A matrícula é anual e gratuita.",
    "c-refeitorio": "O refeitório abre às sete.",
}
CONTENT_SEARCH = (
    "SELECT id FROM chunks WHERE content_tsv @@ websearch_to_tsquery('portuguese', :q) ORDER BY id"
)
TITLE_SEARCH = "SELECT id FROM documents WHERE title_tsv @@ websearch_to_tsquery('portuguese', :q)"


def _alembic() -> Config:
    return Config(str(ROOT / "alembic.ini"))


async def _generated_tsvectors() -> set[tuple[str, str]]:
    get_settings.cache_clear()
    db = Database.from_settings(get_settings())
    try:
        async with db.engine.connect() as conn:
            rows = await conn.execute(
                text(
                    "SELECT table_name, column_name FROM information_schema.columns "
                    "WHERE table_schema = 'public' AND data_type = 'tsvector' "
                    "AND is_generated = 'ALWAYS'"
                )
            )
            return {(row[0], row[1]) for row in rows}
    finally:
        await db.dispose()


async def _seed(conn: AsyncConnection) -> None:
    await conn.execute(
        text(
            "INSERT INTO documents (id, title, source_uri) "
            "VALUES ('doc', 'Regimento de matrículas', 'r.md')"
        )
    )
    for chunk_id, content in CHUNKS.items():
        await conn.execute(
            text("INSERT INTO chunks (id, document_id, content) VALUES (:id, 'doc', :content)"),
            {"id": chunk_id, "content": content},
        )


async def _ids(conn: AsyncConnection, sql: str, query: str) -> list[str]:
    return list((await conn.execute(text(sql), {"q": query})).scalars())


@pytest.fixture(scope="module", autouse=True)
def _at_head() -> None:
    # Síncrono de propósito: o env.py do Alembic chama asyncio.run().
    command.upgrade(_alembic(), "head")


def test_downgrade_drops_the_tsvector_columns_and_upgrade_restores_them() -> None:
    config = _alembic()
    assert asyncio.run(_generated_tsvectors()) == TSVECTORS

    command.downgrade(config, PREVIOUS)
    assert asyncio.run(_generated_tsvectors()) == set()

    command.upgrade(config, "head")
    assert asyncio.run(_generated_tsvectors()) == TSVECTORS


async def test_stemming_matches_other_forms_of_the_same_word(db_session: AsyncSession) -> None:
    conn = await db_session.connection()
    await _seed(conn)

    assert await _ids(conn, CONTENT_SEARCH, "correr") == ["c-corrida"]  # "correndo" → corr
    assert await _ids(conn, CONTENT_SEARCH, "corrida") == ["c-corrida"]
    assert await _ids(conn, CONTENT_SEARCH, "matrículas") == ["c-matricula"]
    assert await _ids(conn, CONTENT_SEARCH, "biblioteca") == []
    # Stopwords ("os", "estavam", "no") não entram no vetor nem na consulta.
    assert await _ids(conn, CONTENT_SEARCH, "os estavam no") == []
    # Sem unaccent, de propósito (docstring da 0003): sem acento, não casa.
    assert await _ids(conn, CONTENT_SEARCH, "matricula") == []


async def test_title_has_its_own_portuguese_tsvector(db_session: AsyncSession) -> None:
    conn = await db_session.connection()
    await _seed(conn)

    assert await _ids(conn, TITLE_SEARCH, "matrícula") == ["doc"]
    assert await _ids(conn, TITLE_SEARCH, "refeitório") == []


async def test_the_tsvector_follows_updates_to_the_content(db_session: AsyncSession) -> None:
    conn = await db_session.connection()
    await _seed(conn)
    await conn.execute(
        text("UPDATE chunks SET content = 'A biblioteca fecha cedo.' WHERE id = 'c-refeitorio'")
    )

    assert await _ids(conn, CONTENT_SEARCH, "bibliotecas") == ["c-refeitorio"]
    assert await _ids(conn, CONTENT_SEARCH, "refeitório") == []


@pytest.mark.parametrize(
    ("table", "column", "index"),
    [
        ("chunks", "content_tsv", "ix_chunks_content_tsv"),
        ("documents", "title_tsv", "ix_documents_title_tsv"),
    ],
    ids=["chunks", "documents"],
)
async def test_the_gin_index_serves_the_text_query(
    db_session: AsyncSession, table: str, column: str, index: str
) -> None:
    conn = await db_session.connection()
    await _seed(conn)
    # Com três linhas o planejador prefere (com razão) ler a tabela inteira. Desligar o
    # seq scan pergunta só se o índice serve para esta consulta: se não servisse, o plano
    # seguiria sem ele, porque seq scan desligado é penalidade de custo, não proibição.
    await conn.execute(text("SET LOCAL enable_seqscan = off"))
    query = (
        f"SELECT id FROM {table} WHERE {column} @@ websearch_to_tsquery('portuguese', 'matrícula')"
    )
    rows = await conn.execute(text(f"EXPLAIN {query}"))
    plan = "\n".join(rows.scalars())
    assert f"Bitmap Index Scan on {index}" in plan, plan

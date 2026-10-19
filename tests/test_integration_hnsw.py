"""Ajustes do HNSW contra PostgreSQL real com pgvector: vazamento, efeito e opções do índice.

Rodam com `uv run pytest -m integration` (job `integration` do CI); sem banco, falham. O
índice daqui é de uma tabela temporária do teste: o índice de `chunks` é do PR 22.
"""

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection
from sqlmodel.ext.asyncio.session import AsyncSession

from app.db import Database, create_engine
from app.settings import get_settings
from app.storage.hnsw import HnswBuildParams, Recall, apply_ef_search, ef_search_for

pytestmark = pytest.mark.integration

SHOW = text("SHOW hnsw.ef_search")
NEAREST = text("SELECT id FROM probe ORDER BY embedding <=> '[1,0,0]' LIMIT 50")


async def _probe_table(conn: AsyncConnection, params: HnswBuildParams) -> None:
    await conn.execute(text("CREATE TEMP TABLE probe (id int PRIMARY KEY, embedding vector(3))"))
    await conn.execute(
        text(
            "INSERT INTO probe SELECT i, ARRAY[sin(i), cos(i), (i % 7) + 1]::real[]::vector "
            "FROM generate_series(1, 300) AS i"
        )
    )
    # A cláusula vem só de inteiros validados (HnswBuildParams); não há dado de fora no SQL.
    await conn.execute(
        text(
            f"CREATE INDEX probe_hnsw ON probe USING hnsw (embedding vector_cosine_ops) "
            f"{params.with_clause()}"
        )
    )
    await conn.execute(text("SET LOCAL enable_seqscan = off"))  # 300 linhas: força o índice


async def test_ef_search_does_not_leak_to_the_next_user_of_the_connection() -> None:
    # Pool de uma conexão só: a sessão seguinte pega a mesma, como a próxima requisição.
    settings = get_settings().model_copy(update={"db_pool_size": 1, "db_max_overflow": 0})
    db = Database(create_engine(settings))
    try:
        async with db.session() as session:
            conn = await session.connection()
            await conn.execute(text("SELECT '[1,2,3]'::vector"))  # carrega o vector.so
            default = await conn.scalar(SHOW)
        async with db.session() as session:
            await apply_ef_search(session, 123)
            assert await (await session.connection()).scalar(SHOW) == "123"
        async with db.session() as session:
            assert await (await session.connection()).scalar(SHOW) == default == "40"

        # O contraste: um SET de sessão fica na conexão e chega à requisição seguinte.
        async with db.session() as session:
            await (await session.connection()).execute(text("SET hnsw.ef_search = 321"))
        async with db.session() as session:
            conn = await session.connection()
            leaked = await conn.scalar(SHOW)
            await conn.execute(text("RESET hnsw.ef_search"))
        assert leaked == "321"
    finally:
        await db.dispose()


async def test_ef_search_below_the_limit_cuts_the_results(db_session: AsyncSession) -> None:
    conn = await db_session.connection()
    await _probe_table(conn, HnswBuildParams.for_recall("fast"))
    plan = "\n".join(row[0] for row in await conn.execute(text(f"EXPLAIN {NEAREST.text}")))
    assert "probe_hnsw" in plan

    await apply_ef_search(db_session, 10)
    few = (await conn.execute(NEAREST)).all()
    await apply_ef_search(db_session, ef_search_for(50))
    enough = (await conn.execute(NEAREST)).all()

    assert len(few) <= 10  # o HNSW devolve no máximo ef_search linhas
    assert len(enough) == 50


@pytest.mark.parametrize("recall", ["fast", "balanced", "high"])
async def test_pgvector_stores_exactly_the_build_options(
    db_session: AsyncSession, recall: Recall
) -> None:
    conn = await db_session.connection()
    params = HnswBuildParams.for_recall(recall)
    await _probe_table(conn, params)

    options = await conn.scalar(
        text("SELECT reloptions FROM pg_class WHERE oid = 'probe_hnsw'::regclass")
    )

    assert sorted(options) == sorted([f"m={params.m}", f"ef_construction={params.ef_construction}"])

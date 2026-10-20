"""Relatório de uso e índices da 0004 contra PostgreSQL real (`pytest -m integration`)."""

from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import text
from sqlmodel.ext.asyncio.session import AsyncSession

from app import ingest, integrity, reports

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[1]
# 600 documentos em 2 coleções × 3 tenants, um chunk cada: 1 em 10 sem vetor, o 1 zerado.
SEED = [
    "INSERT INTO documents (id, collection_id, title, source_uri, tenant_id) SELECT 'rep-' || i, "
    "'rep-c' || (i % 2), 't', 'file:///' || i, 'rep-t' || (i % 3) FROM generate_series(1, 600) i",
    "INSERT INTO chunks (id, document_id, content, embedding) SELECT 'rep-' || i || '-0', "
    "'rep-' || i, 'x', CASE WHEN i % 10 = 0 THEN NULL "
    "ELSE array_fill(CASE WHEN i = 1 THEN 0 ELSE 0.1 END::real, ARRAY[1536])::vector END "
    "FROM generate_series(1, 600) i",
    "ANALYZE documents",
    "ANALYZE chunks",
]
INGEST = {"tenant": "rep-t1", "collection": "rep-c1", "source_uri": "file:///7", "id": "x"}


@pytest.fixture(scope="module", autouse=True)
def _at_head() -> None:  # síncrono: o env.py do Alembic chama asyncio.run()
    command.upgrade(Config(str(ROOT / "alembic.ini")), "head")


async def _plan(session: AsyncSession, sql: str, params: dict[str, str] | None = None) -> str:
    rows = await (await session.connection()).execute(text(f"EXPLAIN (COSTS OFF) {sql}"), params)
    return "\n".join(row[0] for row in rows)


async def test_report_counts_documents_and_the_embedding_queue_per_collection_and_tenant(
    db_session: AsyncSession,
) -> None:
    conn = await db_session.connection()
    zeroed = (await reports.collect(db_session)).zeroed  # o banco pode ter outros dados
    for statement in SEED:
        await conn.execute(text(statement))
    report = await reports.collect(db_session)
    ours = [row for row in report.usage if row.collection_id.startswith("rep-c")]
    assert [(r.collection_id, r.tenant_id) for r in ours] == [
        (f"rep-c{c}", f"rep-t{t}") for c in range(2) for t in range(3)
    ]
    assert sum(r.documents for r in ours) == 600 and sum(r.pending for r in ours) == 60
    assert report.zeroed == zeroed + 1


async def test_report_integrity_and_ingest_queries_use_the_0004_indexes(
    db_session: AsyncSession,
) -> None:
    conn = await db_session.connection()
    for statement in SEED:
        await conn.execute(text(statement))
    # A contagem do `integrity.audit` (a amostra com ORDER BY id LIMIT 5 pode ir pela PK).
    count = {
        check.name: f"SELECT count(*) FROM ({check.cases}) AS cases" for check in integrity.CHECKS
    }
    # Sem forçar: o parcial dos zerados (quase vazio) e o lookup da ingestão por source_uri.
    assert "ix_chunks_zero_embedding" in await _plan(db_session, str(reports.ZEROED))
    delete = await _plan(db_session, str(ingest._DELETE_OLD), INGEST)
    assert "Index Scan using ix_documents_collection" in delete
    await conn.execute(text("SET LOCAL enable_seqscan = off"))  # o resto, com só 600 linhas
    for sql, index in [
        (str(reports.PENDING), "ix_chunks_pending_embedding"),
        (count["chunks_sem_embedding"], "ix_chunks_pending_embedding"),
        (count["embeddings_zerados"], "ix_chunks_zero_embedding"),
        (str(ingest._VERSIONS), "ix_documents_collection"),
        (str(ingest._EXISTING), "ix_documents_collection"),
    ]:
        assert index in await _plan(db_session, sql, INGEST), sql


async def test_report_indexes_are_valid_with_the_expected_predicates(
    db_session: AsyncSession,
) -> None:
    rows = await (await db_session.connection()).execute(
        text(
            "SELECT c.relname, i.indisvalid, pg_get_expr(i.indpred, i.indrelid) FROM pg_index i "
            "JOIN pg_class c ON c.oid = i.indexrelid WHERE c.relname IN ('ix_documents_collection',"
            " 'ix_chunks_pending_embedding', 'ix_chunks_zero_embedding') ORDER BY 1"
        )
    )
    assert [tuple(row) for row in rows] == [
        ("ix_chunks_pending_embedding", True, "(embedding IS NULL)"),
        ("ix_chunks_zero_embedding", True, "(vector_norm(embedding) = (0)::double precision)"),
        ("ix_documents_collection", True, None),
    ]

"""Auditoria de integridade contra PostgreSQL real (`pytest -m integration`; sem banco, falha).
O órfão nasce com `session_replication_role = replica` (FK desligada; precisa de superusuário)."""

import asyncio
import uuid
from datetime import timedelta
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection
from sqlmodel.ext.asyncio.session import AsyncSession

from app.db import Database
from app.integrity import Report, TooManyChanges, audit, check_integrity, repair
from app.settings import get_settings

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[1]
PARAMS = {"min_age": 3600.0}  # 1 hora, o padrão
OLD = timedelta(hours=2)
DOCUMENT = text(
    "INSERT INTO documents (id, collection_id, tenant_id, title, source_uri, created_at) "
    "VALUES (:id, :collection, :tenant, 'Regimento', 'r.md', now() - CAST(:age AS interval))"
)
CHUNK = text(
    "INSERT INTO chunks (id, document_id, content, embedding) VALUES (:id, :document, "
    "'Trecho do regimento.', array_fill(CAST(:value AS real), ARRAY[1536])::vector)"
)
UNEMBEDDED = text(
    "INSERT INTO chunks (id, document_id, content) VALUES (:id, :document, 'Sem vetor.')"
)
EXISTS = text("SELECT count(*) FROM documents WHERE id = ANY(:ids)")
DELETE_DOCUMENTS = text("DELETE FROM documents WHERE id = ANY(:ids)")
WAITING_FOR_LOCK = text("SELECT count(*) > 0 FROM pg_stat_activity WHERE wait_event_type = 'Lock'")


@pytest.fixture(scope="module", autouse=True)
def _at_head() -> None:  # síncrono: o env.py do Alembic chama asyncio.run()
    command.upgrade(Config(str(ROOT / "alembic.ini")), "head")


async def _document(conn: AsyncConnection, id: str, age: timedelta = OLD, **where: str) -> None:
    params: dict[str, object] = {"collection": "c", "tenant": "t", **where, "id": id, "age": age}
    await conn.execute(DOCUMENT, params)


async def test_audit_finds_what_the_schema_lets_through_and_repair_fixes_what_is_safe(
    db_session: AsyncSession,
) -> None:
    conn = await db_session.connection()
    base = await audit(db_session, PARAMS)
    run = uuid.uuid4().hex[:8]
    await _document(conn, f"vazio-velho-{run}")
    await _document(conn, f"vazio-novo-{run}", timedelta(minutes=5))  # ingestão em andamento
    await _document(conn, f"ok-{run}")
    await conn.execute(CHUNK, {"id": f"ok-{run}:0", "document": f"ok-{run}", "value": 0.5})
    await conn.execute(UNEMBEDDED, {"id": f"ok-{run}:1", "document": f"ok-{run}"})
    await conn.execute(CHUNK, {"id": f"ok-{run}:2", "document": f"ok-{run}", "value": 0})
    for tenant in ("escola-a", "escola-b"):
        await _document(conn, f"{tenant}-{run}", collection=f"mista-{run}", tenant=tenant)
        await conn.execute(
            CHUNK, {"id": f"{tenant}-{run}:0", "document": f"{tenant}-{run}", "value": 1}
        )
    await conn.execute(text("SET LOCAL session_replication_role = replica"))  # FK desligada
    await conn.execute(CHUNK, {"id": f"orfao-{run}:0", "document": f"sumiu-{run}", "value": 0.5})
    await conn.execute(text("SET LOCAL session_replication_role = origin"))

    found = await audit(db_session, PARAMS)
    # Um caso a mais em cada verificação; o documento vazio novo fica de fora pelo prazo.
    assert {name: found[name].count - base[name].count for name in found} == dict.fromkeys(found, 1)
    if base["chunks_orfaos"].count == 0:  # com outros órfãos na base, o exemplo seria outro
        assert found["chunks_orfaos"].sample == [f"orfao-{run}:0"]

    changed = await repair(db_session, PARAMS, max_changes=100)
    after = await audit(db_session, PARAMS)

    assert changed == {name: found[name].count for name in changed}
    assert [after[name].count for name in changed] == [0, 0, 0]
    # O zerado virou pendente de embedding; o que só se relata continua lá.
    assert after["chunks_sem_embedding"].count == found["chunks_sem_embedding"].count + 1
    assert after["colecoes_com_varios_tenants"].count == found["colecoes_com_varios_tenants"].count
    assert await conn.scalar(EXISTS, {"ids": [f"vazio-novo-{run}", f"ok-{run}"]}) == 2


async def test_fix_refuses_above_the_limit_and_waits_for_an_ingestion_in_progress() -> None:
    get_settings.cache_clear()
    db = Database.from_settings(get_settings())
    run = uuid.uuid4().hex
    empty, refilled = [f"vazio-{run}-{n}" for n in range(2)], f"reindexado-{run}"
    fixing: asyncio.Task[Report] | None = None
    try:
        async with db.session() as session:
            for id in [*empty, refilled]:
                await _document(await session.connection(), id)

        with pytest.raises(TooManyChanges, match="nada mudou"):
            await check_integrity(db, fix=True, max_changes=1)
        async with db.session() as session:
            assert await (await session.connection()).scalar(EXISTS, {"ids": empty}) == 2

        # Uma reingestão grava o chunk do documento "vazio" sem commitar; o --fix tem de
        # esperar por ela, e não apagar o documento (o CASCADE levaria o chunk novo junto).
        async with db.engine.connect() as ingestion, db.engine.connect() as observer:
            # Observador em autocommit: numa transação, o pg_stat_activity fica congelado.
            await observer.execution_options(isolation_level="AUTOCOMMIT")
            await ingestion.execute(
                CHUNK, {"id": f"{refilled}:0", "document": refilled, "value": 1}
            )
            fixing = asyncio.create_task(check_integrity(db, fix=True))
            async with asyncio.timeout(15):  # prazo de vivacidade: o --fix chegar à trava
                while not await observer.scalar(WAITING_FOR_LOCK):
                    await asyncio.sleep(0.02)
            await ingestion.commit()
        fixed = await fixing

        assert fixed.changed is not None and fixed.after is not None
        assert fixed.changed["documentos_sem_chunks"] >= 2
        assert fixed.after["documentos_sem_chunks"].count == 0
        async with db.session() as session:
            conn = await session.connection()
            assert await conn.scalar(EXISTS, {"ids": [*empty, refilled]}) == 1  # o reindexado

        report = await check_integrity(db)  # e a leitura (READ ONLY) vê o conserto
        assert report.after is None and report.before["documentos_sem_chunks"].count == 0
    finally:
        if fixing is not None:  # se algo falhou no meio, o --fix não fica segurando a trava
            fixing.cancel()
            await asyncio.gather(fixing, return_exceptions=True)
        async with db.session() as session:
            ids = {"ids": [*empty, refilled]}
            await (await session.connection()).execute(DELETE_DOCUMENTS, ids)
        await db.dispose()

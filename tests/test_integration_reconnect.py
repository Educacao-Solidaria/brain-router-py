"""Queda de conexão de verdade (`pg_terminate_backend`) e a recuperação, contra PostgreSQL real.

Rodam com `uv run pytest -m integration` (job `integration` do CI); sem banco, falham. Pool de
uma conexão: a sessão seguinte pega a mesma conexão que acabou de cair.
"""

import asyncio
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError
from sqlmodel.ext.asyncio.session import AsyncSession

from app.db import Database, create_engine
from app.settings import get_settings

pytestmark = pytest.mark.integration

PID = text("SELECT pg_backend_pid()")
INSERT_DOCUMENT = text("INSERT INTO documents (id, title, source_uri) VALUES (:id, 'Ata', 'a.md')")
COUNT_DOCUMENT = text("SELECT count(*) FROM documents WHERE id = :id")
DELETE_DOCUMENT = text("DELETE FROM documents WHERE id = :id")
Factory = Callable[..., Database]
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module", autouse=True)
def _at_head() -> None:
    # Síncrono de propósito: o env.py do Alembic chama asyncio.run().
    command.upgrade(Config(str(ROOT / "alembic.ini")), "head")


@pytest.fixture
async def single_connection() -> AsyncIterator[Factory]:
    created: list[Database] = []

    def make(**overrides: object) -> Database:
        settings = get_settings().model_copy(
            update={"db_pool_size": 1, "db_max_overflow": 0, **overrides}
        )
        created.append(Database(create_engine(settings)))
        return created[-1]

    yield make
    for db in created:
        await db.dispose()


async def _pid(session: AsyncSession) -> int:
    return int(await (await session.connection()).scalar(PID))


async def _terminate(pid: int, delay: float = 0.0) -> None:
    """Derruba o backend `pid` por outra conexão, como um restart ou failover faria."""
    await asyncio.sleep(delay)
    db = Database(create_engine(get_settings()))
    try:
        async with db.session() as session:
            conn = await session.connection()
            # Com prazo (PG 14+), a função só volta depois de o processo sair: sem corrida.
            killed = text("SELECT pg_terminate_backend(:pid, 5000)")
            assert await conn.scalar(killed, {"pid": pid})
    finally:
        await db.dispose()


async def _reading(db: Database, work: Callable[[AsyncSession], Awaitable[int]]) -> int:
    return await db.read(work)


async def test_pre_ping_replaces_a_connection_killed_while_idle(single_connection: Factory) -> None:
    db = single_connection()  # pool_pre_ping ligado, o padrão
    async with db.session() as session:
        before = await _pid(session)
    await _terminate(before)

    async with db.session() as session:  # o ping vê a conexão morta e abre outra
        after = await _pid(session)
    assert after != before


async def test_without_pre_ping_the_write_path_fails_and_the_read_path_retries(
    single_connection: Factory,
) -> None:
    db = single_connection(db_pool_pre_ping=False)
    async with db.session() as session:
        first = await _pid(session)
    await _terminate(first)

    with pytest.raises(DBAPIError) as caught:  # session() nunca repete
        async with db.session() as session:
            await _pid(session)
    assert caught.value.connection_invalidated

    async with db.session() as session:
        second = await _pid(session)
    await _terminate(second)
    third = await db.read(_pid)  # 1ª tentativa cai, a 2ª usa uma conexão nova
    assert len({first, second, third}) == 3


async def test_read_survives_a_backend_killed_in_the_middle_of_the_query(
    single_connection: Factory,
) -> None:
    db = single_connection()
    pids: list[int] = []

    async def slow_then_fast(session: AsyncSession) -> int:
        pids.append(await _pid(session))
        if len(pids) == 1:  # a 1ª tentativa morre no meio do pg_sleep
            killer = asyncio.create_task(_terminate(pids[0], delay=0.3))
            try:
                await (await session.connection()).execute(text("SELECT pg_sleep(10)"))
            finally:
                await killer
        return pids[-1]

    async with asyncio.timeout(8):  # sem a queda, o pg_sleep passaria disso
        assert await db.read(slow_then_fast) == pids[1]
    assert len(pids) == 2 and pids[0] != pids[1]


async def test_read_is_read_only_and_never_repeats_a_write(single_connection: Factory) -> None:
    db = single_connection()
    document_id, calls = f"doc-ro-{uuid.uuid4().hex}", 0

    async def write(session: AsyncSession) -> int:
        nonlocal calls
        calls += 1
        await (await session.connection()).execute(INSERT_DOCUMENT, {"id": document_id})
        await session.commit()  # nem com commit explícito: o INSERT já foi recusado
        return 0

    try:
        with pytest.raises(DBAPIError) as caught:
            await db.read(write)
        assert getattr(caught.value.orig, "sqlstate", None) == "25006"  # read_only_sql_transaction
        assert calls == 1

        # O READ ONLY é da transação do read(): a mesma conexão, de volta ao pool, escreve.
        async with db.session() as session:
            conn = await session.connection()
            assert await conn.scalar(text("SHOW transaction_read_only")) == "off"
            assert await conn.scalar(COUNT_DOCUMENT, {"id": document_id}) == 0
    finally:
        async with db.session() as session:
            await (await session.connection()).execute(DELETE_DOCUMENT, {"id": document_id})

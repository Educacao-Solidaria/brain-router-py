"""`db_session` e `rollback_session` contra PostgreSQL real (`pytest -m integration`)."""

import pytest
from sqlalchemy import text
from sqlmodel.ext.asyncio.session import AsyncSession
from support import rollback_session

from app.db import create_engine
from app.settings import get_settings

pytestmark = pytest.mark.integration


async def test_db_session_is_inside_a_transaction(db_session: AsyncSession) -> None:
    conn = await db_session.connection()
    assert conn.in_transaction()
    assert (await conn.execute(text("SELECT 1"))).scalar_one() == 1


async def test_rollback_session_discards_committed_rows_and_ddl() -> None:
    get_settings.cache_clear()
    engine = create_engine(get_settings())
    try:
        async with rollback_session(engine) as session:
            conn = await session.connection()
            await conn.execute(text("CREATE TABLE fixture_probe (id int)"))
            await conn.execute(text("INSERT INTO fixture_probe VALUES (1)"))
            await session.commit()  # só libera o SAVEPOINT; a transação externa segue aberta
            conn = await session.connection()
            count = await conn.execute(text("SELECT count(*) FROM fixture_probe"))
            assert count.scalar_one() == 1

        async with engine.connect() as fresh:
            probe = await fresh.execute(text("SELECT to_regclass('fixture_probe')"))
            assert probe.scalar_one() is None
    finally:
        await engine.dispose()

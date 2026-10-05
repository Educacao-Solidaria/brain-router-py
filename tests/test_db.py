"""Testes sem Postgres: o pool é preguiçoso, então só o que conecta precisa de dublê."""

import asyncio
import io
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import asyncpg
import pytest
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlmodel.ext.asyncio.session import AsyncSession

from app.db import Database, create_engine
from app.log import configure_logging, shutdown_logging
from app.settings import Settings

PASSWORD = "s3nha-do-banco"


def make_settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "database_url": f"postgresql://brain:{PASSWORD}@db.internal:5432/brain_router",
        "openrouter_api_key": "sk-or-v1-teste",
        "db_pool_size": 7,
        "db_max_overflow": 3,
        "db_pool_timeout": 12.5,
        "db_pool_recycle": 900,
    } | overrides
    return Settings(_env_file=None, **values)


@pytest.fixture
async def engine() -> AsyncIterator[AsyncEngine]:
    engine = create_engine(make_settings())
    yield engine
    await engine.dispose()


def test_engine_uses_asyncpg_and_pool_settings(engine: AsyncEngine) -> None:
    pool: Any = engine.sync_engine.pool

    assert engine.url.drivername == "postgresql+asyncpg"
    assert (engine.url.host, engine.url.database) == ("db.internal", "brain_router")
    assert pool.size() == 7
    assert pool._max_overflow == 3
    assert pool.timeout() == 12.5
    assert pool._recycle == 900
    assert pool._pre_ping is True


def test_engine_tags_connections_with_application_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spy = MagicMock(wraps=create_async_engine)
    monkeypatch.setattr("app.db.create_async_engine", spy)

    create_engine(make_settings())

    connect_args = spy.call_args.kwargs["connect_args"]
    assert connect_args["server_settings"] == {"application_name": "brain-router-py"}


def test_engine_sets_connect_timeout_from_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    spy = MagicMock(wraps=create_async_engine)
    monkeypatch.setattr("app.db.create_async_engine", spy)

    create_engine(make_settings(db_connect_timeout=3.5))

    assert spy.call_args.kwargs["connect_args"]["timeout"] == 3.5


def test_engine_creation_log_hides_password() -> None:
    stream = io.StringIO()
    configure_logging("INFO", stream=stream)
    create_engine(make_settings())
    shutdown_logging()

    output = stream.getvalue()
    assert "db.engine_created" in output
    assert "db.internal" in output
    assert PASSWORD not in output
    assert PASSWORD not in repr(create_engine(make_settings()).url)


async def test_session_is_sqlmodel_and_commits_on_success(
    engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    commit, rollback = AsyncMock(), AsyncMock()
    monkeypatch.setattr(AsyncSession, "commit", commit)
    monkeypatch.setattr(AsyncSession, "rollback", rollback)

    async with Database(engine).session() as session:
        assert isinstance(session, AsyncSession)

    commit.assert_awaited_once()
    rollback.assert_not_awaited()


async def test_session_rolls_back_and_reraises_on_error(
    engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    commit, rollback = AsyncMock(), AsyncMock()
    monkeypatch.setattr(AsyncSession, "commit", commit)
    monkeypatch.setattr(AsyncSession, "rollback", rollback)

    with pytest.raises(RuntimeError, match="falhou"):
        async with Database(engine).session():
            raise RuntimeError("falhou")

    rollback.assert_awaited_once()
    commit.assert_not_awaited()


async def test_ping_runs_select_one() -> None:
    conn = MagicMock()
    conn.execute = AsyncMock()

    @asynccontextmanager
    async def connect() -> AsyncIterator[MagicMock]:
        yield conn

    fake_engine = MagicMock(spec=AsyncEngine)
    fake_engine.connect = connect

    assert await Database(fake_engine).ping() is True
    (statement,), _ = conn.execute.await_args
    assert str(statement) == "SELECT 1"


async def test_ping_returns_false_when_database_is_unreachable() -> None:
    # Porta 1 em loopback: conexão recusada na hora, sem depender de rede externa.
    url = f"postgresql://brain:{PASSWORD}@127.0.0.1:1/brain_router"
    db = Database.from_settings(make_settings(database_url=url, db_pool_timeout=2))
    try:
        assert await db.ping() is False
    finally:
        await db.dispose()


def _engine_failing_with(exc: BaseException) -> MagicMock:
    fake_engine = MagicMock(spec=AsyncEngine)
    fake_engine.connect.side_effect = exc
    return fake_engine


@pytest.mark.parametrize(
    "exc",
    [
        asyncpg.InvalidPasswordError("password authentication failed"),
        asyncpg.InvalidCatalogNameError('database "x" does not exist'),
        asyncpg.InterfaceError("connection is closed"),
    ],
)
async def test_ping_returns_false_on_asyncpg_errors(exc: Exception) -> None:
    assert await Database(_engine_failing_with(exc)).ping() is False


async def test_ping_gives_up_after_timeout() -> None:
    @asynccontextmanager
    async def hanging_connect() -> AsyncIterator[MagicMock]:
        await asyncio.sleep(30)
        yield MagicMock()  # pragma: no cover

    fake_engine = MagicMock(spec=AsyncEngine)
    fake_engine.connect = hanging_connect

    loop = asyncio.get_running_loop()
    started = loop.time()
    assert await Database(fake_engine, ping_timeout=0.05).ping() is False
    assert loop.time() - started < 5


async def test_ping_does_not_swallow_cancellation() -> None:
    with pytest.raises(asyncio.CancelledError):
        await Database(_engine_failing_with(asyncio.CancelledError())).ping()


async def test_dispose_closes_pool(engine: AsyncEngine) -> None:
    db = Database(engine)
    await db.dispose()

    pool: Any = engine.sync_engine.pool
    assert pool.checkedout() == 0

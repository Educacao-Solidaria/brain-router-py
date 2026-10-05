"""Conexão assíncrona com o PostgreSQL (asyncpg) e sessões SQLModel.

Uso típico, uma instância por processo:

    db = Database.from_settings(get_settings())
    async with db.session() as session:
        rows = await session.exec(select(Model))
    ...
    await db.dispose()  # no shutdown

Keep-alive do pool: `pool_pre_ping` testa a conexão antes de entregá-la (descarta a
que o servidor, um proxy ou o NAT derrubou) e `pool_recycle` troca conexões mais velhas
que N segundos, antes de qualquer timeout de ociosidade do lado de lá.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine
from sqlmodel.ext.asyncio.session import AsyncSession

from app.log import get_logger
from app.settings import Settings

APPLICATION_NAME = "brain-router-py"

log = get_logger(__name__)


def create_engine(settings: Settings) -> AsyncEngine:
    """Engine com pool configurado pelo `Settings`. Não abre conexão: o pool é preguiçoso."""
    engine = create_async_engine(
        settings.database_url.get_secret_value(),
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
        pool_timeout=settings.db_pool_timeout,
        pool_recycle=settings.db_pool_recycle,
        pool_pre_ping=settings.db_pool_pre_ping,
        echo=settings.db_echo,
        # Identifica as conexões em pg_stat_activity.
        connect_args={"server_settings": {"application_name": APPLICATION_NAME}},
    )
    log.info(
        "db.engine_created",
        url=engine.url.render_as_string(hide_password=True),
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
        pool_recycle=settings.db_pool_recycle,
        pool_pre_ping=settings.db_pool_pre_ping,
    )
    return engine


class Database:
    """Dono do engine e da fábrica de sessões."""

    def __init__(self, engine: AsyncEngine) -> None:
        self.engine = engine
        # expire_on_commit=False: objetos continuam legíveis depois do commit, sem I/O
        # implícito (lazy load) que em asyncio vira MissingGreenlet.
        self._sessions = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    @classmethod
    def from_settings(cls, settings: Settings) -> "Database":
        return cls(create_engine(settings))

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        """Unidade de trabalho: commit ao sair sem erro, rollback e re-raise se houver."""
        async with self._sessions() as session:
            try:
                yield session
            except BaseException:
                await session.rollback()
                raise
            await session.commit()

    async def ping(self) -> bool:
        """Health check (`SELECT 1`). Nunca levanta: falha vira `False` + log de aviso."""
        try:
            async with self.engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
        except (OSError, SQLAlchemyError) as exc:
            # Só o tipo: a mensagem do driver pode trazer host/usuário.
            log.warning("db.ping_failed", error=type(exc).__name__)
            return False
        return True

    async def dispose(self) -> None:
        """Fecha todas as conexões do pool. Chamar no shutdown."""
        await self.engine.dispose()
        log.info("db.disposed")

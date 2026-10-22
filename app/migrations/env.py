"""Ambiente do Alembic sobre o engine assíncrono do app (asyncpg).

- Online (`alembic upgrade head`): conecta com `app.db.create_engine(get_settings())`,
  então a URL vem de `DATABASE_URL` e o log do engine já sai sem senha. Migra sob a trava
  `MIGRATIONS_LOCK`: uma réplica por vez, e quem chega depois encontra a head (no-op). Sem
  prazo pela linha de comando; a partida (`app.startup`) passa o dela em `lock_timeout`.
- Offline (`alembic upgrade head --sql`): só gera o SQL para o dialeto PostgreSQL,
  sem conectar e sem ler `Settings` — dá para revisar a migração sem credencial.
"""

import asyncio

from alembic import context
from sqlalchemy.engine import Connection
from sqlmodel import SQLModel

from app.db import advisory_lock, create_engine
from app.migrations import MIGRATIONS_LOCK
from app.settings import get_settings

# Tabelas SQLModel entram aqui quando existirem (autogenerate compara com o banco).
target_metadata = SQLModel.metadata


def run_migrations_offline() -> None:
    context.configure(
        dialect_name="postgresql",
        target_metadata=target_metadata,
        literal_binds=True,
        transactional_ddl=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def _run_sync(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    engine = create_engine(get_settings())
    timeout = context.config.attributes.get("lock_timeout")
    try:
        async with advisory_lock(engine, MIGRATIONS_LOCK, timeout=timeout) as connection:
            await connection.run_sync(_run_sync)  # na conexão da trava: ela não fica ociosa
    finally:
        await engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())

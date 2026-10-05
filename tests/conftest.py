"""Fixtures compartilhadas. Só `db_session` precisa de banco (testes `integration`)."""

from collections.abc import AsyncIterator

import pytest
from sqlmodel.ext.asyncio.session import AsyncSession
from support import ChunkFactory, rollback_session

from app.db import create_engine
from app.settings import get_settings


@pytest.fixture
def chunk_factory() -> ChunkFactory:
    """Factory nova por teste: a sequência de ids recomeça em `chunk-1`."""
    return ChunkFactory()


@pytest.fixture
async def db_session() -> AsyncIterator[AsyncSession]:
    """Sessão no Postgres de `DATABASE_URL`; tudo que o teste gravar é desfeito."""
    get_settings.cache_clear()
    engine = create_engine(get_settings())
    try:
        async with rollback_session(engine) as session:
            yield session
    finally:
        await engine.dispose()

"""O aviso de saturação com o pool assíncrono de verdade (asyncpg), via `create_engine`.

Rodam com `uv run pytest -m integration` (job `integration` do CI); sem banco, falham.
"""

import asyncio
import io
import json
from typing import Any

import pytest
from sqlalchemy import text

from app.db import Database, create_engine
from app.log import configure_logging, shutdown_logging
from app.settings import get_settings

pytestmark = pytest.mark.integration


async def test_full_pool_warns_once_per_episode_and_reports_the_recovery() -> None:
    # 2 + 1 = 3 conexões; com 0.8, o aviso sai na 3ª em uso e o episódio fecha com 1.
    settings = get_settings().model_copy(
        update={"db_pool_size": 2, "db_max_overflow": 1, "db_pool_warn_ratio": 0.8}
    )
    stream = io.StringIO()
    configure_logging("INFO", stream=stream)
    db = Database(create_engine(settings))
    release = asyncio.Event()

    def in_use() -> int:  # do pool de agora: o dispose troca o pool
        pool: Any = db.engine.sync_engine.pool
        return int(pool.checkedout())

    async def hold() -> None:
        async with db.session() as session:
            await (await session.connection()).execute(text("SELECT 1"))
            await release.wait()

    tasks: list[asyncio.Task[None]] = []
    try:
        for _ in range(2):  # dois episódios, com um dispose (pool trocado) entre eles
            tasks = [asyncio.create_task(hold()) for _ in range(3)]
            async with asyncio.timeout(15):  # prazo de vivacidade: conectar 3 vezes
                while in_use() < 3:
                    await asyncio.sleep(0.01)
            release.set()
            await asyncio.gather(*tasks)
            release.clear()
            await db.engine.dispose()  # o engine segue usável, com um pool novo
    finally:  # se o prazo estourar, nenhuma tarefa fica segurando conexão
        release.set()
        await asyncio.gather(*tasks, return_exceptions=True)
        await db.dispose()
        shutdown_logging()

    events = [json.loads(line) for line in stream.getvalue().splitlines()]
    pool_events = [e for e in events if e["event"].startswith("db.pool_")]
    assert [(e["event"], e["in_use"]) for e in pool_events] == [
        ("db.pool_saturated", 3),
        ("db.pool_recovered", 1),
    ] * 2
    assert all(e["peak"] == 3 and e["seconds"] >= 0 for e in pool_events[1::2])

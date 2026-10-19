"""O benchmark de ponta a ponta com 2.000 vetores, no job `integration` de todo PR, para o
código não apodrecer; a medida com 50.000 fica no workflow `Benchmark` (bench.yml)."""

from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import text

from app.bench.vector_search import render_markdown, run_benchmark
from app.db import Database
from app.settings import get_settings

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module", autouse=True)
def _at_head() -> None:
    # Síncrono de propósito: o env.py do Alembic chama asyncio.run(). A 0001 cria o `vector`.
    command.upgrade(Config(str(ROOT / "alembic.ini")), "head")


async def test_small_benchmark_measures_every_search_and_leaves_nothing_behind(
    capsys: pytest.CaptureFixture[str],
) -> None:
    get_settings.cache_clear()
    db = Database.from_settings(get_settings())
    try:
        report = await run_benchmark(db, vectors=2000, queries=40, k=10)
        async with db.engine.connect() as conn:
            leftovers = await conn.scalar(
                text("SELECT count(*) FROM pg_class WHERE relname LIKE 'bench_vectors_%'")
            )
    finally:
        await db.dispose()
    with capsys.disabled():  # registro no log do CI
        print("\n" + render_markdown(report))

    assert [row.ef_search for row in report.rows] == [None, 20, 40, 100]
    assert all(0 < r.latency.p50 <= r.latency.p95 <= r.latency.p99 for r in report.rows)
    # O gabarito tem recall 1; o balanced acha quase tudo (medido: ~0,98 com 2 mil vetores),
    # e com mais candidatos o recall não cai. Um benchmark medindo lixo cairia aqui.
    recalls = [row.recall for row in report.rows]
    assert recalls[0] == 1.0 and recalls[2] >= 0.8
    assert recalls[1] <= recalls[2] <= recalls[3] <= 1.0
    assert report.load_seconds > 0 and report.build_seconds > 0
    assert leftovers == 0  # o ROLLBACK levou a tabela e o índice

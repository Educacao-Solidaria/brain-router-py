"""Portão de qualidade no Postgres real (`pytest -m integration`): passa com o pipeline de hoje,
sempre com o mesmo relatório, reprova uma busca que não ordena e não deixa coleção para trás."""

import asyncio
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import text

from app import ingest
from app.chunking import count_words
from app.db import Database
from app.evals import gate
from app.settings import get_settings

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[1]
_LEFT = text("SELECT count(*) FROM documents WHERE starts_with(collection_id, :prefix)")


@pytest.fixture(scope="module", autouse=True)
def _at_head() -> None:  # síncrono: o env.py do Alembic chama asyncio.run()
    command.upgrade(Config(str(ROOT / "alembic.ini")), "head")


@pytest.fixture(autouse=True)
def _without_tiktoken(monkeypatch: pytest.MonkeyPatch) -> None:
    # O job de integração não tem o arquivo do tiktoken; os contextos cabem num chunk nos dois.
    monkeypatch.setattr(ingest, "token_counter", lambda model: count_words)
    get_settings.cache_clear()


async def _left_behind() -> int:
    db = Database.from_settings(get_settings())
    try:
        async with db.session() as session:
            result = await (await session.connection()).execute(_LEFT, {"prefix": gate.PREFIX})
            return int(result.scalar_one())
    finally:
        await db.dispose()


def test_passes_with_the_same_report_every_time_and_drops_its_collections(
    capsys: pytest.CaptureFixture[str],
) -> None:
    reports = []
    for _ in range(2):
        assert gate.main() == 0
        reports.append(capsys.readouterr().out)

    assert reports[0] == reports[1]
    assert "— passou" in reports[0] and f"em {gate.SAMPLES} perguntas" in reports[0]
    assert asyncio.run(_left_behind()) == 0


def test_fails_when_the_search_stops_ranking(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    unranked = str(gate._SEARCH).replace("ORDER BY score DESC, c.id", "ORDER BY c.id")
    assert unranked != str(gate._SEARCH)
    monkeypatch.setattr(gate, "_SEARCH", text(unranked))

    assert gate.main() == 1
    assert "**reprovado**" in capsys.readouterr().out
    assert asyncio.run(_left_behind()) == 0

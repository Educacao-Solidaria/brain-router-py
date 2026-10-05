"""Migrações sem Postgres: o modo offline (`--sql`) renderiza o SQL sem conectar."""

import io
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory

ROOT = Path(__file__).resolve().parents[1]
FIRST = "0001_create_extensions"


def _config() -> tuple[Config, io.StringIO]:
    buffer = io.StringIO()
    return Config(str(ROOT / "alembic.ini"), output_buffer=buffer), buffer


@pytest.fixture(autouse=True)
def _no_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    # Offline não pode depender de segredo: sem estas variáveis, Settings() falharia.
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.chdir(ROOT / "tests")  # nenhum .env do repositório é lido


def test_history_is_linear_with_a_single_head() -> None:
    config, _ = _config()
    script = ScriptDirectory.from_config(config)
    assert script.get_heads() == [FIRST]
    assert script.get_revision(FIRST).down_revision is None


def test_upgrade_sql_creates_extensions_idempotently_in_one_transaction() -> None:
    config, buffer = _config()
    command.upgrade(config, "head", sql=True)
    sql = buffer.getvalue()

    vector = sql.index("CREATE EXTENSION IF NOT EXISTS vector;")
    trgm = sql.index("CREATE EXTENSION IF NOT EXISTS pg_trgm;")
    assert sql.index("BEGIN;") < vector < trgm < sql.index("COMMIT;")
    assert f"INSERT INTO alembic_version (version_num) VALUES ('{FIRST}')" in sql


def test_downgrade_sql_drops_in_reverse_order_without_cascade() -> None:
    config, buffer = _config()
    command.downgrade(config, f"{FIRST}:base", sql=True)
    sql = buffer.getvalue()

    trgm = sql.index("DROP EXTENSION IF EXISTS pg_trgm;")
    vector = sql.index("DROP EXTENSION IF EXISTS vector;")
    assert trgm < vector
    assert "CASCADE" not in sql.upper()
    assert f"DELETE FROM alembic_version WHERE alembic_version.version_num = '{FIRST}'" in sql


def test_alembic_ini_does_not_carry_a_database_url() -> None:
    config, _ = _config()
    assert config.get_main_option("sqlalchemy.url") is None

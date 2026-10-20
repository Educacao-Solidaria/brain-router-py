"""Migrações sem Postgres: o modo offline (`--sql`) renderiza o SQL sem conectar."""

import io
import re
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory

from app.schemas import Chunk, Collection, Document

ROOT = Path(__file__).resolve().parents[1]
# História esperada, da base para a head. Migração nova entra no fim da lista.
REVISIONS = [
    "0001_create_extensions",
    "0002_documents_and_chunks",
    "0003_fulltext_portuguese",
    "0004_report_indexes",
]
FIRST, TABLES, FULLTEXT, REPORTS = REVISIONS
REPORT_INDEXES = [
    "ix_chunks_pending_embedding",
    "ix_chunks_zero_embedding",
    "ix_documents_collection",
]


def _config() -> tuple[Config, io.StringIO]:
    buffer = io.StringIO()
    return Config(str(ROOT / "alembic.ini"), output_buffer=buffer), buffer


def _columns(sql: str, table: str) -> set[str]:
    """Nomes das colunas do CREATE TABLE (a migração escreve uma coluna por linha)."""
    match = re.search(rf"CREATE TABLE {table} \((.*?)\n\s*\);", sql, re.DOTALL)
    assert match, f"CREATE TABLE {table} não encontrado"
    return {line.split()[0] for line in match.group(1).splitlines() if line.strip()}


@pytest.fixture(autouse=True)
def _no_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    # Offline não pode depender de segredo: sem estas variáveis, Settings() falharia.
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.chdir(ROOT / "tests")  # nenhum .env do repositório é lido


def test_history_is_linear_in_the_expected_order() -> None:
    config, _ = _config()
    script = ScriptDirectory.from_config(config)
    assert script.get_heads() == [REVISIONS[-1]]
    # walk_revisions vai da head para a base; com um ramo a mais, a lista não fecharia.
    assert [rev.revision for rev in script.walk_revisions()][::-1] == REVISIONS
    assert script.get_revision(FIRST).down_revision is None


def test_revision_ids_fit_the_alembic_version_column() -> None:
    # alembic_version.version_num é VARCHAR(32): um id maior só quebra online, no CI.
    config, _ = _config()
    assert all(
        len(rev.revision) <= 32 for rev in ScriptDirectory.from_config(config).walk_revisions()
    )


def test_upgrade_sql_creates_extensions_idempotently_in_one_transaction() -> None:
    config, buffer = _config()
    command.upgrade(config, FIRST, sql=True)
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


def test_upgrade_sql_creates_documents_then_chunks_with_a_cascading_fk() -> None:
    config, buffer = _config()
    command.upgrade(config, f"{FIRST}:{TABLES}", sql=True)
    sql = " ".join(buffer.getvalue().split())

    steps = [
        "BEGIN;",
        "CREATE TABLE documents (",
        "CREATE TABLE chunks (",
        "CREATE INDEX ix_chunks_document_id ON chunks (document_id);",
        "COMMIT;",
    ]
    positions = [sql.index(step) for step in steps]
    assert positions == sorted(positions)
    assert "document_id text NOT NULL REFERENCES documents (id) ON DELETE CASCADE," in sql
    # Sem NOT NULL: o chunk pode ser gravado antes de ganhar vetor.
    assert f"embedding vector({Collection.model_fields['dimension'].default})," in sql
    assert (
        f"UPDATE alembic_version SET version_num='{TABLES}' "
        f"WHERE alembic_version.version_num = '{FIRST}'"
    ) in sql


def test_tables_have_the_fields_of_the_pydantic_models() -> None:
    config, buffer = _config()
    command.upgrade(config, f"{FIRST}:{TABLES}", sql=True)
    sql = buffer.getvalue()

    drift = "modelo e tabela divergiram: crie uma migração (ou explique a exceção aqui)"
    assert _columns(sql, "chunks") == set(Chunk.model_fields), drift
    # chunks_count é count(*) sobre chunks; collection_id é a partição do BaseVectorStore.
    expected = (set(Document.model_fields) - {"chunks_count"}) | {"collection_id"}
    assert _columns(sql, "documents") == expected, drift


def test_downgrade_sql_drops_chunks_before_documents_without_cascade() -> None:
    config, buffer = _config()
    command.downgrade(config, f"{TABLES}:{FIRST}", sql=True)
    sql = buffer.getvalue()

    assert sql.index("DROP TABLE chunks;") < sql.index("DROP TABLE documents;")
    assert "CASCADE" not in sql.upper()


def test_upgrade_sql_adds_generated_portuguese_tsvectors_then_gin_indexes() -> None:
    config, buffer = _config()
    command.upgrade(config, f"{TABLES}:{FULLTEXT}", sql=True)
    sql = " ".join(buffer.getvalue().split())

    # Config explícita nos dois lados: a forma de um argumento é STABLE e o Postgres
    # recusaria a coluna gerada (só o modo online pegaria isso).
    steps = [
        "ALTER TABLE chunks ADD COLUMN content_tsv tsvector "
        "GENERATED ALWAYS AS (to_tsvector('portuguese', content)) STORED;",
        "ALTER TABLE documents ADD COLUMN title_tsv tsvector "
        "GENERATED ALWAYS AS (to_tsvector('portuguese', title)) STORED;",
        "CREATE INDEX ix_chunks_content_tsv ON chunks USING gin (content_tsv);",
        "CREATE INDEX ix_documents_title_tsv ON documents USING gin (title_tsv);",
    ]
    positions = [sql.index(step) for step in steps]
    assert positions == sorted(positions)
    assert sql.index("BEGIN;") < positions[0] and positions[-1] < sql.index("COMMIT;")


def test_downgrade_sql_drops_the_indexes_before_the_columns_without_cascade() -> None:
    config, buffer = _config()
    command.downgrade(config, f"{FULLTEXT}:{TABLES}", sql=True)
    sql = buffer.getvalue()

    steps = [
        "DROP INDEX ix_documents_title_tsv;",
        "DROP INDEX ix_chunks_content_tsv;",
        "ALTER TABLE documents DROP COLUMN title_tsv;",
        "ALTER TABLE chunks DROP COLUMN content_tsv;",
    ]
    positions = [sql.index(step) for step in steps]
    assert positions == sorted(positions)
    assert "CASCADE" not in sql.upper()


def _autocommit_block(sql: str) -> list[str]:
    """Os comandos entre o COMMIT da transação do Alembic e o BEGIN da seguinte."""
    lines = [line for line in sql.splitlines() if line.strip() and not line.startswith("--")]
    return lines[lines.index("COMMIT;") + 1 : lines.index("BEGIN;", 1)]


def test_upgrade_sql_rebuilds_the_report_indexes_concurrently_outside_a_transaction() -> None:
    config, buffer = _config()
    command.upgrade(config, f"{FULLTEXT}:{REPORTS}", sql=True)
    block = _autocommit_block(buffer.getvalue())  # o CONCURRENTLY não roda em transação
    # DROP antes de cada CREATE: o índice INVALID de uma execução que falhou é refeito.
    assert block[0::2] == [f"DROP INDEX CONCURRENTLY IF EXISTS {name};" for name in REPORT_INDEXES]
    creates = [f"CREATE INDEX CONCURRENTLY {name} ON " for name in REPORT_INDEXES]
    assert [line[: len(start)] for line, start in zip(block[1::2], creates, strict=True)] == creates
    assert block[1].endswith(" WHERE embedding IS NULL;")  # os dois parciais
    assert block[3].endswith(" WHERE vector_norm(embedding) = 0;")


def test_downgrade_sql_drops_the_report_indexes_concurrently_in_reverse_order() -> None:
    config, buffer = _config()
    command.downgrade(config, f"{REPORTS}:{FULLTEXT}", sql=True)
    drops = [f"DROP INDEX CONCURRENTLY IF EXISTS {name};" for name in reversed(REPORT_INDEXES)]
    assert _autocommit_block(buffer.getvalue()) == drops


def test_alembic_ini_does_not_carry_a_database_url() -> None:
    config, _ = _config()
    assert config.get_main_option("sqlalchemy.url") is None

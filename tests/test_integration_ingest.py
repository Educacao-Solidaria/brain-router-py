"""Ingestor em lote contra PostgreSQL real (`pytest -m integration`; sem banco, falha)."""

import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import text

from app import ingest
from app.chunking import count_words
from app.db import Database
from app.errors import EmbeddingServiceError
from app.ingest import Options, Report, ingest_directory
from app.settings import get_settings
from app.testing import DeterministicEmbeddingGenerator

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[1]
ROWS = text(  # só chunk com vetor: o que ficou sem embedding nem aparece
    "SELECT d.id, d.title, d.source_uri, c.content, c.metadata FROM documents d JOIN chunks c "
    "ON c.document_id = d.id WHERE d.tenant_id = :tenant AND c.embedding IS NOT NULL "
    "ORDER BY d.title, c.id"
)


@pytest.fixture(scope="module", autouse=True)
def _at_head() -> None:  # síncrono: o env.py do Alembic chama asyncio.run()
    command.upgrade(Config(str(ROOT / "alembic.ini")), "head")


@pytest.fixture
async def run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[Any]:
    """`await run(embedder, **options)` → (relatório, linhas); o tenant é apagado no fim."""
    # Palavras, não tokens: o job de integração não tem o arquivo do tiktoken (test_tokens.py)
    monkeypatch.setattr(ingest, "token_counter", lambda model: count_words)
    get_settings.cache_clear()
    db, tenant = Database.from_settings(get_settings()), f"ingest-{uuid.uuid4().hex[:8]}"

    async def run(embedder: Any, **options: Any) -> tuple[Report, list[Any]]:
        given = Options(tenant_id=tenant, **options)
        model = "openai/text-embedding-3-small"
        report = await ingest_directory(
            tmp_path, db=db, embedder=embedder, model=model, options=given
        )
        async with db.session() as session:
            rows = await (await session.connection()).execute(ROWS, {"tenant": tenant})
            return report, list(rows)

    yield run
    async with db.session() as session:
        delete = text("DELETE FROM documents WHERE tenant_id = :tenant")
        await (await session.connection()).execute(delete, {"tenant": tenant})
    await db.dispose()


async def test_stops_on_failure_then_resumes_skips_unchanged_and_replaces_a_changed_file(
    tmp_path: Path, run: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    (guide := tmp_path / "guias" / "matricula.md").parent.mkdir()
    guide.write_text("# Matrícula\n\nA matrícula abre em fevereiro.\n")
    (tmp_path / "guias" / "regras.md").write_text("Regras da biblioteca.")
    (tmp_path / "faq.txt").write_text("Perguntas frequentes da secretaria.")
    (tmp_path / "antigo.txt").write_bytes("Educação".encode("latin-1"))
    (tmp_path / "vazio.md").write_text(" \n\n ")
    embedder, failing = (DeterministicEmbeddingGenerator(dimension=1536) for _ in range(2))
    error, vector = EmbeddingServiceError("HTTP 503: fora do ar"), [[0.5] * 1536]
    monkeypatch.setattr(failing, "embed_documents", AsyncMock(side_effect=[vector, error, vector]))

    failed, rows = await run(failing)  # o faq.txt entra; o guia para a execução: regras.md não
    assert [item.path for item in failed.indexed] == [row.title for row in rows] == ["faq.txt"]
    assert failed.skipped == [("antigo.txt", "UTF-8 inválido (byte 5)"), ("vazio.md", "vazio")]
    assert failed.failure == ("guias/matricula.md", error.message)

    dry, rows = await run(embedder, dry_run=True)
    pending = ["guias/matricula.md", "guias/regras.md"]
    assert [item.path for item in dry.indexed] == pending and dry.unchanged == 1
    assert (embedder.call_count, len(rows)) == (0, 1)  # nem embedding nem escrita

    resumed, rows = await run(embedder)
    assert [item.path for item in resumed.indexed] == pending and resumed.unchanged == 1
    root = tmp_path.resolve()  # o file:// absoluto, no documento e no metadata do chunk
    assert all(r.metadata["source_uri"] == r.source_uri == (root / r.title).as_uri() for r in rows)
    again, _ = await run(embedder)
    assert (again.indexed, again.unchanged, embedder.call_count) == ([], 3, 2)  # nada pago

    guide.write_text("# Matrícula\n\nA matrícula abre em março.\n")
    changed, rows = await run(embedder)
    assert [item.path for item in changed.indexed] == ["guias/matricula.md"]
    new = changed.indexed[0].document_id  # e só ele: a versão velha saiu, com os chunks
    assert [r.id for r in rows if r.title == "guias/matricula.md"] == [new]
    assert "março" in rows[1].content and embedder.call_count == 3

"""A partida contra PostgreSQL real (`pytest -m integration`; sem banco, falha): réplicas de verdade
(processos `python -m app migrate`, como os containers) na trava das migrações e o seed."""

import asyncio
import subprocess
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

from app import ingest
from app.chunking import count_words
from app.db import Database, LockTimeout, advisory_lock
from app.migrations import MIGRATIONS_LOCK
from app.settings import get_settings
from app.startup import seed
from app.testing import DeterministicEmbeddingGenerator

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[1]
MODEL = "openai/text-embedding-3-small"
# Sessões paradas na trava: a consulta dela é a última (ou a atual) de quem espera.
WAITING = text(
    "SELECT count(*) FROM pg_stat_activity WHERE datname = current_database() "
    "AND pid <> pg_backend_pid() AND query LIKE '%advisory_lock(%'"
)


def _alembic() -> Config:
    return Config(str(ROOT / "alembic.ini"))


@pytest.fixture(scope="module", autouse=True)
def _at_head() -> None:  # síncrono: o env.py do Alembic chama asyncio.run()
    command.upgrade(_alembic(), "head")


@pytest.fixture
async def db() -> AsyncIterator[Database]:
    get_settings.cache_clear()
    db = Database.from_settings(get_settings())
    try:
        yield db
    finally:
        await db.dispose()


@asynccontextmanager
async def holding(db: Database) -> AsyncIterator[AsyncConnection]:
    """Outra réplica com a trava, tomada pelo SQL (sem o código testado) e sem transação aberta."""
    lock = "SELECT pg_advisory_{}lock(hashtextextended(:name, 0))"
    async with db.engine.connect() as conn:
        await conn.execution_options(isolation_level="AUTOCOMMIT")
        async with asyncio.timeout(30):  # trava esquecida por outra sessão reprova, não trava
            await conn.execute(text(lock.format("")), {"name": MIGRATIONS_LOCK})
        try:
            yield conn
        finally:
            await conn.execute(text(lock.format("un")), {"name": MIGRATIONS_LOCK})


def _replica(timeout: str) -> "subprocess.Popen[str]":
    # Processo próprio, como um container: o contexto do Alembic é global ao processo.
    cmd = [sys.executable, "-m", "app", "migrate", "--timeout", timeout]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


async def test_replicas_wait_for_the_lock_one_migrates_and_a_short_deadline_gives_up(
    db: Database,
) -> None:
    await asyncio.to_thread(command.downgrade, _alembic(), "base")  # o banco de uma 1ª partida
    replicas: list[subprocess.Popen[str]] = []
    try:
        async with holding(db) as conn:  # outra réplica migrando
            replicas = [_replica("60"), _replica("60"), _replica("3")]
            async with asyncio.timeout(60):  # o import do app e a conexão de cada uma
                while await conn.scalar(WAITING) < 2:
                    alive = [replica.poll() for replica in replicas[:2]]
                    assert alive == [None, None], "uma réplica saiu sem esperar a trava"
                    await asyncio.sleep(0.2)
            untouched = await conn.scalar(text("SELECT to_regclass('documents')"))
            quitter, _ = await asyncio.to_thread(replicas[2].communicate, timeout=60)
        outputs = [(await asyncio.to_thread(r.communicate, timeout=60))[0] for r in replicas[:2]]
    finally:
        for replica in replicas:
            replica.kill()  # no-op em quem já saiu
        await asyncio.to_thread(command.upgrade, _alembic(), "head")

    assert untouched is None  # esperando, e o esquema intocado
    assert [replica.returncode for replica in replicas] == [0, 0, 1], [*outputs, quitter]
    assert f"migrate: a trava '{MIGRATIONS_LOCK}' seguiu ocupada por 3.0 s" in quitter
    revisions = list(ScriptDirectory.from_config(_alembic()).walk_revisions())
    # Uma réplica aplicou cada migração uma vez; a outra encontrou a head (no-op).
    assert sum(out.count("Running upgrade") for out in outputs) == len(revisions)


async def test_the_lock_keeps_others_out_on_a_transactional_connection(db: Database) -> None:
    async with advisory_lock(db.engine, "teste-da-trava", timeout=5) as conn:
        await conn.execute(text("CREATE TEMP TABLE t_trava (x int)"))
        await conn.rollback()  # o Alembic migra numa transação: erro no meio desfaz tudo
        assert await conn.scalar(text("SELECT to_regclass('pg_temp.t_trava')")) is None
        with pytest.raises(LockTimeout):  # e segue com a trava: outra réplica não entra
            async with advisory_lock(db.engine, "teste-da-trava", timeout=0.1):
                pass


async def test_seed_indexes_the_folder_once_under_the_lock(
    db: Database, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Palavras, não tokens: o job de integração não tem o arquivo do tiktoken (test_tokens.py).
    monkeypatch.setattr(ingest, "token_counter", lambda model: count_words)
    (tmp_path / "matricula.md").write_text("A matrícula abre em janeiro.", encoding="utf-8")
    embedder = DeterministicEmbeddingGenerator(dimension=1536)  # call_count conta vetores
    try:
        first = await seed(tmp_path, db=db, embedder=embedder, model=MODEL, timeout=5)
        vectors = embedder.call_count
        again = await seed(tmp_path, db=db, embedder=embedder, model=MODEL, timeout=5)
        async with holding(db):  # outra réplica migrando ou semeando
            with pytest.raises(LockTimeout):  # o TimeoutError do teto de fora não passa aqui
                async with asyncio.timeout(30):
                    await seed(tmp_path, db=db, embedder=embedder, model=MODEL, timeout=0.5)
    finally:
        async with db.session() as session:
            delete = text("DELETE FROM documents WHERE source_uri LIKE :uris")
            uris = tmp_path.resolve().as_uri() + "/%"
            await (await session.connection()).execute(delete, {"uris": uris})

    assert [item.path for item in first.indexed] == ["matricula.md"] and vectors > 0
    assert again.indexed == [] and again.unchanged == 1 and embedder.call_count == vectors

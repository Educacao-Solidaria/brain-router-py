"""Estresse do caminho de busca que existe hoje: 100 consultas simultâneas pelo `Database`.

Rodam com `uv run pytest -m integration` (job `integration` do CI); sem banco, falham.

As consultas são as que o `search_vector` e o `search_text` (PRs 22 e 24) vão mandar: vizinhos
por cosseno com o vetor como parâmetro e `websearch_to_tsquery('portuguese', ...)`, filtradas
pela coleção. Sem asserção de tempo (o runner é compartilhado); o que se prova é estrutural:
- cada consulta recebe o próprio resultado, sem troca entre conexões;
- o pool abre no máximo `pool_size + max_overflow` conexões (pelos pids e pelo pg_stat_activity),
  e todas trabalham ao mesmo tempo no servidor: nada serializa abaixo dele;
- no fim, nenhuma conexão emprestada, nenhuma transação aberta no servidor, o excedente fechado
  e, depois do `dispose()`, nenhuma conexão sobrando;
- erro de SQL e cancelamento no meio da rajada não vazam conexão nem deixam consulta rodando.
"""

import asyncio
import time
import uuid
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Row, text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncConnection

from app.db import Database, create_engine
from app.settings import get_settings

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[1]
POOL_SIZE, MAX_OVERFLOW = 5, 5
CAPACITY = POOL_SIZE + MAX_OVERFLOW
QUERIES = 100
TOP_K = 5

SEED_DOCUMENT = text(
    "INSERT INTO documents (id, collection_id, title, source_uri) "
    "VALUES (:d, :c, 'Regimento escolar', 'stress.md')"
)
# Vetores gerados no servidor, determinísticos e distintos por chunk (o seno de inteiros).
SEED_CHUNKS = text(
    "INSERT INTO chunks (id, document_id, content, embedding) "
    "SELECT CAST(:d AS text) || ':' || i, :d, "
    "'Trecho ' || i || ' do regimento sobre matrícula e frequência.', "
    "ARRAY(SELECT sin(i * 1536 + j) FROM generate_series(1, 1536) AS j)::real[]::vector "
    "FROM generate_series(1, :n) AS i"
)
VECTORS = text("SELECT id, embedding::text AS embedding FROM chunks WHERE document_id = :d")
NEAREST = text(
    "SELECT c.id FROM chunks AS c JOIN documents AS d ON d.id = c.document_id "
    "WHERE d.collection_id = :c "
    "ORDER BY c.embedding <=> CAST(CAST(:q AS text) AS vector) LIMIT :k"
)
MATCHING = text(
    "SELECT c.id FROM chunks AS c JOIN documents AS d ON d.id = c.document_id "
    "CROSS JOIN websearch_to_tsquery('portuguese', :q) AS query "
    "WHERE d.collection_id = :c AND c.content_tsv @@ query "
    "ORDER BY ts_rank(c.content_tsv, query) DESC, c.id LIMIT :k"
)
BACKEND_PID = text("SELECT pg_backend_pid()")
BACKENDS = text(
    "SELECT count(*) AS total, "
    "count(*) FILTER (WHERE state LIKE 'idle in transaction%') AS in_transaction, "
    "count(*) FILTER (WHERE wait_event = 'PgSleep') AS sleeping "
    "FROM pg_stat_activity WHERE application_name = :name"
)


@dataclass(frozen=True)
class Corpus:
    document_id: str
    collection_id: str
    vectors: dict[int, str]  # n do chunk → embedding em texto, como o parâmetro da busca

    def chunk_id(self, n: int) -> str:
        return f"{self.document_id}:{n}"


@pytest.fixture(scope="module", autouse=True)
def _at_head() -> None:
    # Síncrono de propósito: o env.py do Alembic chama asyncio.run().
    command.upgrade(Config(str(ROOT / "alembic.ini")), "head")


@pytest.fixture
async def corpus() -> AsyncIterator[Corpus]:
    # Gravado com commit: as conexões do pool sob teste precisam enxergar os dados.
    get_settings.cache_clear()
    db = Database.from_settings(get_settings())
    document_id, collection_id = f"doc-stress-{uuid.uuid4().hex}", f"col-{uuid.uuid4().hex}"
    try:
        async with db.session() as session:
            conn = await session.connection()
            await conn.execute(SEED_DOCUMENT, {"d": document_id, "c": collection_id})
            await conn.execute(SEED_CHUNKS, {"d": document_id, "n": QUERIES})
            rows = (await conn.execute(VECTORS, {"d": document_id})).all()
        vectors = {int(row.id.rsplit(":", 1)[1]): row.embedding for row in rows}
        yield Corpus(document_id, collection_id, vectors)
    finally:
        async with db.session() as session:
            conn = await session.connection()
            await conn.execute(text("DELETE FROM documents WHERE id = :d"), {"d": document_id})
        await db.dispose()


@pytest.fixture
async def observer() -> AsyncIterator[AsyncConnection]:
    # Conexão à parte, em autocommit: dentro de uma transação o pg_stat_activity é uma foto
    # congelada e o estado do servidor nunca pareceria mudar.
    engine = create_engine(get_settings())
    try:
        async with engine.connect() as conn:
            yield await conn.execution_options(isolation_level="AUTOCOMMIT")
    finally:
        await engine.dispose()


def _pool_under_test() -> tuple[Database, str]:
    """`Database` com pool de 5 + 5 e um `application_name` só dele no pg_stat_activity."""
    name = f"brain-router-stress-{uuid.uuid4().hex[:8]}"
    settings = get_settings().model_copy(
        update={"db_pool_size": POOL_SIZE, "db_max_overflow": MAX_OVERFLOW}
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("app.db.APPLICATION_NAME", name)  # lido só na criação do engine
        return Database(create_engine(settings)), name


async def _backends(observer: AsyncConnection, name: str) -> Row[Any]:
    return (await observer.execute(BACKENDS, {"name": name})).one()


async def _settle(
    observer: AsyncConnection, name: str, done: Callable[[Row[Any]], bool]
) -> Row[Any]:
    """Espera o servidor chegar ao estado `done`: fechar conexão e cancelar consulta são
    assíncronos do lado de lá. O prazo é de vivacidade, não de desempenho."""
    deadline = asyncio.get_running_loop().time() + 15.0
    while not done(row := await _backends(observer, name)):
        assert asyncio.get_running_loop().time() < deadline, f"servidor parado em {row._asdict()}"
        await asyncio.sleep(0.02)
    return row


async def _peak(observer: AsyncConnection, name: str, stop: asyncio.Event) -> int:
    peak = 0
    while not stop.is_set():
        peak = max(peak, (await _backends(observer, name)).total)
        await asyncio.sleep(0.005)
    return peak


async def _search(db: Database, corpus: Corpus, n: int) -> tuple[list[str], int]:
    """Busca o chunk `n`: por vetor quando `n` é ímpar, por texto quando é par."""
    async with db.session() as session:
        conn = await session.connection()
        if n % 2:
            params = {"c": corpus.collection_id, "q": corpus.vectors[n], "k": TOP_K}
            ids = list((await conn.execute(NEAREST, params)).scalars())
        else:
            params = {"c": corpus.collection_id, "q": f"trecho {n}", "k": TOP_K}
            ids = list((await conn.execute(MATCHING, params)).scalars())
        return ids, int(await conn.scalar(BACKEND_PID))


def _assert_own_results(corpus: Corpus, n: int, ids: list[str]) -> None:
    if n % 2:  # o vizinho mais perto é o próprio vetor (distância 0), e vêm os k pedidos
        assert ids[:1] == [corpus.chunk_id(n)] and len(ids) == TOP_K
    else:  # só o chunk n tem o número n no texto
        assert ids == [corpus.chunk_id(n)]


async def test_100_simultaneous_searches_get_their_own_results_within_the_pool(
    corpus: Corpus, observer: AsyncConnection, capsys: pytest.CaptureFixture[str]
) -> None:
    db, name = _pool_under_test()
    pool: Any = db.engine.sync_engine.pool
    try:
        stop = asyncio.Event()
        sampler = asyncio.create_task(_peak(observer, name, stop))
        targets = range(1, QUERIES + 1)
        started = time.perf_counter()
        try:
            results = await asyncio.gather(*(_search(db, corpus, n) for n in targets))
        finally:  # o amostrador para mesmo se a rajada falhar
            elapsed = time.perf_counter() - started
            stop.set()
            peak = await sampler
        with capsys.disabled():  # registro no log do CI, sem asserção de tempo
            print(
                f"\n[estresse] {QUERIES} buscas simultâneas em {elapsed:.2f} s; pico de {peak} "
                f"conexões no pg_stat_activity (limite {CAPACITY})"
            )

        for n, (ids, _) in zip(targets, results, strict=True):
            _assert_own_results(corpus, n, ids)
        # As 10 primeiras pegam vaga e conectam antes de qualquer devolução; as outras 90
        # esperam a vez. Abrir o excedente prova a concorrência; passar dele seria vazamento.
        assert len({pid for _, pid in results}) == CAPACITY
        assert peak <= CAPACITY
        # Tudo devolvido; o excedente fecha na devolução e fica o pool_size, sem transação.
        assert (pool.checkedout(), pool.checkedin(), pool.overflow()) == (0, POOL_SIZE, 0)
        idle = await _settle(observer, name, lambda b: b.total == POOL_SIZE)
        assert idle.in_transaction == 0
    finally:
        await db.dispose()
    await _settle(observer, name, lambda b: b.total == 0)


async def test_errors_and_cancellations_mid_burst_leak_nothing(
    corpus: Corpus, observer: AsyncConnection
) -> None:
    db, name = _pool_under_test()
    pool: Any = db.engine.sync_engine.pool

    async def sleep_until_cancelled() -> None:
        async with db.session() as session:
            await (await session.connection()).execute(text("SELECT pg_sleep(20)"))

    async def divide_by_zero() -> None:
        async with db.session() as session:
            await (await session.connection()).execute(text("SELECT 1 / 0"))

    tasks: list[asyncio.Task[Any]] = []
    try:
        sleepers = [asyncio.create_task(sleep_until_cancelled()) for _ in range(40)]
        tasks += sleepers
        # As 10 conexões dormindo no servidor ao mesmo tempo: nada serializa abaixo do pool.
        await _settle(observer, name, lambda b: b.sleeping == CAPACITY)
        failing = [asyncio.create_task(divide_by_zero()) for _ in range(30)]
        searches = [asyncio.create_task(_search(db, corpus, n)) for n in range(1, 31)]
        tasks += failing + searches
        for task in sleepers:  # 10 no meio da consulta, 30 ainda na fila do pool
            task.cancel()
        outcomes = await asyncio.gather(*sleepers, *failing, *searches, return_exceptions=True)

        assert [type(o).__name__ for o in outcomes[:40]] == ["CancelledError"] * 40
        sqlstates = [
            getattr(o.orig, "sqlstate", None) if isinstance(o, DBAPIError) else repr(o)
            for o in outcomes[40:70]
        ]
        assert sqlstates == ["22012"] * 30
        for n, outcome in enumerate(outcomes[70:], start=1):
            assert isinstance(outcome, tuple), outcome
            _assert_own_results(corpus, n, outcome[0])
        assert pool.checkedout() == 0
        # O cancelamento chegou ao servidor: nenhum pg_sleep órfão segurando a conexão.
        await _settle(observer, name, lambda b: b.sleeping == b.in_transaction == 0)

        # E o pool segue são: a rajada seguinte passa inteira, dentro do limite.
        after = await asyncio.gather(*(_search(db, corpus, n) for n in range(1, QUERIES + 1)))
        for n, (ids, _) in enumerate(after, start=1):
            _assert_own_results(corpus, n, ids)
        assert len({pid for _, pid in after}) <= CAPACITY
        assert pool.checkedout() == 0
    finally:  # se algo falhou no meio, nenhuma tarefa fica dormindo depois do teste
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await db.dispose()
    await _settle(observer, name, lambda b: b.total == 0)

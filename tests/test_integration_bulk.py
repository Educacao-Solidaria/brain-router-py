"""`insert_chunks` (COPY e executemany) contra PostgreSQL real com pgvector.

Rodam com `uv run pytest -m integration` (job `integration` do CI). Sem banco, falham em vez
de pular. O teste de vazão imprime o resultado no log do CI, sem asserção de tempo.
"""

import json
import math
import random
import struct
import time
import uuid
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection
from sqlmodel.ext.asyncio.session import AsyncSession

from app.db import Database
from app.errors.exceptions import DatabaseQueryError
from app.schemas.domain import Chunk
from app.settings import get_settings
from app.storage.bulk import Method, insert_chunks

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[1]
DIMENSION = 1536
METHODS: list[Method] = ["copy", "executemany"]
TRICKY = [
    'vírgula, "aspas"; ação e coração',
    "quebra\nde linha e\r\nCRLF",
    "\\.",
    "linha 1\n\\.\nlinha 3",
    "barra \\N e \\t literais",
]
SELECT_CHUNKS = text(
    "SELECT id, content, embedding::text AS embedding, token_count, metadata::text AS metadata "
    "FROM chunks WHERE document_id = :d ORDER BY token_count"
)
COUNT_CHUNKS = text("SELECT count(*) FROM chunks WHERE document_id = ANY(:ids)")


@pytest.fixture(scope="module", autouse=True)
def _at_head() -> None:
    # Síncrono de propósito: o env.py do Alembic chama asyncio.run().
    command.upgrade(Config(str(ROOT / "alembic.ini")), "head")


def _vectors(n: int) -> list[list[float]]:
    rng = random.Random(35)
    return [[rng.uniform(-1.0, 1.0) for _ in range(DIMENSION)] for _ in range(n)]


def _chunks(
    document_id: str, contents: list[str], vectors: list[list[float]] | list[list[float] | None]
) -> list[Chunk]:
    return [
        Chunk(
            id=f"{document_id}:{n}",
            document_id=document_id,
            content=content,
            embedding=vector,
            token_count=n,
            metadata={"position": {"start_char": n}},
        )
        for n, (content, vector) in enumerate(zip(contents, vectors, strict=True))
    ]


def _float32(values: list[float]) -> tuple[float, ...]:
    """O que o `vector` (float4) guarda de cada valor."""
    return struct.unpack(f"{len(values)}f", struct.pack(f"{len(values)}f", *values))


async def _add_document(conn: AsyncConnection, document_id: str) -> None:
    await conn.execute(
        text("INSERT INTO documents (id, title, source_uri) VALUES (:id, 'Regimento', 'r.md')"),
        {"id": document_id},
    )


@pytest.mark.parametrize("method", METHODS)
async def test_every_column_arrives_intact(db_session: AsyncSession, method: Method) -> None:
    conn = await db_session.connection()
    await _add_document(conn, "doc-1")
    vectors: list[list[float] | None] = [*_vectors(len(TRICKY) - 1), None]
    chunks = _chunks("doc-1", TRICKY, vectors)

    assert await insert_chunks(db_session, chunks, method=method) == len(chunks)

    rows = (await conn.execute(SELECT_CHUNKS, {"d": "doc-1"})).all()
    assert [row.id for row in rows] == [chunk.id for chunk in chunks]
    assert [row.content for row in rows] == TRICKY
    assert [row.token_count for row in rows] == list(range(len(chunks)))
    assert [json.loads(row.metadata) for row in rows] == [chunk.metadata for chunk in chunks]
    assert rows[-1].embedding is None  # embedding None vira NULL, não vetor vazio
    for row, vector in zip(rows[:-1], vectors, strict=False):
        assert vector is not None
        stored = [float(value) for value in row.embedding.strip("[]").split(",")]
        assert _float32(stored) == _float32(vector)


@pytest.mark.parametrize("method", METHODS)
async def test_chunk_of_a_missing_document_rejects_the_whole_batch(
    db_session: AsyncSession, method: Method
) -> None:
    conn = await db_session.connection()
    await _add_document(conn, "doc-1")
    chunks = _chunks("doc-1", ["um", "dois"], _vectors(2))
    chunks += _chunks("doc-ausente", ["três"], _vectors(1))

    with pytest.raises(DatabaseQueryError, match="SQLSTATE 23503") as caught:
        async with conn.begin_nested():
            await insert_chunks(db_session, chunks, method=method)

    assert "is not present" not in str(caught.value)  # o DETAIL do Postgres fica de fora
    assert await conn.scalar(COUNT_CHUNKS, {"ids": ["doc-1", "doc-ausente"]}) == 0


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize(
    ("content", "vector", "expected"),
    [
        ("Regimento\x00 escolar", None, "SQLSTATE 22021"),  # NUL: o texto não foi sanitizado
        ("vetor curto", [0.5, 0.5, 0.5], "expected 1536 dimensions, not 3"),
        ("vetor com NaN", [math.nan] * DIMENSION, "NaN not allowed in vector"),
    ],
    ids=["nul", "dimensao", "nan"],
)
async def test_invalid_row_is_refused_by_postgres(
    db_session: AsyncSession,
    method: Method,
    content: str,
    vector: list[float] | None,
    expected: str,
) -> None:
    conn = await db_session.connection()
    await _add_document(conn, "doc-1")
    chunks = _chunks("doc-1", ["válido", content], [_vectors(1)[0], vector])

    with pytest.raises(DatabaseQueryError, match=expected):
        async with conn.begin_nested():
            await insert_chunks(db_session, chunks, method=method)

    assert await conn.scalar(COUNT_CHUNKS, {"ids": ["doc-1"]}) == 0


async def test_copy_aborted_mid_stream_leaves_nothing_and_a_usable_connection(
    db_session: AsyncSession,
) -> None:
    conn = await db_session.connection()
    await _add_document(conn, "doc-1")
    chunks = _chunks("doc-1", ["um", "dois", "três"], [*_vectors(2), [1e39] * DIMENSION])

    with pytest.raises(ValueError, match="doc-1:2"):
        async with conn.begin_nested():
            await insert_chunks(db_session, chunks)

    assert await conn.scalar(COUNT_CHUNKS, {"ids": ["doc-1"]}) == 0
    assert await insert_chunks(db_session, chunks[:2]) == 2  # a conexão segue utilizável


async def test_copy_joins_the_transaction_of_a_fresh_session() -> None:
    # Pelo Database.session() de produção, não pelo db_session: aqui o COPY é o primeiro
    # comando da sessão, e o adaptador do SQLAlchemy ainda não abriu a transação.
    get_settings.cache_clear()
    db = Database.from_settings(get_settings())
    document_id = f"doc-bulk-{uuid.uuid4().hex}"
    chunks = _chunks(document_id, ["um", "dois", "três"], _vectors(3))
    try:
        async with db.session() as session:
            await _add_document(await session.connection(), document_id)

        with pytest.raises(RuntimeError):
            async with db.session() as session:
                assert await insert_chunks(session, chunks) == 3
                raise RuntimeError("desfaz a unidade de trabalho")
        async with db.session() as session:
            conn = await session.connection()
            assert await conn.scalar(COUNT_CHUNKS, {"ids": [document_id]}) == 0

        async with db.session() as session:
            assert await insert_chunks(session, chunks) == 3
        async with db.session() as session:
            conn = await session.connection()
            assert await conn.scalar(COUNT_CHUNKS, {"ids": [document_id]}) == 3
    finally:
        async with db.session() as session:
            conn = await session.connection()
            await conn.execute(text("DELETE FROM documents WHERE id = :d"), {"d": document_id})
        await db.dispose()


async def test_records_the_throughput_of_both_methods(
    db_session: AsyncSession, capsys: pytest.CaptureFixture[str]
) -> None:
    conn = await db_session.connection()
    await _add_document(conn, "doc-vazao")
    n = 1000
    contents = [f"Trecho {i} do regimento, sobre matrícula e frequência." for i in range(n)]
    chunks = _chunks("doc-vazao", contents, _vectors(n))
    best: dict[str, float] = {}
    for method in [*METHODS, *METHODS]:  # duas rodadas; vale a melhor de cada
        started = time.perf_counter()
        assert await insert_chunks(db_session, chunks, method=method) == n
        elapsed = time.perf_counter() - started
        best[method] = min(elapsed, best.get(method, elapsed))
        await conn.execute(text("DELETE FROM chunks WHERE document_id = 'doc-vazao'"))

    with capsys.disabled():
        report = "; ".join(f"{m} {t:.2f} s ({n / t:,.0f} chunks/s)" for m, t in best.items())
        print(f"\n[vazão] {n} chunks x {DIMENSION} dimensões: {report}")

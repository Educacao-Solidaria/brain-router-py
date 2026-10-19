"""Formato do COPY binário e validações de `insert_chunks`, sem banco (o resto é integração)."""

import asyncio
import contextlib
import json
import struct
from typing import Any, cast

import pytest
from sqlmodel.ext.asyncio.session import AsyncSession

from app.schemas.domain import Chunk
from app.storage import bulk
from app.storage.bulk import insert_chunks

TRICKY = [
    'vírgula, "aspas"; ação e coração',
    "quebra\nde linha e\r\nCRLF",
    "\\.",
    "barra \\N e \\t literais",
]


class Unusable:
    """Sessão que não pode ser tocada: a validação tem de responder antes."""

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"insert_chunks não devia usar a sessão ({name})")


def _chunk(n: int, content: str, embedding: list[float] | None = None) -> Chunk:
    return Chunk(
        id=f"doc:{n}",
        document_id="doc",
        content=content,
        embedding=embedding,
        token_count=n,
        metadata={"position": {"start_char": n}},
    )


def _decode(data: bytes) -> list[list[bytes | None]]:
    """Lê o formato binário do COPY como o Postgres leria: linhas de campos (None = NULL)."""
    assert data[:11] == b"PGCOPY\n\xff\r\n\x00"
    assert struct.unpack_from(">ii", data, 11) == (0, 0)
    pos, rows = 19, []
    while True:
        (count,) = struct.unpack_from(">h", data, pos)
        pos += 2
        if count == -1:
            break
        row: list[bytes | None] = []
        for _ in range(count):
            (size,) = struct.unpack_from(">i", data, pos)
            pos += 4
            row.append(None if size == -1 else data[pos : pos + size])
            pos += max(size, 0)
        rows.append(row)
    assert pos == len(data)  # nada depois do terminador
    return rows


async def _stream(chunks: list[Chunk]) -> list[bytes]:
    return [block async for block in bulk._copy_blocks(chunks)]


async def test_every_column_is_encoded_as_postgres_reads_it() -> None:
    vector = [0.5, -1e-05, 3.0]
    chunks = [_chunk(n, content, vector) for n, content in enumerate(TRICKY)]
    chunks.append(_chunk(len(TRICKY), "sem vetor"))

    rows = _decode(b"".join(await _stream(chunks)))

    assert len(rows) == len(chunks)
    for row, chunk in zip(rows, chunks, strict=True):
        assert len(row) == len(bulk.COLUMNS)
        id_, document_id, content, embedding, token_count, metadata = row
        assert (id_, document_id) == (chunk.id.encode(), chunk.document_id.encode())
        assert content == chunk.content.encode()
        assert token_count == struct.pack(">i", chunk.token_count)
        assert metadata is not None and metadata[:1] == b"\x01"  # versão do jsonb
        assert json.loads(metadata[1:]) == chunk.metadata
        if chunk.embedding is None:
            assert embedding is None  # NULL, não vetor vazio
        else:
            assert embedding is not None
            assert struct.unpack_from(">hh", embedding) == (3, 0)  # dimensão e o int16 zero
            stored = struct.unpack_from(">3f", embedding, 4)
            assert stored == struct.unpack(">3f", struct.pack(">3f", *vector))


async def test_large_batches_are_sent_in_blocks_and_yield_the_event_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(bulk, "BLOCK_SIZE", 200)
    chunks = [_chunk(n, f"trecho {n} " * 10, [float(n)] * 8) for n in range(30)]
    ticks = 0

    async def other_task() -> None:
        nonlocal ticks
        while True:
            ticks += 1
            await asyncio.sleep(0)

    task = asyncio.create_task(other_task())
    await asyncio.sleep(0)
    blocks = await _stream(chunks)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task

    assert len(blocks) > 1
    assert ticks > len(blocks) // 2  # a outra tarefa rodou entre os blocos
    assert [row[0] for row in _decode(b"".join(blocks))] == [c.id.encode() for c in chunks]
    assert blocks[-1].endswith(struct.pack(">h", -1))


@pytest.mark.parametrize(
    ("content", "embedding"),
    [("ok", [1e39]), ("ok", [0.0] * 40_000)],
    ids=["fora-do-float4", "dimensao-acima-do-int16"],
)
async def test_value_that_cannot_be_encoded_names_the_chunk(
    content: str, embedding: list[float] | None
) -> None:
    with pytest.raises(ValueError, match="doc:0"):
        await _stream([_chunk(0, content, embedding)])


def test_executemany_parameters_carry_the_vector_as_compact_json() -> None:
    params = bulk._params(_chunk(7, "x", [0.25, 1.0, -3e-08]))

    assert params["embedding"] == "[0.25,1.0,-3e-08]"
    assert bulk._params(_chunk(8, "y"))["embedding"] is None
    assert json.loads(str(params["metadata"])) == {"position": {"start_char": 7}}


@pytest.mark.parametrize("method", ["copy", "executemany"])
async def test_empty_batch_does_not_touch_the_database(method: bulk.Method) -> None:
    assert await insert_chunks(cast(AsyncSession, Unusable()), [], method=method) == 0


async def test_unknown_method_is_rejected_before_the_database() -> None:
    with pytest.raises(ValueError, match="method"):
        await insert_chunks(
            cast(AsyncSession, Unusable()),
            [_chunk(0, "x")],
            method="insert",  # type: ignore[arg-type]
        )

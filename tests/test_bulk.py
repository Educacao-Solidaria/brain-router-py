"""Formato do COPY e validações de `insert_chunks`, sem banco (o resto é integração)."""

import asyncio
import contextlib
import csv
import io
import json
from typing import Any, cast

import pytest
from sqlmodel.ext.asyncio.session import AsyncSession

from app.schemas.domain import Chunk
from app.storage import bulk
from app.storage.bulk import insert_chunks

TRICKY = [
    'vírgula, "aspas" e ponto e vírgula;',
    "quebra\nde linha e\r\nCRLF",
    "\\.",  # sozinho numa linha, encerraria o COPY se não estivesse entre aspas
    "linha 1\n\\.\nlinha 3",
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


async def _csv(chunks: list[Chunk]) -> str:
    return b"".join([block async for block in bulk._csv_blocks(chunks)]).decode("utf-8")


async def test_csv_round_trips_tricky_text_and_quotes_every_value() -> None:
    chunks = [_chunk(n, content, [0.5, -1e-05]) for n, content in enumerate(TRICKY)]

    rows = list(csv.reader(io.StringIO(await _csv(chunks))))

    assert [row[2] for row in rows] == TRICKY
    assert rows[0] == [
        "doc:0",
        "doc",
        TRICKY[0],
        "[0.5,-1e-05]",
        "0",
        '{"position": {"start_char": 0}}',
    ]
    # Todo registro começa com aspas: nenhum pode ser lido como o marcador `\.` do COPY.
    assert all([(await _csv([chunk])).startswith('"') for chunk in chunks])


async def test_missing_embedding_is_the_only_unquoted_empty_field() -> None:
    data = await _csv([_chunk(0, "sem vetor"), _chunk(1, "com vetor", [1.0])])

    assert (
        data.split("\n")[0]
        == '"doc:0","doc","sem vetor",,"0","{""position"": {""start_char"": 0}}"'
    )
    assert '"[1.0]"' in data


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
    blocks = [block async for block in bulk._csv_blocks(chunks)]
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task

    assert len(blocks) > 1
    assert ticks > len(blocks) // 2  # a outra tarefa rodou entre os blocos
    rows = list(csv.reader(io.StringIO(b"".join(blocks).decode("utf-8"))))
    assert [row[0] for row in rows] == [chunk.id for chunk in chunks]


async def test_vector_and_metadata_are_compact_json() -> None:
    row = bulk._row(_chunk(7, "x", [0.25, 1.0, -3e-08]))

    assert row[3] == "[0.25,1.0,-3e-08]"
    assert json.loads(row[5]) == {"position": {"start_char": 7}}


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

"""Inserção em lote de chunks com embedding na tabela `chunks`: COPY binário ou executemany.

`insert_chunks(session, chunks, *, method="copy")` grava na transação da sessão e devolve
quantos entraram. Tudo ou nada: qualquer linha recusada derruba o lote inteiro.

- `copy` (padrão): `COPY ... FROM STDIN (FORMAT binary)` pela conexão asyncpg da sessão, com
  o formato montado aqui por `struct` (o `vector` como no `vector_send`: int16 dimensão,
  int16 zero, float4 big-endian). Em texto, formatar e reler 1,5 mil floats por chunk custa
  mais que o protocolo: o COPY em CSV mediu o mesmo que o executemany. Sem codec de `vector`
  no asyncpg, que mudaria o tipo lido pelo resto do código nas conexões do pool. Blocos de
  ~1 MB gerados sob demanda: memória constante e event loop livre entre um bloco e outro.
- `executemany`: INSERT preparado uma vez, vetor em texto com CAST, sem a conexão crua.

O banco valida FK para `documents`, CHECKs, dimensão do `vector(1536)` e NaN/inf; o texto
deve vir do `sanitize_text` (NUL é recusado). Erro do banco vira `DatabaseQueryError` com
SQLSTATE e mensagem, sem o DETAIL (que pode trazer a linha inteira, texto e vetor), e deixa
a transação abortada para quem chamou desfazer (o `Database.session()` já faz). No COPY,
valor que não cabe no formato (fora do float4, mais de 32767 dimensões) vira `ValueError`
com o id do chunk.
"""

import asyncio
import json
import struct
from collections.abc import AsyncIterator, Sequence
from typing import Literal

import asyncpg
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError
from sqlmodel.ext.asyncio.session import AsyncSession

from app.errors.exceptions import DatabaseQueryError
from app.schemas.domain import Chunk

Method = Literal["copy", "executemany"]
COLUMNS = ("id", "document_id", "content", "embedding", "token_count", "metadata")
BLOCK_SIZE = 1 << 20  # bytes por bloco enviado ao COPY

# Formato binário do COPY: assinatura, flags e extensão do cabeçalho; cada linha com o
# número de campos e, por campo, o tamanho (-1 = NULL) e os bytes; -1 no fim.
_HEADER = b"PGCOPY\n\xff\r\n\x00" + struct.pack(">ii", 0, 0)
_FIELDS = struct.pack(">h", len(COLUMNS))
_NULL = struct.pack(">i", -1)
_TRAILER = struct.pack(">h", -1)
_JSONB_VERSION = b"\x01"

_INSERT = text(
    "INSERT INTO chunks (id, document_id, content, embedding, token_count, metadata) VALUES "
    "(:id, :document_id, :content, CAST(CAST(:embedding AS text) AS vector), :token_count, "
    "CAST(CAST(:metadata AS text) AS jsonb))"
)


def _field(data: bytes) -> bytes:
    return struct.pack(">i", len(data)) + data


def _encode(chunk: Chunk) -> bytes:
    """Uma linha do COPY binário, na ordem de `COLUMNS`."""
    try:
        if chunk.embedding is None:
            vector = _NULL
        else:
            dim = len(chunk.embedding)
            vector = _field(struct.pack(f">hh{dim}f", dim, 0, *chunk.embedding))
        return b"".join(
            (
                _FIELDS,
                _field(chunk.id.encode()),
                _field(chunk.document_id.encode()),
                _field(chunk.content.encode()),
                vector,
                _field(struct.pack(">i", chunk.token_count)),
                _field(_JSONB_VERSION + json.dumps(chunk.metadata).encode()),
            )
        )
    except (UnicodeEncodeError, OverflowError, struct.error) as exc:
        raise ValueError(f"chunk {chunk.id!r} não cabe no COPY: {exc}") from exc


async def _copy_blocks(chunks: Sequence[Chunk]) -> AsyncIterator[bytes]:
    block, size = [_HEADER], len(_HEADER)
    for chunk in chunks:
        row = _encode(chunk)
        block.append(row)
        size += len(row)
        if size >= BLOCK_SIZE:
            yield b"".join(block)
            block, size = [], 0
            await asyncio.sleep(0)  # outras tarefas rodam enquanto o lote é montado
    block.append(_TRAILER)
    yield b"".join(block)


def _params(chunk: Chunk) -> dict[str, str | int | None]:
    vector = None if chunk.embedding is None else json.dumps(chunk.embedding, separators=(",", ":"))
    return {
        "id": chunk.id,
        "document_id": chunk.document_id,
        "content": chunk.content,
        "embedding": vector,
        "token_count": chunk.token_count,
        "metadata": json.dumps(chunk.metadata),
    }


async def _copy(session: AsyncSession, chunks: Sequence[Chunk]) -> int:
    connection = await session.connection()
    raw = (await connection.get_raw_connection()).driver_connection
    if raw is None:
        raise RuntimeError("conexão do pool sem driver: foi invalidada")
    if not raw.is_in_transaction():
        # O adaptador do SQLAlchemy só abre a transação no primeiro comando. Sem isto, um
        # COPY logo no início da sessão rodaria em autocommit, fora do rollback dela.
        await connection.exec_driver_sql("SELECT 1")
    status = await raw.copy_to_table(
        "chunks", source=_copy_blocks(chunks), columns=COLUMNS, format="binary"
    )
    return int(status.split()[-1])  # "COPY <n>"


def _describe(exc: Exception) -> str:
    # O DBAPIError do SQLAlchemy embrulha o erro do asyncpg em `orig.__cause__`. `.message`
    # é só a mensagem primária: o `str()` do asyncpg acrescenta o DETAIL.
    error = getattr(getattr(exc, "orig", None), "__cause__", None) or exc
    sqlstate = getattr(error, "sqlstate", None) or "?"
    return f"SQLSTATE {sqlstate}: {getattr(error, 'message', None) or type(error).__name__}"


async def insert_chunks(
    session: AsyncSession, chunks: Sequence[Chunk], *, method: Method = "copy"
) -> int:
    """Grava `chunks` na transação de `session` e devolve quantos entraram (tudo ou nada)."""
    if method not in ("copy", "executemany"):
        raise ValueError(f"method deve ser 'copy' ou 'executemany', veio {method!r}")
    if not chunks:
        return 0
    try:
        if method == "copy":
            return await _copy(session, chunks)
        connection = await session.connection()
        await connection.execute(_INSERT, [_params(chunk) for chunk in chunks])
        return len(chunks)
    # InterfaceError: falha do lado do cliente (conexão caída no meio do COPY). No
    # executemany ela chega como DBAPIError; aqui, crua. Os dois métodos falham igual.
    except (asyncpg.PostgresError, asyncpg.InterfaceError, DBAPIError) as exc:
        raise DatabaseQueryError(f"insert_chunks ({method}): {_describe(exc)}") from exc

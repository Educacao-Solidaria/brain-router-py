"""Inserção em lote de chunks com embedding na tabela `chunks`: COPY ou executemany.

`insert_chunks(session, chunks, *, method="copy")` grava na transação da sessão e devolve
quantos chunks entraram. Tudo ou nada: o lote é um comando só (COPY) ou uma execução só do
asyncpg (executemany), e qualquer linha recusada derruba o lote inteiro.

- `copy` (padrão): `COPY chunks (...) FROM STDIN (FORMAT csv)` pela conexão asyncpg da
  sessão, com os dados gerados em blocos de ~1 MB: memória constante, e o event loop fica
  livre entre um bloco e outro (formatar milhares de vetores leva centenas de ms).
- `executemany`: o INSERT preparado uma vez e executado para todas as linhas (a lista de
  parâmetros vira `executemany` no asyncpg). Mais lento em lote grande; não usa a conexão
  crua.

Formato CSV, não o binário: o COPY binário do asyncpg exige codec para todas as colunas, e
o projeto não tem codec de `vector`. Registrar um numa conexão do pool mudaria o tipo que o
resto do código recebe ao ler `embedding`. Em CSV o vetor vai como o literal `[0.1,0.2]` e o
Postgres converte, como no INSERT com CAST. `csv.QUOTE_NOTNULL` põe todo valor entre aspas
e deixa só o `None` vazio (o NULL do COPY CSV): texto vazio não vira NULL, e uma linha
`\\.` dentro do conteúdo não encerra o COPY.

O banco valida, e aqui não se repete: a FK para `documents` (chunk de documento que não
existe derruba o lote), os CHECKs, a dimensão do `vector(1536)` e NaN/inf no vetor. O texto
deve vir do `sanitize_text` (o Postgres recusa NUL). Erro do banco vira `DatabaseQueryError`
com o SQLSTATE e a mensagem, sem o DETAIL, que pode trazer a linha inteira (texto e vetor).
Depois dele a transação fica abortada: quem chamou desfaz, como o `Database.session()` faz.
"""

import asyncio
import csv
import io
import json
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
BLOCK_SIZE = 1 << 20  # caracteres de CSV por bloco enviado ao COPY

_INSERT = text(
    "INSERT INTO chunks (id, document_id, content, embedding, token_count, metadata) VALUES "
    "(:id, :document_id, :content, CAST(CAST(:embedding AS text) AS vector), :token_count, "
    "CAST(CAST(:metadata AS text) AS jsonb))"
)

_Row = tuple[str, str, str, str | None, int, str]


def _row(chunk: Chunk) -> _Row:
    vector = None if chunk.embedding is None else json.dumps(chunk.embedding, separators=(",", ":"))
    metadata = json.dumps(chunk.metadata)
    return (chunk.id, chunk.document_id, chunk.content, vector, chunk.token_count, metadata)


async def _csv_blocks(chunks: Sequence[Chunk]) -> AsyncIterator[bytes]:
    buffer = io.StringIO()
    writer = csv.writer(buffer, quoting=csv.QUOTE_NOTNULL, lineterminator="\n")
    for chunk in chunks:
        writer.writerow(_row(chunk))
        if buffer.tell() >= BLOCK_SIZE:
            yield buffer.getvalue().encode("utf-8")
            buffer.seek(0)
            buffer.truncate()
            await asyncio.sleep(0)  # outras tarefas rodam enquanto o lote é formatado
    if buffer.tell():
        yield buffer.getvalue().encode("utf-8")


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
        "chunks", source=_csv_blocks(chunks), columns=COLUMNS, format="csv"
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
        params = [dict(zip(COLUMNS, _row(chunk), strict=True)) for chunk in chunks]
        await connection.execute(_INSERT, params)
        return len(chunks)
    except (asyncpg.PostgresError, DBAPIError) as exc:
        raise DatabaseQueryError(f"insert_chunks ({method}): {_describe(exc)}") from exc

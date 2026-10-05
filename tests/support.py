"""Peças reutilizáveis dos testes: factory de chunks e sessão de banco com rollback.

As fixtures que as expõem ficam em `conftest.py` (`chunk_factory`, `db_session`).
"""

import hashlib
import itertools
import math
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.ext.asyncio import AsyncEngine
from sqlmodel.ext.asyncio.session import AsyncSession

from app.schemas.domain import Chunk


def fake_embedding(text: str, dim: int = 8) -> list[float]:
    """Vetor unitário determinístico derivado do texto (mesmo texto → mesmo vetor).

    ponytail: hash simples, sem semântica; quando o mock de embeddings do RAGPY-14
    entrar na main, a factory passa a usá-lo.
    """
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    raw = [digest[i % len(digest)] / 255.0 - 0.5 for i in range(dim)]
    norm = math.sqrt(sum(x * x for x in raw)) or 1.0
    return [x / norm for x in raw]


@dataclass
class ChunkFactory:
    """Cria `Chunk` válidos com ids sequenciais (`chunk-1`, `chunk-2`, ...) e texto em pt-br.

    Tudo o que vier em `overrides` passa pela validação do próprio `Chunk`.
    """

    document_id: str = "doc-1"
    dim: int = 8
    _sequence: Iterator[int] = field(default_factory=lambda: itertools.count(1))

    def __call__(
        self, content: str | None = None, *, embed: bool = False, **overrides: Any
    ) -> Chunk:
        n = next(self._sequence)
        text = content if content is not None else f"Trecho {n} sobre matrícula e frequência."
        values: dict[str, Any] = {
            "id": f"chunk-{n}",
            "document_id": self.document_id,
            "content": text,
            "token_count": len(text.split()),
        }
        if embed:
            values["embedding"] = fake_embedding(text, self.dim)
        return Chunk(**values | overrides)

    def batch(self, size: int, *, embed: bool = False, **overrides: Any) -> list[Chunk]:
        return [self(embed=embed, **overrides) for _ in range(size)]


@asynccontextmanager
async def rollback_session(engine: AsyncEngine) -> AsyncIterator[AsyncSession]:
    """Sessão presa a uma transação externa que é sempre desfeita no fim.

    `commit()` dentro do teste só libera um SAVEPOINT (`create_savepoint`); o
    `ROLLBACK` da transação externa apaga tudo, inclusive DDL, sem limpar à mão.
    """
    async with engine.connect() as connection:
        outer = await connection.begin()
        session = AsyncSession(
            bind=connection,
            join_transaction_mode="create_savepoint",
            expire_on_commit=False,
        )
        try:
            yield session
        finally:
            await session.close()
            await outer.rollback()

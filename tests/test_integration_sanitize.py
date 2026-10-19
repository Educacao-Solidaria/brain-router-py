"""O que o sanitizador evita, contra PostgreSQL real: NUL recusado e busca textual furada.

Rodam com `uv run pytest -m integration` (job `integration` do CI); sem banco, falham.
"""

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError
from sqlmodel.ext.asyncio.session import AsyncSession

from app.sanitize import sanitize_text

pytestmark = pytest.mark.integration

MATCHES = text(
    "SELECT to_tsvector('portuguese', :content) @@ websearch_to_tsquery('portuguese', :query)"
)


async def test_postgres_refuses_nul_and_accepts_the_sanitized_text(
    db_session: AsyncSession,
) -> None:
    raw = "Regimento\x00 escolar"
    with pytest.raises(DBAPIError) as caught:
        async with db_session.begin_nested():
            await db_session.execute(text("SELECT CAST(:t AS text)"), {"t": raw})
    # 22021 = character_not_in_repertoire ("invalid byte sequence for encoding UTF8: 0x00").
    assert getattr(caught.value.orig, "sqlstate", None) == "22021"

    clean = sanitize_text(raw)
    stored = await db_session.scalar(text("SELECT CAST(:t AS text)"), {"t": clean})
    assert stored == "Regimento escolar"


@pytest.mark.parametrize(
    "hidden", ["\u200b", "\u00ad", "\u2060"], ids=["zwsp", "soft-hyphen", "word-joiner"]
)
async def test_invisible_character_breaks_full_text_search_until_sanitized(
    db_session: AsyncSession, hidden: str
) -> None:
    raw = f"Prazo da ma{hidden}trícula: até março."
    params = {"query": "matrícula"}

    assert await db_session.scalar(MATCHES, {"content": raw, **params}) is False
    assert await db_session.scalar(MATCHES, {"content": sanitize_text(raw), **params}) is True

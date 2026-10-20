"""Relatório de uso do banco: `brain-router report` (só leitura, pelo `Database.read()`).

Documentos por coleção e tenant leem a tabela toda (nenhum índice deixa a contagem total mais
rápida). A fila de embedding (`ix_chunks_pending_embedding`) e os zerados
(`ix_chunks_zero_embedding`) leem só os índices parciais da migração 0004; o teste de
integração prova pelo EXPLAIN.
ponytail: com milhões de documentos, um contador por coleção em vez do count(*).
"""

from dataclasses import dataclass

from sqlalchemy import text
from sqlmodel.ext.asyncio.session import AsyncSession

from app.db import Database
from app.ingest import display

DOCUMENTS = text(
    "SELECT collection_id, tenant_id, count(*) FROM documents "
    "GROUP BY collection_id, tenant_id ORDER BY collection_id, tenant_id"
)
PENDING = text(
    "SELECT d.collection_id, d.tenant_id, count(*) FROM chunks AS c "
    "JOIN documents AS d ON d.id = c.document_id WHERE c.embedding IS NULL "
    "GROUP BY d.collection_id, d.tenant_id"
)
ZEROED = text("SELECT count(*) FROM chunks WHERE vector_norm(embedding) = 0")  # como o integrity


@dataclass(frozen=True)
class Usage:
    collection_id: str
    tenant_id: str
    documents: int
    pending: int  # chunks sem embedding: fora da busca vetorial até o vetor chegar


@dataclass(frozen=True)
class Report:
    usage: list[Usage]
    zeroed: int  # chunks com vetor de norma 0 (o `check-integrity --fix` os devolve à fila)


async def collect(session: AsyncSession) -> Report:
    conn = await session.connection()
    pending = {(c, t): n for c, t, n in await conn.execute(PENDING)}
    rows = await conn.execute(DOCUMENTS)  # todo chunk tem documento (FK): nada fica de fora
    usage = [Usage(c, t, n, pending.get((c, t), 0)) for c, t, n in rows]
    return Report(usage, int(await conn.scalar(ZEROED) or 0))


async def usage_report(db: Database) -> Report:
    return await db.read(collect)


def render(report: Report) -> str:
    lines = ["Uso por coleção e tenant (somente leitura)", ""]
    lines.append(f"{'coleção':<24} {'tenant':<24} {'documentos':>10} {'sem embedding':>13}")
    for row in report.usage:  # coleção e tenant vêm de fora: sem controle de terminal
        names = f"{display(row.collection_id):<24} {display(row.tenant_id):<24}"
        lines.append(f"{names} {row.documents:>10} {row.pending:>13}")
    if not report.usage:
        lines.append("(nenhum documento)")
    lines += ["", f"chunks com embedding zerado: {report.zeroed}"]
    return "\n".join(lines) + "\n"

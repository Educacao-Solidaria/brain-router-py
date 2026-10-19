"""Auditoria da consistência entre documentos e vetores: `brain-router check-integrity`.

Somente leitura por padrão: roda no `Database.read()` (READ ONLY) e só relata. Com `--fix`,
corrige o que tem conserto seguro numa transação só, com a contagem antes e depois, acima de
`max_changes` linhas recusa e desfaz, e a escrita em `chunks` espera até o commit.

O schema (0002/0003) já garante: chunk com documento (FK NOT NULL, ON DELETE CASCADE), vetor
de 1536 dimensões sem NaN nem infinito, conteúdo e título não vazios, `token_count >= 0` e
`metadata` objeto. O que ele deixa passar, e se procura aqui ([correção do --fix]):
- `chunks_orfaos`: só com a FK contornada (`session_replication_role`, `DISABLE TRIGGER`),
  mas o órfão aparece na busca sem dono [apagar];
- `documentos_sem_chunks`: ingestão que parou no meio, criada há mais de `min_age_hours`
  (para não pegar uma em andamento), fora de toda busca [apagar];
- `embeddings_zerados`: o cosseno dá NaN e o chunk sai do ranking [NULL: volta à fila];
- `chunks_sem_embedding`: fora da busca vetorial [só relata];
- `colecoes_com_varios_tenants`: a busca por coleção misturaria tenants [só relata].
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass

from sqlalchemy import text
from sqlmodel.ext.asyncio.session import AsyncSession

from app.db import Database

SAMPLE = 5  # ids de exemplo por verificação

_ORPHAN = "NOT EXISTS (SELECT 1 FROM documents AS d WHERE d.id = c.document_id)"
_EMPTY = (
    "d.created_at < now() - make_interval(secs => :min_age) "
    "AND NOT EXISTS (SELECT 1 FROM chunks AS c WHERE c.document_id = d.id)"
)
_ZERO = "vector_norm(c.embedding) = 0"


@dataclass(frozen=True)
class Check:
    name: str
    cases: str  # SELECT de uma coluna: o id de cada caso
    fix: str | None = None  # o comando que corrige todos, no --fix


CHECKS = (
    Check(
        "chunks_orfaos",
        f"SELECT c.id FROM chunks AS c WHERE {_ORPHAN}",
        f"DELETE FROM chunks AS c WHERE {_ORPHAN}",
    ),
    Check(
        "documentos_sem_chunks",
        f"SELECT d.id FROM documents AS d WHERE {_EMPTY}",
        f"DELETE FROM documents AS d WHERE {_EMPTY}",
    ),
    Check(
        "embeddings_zerados",
        f"SELECT c.id FROM chunks AS c WHERE {_ZERO}",
        f"UPDATE chunks AS c SET embedding = NULL WHERE {_ZERO}",
    ),
    Check("chunks_sem_embedding", "SELECT c.id FROM chunks AS c WHERE c.embedding IS NULL"),
    Check(
        "colecoes_com_varios_tenants",
        "SELECT collection_id FROM documents GROUP BY collection_id "
        "HAVING count(DISTINCT tenant_id) > 1",
    ),
)


@dataclass(frozen=True)
class Finding:
    count: int
    sample: list[str]


@dataclass(frozen=True)
class Report:
    before: dict[str, Finding]
    changed: dict[str, int] | None = None  # None = só leitura
    after: dict[str, Finding] | None = None

    @property
    def remaining(self) -> int:
        """Casos que sobraram no fim (o que decide o código de saída)."""
        final = self.after if self.after is not None else self.before
        return sum(finding.count for finding in final.values())


class TooManyChanges(Exception):
    def __init__(self, pending: int, limit: int) -> None:
        super().__init__(f"--fix mudaria {pending} linhas, acima do limite de {limit}; nada mudou")


def check_limits(min_age_hours: float, max_changes: int) -> None:
    if not (math.isfinite(min_age_hours) and min_age_hours >= 0):
        raise ValueError(f"min_age_hours deve ser finito e >= 0, veio {min_age_hours!r}")
    if type(max_changes) is not int or max_changes < 0:
        raise ValueError(f"max_changes deve ser um inteiro >= 0, veio {max_changes!r}")


async def audit(session: AsyncSession, params: Mapping[str, object]) -> dict[str, Finding]:
    conn = await session.connection()
    findings: dict[str, Finding] = {}
    for check in CHECKS:
        count = await conn.scalar(text(f"SELECT count(*) FROM ({check.cases}) AS cases"), params)
        sample = await conn.scalars(text(f"{check.cases} ORDER BY 1 LIMIT {SAMPLE}"), params)
        findings[check.name] = Finding(int(count or 0), [str(key) for key in sample])
    return findings


async def repair(
    session: AsyncSession, params: Mapping[str, object], *, max_changes: int
) -> dict[str, int]:
    """Roda as correções na transação de `session`; acima do limite, levanta (e quem chamou
    desfaz). Quem chama confere antes também: aqui é a guarda contra o que chegou no meio."""
    conn = await session.connection()
    changed: dict[str, int] = {}
    for check in CHECKS:
        if check.fix is not None:
            changed[check.name] = (await conn.execute(text(check.fix), params)).rowcount
            if sum(changed.values()) > max_changes:
                raise TooManyChanges(sum(changed.values()), max_changes)
    return changed


async def check_integrity(
    db: Database, *, fix: bool = False, min_age_hours: float = 1.0, max_changes: int = 1000
) -> Report:
    check_limits(min_age_hours, max_changes)
    params = {"min_age": min_age_hours * 3600}
    if not fix:
        return Report(before=await db.read(lambda session: audit(session, params)))
    async with db.session() as session:  # tudo numa transação: commit só se nada levantar
        # Espera as escritas em chunks em andamento e barra as novas até o commit: sem isso, o
        # documento "vazio" que uma ingestão enche agora sairia, e o CASCADE levaria os chunks.
        await (await session.connection()).execute(text("LOCK TABLE chunks IN SHARE MODE"))
        before = await audit(session, params)
        pending = sum(before[check.name].count for check in CHECKS if check.fix is not None)
        if pending > max_changes:
            raise TooManyChanges(pending, max_changes)
        changed = await repair(session, params, max_changes=max_changes)
        after = await audit(session, params)
    return Report(before=before, changed=changed, after=after)


def render(report: Report) -> str:
    if report.after is None or report.changed is None:
        lines = ["Integridade de documentos e vetores (somente leitura: nada foi alterado)", ""]
        lines.append(f"{'verificação':<30} {'casos':>7}  exemplos")
        for name, finding in report.before.items():
            lines.append(f"{name:<30} {finding.count:>7}  {', '.join(finding.sample)}".rstrip())
        return "\n".join(lines) + "\n"
    lines = ["Integridade de documentos e vetores (--fix: corrigido numa transação)", ""]
    lines.append(f"{'verificação':<30} {'antes':>7} {'alterados':>10} {'depois':>7}")
    for name, finding in report.before.items():
        changed = report.changed.get(name)
        lines.append(
            f"{name:<30} {finding.count:>7} {'-' if changed is None else changed:>10} "
            f"{report.after[name].count:>7}"
        )
    return "\n".join(lines) + "\n"

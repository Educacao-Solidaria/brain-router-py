"""Portão de qualidade do RAG no CI: o score médio do pipeline num dataset sintético, mínimo 0,85.

`python -m app.evals.gate` (job `quality-gate` do CI, no Postgres de `DATABASE_URL`) imprime a
tabela em Markdown, que o workflow copia para o `$GITHUB_STEP_SUMMARY`, e sai 1 com o score
abaixo de `THRESHOLD`.

O que o score mede: cada pergunta de `generate_dataset(200, seed=0, distractors=4)` ganha uma
coleção própria com os seus 5 contextos (o trecho que a responde e um de cada outro tema),
gravados pelo `index_handler` (o `POST /api/v1/index`: limpeza, chunking, COPY). A busca léxica
devolve o 1º resultado, e o `FaithfulnessEvaluator` da Fase 1 dá a fração das palavras da
resposta de referência que estão nele (as de 3+ letras: número de 2 dígitos não conta); o score
é a média. Ou seja: "a busca pôs em 1º o trecho com o que responde à pergunta?" — recuperação,
não a fidelidade de uma resposta gerada (não há LLM aqui: determinístico e sem chamada paga). A
busca densa fica de fora: o mock de embeddings é um hash, sem semântica. As coleções
`quality-gate/...` são apagadas no fim; os nomes são fixos (ids estáveis entre execuções), então
não rode dois portões ao mesmo tempo no mesmo banco.
"""

import asyncio
import sys
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from statistics import fmean

from sqlalchemy import text
from sqlmodel.ext.asyncio.session import AsyncSession

from app.clients import MockEmbeddingClient
from app.db import Database
from app.evals.dataset import generate_dataset
from app.evals.interfaces import BaseEvaluator, EvalInput
from app.evals.metrics import FaithfulnessEvaluator
from app.ingest import index_handler
from app.log import configure_logging, shutdown_logging
from app.schemas.domain import Chunk, SearchResult
from app.schemas.mcp import IndexDocumentInput, IndexDocumentOutput
from app.settings import get_settings
from app.storage.interfaces import BaseTextSearch
from app.tokens import DEFAULT_MODEL

THRESHOLD = 0.85
SAMPLES, SEED, DISTRACTORS = 200, 0, 4  # 4 distratores = um contexto de cada outro tema
PREFIX = "quality-gate/"  # coleção de cada pergunta: quality-gate/sintetico-0-17

# Qualquer termo da pergunta (OR), na ordem do `ts_rank_cd`: com o AND do `websearch_to_tsquery`
# (o dublê do e2e), 165 das 200 perguntas voltam sem resultado. Empate exato (12 perguntas de
# bolsa contra a matrícula do mesmo curso: o `ts_rank_cd` não pesa a raridade do termo) sai pelo
# id do chunk, que é estável entre execuções.
_SEARCH = text(
    "SELECT c.id, c.document_id, c.content, c.token_count, c.metadata, "
    "ts_rank_cd(c.content_tsv, q) AS score FROM chunks AS c JOIN documents AS d "
    "ON d.id = c.document_id, CAST(replace(CAST(plainto_tsquery('portuguese', :query) AS text), "
    "' & ', ' | ') AS tsquery) AS q WHERE d.collection_id = :collection AND c.content_tsv @@ q "
    "ORDER BY score DESC, c.id LIMIT :limit"
)
_DROP = text("DELETE FROM documents WHERE collection_id = ANY(:collections)")  # chunks no CASCADE

Index = Callable[[IndexDocumentInput], Awaitable[IndexDocumentOutput]]


class AnyTermSearch(BaseTextSearch):
    """Busca léxica sobre o `content_tsv` da 0003 (config `portuguese` nos dois lados).

    ponytail: dublê do motor textual do Luan (PR 24, ainda fora da main). Quando ele entrar,
    o `_run` passa a usá-lo e o portão mede a busca de produção.
    """

    def __init__(self, db: Database) -> None:
        self.db = db

    async def search_text(
        self, collection_name: str, query: str, limit: int = 10
    ) -> list[SearchResult]:
        async def run(session: AsyncSession) -> list[SearchResult]:
            params = {"collection": collection_name, "query": query, "limit": limit}
            rows = await (await session.connection()).execute(_SEARCH, params)
            return [
                SearchResult(
                    chunk=Chunk(
                        id=row.id,
                        document_id=row.document_id,
                        content=row.content,
                        token_count=row.token_count,
                        metadata=row.metadata,
                    ),
                    sparse_score=row.score,
                    rrf_score=row.score,
                    rank=rank,
                )
                for rank, row in enumerate(rows, 1)
            ]

        return await self.db.read(run)


@dataclass(frozen=True)
class Graded:
    topic: str
    score: float
    first_is_gold: bool  # o 1º resultado é o trecho que responde à pergunta


async def grade(
    samples: Sequence[EvalInput],
    *,
    index: Callable[[str], Index],
    search: BaseTextSearch,
    evaluator: BaseEvaluator,
) -> list[Graded]:
    """Grava os contextos de cada amostra numa coleção só dela e avalia o 1º resultado.

    As amostras são as do `generate_dataset` (`metadata` com `id`, `topic` e `gold_index`).
    """
    graded = []
    for sample in samples:
        collection = f"{PREFIX}{sample.metadata['id']}"
        store = index(collection)
        for n, context in enumerate(sample.contexts):
            body = IndexDocumentInput(title="trecho", content=context, source_uri=f"eval://{n}")
            await store(body)
        first = (await search.search_text(collection, sample.query, limit=1))[:1]
        update = {"contexts": [result.chunk.content for result in first]}
        output = await evaluator.evaluate(sample.model_copy(update=update))
        gold = f"eval://{sample.metadata['gold_index']}"
        hit = any(result.chunk.metadata.get("source_uri") == gold for result in first)
        graded.append(Graded(sample.metadata["topic"], output.score, hit))
    return graded


def render(graded: Sequence[Graded]) -> tuple[float, str]:
    """O score médio (4 casas, o mesmo valor que se compara com o limiar) e o relatório."""
    if not graded:
        raise ValueError("nenhuma pergunta avaliada: o portão não passa no vazio")
    score = round(fmean(g.score for g in graded), 4)
    verdict = "passou" if score >= THRESHOLD else "**reprovado**"
    lines = [
        f"### Quality gate do RAG: {score:.4f} (mínimo {THRESHOLD}) — {verdict}",
        "",
        f"Média, em {len(graded)} perguntas do dataset sintético (seed {SEED}), do "
        "`FaithfulnessEvaluator` entre a resposta de referência e o 1º resultado da busca léxica "
        f"no Postgres, entre o trecho que responde e {DISTRACTORS} de outros temas gravados pelo "
        "`index_handler`. Mede a recuperação, não uma resposta gerada (sem LLM nem busca densa).",
        "",
        "| Tema | Perguntas | Faithfulness médio | Trecho certo em 1º |",
        "|---|---:|---:|---:|",
    ]
    for topic in sorted({g.topic for g in graded}):
        rows = [g for g in graded if g.topic == topic]
        hits = sum(g.first_is_gold for g in rows)
        lines.append(f"| {topic} | {len(rows)} | {fmean(g.score for g in rows):.4f} | {hits} |")
    return score, "\n".join(lines) + "\n"


async def _run() -> list[Graded]:
    settings = get_settings()
    if settings.is_production:  # grava e apaga coleções no banco
        raise SystemExit("quality gate recusado com APP_ENV=production")
    db, embedder = Database.from_settings(settings), MockEmbeddingClient(dimension=1536)
    samples = generate_dataset(SAMPLES, seed=SEED, distractors=DISTRACTORS)

    def index(collection: str) -> Index:
        return index_handler(db, embedder, model=DEFAULT_MODEL, collection_id=collection)

    try:
        search = AnyTermSearch(db)
        return await grade(samples, index=index, search=search, evaluator=FaithfulnessEvaluator())
    finally:
        try:
            collections = [f"{PREFIX}{sample.metadata['id']}" for sample in samples]
            async with db.session() as session:
                await (await session.connection()).execute(_DROP, {"collections": collections})
        finally:
            await db.dispose()


def main() -> int:
    configure_logging("WARNING", json=False, stream=sys.stderr)  # o stdout é do relatório
    try:
        score, report = render(asyncio.run(_run()))
    finally:
        shutdown_logging()
    print(report, end="")
    return 0 if score >= THRESHOLD else 1


if __name__ == "__main__":
    raise SystemExit(main())

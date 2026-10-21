"""Portão de qualidade (`app.evals.gate`) sem banco: a nota de cada pergunta, o limiar e a saída."""

from collections.abc import Callable

import pytest

from app.evals import gate
from app.evals.dataset import generate_dataset
from app.evals.metrics import FaithfulnessEvaluator
from app.schemas.domain import Chunk, SearchResult
from app.schemas.mcp import IndexDocumentInput, IndexDocumentOutput
from app.settings import get_settings
from app.storage.interfaces import BaseTextSearch

SAMPLES = generate_dataset(30, seed=4, distractors=gate.DISTRACTORS)
GOLD = {f"{gate.PREFIX}{s.metadata['id']}": f"eval://{s.metadata['gold_index']}" for s in SAMPLES}
Order = Callable[[str, list[IndexDocumentInput]], list[IndexDocumentInput]]


class Pool(BaseTextSearch):
    """Índice e busca de mentira: guarda o que o portão grava e devolve tudo na ordem de `order`
    (ignora o `limit`, como um motor defeituoso: o portão tem de avaliar só o 1º)."""

    def __init__(self, order: Order) -> None:
        self.order = order
        self.limits: list[int] = []
        self.stored: dict[str, list[IndexDocumentInput]] = {}

    def index(self, collection: str) -> gate.Index:
        async def store(body: IndexDocumentInput) -> IndexDocumentOutput:
            self.stored.setdefault(collection, []).append(body)
            # O id do documento é o da coleção, o mesmo que a busca devolve no chunk.
            return IndexDocumentOutput(
                document_id=collection, chunks_created=1, success=True, message=""
            )

        return store

    async def search_text(
        self, collection_name: str, query: str, limit: int = 10
    ) -> list[SearchResult]:
        self.limits.append(limit)
        found = self.order(collection_name, self.stored[collection_name])
        return [
            SearchResult(
                chunk=Chunk(
                    id=body.source_uri,
                    document_id=collection_name,
                    content=body.content,
                    metadata={"source_uri": body.source_uri},
                ),
                rrf_score=1.0,
                rank=rank,
            )
            for rank, body in enumerate(found, 1)
        ]


async def _grade(pool: Pool) -> list[gate.Graded]:
    evaluator = FaithfulnessEvaluator()
    return await gate.grade(SAMPLES, index=pool.index, search=pool, evaluator=evaluator)


async def test_gold_first_scores_one_and_each_question_gets_its_own_collection() -> None:
    pool = Pool(
        lambda collection, bodies: sorted(bodies, key=lambda b: b.source_uri != GOLD[collection])
    )

    graded = await _grade(pool)

    assert [(g.score, g.first_is_gold) for g in graded] == [(1.0, True)] * len(SAMPLES)
    assert [g.topic for g in graded] == [s.metadata["topic"] for s in SAMPLES]
    assert gate.render(graded)[0] == 1.0 and pool.limits == [1] * len(SAMPLES)
    for sample in SAMPLES:
        bodies = pool.stored[f"{gate.PREFIX}{sample.metadata['id']}"]
        assert [body.content for body in bodies] == sample.contexts
        assert [body.source_uri for body in bodies] == [f"eval://{n}" for n in range(5)]


async def test_a_wrong_passage_first_fails_the_gate() -> None:
    pool = Pool(
        lambda collection, bodies: sorted(bodies, key=lambda b: b.source_uri == GOLD[collection])
    )

    graded = await _grade(pool)

    assert not any(g.first_is_gold for g in graded)
    assert all(g.score < 1.0 for g in graded)
    score, report = gate.render(graded)
    assert score < gate.THRESHOLD and "**reprovado**" in report


async def test_a_search_that_finds_nothing_scores_zero() -> None:
    graded = await _grade(Pool(lambda collection, bodies: []))

    assert {(g.score, g.first_is_gold) for g in graded} == {(0.0, False)}


@pytest.mark.parametrize(
    ("scores", "passed"),
    [([0.85], True), ([0.8499], False), ([0.8, 0.9], True), ([0.8, 0.8998], False)],
)
def test_the_threshold_is_inclusive_on_the_printed_value(scores: list[float], passed: bool) -> None:
    score, report = gate.render([gate.Graded("bolsa", s, True) for s in scores])

    assert (score >= gate.THRESHOLD) is passed
    assert f"### Quality gate do RAG: {score:.4f} (mínimo 0.85)" in report
    assert ("— passou" in report) is passed


def test_report_has_one_row_per_topic() -> None:
    graded = [
        gate.Graded("matrícula", 1.0, True),
        gate.Graded("bolsa", 0.5, False),
        gate.Graded("bolsa", 1.0, True),
    ]

    score, report = gate.render(graded)

    assert score == 0.8333
    rows = [line for line in report.splitlines() if line.startswith("| ") and "Tema" not in line]
    assert rows == ["| bolsa | 2 | 0.7500 | 1 |", "| matrícula | 1 | 1.0000 | 1 |"]


def test_an_empty_run_never_passes() -> None:
    with pytest.raises(ValueError, match="nenhuma pergunta"):
        gate.render([])


@pytest.mark.parametrize(("score", "code"), [(0.85, 0), (0.8499, 1)])
def test_main_prints_only_the_report_and_exits_by_the_threshold(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], score: float, code: int
) -> None:
    async def run() -> list[gate.Graded]:
        return [gate.Graded("bolsa", score, True)]

    monkeypatch.setattr(gate, "_run", run)

    assert gate.main() == code
    assert capsys.readouterr().out == gate.render([gate.Graded("bolsa", score, True)])[1]


def test_refuses_to_run_against_production(monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in [("APP_ENV", "production"), ("DATABASE_URL", "postgresql://u:p@db/b")]:
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-x")
    get_settings.cache_clear()
    try:
        with pytest.raises(SystemExit, match="APP_ENV=production"):
            gate.main()
    finally:
        get_settings.cache_clear()

"""Partes do benchmark que não precisam de banco: percentis, recall, vetores, relatório e CLI."""

import argparse
from unittest.mock import MagicMock

import pytest

from app.bench import vector_search as bench
from app.bench.vector_search import Latency, Report, Row, recall_at_k, render_markdown, summarize
from app.storage.hnsw import HnswBuildParams


def _fields(latency: Latency) -> tuple[float, float, float, float]:
    return latency.p50, latency.p95, latency.p99, latency.mean


@pytest.mark.parametrize(
    ("samples", "expected"),
    [
        ([float(v) for v in range(100, 0, -1)], (50.5, 95.05, 99.01, 50.5)),  # fora de ordem
        ([3.0, 1.0], (2.0, 2.9, 2.98, 2.0)),  # interpola entre as duas
        ([0.25], (0.25, 0.25, 0.25, 0.25)),  # o quantiles do 3.12 recusaria uma só
    ],
)
def test_summarize_interpolates_percentiles_between_samples(
    samples: list[float], expected: tuple[float, ...]
) -> None:
    assert _fields(summarize(samples)) == pytest.approx(expected)


def test_summarize_refuses_no_samples() -> None:
    with pytest.raises(ValueError, match="sem medições"):
        summarize([])


@pytest.mark.parametrize(
    ("found", "truth", "expected"),
    [
        ([[1, 2, 3]], [[3, 2, 1]], 1.0),  # a ordem não importa
        ([[4, 5, 6]], [[1, 2, 3]], 0.0),
        ([[1, 2, 9, 9], [1, 2, 3, 4]], [[1, 2, 3, 4], [1, 2, 3, 4]], 0.75),  # média das consultas
        ([[1, 1, 1]], [[1, 2, 3]], 1 / 3),  # repetido conta uma vez
    ],
)
def test_recall_is_the_mean_share_of_the_exact_neighbors_found(
    found: list[list[int]], truth: list[list[int]], expected: float
) -> None:
    assert recall_at_k(found, truth) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("found", "truth"),
    [([], []), ([[1]], [[1], [2]]), ([[1], []], [[1], []])],
    ids=["sem-consultas", "tamanhos-diferentes", "gabarito-vazio"],
)
def test_recall_refuses_inputs_without_a_meaningful_answer(
    found: list[list[int]], truth: list[list[int]]
) -> None:
    with pytest.raises(ValueError):
        recall_at_k(found, truth)


def test_query_vectors_are_deterministic_bounded_pgvector_text() -> None:
    first = bench.query_vectors(2)
    values = [float(v) for v in first[0].removeprefix("[").removesuffix("]").split(",")]

    assert first == bench.query_vectors(2) != bench.query_vectors(2, seed=7)
    assert first[0] != first[1] and bench.query_vectors(0) == []
    assert len(values) == bench.DIMENSION and all(-1.0 <= v <= 1.0 for v in values)


def _report() -> Report:
    exact = Row("exata (sem índice)", None, Latency(0.1, 0.12, 0.125, 0.105), 1.0)
    fast = Row("HNSW fast", 20, Latency(0.0009, 0.0011, 0.00125, 0.001), 0.4321)
    build = HnswBuildParams.for_recall("balanced")
    return Report(
        50_000, 300, 10, "PostgreSQL 17.6, pgvector 0.8.1", build, 5.04, 61.27, [exact, fast]
    )


def test_markdown_report_has_context_and_one_line_per_search() -> None:
    lines = render_markdown(_report()).splitlines()

    assert lines[0] == "### Busca vetorial: 50.000 vetores × 1536 dimensões"
    for fragment in (
        "pgvector 0.8.1",
        "carga em 5.0 s, índice em 61.3 s",
        "300 consultas",
        "k = 10",
    ):
        assert fragment in lines[2]
    assert "pior caso" in lines[4]
    assert lines[6].endswith("| recall@10 |")
    assert lines[-2] == "| exata (sem índice) | — | 100.00 | 120.00 | 125.00 | 105.00 | 1.000 |"
    assert lines[-1] == "| HNSW fast | 20 | 0.90 | 1.10 | 1.25 | 1.00 | 0.432 |"


@pytest.mark.parametrize(
    "argv",
    [
        ["--vectors", "0"],
        ["--queries", "0"],
        ["--k", "-3"],
        ["--vectors", "dez"],
        ["--vectors", "10", "--k", "11"],  # mais vizinhos que vetores
        ["--k", "1001"],  # acima do teto do ef_search
    ],
)
def test_cli_refuses_degenerate_sizes_before_touching_the_database(
    argv: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(bench, "_run", MagicMock(side_effect=AssertionError("não devia rodar")))
    with pytest.raises(SystemExit) as exit_info:
        bench.main(argv)
    assert exit_info.value.code == 2


@pytest.mark.parametrize(("vectors", "queries", "k"), [(5, 1, 10), (10, 0, 10), (10, 1, 0)])
async def test_run_benchmark_validates_before_connecting(
    vectors: int, queries: int, k: int
) -> None:
    db = MagicMock()  # sem spec: o `engine` é atributo de instância do Database
    with pytest.raises(ValueError, match="precisa de 1 <= k"):
        await bench.run_benchmark(db, vectors=vectors, queries=queries, k=k)
    db.engine.connect.assert_not_called()


def test_main_prints_only_the_report_on_stdout(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    seen: list[argparse.Namespace] = []

    async def fake_run(args: argparse.Namespace) -> Report:
        seen.append(args)
        return _report()

    monkeypatch.setattr(bench, "_run", fake_run)
    assert bench.main(["--vectors", "2000", "--queries", "30"]) == 0
    assert capsys.readouterr().out == render_markdown(_report())  # o log foi para o stderr
    assert (seen[0].vectors, seen[0].queries, seen[0].k) == (2000, 30, 10)

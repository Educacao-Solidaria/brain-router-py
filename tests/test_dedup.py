"""Normalização min-max de scores e deduplicação de resultados sobrepostos."""

import random
from typing import Any

import pytest

from app.chunking import chunk_text
from app.schemas import Chunk, ChunkPosition, SearchResult
from app.search.dedup import deduplicate, min_max_normalize

NAN, INF = float("nan"), float("inf")


def _result(
    chunk_id: str,
    content: str,
    *,
    document: str = "doc-1",
    score: float = 0.5,
    span: tuple[int, int] | None = None,
    dense: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> SearchResult:
    if span is not None:
        metadata = {"position": ChunkPosition(start_char=span[0], end_char=span[1]).model_dump()}
    chunk = Chunk(id=chunk_id, document_id=document, content=content, metadata=metadata or {})
    return SearchResult(chunk=chunk, dense_score=dense, rrf_score=score, rank=1)


def test_min_max_maps_the_lowest_to_zero_and_the_highest_to_one() -> None:
    assert min_max_normalize([2.0, 4.0, 3.0]) == [0.0, 1.0, 0.5]
    assert min_max_normalize([-1.0, 1.0]) == [0.0, 1.0]


def test_min_max_of_an_empty_list_and_of_equal_scores() -> None:
    assert min_max_normalize([]) == []
    assert min_max_normalize([0.7]) == [1.0]  # sem divisão por zero
    assert min_max_normalize([3.0, 3.0, 3.0]) == [1.0, 1.0, 1.0]
    assert min_max_normalize([0.0, -0.0]) == [1.0, 1.0]


def test_none_nan_and_infinity_become_zero_without_moving_the_range() -> None:
    assert min_max_normalize([NAN, 1.0, 3.0, None, INF, -INF, 2.0]) == [
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.5,
    ]
    assert min_max_normalize([NAN, None]) == [0.0, 0.0]
    assert min_max_normalize([NAN, 5.0]) == [0.0, 1.0]


def test_extreme_magnitudes_do_not_overflow() -> None:
    assert min_max_normalize([-1.7e308, 1.7e308, 0.0]) == [0.0, 1.0, 0.5]


def test_min_max_output_is_always_between_zero_and_one() -> None:
    rng = random.Random(29)
    odd: list[float | None] = [0.0, -0.0, 5e-324, -5e-324, 1.7e308, -1.7e308, NAN, INF, -INF, None]
    for _ in range(500):
        scores = [
            rng.choice(odd) if rng.random() < 0.3 else rng.uniform(-1e3, 1e3)
            for _ in range(rng.randint(0, 8))
        ]
        normalized = min_max_normalize(scores)
        assert len(normalized) == len(scores)
        assert all(0.0 <= value <= 1.0 for value in normalized), scores


def test_the_same_chunk_from_both_engines_keeps_the_better_score() -> None:
    dense = _result("c1", "A matrícula é anual.", score=0.4)
    sparse = _result("c1", "A matrícula é anual.", score=0.7)
    assert deduplicate([dense, sparse]) == [sparse]


def test_the_same_text_in_another_document_is_a_repeat() -> None:
    first = _result("a", "A matrícula é anual.", document="doc-a", score=0.2)
    copy = _result("b", "  a MATRÍCULA é\nanual. ", document="doc-b", score=0.9)
    assert deduplicate([first, copy]) == [copy]


def test_text_contained_in_another_of_the_same_document_is_a_repeat() -> None:
    small = _result("s", "A matrícula é anual.", score=0.9)
    big = _result("b", "Regras gerais. A matrícula é anual. Depois, a rematrícula.", score=0.3)
    # Fica o de maior score, mesmo sendo o menor: é o trecho que a consulta achou melhor.
    assert deduplicate([big, small]) == [small]


def test_containment_across_documents_keeps_both_sources() -> None:
    small = _result("s", "A matrícula é anual.", document="doc-a", score=0.9)
    big = _result("b", "Regras gerais. A matrícula é anual.", document="doc-b", score=0.3)
    assert deduplicate([small, big]) == [small, big]


def test_containment_counts_whole_words_only() -> None:
    word = _result("a", "ano", score=0.9)
    longer_word = _result("b", "A anotação letiva.", score=0.1)
    assert deduplicate([word, longer_word]) == [word, longer_word]


def test_spans_of_the_same_document_that_overlap_enough_are_a_repeat() -> None:
    first = _result("c1", "texto um", span=(0, 100), score=0.5)
    mostly_same = _result("c2", "texto dois", span=(20, 120), score=0.6)  # 80 de 100
    neighbour = _result("c3", "texto três", span=(110, 210), score=0.4)  # 10 de 100
    assert deduplicate([first, mostly_same, neighbour]) == [mostly_same, neighbour]
    assert deduplicate([first, mostly_same, neighbour], min_overlap=0.05) == [mostly_same]


def test_spans_of_different_documents_are_never_compared() -> None:
    a = _result("a", "um", document="doc-a", span=(0, 100))
    b = _result("b", "dois", document="doc-b", span=(0, 100))
    assert deduplicate([a, b]) == [a, b]


@pytest.mark.parametrize(
    "position",
    ["0-100", {"start_char": -1, "end_char": 50}, {"start_char": 9, "end_char": 9}],
    ids=["texto", "negativo", "vazio"],
)
def test_malformed_positions_only_turn_the_span_check_off(position: Any) -> None:
    odd = _result("a", "um", metadata={"position": position})
    other = _result("b", "dois", span=(0, 100))
    assert deduplicate([odd, other]) == [odd, other]


def test_consecutive_chunks_from_the_chunker_are_not_repeats() -> None:
    # Três frases por chunk e uma de sobreposição: 1/3 em comum, abaixo do limiar de 0,5.
    text = " ".join(f"Frase número {n} fala de matrícula." for n in range(1, 12))
    chunks = chunk_text(text, "doc", max_tokens=18, overlap_tokens=6)
    results = [SearchResult(chunk=chunk, rrf_score=1.0, rank=1) for chunk in chunks]
    assert len(results) == 5
    assert deduplicate(results) == results


def test_missing_or_nan_scores_lose_and_a_tie_keeps_the_longer_text() -> None:
    nan = _result("n", "A matrícula é anual.", score=NAN)
    real = _result("r", "A matrícula é anual.", document="doc-2", score=0.01)
    assert deduplicate([nan, real]) == [real]

    short = _result("s", "A matrícula é anual.", dense=0.5)
    long = _result("l", "Regras. A matrícula é anual.", dense=0.5)
    missing = _result("m", "A matrícula é anual.", dense=None)
    assert deduplicate([short, long], score=lambda r: r.dense_score) == [long]
    assert deduplicate([missing, short], score=lambda r: r.dense_score) == [short]


def test_survivors_keep_the_input_order() -> None:
    results = [_result("a", "um", score=0.1), _result("b", "dois", score=0.9), _result("c", "três")]
    assert deduplicate(results) == results
    assert deduplicate([]) == []


@pytest.mark.parametrize("value", [-0.1, 1.5, NAN])
def test_min_overlap_outside_zero_to_one_is_rejected(value: float) -> None:
    with pytest.raises(ValueError, match="min_overlap"):
        deduplicate([], min_overlap=value)

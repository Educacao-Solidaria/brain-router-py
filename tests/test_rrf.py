"""Testes unitários para o cálculo matemático e estruturas do Reciprocal Rank Fusion (RRF)."""

import pytest

from app.search.rrf import (
    RankedCandidate,
    RRFConfig,
    compute_rrf,
)


def test_rrf_identical_ranks() -> None:
    # Dois sistemas retornam os mesmos itens nas mesmas posições
    dense_run = [
        RankedCandidate[str](id="doc_a", rank=1, score=0.95),
        RankedCandidate[str](id="doc_b", rank=2, score=0.80),
    ]
    sparse_run = [
        RankedCandidate[str](id="doc_a", rank=1, score=12.5),
        RankedCandidate[str](id="doc_b", rank=2, score=8.1),
    ]

    results = compute_rrf({"dense": dense_run, "sparse": sparse_run}, RRFConfig(k=60))

    assert len(results) == 2
    assert results[0].id == "doc_a"
    assert results[0].rank == 1
    # 1/(60+1) + 1/(60+1) = 2/61 ≈ 0.032787
    expected_score = round(2.0 / 61.0, 6)
    assert abs(results[0].rrf_score - expected_score) < 1e-5

    assert results[1].id == "doc_b"
    assert results[1].rank == 2
    # 2/(60+2) = 2/62 ≈ 0.032258
    assert results[0].rrf_score > results[1].rrf_score


def test_rrf_complementary_lists() -> None:
    # Item A só no denso (rank 1), Item B só no esparso (rank 1), Item C em ambos (rank 2)
    dense_run = [
        RankedCandidate[str](id="doc_a", rank=1, score=0.9),
        RankedCandidate[str](id="doc_c", rank=2, score=0.7),
    ]
    sparse_run = [
        RankedCandidate[str](id="doc_b", rank=1, score=10.0),
        RankedCandidate[str](id="doc_c", rank=2, score=9.0),
    ]

    results = compute_rrf({"dense": dense_run, "sparse": sparse_run}, RRFConfig(k=60))

    # doc_c aparece em ambos no rank 2: 1/62 + 1/62 = 2/62 = 0.032258
    # doc_a e doc_b aparecem apenas em 1 no rank 1: 1/61 = 0.016393
    # Portanto doc_c deve vencer e ser rank 1!
    assert results[0].id == "doc_c"
    assert results[0].rank == 1
    assert results[0].ranks_by_system == {"dense": 2, "sparse": 2}


def test_rrf_weights() -> None:
    dense_run = [RankedCandidate[str](id="doc_dense", rank=1, score=0.9)]
    sparse_run = [RankedCandidate[str](id="doc_sparse", rank=1, score=10.0)]

    # Dando o dobro de peso ao sistema denso
    config = RRFConfig(k=60, weights={"dense": 2.0, "sparse": 1.0})
    results = compute_rrf({"dense": dense_run, "sparse": sparse_run}, config)

    assert results[0].id == "doc_dense"
    assert results[1].id == "doc_sparse"
    assert results[0].rrf_score > results[1].rrf_score


def test_rrf_payload_preservation() -> None:
    payload_obj = {"title": "RAG com Hybrid Search"}
    dense_run = [RankedCandidate[dict[str, str]](id="doc_1", rank=1, payload=payload_obj)]

    results = compute_rrf({"dense": dense_run})
    assert results[0].payload == payload_obj


def test_rrf_invalid_k_and_weight() -> None:
    with pytest.raises(ValueError):
        RRFConfig(k=0)

    with pytest.raises(ValueError):
        compute_rrf(
            {"dense": [RankedCandidate[str](id="1", rank=1)]},
            RRFConfig(k=60, weights={"dense": -0.5}),
        )

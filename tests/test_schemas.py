import pytest
from pydantic import ValidationError

from app.schemas.domain import Chunk, SearchResult
from app.schemas.mcp import (
    EvalMetric,
    HybridSearchInput,
)


def test_chunk_creation_valid() -> None:
    chunk = Chunk(
        id="chk_01",
        document_id="doc_01",
        content="Texto de exemplo para indexação vetorial.",
        embedding=[0.1, 0.2, 0.3],
        token_count=8,
        metadata={"section": "introducao"},
    )
    assert chunk.id == "chk_01"
    assert chunk.embedding is not None and len(chunk.embedding) == 3
    assert chunk.token_count == 8


def test_chunk_empty_content_raises() -> None:
    with pytest.raises(ValidationError):
        Chunk(id="chk_02", document_id="doc_01", content="")


def test_hybrid_search_input_validation() -> None:
    # Validação dos limites de alpha (0.0 a 1.0)
    valid_input = HybridSearchInput(query="o que e o trynux?", alpha=0.7, top_k=10)
    assert valid_input.top_k == 10
    assert valid_input.alpha == 0.7

    with pytest.raises(ValidationError):
        HybridSearchInput(query="teste", alpha=1.5)  # Alpha acima de 1.0


def test_eval_metric_score_bounds() -> None:
    metric = EvalMetric(
        name="faithfulness",
        score=0.95,
        passed=True,
        reasoning="Todas as sentenças constam no contexto.",
    )
    assert metric.score == 0.95
    assert metric.passed is True

    with pytest.raises(ValidationError):
        EvalMetric(name="relevance", score=1.2, passed=True)  # Score > 1.0


def test_search_result_rrf() -> None:
    chunk = Chunk(id="chk_03", document_id="doc_02", content="Conteudo recuperado.")
    res = SearchResult(
        chunk=chunk,
        dense_score=0.91,
        sparse_score=12.4,
        rrf_score=0.032,
        rank=1,
    )
    assert res.rank == 1
    assert res.rrf_score == 0.032

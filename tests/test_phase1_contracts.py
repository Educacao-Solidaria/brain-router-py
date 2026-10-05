"""Validação integral dos contratos e schemas Pydantic da Fase 1."""

import inspect

import pytest
from pydantic import ValidationError

from app.clients.interfaces import (
    BaseEmbeddingClient,
    BaseRerankerClient,
    MockEmbeddingClient,
    MockRerankerClient,
)
from app.errors.exceptions import (
    BrainRouterError,
    CollectionNotFoundError,
    DatabaseQueryError,
    DocumentNotFoundError,
    EmbeddingServiceError,
    RerankServiceError,
)
from app.schemas.collection import ChunkFilter, ChunkPosition, Collection, EmbeddingVector
from app.schemas.domain import Chunk, Document, SearchResult
from app.schemas.mcp import (
    EvalMetric,
    EvalRunOutput,
    HybridSearchInput,
    IndexDocumentInput,
)
from app.storage.interfaces import BaseTextSearch, BaseVectorStore


def test_contract_chunk_and_document_serialization() -> None:
    doc = Document(
        id="doc-phase1-01",
        title="Especificação da Arquitetura RAG",
        source_uri="docs/architecture.md",
        tenant_id="tenant-core",
        metadata={"version": "1.0", "author": "Luan"},
    )
    assert doc.id == "doc-phase1-01"
    assert doc.tenant_id == "tenant-core"

    chunk = Chunk(
        id="chunk-01",
        document_id=doc.id,
        content="Texto do chunk para teste de serialização.",
        embedding=[0.05] * 128,
        token_count=10,
        metadata={"pos": 0},
    )

    dumped = chunk.model_dump()
    assert dumped["id"] == "chunk-01"
    assert len(dumped["embedding"]) == 128

    json_str = chunk.model_dump_json()
    assert "chunk-01" in json_str
    reconstructed = Chunk.model_validate_json(json_str)
    assert reconstructed.id == chunk.id
    assert reconstructed.embedding == chunk.embedding


def test_contract_search_result_invariants() -> None:
    chunk = Chunk(id="chk-valid", document_id="doc-01", content="Validação de rank")
    res = SearchResult(
        chunk=chunk,
        dense_score=0.88,
        sparse_score=14.2,
        rrf_score=0.0325,
        rank=1,
    )
    assert res.rank == 1

    # Rank deve ser estritamente >= 1
    with pytest.raises(ValidationError):
        SearchResult(chunk=chunk, dense_score=0.5, rrf_score=0.1, rank=0)


def test_contract_hybrid_search_input_bounds() -> None:
    # 1. Alpha deve estar entre 0.0 e 1.0
    valid_lower = HybridSearchInput(query="busca", alpha=0.0)
    assert valid_lower.alpha == 0.0

    valid_upper = HybridSearchInput(query="busca", alpha=1.0)
    assert valid_upper.alpha == 1.0

    with pytest.raises(ValidationError):
        HybridSearchInput(query="busca", alpha=-0.01)

    with pytest.raises(ValidationError):
        HybridSearchInput(query="busca", alpha=1.01)

    # 2. top_k deve estar entre 1 e 50
    valid_k = HybridSearchInput(query="busca", top_k=50)
    assert valid_k.top_k == 50

    with pytest.raises(ValidationError):
        HybridSearchInput(query="busca", top_k=0)

    with pytest.raises(ValidationError):
        HybridSearchInput(query="busca", top_k=51)


def test_contract_index_document_input_bounds() -> None:
    # chunk_size entre 50 e 4000
    valid = IndexDocumentInput(
        title="Doc Teste",
        content="Conteúdo completo do documento para indexação",
        source_uri="uri://test",
        chunk_size=500,
        chunk_overlap=50,
    )
    assert valid.chunk_size == 500

    with pytest.raises(ValidationError):
        IndexDocumentInput(
            title="Doc",
            content="Conteúdo",
            source_uri="uri://test",
            chunk_size=40,  # Menor que 50
        )

    with pytest.raises(ValidationError):
        IndexDocumentInput(
            title="Doc",
            content="Conteúdo",
            source_uri="uri://test",
            chunk_size=5000,  # Maior que 4000
        )


def test_contract_eval_schemas() -> None:
    metric = EvalMetric(name="faithfulness", score=0.85, passed=True, reasoning="Aderente")
    out = EvalRunOutput(
        metrics=[metric],
        overall_score=0.85,
        passed_all=True,
        evaluation_id="eval-123",
    )
    assert out.overall_score == 0.85
    assert out.passed_all is True


def test_contract_collection_and_embedding_vector_schemas() -> None:
    col = Collection(
        id="col-001",
        name="financeiro-fies",
        description="Base de documentos de crédito",
        tenant_id="tenant-fies",
        dimension=1536,
    )
    assert col.dimension == 1536
    assert col.name == "financeiro-fies"

    vec = EmbeddingVector(
        values=[0.1, 0.2, 0.3],
        dimension=3,
        model="text-embedding-3-small",
        is_normalized=True,
    )
    assert vec.dimension == 3

    # Divergência de dimensão deve lançar erro
    with pytest.raises(ValidationError):
        EmbeddingVector(
            values=[0.1, 0.2],
            dimension=3,
            model="text-embedding-3-small",
        )

    chunk_pos = ChunkPosition(start_char=0, end_char=100, line_start=1, line_end=5)
    assert chunk_pos.start_char == 0
    assert chunk_pos.line_end == 5

    flt = ChunkFilter(collection_id="col-001", tags=["fies", "contratos"])
    assert flt.collection_id == "col-001"
    assert len(flt.tags) == 2


def test_contract_error_hierarchy() -> None:
    exceptions = [
        DatabaseQueryError("falha no banco"),
        DocumentNotFoundError("doc-123"),
        CollectionNotFoundError("col-123"),
        EmbeddingServiceError("erro no embedding"),
        RerankServiceError("erro no rerank"),
    ]
    for exc in exceptions:
        assert isinstance(exc, BrainRouterError)
        assert len(str(exc)) > 0
        data = exc.to_dict()
        assert "error" in data
        assert "code" in data["error"]


def test_contract_interface_signatures() -> None:
    assert inspect.isabstract(BaseVectorStore)
    assert hasattr(BaseVectorStore, "add_chunks")
    assert hasattr(BaseVectorStore, "search_vector")

    assert inspect.isabstract(BaseTextSearch)
    assert hasattr(BaseTextSearch, "search_text")

    assert inspect.isabstract(BaseEmbeddingClient)
    assert hasattr(BaseEmbeddingClient, "embed_query")
    assert hasattr(BaseEmbeddingClient, "embed_documents")

    assert inspect.isabstract(BaseRerankerClient)
    assert hasattr(BaseRerankerClient, "rerank")

    mock_client = MockEmbeddingClient(dimension=64)
    assert mock_client.dimension == 64

    mock_reranker = MockRerankerClient()
    assert hasattr(mock_reranker, "rerank")

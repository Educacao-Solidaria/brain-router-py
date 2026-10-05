import pytest

from app.clients.interfaces import MockEmbeddingClient, MockRerankerClient
from app.errors.exceptions import EmbeddingServiceError, RerankServiceError
from app.schemas.domain import Chunk


@pytest.mark.asyncio
async def test_mock_embedding_client_dimensions_and_norm() -> None:
    client = MockEmbeddingClient(dimension=64)
    assert client.dimension == 64

    vec = await client.embed_query("teste de busca vetorial")
    assert len(vec) == 64

    # Norma L2 deve ser aproximadamente 1.0
    l2_norm = sum(x * x for x in vec) ** 0.5
    assert pytest.approx(l2_norm, 0.001) == 1.0

    # Lote de documentos
    docs = ["Doc um", "Doc dois", "Doc tres"]
    batch_vecs = await client.embed_documents(docs)
    assert len(batch_vecs) == 3
    assert len(batch_vecs[0]) == 64


@pytest.mark.asyncio
async def test_mock_embedding_client_empty_text_error() -> None:
    client = MockEmbeddingClient()
    with pytest.raises(EmbeddingServiceError):
        await client.embed_query("   ")


@pytest.mark.asyncio
async def test_mock_reranker_client() -> None:
    reranker = MockRerankerClient()
    chunks = [
        Chunk(id="c1", document_id="d1", content="Inteligencia artificial e redes neurais"),
        Chunk(id="c2", document_id="d1", content="Receita de bolo de chocolate caseiro"),
        Chunk(
            id="c3",
            document_id="d2",
            content="Redes neurais profundas e aprendizado supervisionado",
        ),
    ]

    results = await reranker.rerank("redes neurais", chunks, top_n=2)
    assert len(results) == 2
    # c1 e c3 tem 'redes neurais', c2 não tem
    top_ids = [r.chunk.id for r in results]
    assert "c2" not in top_ids
    assert results[0].rank == 1
    assert results[1].rank == 2


@pytest.mark.asyncio
async def test_mock_reranker_client_empty_query() -> None:
    reranker = MockRerankerClient()
    with pytest.raises(RerankServiceError):
        await reranker.rerank("", [])

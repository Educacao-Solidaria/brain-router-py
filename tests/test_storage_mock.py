import pytest
from app.errors.exceptions import CollectionNotFoundError
from app.schemas.domain import Chunk
from app.storage.interfaces import InMemoryVectorStore


@pytest.mark.asyncio
async def test_in_memory_vector_store_crud() -> None:
    store = InMemoryVectorStore()
    col = "docs-test"

    chunks = [
        Chunk(
            id="c1",
            document_id="doc1",
            content="Introducao ao ecossistema Open Source e fluxo MCP",
            position=0,
            embedding=[1.0, 0.0, 0.0],
        ),
        Chunk(
            id="c2",
            document_id="doc1",
            content="Instrucoes sobre arquitetura de microservicos e Rust",
            position=1,
            embedding=[0.0, 1.0, 0.0],
        ),
        Chunk(
            id="c3",
            document_id="doc2",
            content="Visao geral do Gateway em Go com OpenRouter",
            position=0,
            embedding=[0.8, 0.6, 0.0],
        ),
    ]

    added = await store.add_chunks(col, chunks)
    assert added == 3
    assert await store.count_chunks(col) == 3

    # Busca vetorial orientada ao vetor [1.0, 0.0, 0.0]
    query_vec = [1.0, 0.0, 0.0]
    results = await store.search_vector(col, query_vec, limit=2)
    assert len(results) == 2
    assert results[0].chunk.id == "c1"
    assert pytest.approx(results[0].dense_score, 0.001) == 1.0
    assert results[0].rank == 1
    assert results[1].chunk.id == "c3"
    assert pytest.approx(results[1].dense_score, 0.001) == 0.8
    assert results[1].rank == 2

    # Busca textual por palavra-chave
    text_results = await store.search_text(col, "OpenRouter Gateway", limit=5)
    assert len(text_results) >= 1
    assert text_results[0].chunk.id == "c3"
    assert text_results[0].sparse_score is not None

    # Deletar coleção
    assert await store.delete_collection(col) is True
    assert await store.count_chunks(col) == 0


@pytest.mark.asyncio
async def test_in_memory_vector_store_collection_not_found() -> None:
    store = InMemoryVectorStore()
    with pytest.raises(CollectionNotFoundError):
        await store.search_vector("inexistente", [0.1, 0.2])

    with pytest.raises(CollectionNotFoundError):
        await store.search_text("inexistente", "termo")

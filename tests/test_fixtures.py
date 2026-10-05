"""Contrato das fixtures de `conftest.py`/`support.py` (sem banco)."""

import math

import pytest
from pydantic import ValidationError
from support import ChunkFactory, fake_embedding

from app.storage.interfaces import InMemoryVectorStore


def test_ids_are_sequential_and_restart_per_test(chunk_factory: ChunkFactory) -> None:
    chunks = chunk_factory.batch(3)

    assert [c.id for c in chunks] == ["chunk-1", "chunk-2", "chunk-3"]
    assert {c.document_id for c in chunks} == {"doc-1"}
    assert all(c.embedding is None for c in chunks)


def test_default_content_is_portuguese_and_counts_tokens(chunk_factory: ChunkFactory) -> None:
    chunk = chunk_factory()

    assert chunk.content == "Trecho 1 sobre matrícula e frequência."
    assert chunk.token_count == 6


def test_overrides_go_through_chunk_validation(chunk_factory: ChunkFactory) -> None:
    chunk = chunk_factory("Ementa da disciplina", id="fixo", metadata={"secao": "ementa"})
    assert (chunk.id, chunk.metadata) == ("fixo", {"secao": "ementa"})

    with pytest.raises(ValidationError):
        chunk_factory("")  # content tem min_length=1
    with pytest.raises(ValidationError):
        chunk_factory(token_count=-1)


def test_fake_embedding_is_deterministic_unit_vector() -> None:
    vector = fake_embedding("Calendário de provas", dim=16)

    assert vector == fake_embedding("Calendário de provas", dim=16)
    assert vector != fake_embedding("Calendário de férias", dim=16)
    assert len(vector) == 16
    assert math.isclose(math.fsum(x * x for x in vector), 1.0)


def test_embedded_chunks_follow_factory_dimension() -> None:
    chunks = ChunkFactory(document_id="doc-9", dim=4).batch(2, embed=True)

    assert all(c.embedding is not None and len(c.embedding) == 4 for c in chunks)
    assert {c.document_id for c in chunks} == {"doc-9"}


async def test_factory_feeds_the_in_memory_store(chunk_factory: ChunkFactory) -> None:
    store = InMemoryVectorStore()
    target = chunk_factory("Regras de rematrícula do semestre", embed=True)
    await store.add_chunks("docs", [*chunk_factory.batch(4, embed=True), target])

    assert target.embedding is not None
    (best,) = await store.search_vector("docs", target.embedding, limit=1)
    assert best.chunk.id == target.id
    assert math.isclose(best.dense_score or 0.0, 1.0)

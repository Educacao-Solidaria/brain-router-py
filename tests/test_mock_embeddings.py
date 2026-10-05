"""Testes unitários para o gerador determinístico de embeddings sintéticos."""

import math

import pytest

from app.errors.exceptions import EmbeddingServiceError
from app.testing.embedding_mock import DeterministicEmbeddingGenerator


@pytest.mark.asyncio
async def test_deterministic_reproducibility() -> None:
    gen1 = DeterministicEmbeddingGenerator(dimension=256, seed=42)
    gen2 = DeterministicEmbeddingGenerator(dimension=256, seed=42)

    vec1 = await gen1.embed_query("pesquisa vetorial com pgvector")
    vec2 = await gen2.embed_query("pesquisa vetorial com pgvector")

    assert len(vec1) == 256
    assert len(vec2) == 256
    assert vec1 == vec2, "Mesmo texto e seed devem produzir vetores idênticos."


@pytest.mark.asyncio
async def test_distinct_seeds_yield_distinct_vectors() -> None:
    gen1 = DeterministicEmbeddingGenerator(dimension=128, seed=1)
    gen2 = DeterministicEmbeddingGenerator(dimension=128, seed=2)

    vec1 = await gen1.embed_query("busca hibrida")
    vec2 = await gen2.embed_query("busca hibrida")

    assert vec1 != vec2, "Seeds distintas devem produzir vetores diferentes."


@pytest.mark.asyncio
async def test_unit_norm_contract() -> None:
    gen = DeterministicEmbeddingGenerator(dimension=384, seed=100)
    vec = await gen.embed_query("verificação de norma euclidiana unitária")

    norm = math.sqrt(sum(x * x for x in vec))
    assert abs(norm - 1.0) < 1e-5, f"Norma deve ser 1.0, obteve {norm}"


@pytest.mark.asyncio
async def test_batch_embedding_consistency() -> None:
    gen = DeterministicEmbeddingGenerator(dimension=64, seed=55)
    texts = [
        "primeiro documento",
        "segundo documento",
        "terceiro documento",
    ]

    batch = await gen.embed_documents(texts)
    assert len(batch) == 3
    for v in batch:
        assert len(v) == 64

    # Vetor individual deve coincidir com o item do lote
    single = await gen.embed_query("segundo documento")
    assert batch[1] == single


@pytest.mark.asyncio
async def test_empty_text_raises_error() -> None:
    gen = DeterministicEmbeddingGenerator()
    with pytest.raises(EmbeddingServiceError):
        await gen.embed_query("   ")


@pytest.mark.asyncio
async def test_stats_and_reset() -> None:
    gen = DeterministicEmbeddingGenerator(dimension=128)
    assert gen.call_count == 0
    assert gen.total_tokens_estimated == 0

    await gen.embed_query("palavra1 palavra2 palavra3")
    assert gen.call_count == 1
    assert gen.total_tokens_estimated == 3

    gen.reset_stats()
    assert gen.call_count == 0
    assert gen.total_tokens_estimated == 0


@pytest.mark.asyncio
async def test_simulated_failure() -> None:
    gen = DeterministicEmbeddingGenerator(fail_rate=1.0)
    with pytest.raises(EmbeddingServiceError, match="Falha simulada"):
        await gen.embed_query("texto com falha garantida")

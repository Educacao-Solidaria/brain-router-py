import pytest
from pydantic import ValidationError

from app.schemas.collection import ChunkFilter, ChunkPosition, Collection, EmbeddingVector


def test_collection_creation() -> None:
    col = Collection(
        id="col_tech",
        name="Artigos Tecnicos",
        description="Base de documentos de arquitetura",
        embedding_model="openai/text-embedding-3-small",
        dimension=1536,
    )
    assert col.name == "Artigos Tecnicos"
    assert col.dimension == 1536
    assert col.tenant_id == "default"


def test_embedding_vector_dimension_validation() -> None:
    # Deve ser válido quando dimension bate com a quantidade de valores
    vec = EmbeddingVector(
        values=[0.1, 0.2, 0.3],
        dimension=3,
        model="text-embedding-3-small",
        is_normalized=True,
    )
    assert len(vec.values) == 3

    # Divergência de dimensão deve disparar ValidationError
    with pytest.raises(ValidationError):
        EmbeddingVector(
            values=[0.1, 0.2],
            dimension=3,
            model="text-embedding-3-small",
        )


def test_chunk_position() -> None:
    pos = ChunkPosition(start_char=100, end_char=250, line_start=5, line_end=12)
    assert pos.start_char == 100
    assert pos.line_start == 5


def test_chunk_filter() -> None:
    f = ChunkFilter(collection_id="col_01", tags=["arquitetura", "ia"])
    assert f.collection_id == "col_01"
    assert len(f.tags) == 2

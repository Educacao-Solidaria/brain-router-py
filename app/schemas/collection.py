from datetime import UTC, datetime

from pydantic import BaseModel, Field, ValidationInfo, field_validator


class Collection(BaseModel):
    """Representa uma coleção lógica particionada de documentos e embeddings."""

    id: str = Field(..., description="Identificador único da coleção")
    name: str = Field(..., min_length=2, max_length=100, description="Nome legível da coleção")
    description: str | None = Field(None, description="Descrição do propósito da base")
    tenant_id: str = Field(default="default", description="Tenant dono da coleção")
    embedding_model: str = Field(
        default="openai/text-embedding-3-small",
        description="Modelo de embedding associado à coleção",
    )
    dimension: int = Field(1536, ge=128, le=4096, description="Dimensão dos vetores")
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    document_count: int = Field(0, ge=0)


class ChunkPosition(BaseModel):
    """Localização exata do chunk no documento original."""

    start_char: int = Field(0, ge=0)
    end_char: int = Field(0, ge=0)
    line_start: int | None = Field(None, ge=1)
    line_end: int | None = Field(None, ge=1)


class EmbeddingVector(BaseModel):
    """Vetor de embedding fortemente tipado e validado."""

    values: list[float] = Field(..., min_length=1)
    dimension: int = Field(..., ge=1)
    model: str = Field(..., min_length=1)
    is_normalized: bool = Field(default=False)

    @field_validator("dimension")
    @classmethod
    def validate_dimension_match(cls, v: int, info: ValidationInfo) -> int:
        values = info.data.get("values")
        if values is not None and len(values) != v:
            raise ValueError(f"Tamanho de values ({len(values)}) diverge de dimension ({v})")
        return v


class ChunkFilter(BaseModel):
    """Filtros para busca particionada por coleção e atributos."""

    collection_id: str | None = Field(None, description="Filtrar por ID de coleção específica")
    tags: list[str] = Field(default_factory=list, description="Lista de tags necessárias")
    created_after: datetime | None = Field(None, description="Corte temporal inicial")
    created_before: datetime | None = Field(None, description="Corte temporal final")

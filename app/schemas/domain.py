from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """Representa um fragmento de texto indexado para busca híbrida."""

    id: str = Field(..., description="Identificador único do chunk")
    document_id: str = Field(..., description="ID do documento pai")
    content: str = Field(..., min_length=1, description="Texto do fragmento")
    embedding: Optional[List[float]] = Field(
        None, description="Vetor denso de embedding (ex: 1536 dimensões)"
    )
    token_count: int = Field(0, ge=0, description="Contagem estimada de tokens")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Metadados arbitrários (tenant, seção, etc)"
    )


class Document(BaseModel):
    """Representa um documento completo ingerido no sistema."""

    id: str = Field(..., description="Identificador único do documento")
    title: str = Field(..., min_length=1, description="Título do documento")
    source_uri: str = Field(..., description="URI de origem (arquivo, URL, etc)")
    tenant_id: str = Field(
        default="default", description="Identificador do tenant para isolamento"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="Data de criação"
    )
    chunks_count: int = Field(0, ge=0, description="Quantidade total de chunks gerados")


class SearchResult(BaseModel):
    """Resultado individual retornado pela busca híbrida."""

    chunk: Chunk = Field(..., description="Chunk recuperado")
    dense_score: Optional[float] = Field(
        None, description="Score de similaridade cosseno (0 a 1)"
    )
    sparse_score: Optional[float] = Field(
        None, description="Score da busca textual BM25/tsvector"
    )
    rrf_score: float = Field(
        ..., description="Score fundido via Reciprocal Rank Fusion"
    )
    rank: int = Field(..., ge=1, description="Posição no ranking final")

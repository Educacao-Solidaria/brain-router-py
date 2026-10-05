from typing import Any

from pydantic import BaseModel, Field

from app.schemas.domain import SearchResult


class HybridSearchInput(BaseModel):
    """Parâmetros de entrada da MCP Tool `hybrid_search`."""

    query: str = Field(..., min_length=1, description="Texto da consulta do usuário")
    top_k: int = Field(5, ge=1, le=50, description="Número de resultados a retornar")
    alpha: float = Field(
        0.5, ge=0.0, le=1.0, description="Ponderação entre busca densa (1.0) e esparsa (0.0)"
    )
    tenant_id: str | None = Field(None, description="Filtro opcional por tenant específico")
    filter_metadata: dict[str, Any] | None = Field(
        None, description="Filtros adicionais por metadados"
    )


class HybridSearchOutput(BaseModel):
    """Retorno estruturado da MCP Tool `hybrid_search`."""

    query: str
    results: list[SearchResult]
    total_found: int
    execution_time_ms: float


class IndexDocumentInput(BaseModel):
    """Parâmetros de entrada da MCP Tool `index_document`."""

    title: str = Field(..., min_length=1, description="Título do documento")
    content: str = Field(..., min_length=1, description="Conteúdo textual completo")
    source_uri: str = Field(..., description="Caminho ou URL de origem")
    tenant_id: str = Field(default="default", description="Identificador do tenant")
    chunk_size: int = Field(500, ge=50, le=4000, description="Tamanho médio de tokens por chunk")
    chunk_overlap: int = Field(50, ge=0, le=500, description="Sobreposição entre chunks")


class IndexDocumentOutput(BaseModel):
    """Retorno estruturado da MCP Tool `index_document`."""

    document_id: str
    chunks_created: int
    success: bool
    message: str


class EvalMetric(BaseModel):
    """Métrica individual de avaliação de qualidade."""

    name: str = Field(..., description="Nome da métrica (ex: faithfulness, relevance)")
    score: float = Field(..., ge=0.0, le=1.0, description="Pontuação normalizada entre 0 e 1")
    passed: bool = Field(..., description="Indica se atingiu o limiar mínimo acordado")
    reasoning: str | None = Field(None, description="Justificativa emitida pelo LLM judge")


class EvalRunInput(BaseModel):
    """Parâmetros de entrada da MCP Tool `run_eval`."""

    query: str = Field(..., description="Pergunta enviada ao modelo")
    response: str = Field(..., description="Resposta gerada pelo pipeline")
    contexts: list[str] = Field(..., description="Trechos recuperados pelo RAG")
    ground_truth: str | None = Field(None, description="Resposta ideal de referência")


class EvalRunOutput(BaseModel):
    """Retorno estruturado da MCP Tool `run_eval`."""

    metrics: list[EvalMetric]
    overall_score: float = Field(..., ge=0.0, le=1.0)
    passed_all: bool
    evaluation_id: str

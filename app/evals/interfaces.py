"""Contratos abstratos e modelos Pydantic para avaliação de pipelines RAG."""

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field


class EvalInput(BaseModel):
    """Payload de entrada padronizado para avaliação de qualidade RAG."""

    query: str = Field(..., min_length=1, description="Pergunta ou instrução original")
    contexts: list[str] = Field(
        default_factory=list, description="Lista de trechos de contexto recuperados"
    )
    answer: str = Field(..., description="Resposta gerada pelo modelo")
    ground_truth: str | None = Field(
        default=None, description="Resposta de referência esperada (opcional)"
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Metadados de rastreio")


class EvalOutput(BaseModel):
    """Resultado estruturado de uma métrica de avaliação."""

    metric_name: str = Field(
        ..., description="Identificador da métrica (ex: faithfulness, relevance)"
    )
    score: float = Field(..., ge=0.0, le=1.0, description="Pontuação normalizada entre 0.0 e 1.0")
    passed: bool = Field(..., description="Indica se atingiu o limiar de qualidade estabelecido")
    threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Limiar mínimo exigido")
    reason: str = Field(..., description="Justificativa ou detalhamento analítico da pontuação")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Métricas secundárias e rastreio"
    )


class BaseEvaluator(ABC):
    """Interface abstrata base para qualquer avaliador de RAG."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Nome único identificador da métrica."""
        ...

    @property
    def default_threshold(self) -> float:
        """Limiar padrão de aprovação da métrica."""
        return 0.70

    @abstractmethod
    async def evaluate(self, input_data: EvalInput, threshold: float | None = None) -> EvalOutput:
        """Executa a avaliação assíncrona sobre os dados de entrada."""
        ...

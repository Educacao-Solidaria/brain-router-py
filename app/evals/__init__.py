"""Módulo de avaliação e métricas de qualidade de RAG."""

from app.evals.interfaces import BaseEvaluator, EvalInput, EvalOutput
from app.evals.metrics import (
    AnswerRelevanceEvaluator,
    ContextRelevanceEvaluator,
    FaithfulnessEvaluator,
)

__all__ = [
    "BaseEvaluator",
    "EvalInput",
    "EvalOutput",
    "FaithfulnessEvaluator",
    "ContextRelevanceEvaluator",
    "AnswerRelevanceEvaluator",
]

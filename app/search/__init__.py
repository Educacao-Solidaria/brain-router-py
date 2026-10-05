"""Módulo de algoritmos de busca e fusão para o motor RAG."""

from app.search.rrf import (
    RankedCandidate,
    RRFConfig,
    RRFMergedResult,
    compute_rrf,
)

__all__ = [
    "RankedCandidate",
    "RRFConfig",
    "RRFMergedResult",
    "compute_rrf",
]

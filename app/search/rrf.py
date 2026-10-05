"""Implementação matemática canônica do algoritmo Reciprocal Rank Fusion (RRF)."""

from pydantic import BaseModel, Field


class RankedCandidate[T](BaseModel):
    """Candidato individual em uma lista ranqueada de busca."""

    id: str = Field(..., description="Identificador único do documento ou chunk")
    rank: int = Field(..., ge=1, description="Posição 1-based no ranking retornado")
    score: float | None = Field(default=None, description="Score bruto retornado pelo motor")
    payload: T | None = Field(default=None, description="Objeto associado ao candidato")

    model_config = {"arbitrary_types_allowed": True}


class RRFConfig(BaseModel):
    """Configurações paramétricas da fusão RRF."""

    k: int = Field(default=60, gt=0, description="Constante de amortecimento de rank (padrão 60)")
    weights: dict[str, float] = Field(
        default_factory=dict,
        description="Ponderações relativas por motor de busca (ex: {'dense': 1.0, 'sparse': 0.8})",
    )


class RRFMergedResult[T](BaseModel):
    """Resultado consolidado da fusão recíproca de rankings."""

    id: str
    rrf_score: float = Field(..., ge=0.0, description="Score acumulado pela fórmula RRF")
    rank: int = Field(..., ge=1, description="Posição final consolidada 1-based")
    ranks_by_system: dict[str, int] = Field(
        default_factory=dict, description="Rank 1-based em cada motor originário"
    )
    scores_by_system: dict[str, float] = Field(
        default_factory=dict, description="Score original em cada motor originário"
    )
    payload: T | None = None

    model_config = {"arbitrary_types_allowed": True}


def compute_rrf[T](
    ranked_runs: dict[str, list[RankedCandidate[T]]],
    config: RRFConfig | None = None,
) -> list[RRFMergedResult[T]]:
    """Calcula o Reciprocal Rank Fusion sobre múltiplos rankings de busca.

    Fórmula:
        RRF(d) = sum_{m in M} ( w_m / (k + rank_m(d)) )
    """
    cfg = config or RRFConfig()
    k = cfg.k

    accumulated_scores: dict[str, float] = {}
    ranks_per_system: dict[str, dict[str, int]] = {}
    scores_per_system: dict[str, dict[str, float]] = {}
    payload_map: dict[str, T | None] = {}

    for system_name, candidates in ranked_runs.items():
        weight = cfg.weights.get(system_name, 1.0)
        if weight < 0:
            raise ValueError(f"O peso do motor '{system_name}' não pode ser negativo ({weight}).")

        for cand in candidates:
            cand_id = cand.id
            if cand_id not in accumulated_scores:
                accumulated_scores[cand_id] = 0.0
                ranks_per_system[cand_id] = {}
                scores_per_system[cand_id] = {}
                payload_map[cand_id] = cand.payload

            # Fórmula RRF com peso relativo
            contribution = weight / (k + cand.rank)
            accumulated_scores[cand_id] += contribution
            ranks_per_system[cand_id][system_name] = cand.rank
            if cand.score is not None:
                scores_per_system[cand_id][system_name] = cand.score

    # Ordenação decrescente de pontuação com desempate determinístico por ID
    sorted_items = sorted(
        accumulated_scores.items(),
        key=lambda item: (-item[1], item[0]),
    )

    results: list[RRFMergedResult[T]] = []
    for final_rank, (doc_id, total_score) in enumerate(sorted_items, start=1):
        results.append(
            RRFMergedResult[T](
                id=doc_id,
                rrf_score=round(total_score, 6),
                rank=final_rank,
                ranks_by_system=ranks_per_system[doc_id],
                scores_by_system=scores_per_system[doc_id],
                payload=payload_map[doc_id],
            )
        )

    return results

"""Normalização de scores e deduplicação de resultados, antes do ranking.

- `min_max_normalize` leva os scores de um motor para [0, 1], para que cosseno e
  `ts_rank` (escalas diferentes) possam ser combinados, como pede o `alpha` do
  `HybridSearchInput`.
- `deduplicate` tira o resultado repetido ou contido em outro, mantendo o de maior score.
"""

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from pydantic import ValidationError

from app.schemas import ChunkPosition, SearchResult

Score = Callable[[SearchResult], float | None]


def min_max_normalize(scores: Sequence[float | None]) -> list[float]:
    """O menor score vira 0 e o maior vira 1, na mesma ordem da entrada.

    - Lista vazia → `[]`.
    - Todos iguais (inclusive um só) → todos `1.0`: sem como diferenciar, cada um vale o
      melhor da lista. Com 0, o único resultado de um motor sairia como o pior.
    - `None`, NaN e ±inf → `0.0`, e ficam fora do mínimo e do máximo.
    """
    finite = [score for score in scores if score is not None and math.isfinite(score)]
    if not finite:
        return [0.0] * len(scores)
    low, high = min(finite), max(finite)
    if high == low:
        return [1.0 if score is not None and math.isfinite(score) else 0.0 for score in scores]
    # `high - low` estoura para |score| perto de 1e308; um quarto (exato em ponto
    # flutuante) mantém a conta finita sem mudar a proporção.
    scale = 0.25 if math.isinf(high - low) else 1.0
    low, high = low * scale, high * scale
    span = high - low
    return [
        (score * scale - low) / span if score is not None and math.isfinite(score) else 0.0
        for score in scores
    ]


def _rrf_score(result: SearchResult) -> float:
    return result.rrf_score


@dataclass(frozen=True, slots=True)
class _Candidate:
    order: int  # posição na entrada
    result: SearchResult
    padded: str  # texto normalizado entre espaços: contém só palavra inteira
    span: tuple[int, int] | None
    score: float


def _span(result: SearchResult) -> tuple[int, int] | None:
    """O span de `metadata["position"]`, se for um `ChunkPosition` válido e não vazio."""
    raw = result.chunk.metadata.get("position")
    if raw is None:
        return None
    try:
        position = ChunkPosition.model_validate(raw)
    except ValidationError:
        return None  # metadata é livre: posição malformada só desliga o critério de span
    if position.end_char <= position.start_char:
        return None
    return position.start_char, position.end_char


def _same_passage(a: _Candidate, b: _Candidate, min_overlap: float) -> bool:
    if a.result.chunk.id == b.result.chunk.id or a.padded == b.padded:
        return True
    if a.result.chunk.document_id != b.result.chunk.document_id:
        return False
    if a.padded in b.padded or b.padded in a.padded:
        return True
    if a.span is None or b.span is None:
        return False
    shared = min(a.span[1], b.span[1]) - max(a.span[0], b.span[0])
    shortest = min(a.span[1] - a.span[0], b.span[1] - b.span[0])
    return shared > 0 and shared >= min_overlap * shortest


def deduplicate(
    results: Sequence[SearchResult], *, score: Score = _rrf_score, min_overlap: float = 0.5
) -> list[SearchResult]:
    """Remove os resultados que repetem outro, mantendo o de maior `score`.

    Dois resultados são o mesmo trecho quando:
    - têm o mesmo `chunk.id` (o mesmo chunk veio dos dois motores);
    - têm o mesmo texto, ignorando caixa e espaços, de qualquer documento;
    - são do mesmo documento e o texto de um contém o do outro (palavras inteiras), ou
      os spans de `metadata["position"]` se sobrepõem em pelo menos `min_overlap` do
      menor (0 = qualquer sobreposição). Documentos diferentes não se comparam por
      contenção: cada um é uma fonte que a resposta pode citar.

    Fica o de maior score (`None` e NaN perdem de qualquer número); no empate, o mais
    longo, que cobre mais. Se o trecho contido tem o score maior, é o maior que sai:
    perde-se o contexto em volta, mas fica o trecho que a consulta achou melhor (o span
    continua em `metadata` para quem quiser reabrir a vizinhança). A saída preserva a
    ordem da entrada: quem rankeia é a etapa seguinte.

    ponytail: compara todos com todos os mantidos, O(n²). Serve para os candidatos de
    uma consulta (até 2 × top_k ≤ 100); para milhares, indexar por documento e texto.
    """
    if not 0.0 <= min_overlap <= 1.0:
        raise ValueError(f"min_overlap deve estar em [0, 1], veio {min_overlap}")
    candidates = []
    for order, result in enumerate(results):
        value = score(result)
        candidates.append(
            _Candidate(
                order=order,
                result=result,
                padded=f" {' '.join(result.chunk.content.split()).casefold()} ",
                span=_span(result),
                score=value if value is not None and not math.isnan(value) else -math.inf,
            )
        )
    kept: list[_Candidate] = []
    for candidate in sorted(candidates, key=lambda c: (-c.score, -len(c.padded), c.order)):
        if not any(_same_passage(candidate, other, min_overlap) for other in kept):
            kept.append(candidate)
    return [candidate.result for candidate in sorted(kept, key=lambda c: c.order)]

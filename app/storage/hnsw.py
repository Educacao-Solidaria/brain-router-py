"""Afinamento do HNSW do pgvector: `m` e `ef_construction` no índice, `ef_search` por consulta.

O índice HNSW (cosine) do `chunks.embedding` é do PR 22 (Luan); este módulo não o cria.
Entrega os parâmetros validados para o CREATE INDEX dele e o ajuste por consulta.

- `HnswBuildParams`: `m` (ligações por nó em cada camada) e `ef_construction` (candidatos
  examinados ao inserir) só valem na construção do índice: mudar depois exige REINDEX.
  Faixas do pgvector: `m` em [2, 100], `ef_construction` em [4, 1000] e >= 2·m. Os perfis
  usam `ef_construction = 4·m`, a proporção do padrão do pgvector (16, 64): `fast` gasta
  menos memória e constrói mais rápido, `high` dá mais recall com mais memória por vetor.
- `ef_search_for(limit, ...)`: tamanho da lista de candidatos da busca. Sem varredura
  iterativa, o HNSW devolve no máximo `ef_search` linhas: um LIMIT 100 com o padrão de 40
  volta com 40. Por isso o valor nunca fica abaixo do `limit`; cresce com o perfil de recall
  e dobra com filtro, que o Postgres aplica depois do índice e que descarta candidatos.
- `apply_ef_search(session, ef_search)`: `set_config('hnsw.ef_search', :v, true)` na
  transação corrente. O `true` é o `SET LOCAL`: o valor some no fim da transação. Um `SET`
  de sessão ficaria na conexão e vazaria para a próxima requisição que a pegasse do pool.
  `SET` não aceita parâmetro; `set_config` aceita, e o inteiro é validado antes, então
  nenhum valor vira texto de SQL.

ponytail: com filtro muito seletivo, dobrar o `ef_search` pode não bastar; o passo seguinte
é a varredura iterativa do pgvector 0.8 (`hnsw.iterative_scan = relaxed_order`), depois de
confirmar a versão do pgvector em produção.
"""

from dataclasses import dataclass
from typing import Literal, Self

from sqlalchemy import text
from sqlmodel.ext.asyncio.session import AsyncSession

Recall = Literal["fast", "balanced", "high"]

M_RANGE = (2, 100)
EF_CONSTRUCTION_RANGE = (4, 1000)
EF_SEARCH_RANGE = (1, 1000)
# Por perfil: o m do índice, o multiplicador do limit e o piso do ef_search (40 é o padrão).
_PROFILES: dict[str, tuple[int, int, int]] = {
    "fast": (12, 1, 20),
    "balanced": (16, 2, 40),
    "high": (32, 4, 100),
}
_SET_EF_SEARCH = text("SELECT set_config('hnsw.ef_search', :value, true)")


def _int_in(name: str, value: object, bounds: tuple[int, int]) -> int:
    low, high = bounds
    # bool é subclasse de int: True passaria como 1.
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} deve ser inteiro em [{low}, {high}], veio {value!r}")
    return value


def _profile(recall: str) -> tuple[int, int, int]:
    if recall not in _PROFILES:
        raise ValueError(f"recall deve ser um de {sorted(_PROFILES)}, veio {recall!r}")
    return _PROFILES[recall]


@dataclass(frozen=True, slots=True)
class HnswBuildParams:
    """`m` e `ef_construction` validados; o padrão é o do pgvector."""

    m: int = 16
    ef_construction: int = 64

    def __post_init__(self) -> None:
        _int_in("m", self.m, M_RANGE)
        _int_in("ef_construction", self.ef_construction, EF_CONSTRUCTION_RANGE)
        if self.ef_construction < 2 * self.m:
            raise ValueError(
                f"ef_construction ({self.ef_construction}) deve ser >= 2 * m ({2 * self.m})"
            )

    @classmethod
    def for_recall(cls, recall: Recall) -> Self:
        m = _profile(recall)[0]
        return cls(m=m, ef_construction=4 * m)

    def with_clause(self) -> str:
        """`WITH (...)` do CREATE INDEX ... USING hnsw; só inteiros já validados entram aqui."""
        return f"WITH (m = {self.m:d}, ef_construction = {self.ef_construction:d})"


def ef_search_for(limit: int, *, recall: Recall = "balanced", filtered: bool = False) -> int:
    """O `ef_search` para devolver `limit` vizinhos: nunca abaixo de `limit`, no máximo 1000."""
    limit = _int_in("limit", limit, EF_SEARCH_RANGE)
    _, factor, floor = _profile(recall)
    ef_search = max(limit * factor, floor) * (2 if filtered else 1)
    return min(max(ef_search, limit), EF_SEARCH_RANGE[1])


async def apply_ef_search(session: AsyncSession, ef_search: int) -> None:
    """Fixa `hnsw.ef_search` só até o fim da transação corrente de `session` (SET LOCAL)."""
    value = _int_in("ef_search", ef_search, EF_SEARCH_RANGE)
    connection = await session.connection()
    await connection.execute(_SET_EF_SEARCH, {"value": str(value)})

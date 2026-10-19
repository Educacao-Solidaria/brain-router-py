"""Parâmetros do HNSW (`app.storage.hnsw`) sem banco: faixas, perfis e o SQL emitido."""

from typing import Any, cast

import pytest
from sqlmodel.ext.asyncio.session import AsyncSession

from app.storage.hnsw import HnswBuildParams, Recall, apply_ef_search, ef_search_for

PROFILES: list[Recall] = ["fast", "balanced", "high"]
# Se apply_ef_search tocar na sessão, `None` quebra o teste: a validação vem antes do banco.
NO_SESSION = cast(AsyncSession, None)


class Recorder:
    """Sessão falsa: guarda o que seria executado."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []

    async def connection(self) -> "Recorder":
        return self

    async def execute(self, statement: Any, params: Any) -> None:
        self.calls.append((str(statement), params))


def test_defaults_are_the_pgvector_defaults() -> None:
    assert HnswBuildParams() == HnswBuildParams(m=16, ef_construction=64)
    assert HnswBuildParams().with_clause() == "WITH (m = 16, ef_construction = 64)"


@pytest.mark.parametrize(
    ("recall", "m", "ef_construction"),
    [("fast", 12, 48), ("balanced", 16, 64), ("high", 32, 128)],
)
def test_profiles_keep_ef_construction_at_four_times_m(
    recall: Recall, m: int, ef_construction: int
) -> None:
    assert HnswBuildParams.for_recall(recall) == HnswBuildParams(m, ef_construction)


@pytest.mark.parametrize(
    "options",
    [
        {"m": 1},
        {"m": 101, "ef_construction": 1000},
        {"ef_construction": 3},
        {"ef_construction": 1001},
        {"m": 16, "ef_construction": 31},  # abaixo de 2 * m: o pgvector recusaria o índice
        {"m": True},
        {"m": 16.0},
        {"ef_construction": "64"},
        {"m": "16); DROP TABLE chunks; --"},
    ],
)
def test_build_params_outside_the_pgvector_ranges_are_rejected(options: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        HnswBuildParams(**options)


def test_build_params_accept_the_edges_of_the_ranges() -> None:
    assert HnswBuildParams(m=2, ef_construction=4).with_clause() == (
        "WITH (m = 2, ef_construction = 4)"
    )
    assert HnswBuildParams(m=100, ef_construction=1000).ef_construction == 1000


@pytest.mark.parametrize(
    ("limit", "recall", "filtered", "expected"),
    [
        (10, "balanced", False, 40),  # o piso é o padrão do pgvector
        (50, "balanced", False, 100),
        (100, "balanced", False, 200),  # 2 x top_k=50 de candidatos densos
        (10, "fast", False, 20),
        (10, "high", False, 100),
        (10, "balanced", True, 80),  # filtro dobra: candidatos somem depois do índice
        (600, "high", False, 1000),  # teto do pgvector
        (1000, "fast", True, 1000),
        (1, "fast", False, 20),
    ],
)
def test_ef_search_for(limit: int, recall: Recall, filtered: bool, expected: int) -> None:
    assert ef_search_for(limit, recall=recall, filtered=filtered) == expected


def test_property_ef_search_never_below_limit_and_never_above_the_ceiling() -> None:
    for recall in PROFILES:
        for filtered in (False, True):
            values = [ef_search_for(n, recall=recall, filtered=filtered) for n in range(1, 1001)]
            assert all(n <= ef <= 1000 for n, ef in enumerate(values, start=1))
            assert values == sorted(values)  # mais vizinhos pedidos, nunca menos candidatos


@pytest.mark.parametrize("limit", [0, -1, 1001, True, 1.5, "10"])
def test_limit_outside_what_hnsw_can_return_is_rejected(limit: Any) -> None:
    with pytest.raises(ValueError, match="limit"):
        ef_search_for(limit)


def test_unknown_profile_is_rejected() -> None:
    with pytest.raises(ValueError, match="recall"):
        ef_search_for(10, recall="max")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="recall"):
        HnswBuildParams.for_recall("max")  # type: ignore[arg-type]


async def test_apply_ef_search_is_local_to_the_transaction_and_bound_as_a_parameter() -> None:
    session = Recorder()

    await apply_ef_search(cast(AsyncSession, session), 120)

    ((statement, params),) = session.calls
    assert statement == "SELECT set_config('hnsw.ef_search', :value, true)"
    assert params == {"value": "120"}


@pytest.mark.parametrize("value", [0, 1001, -5, True, 100.0, "100", "100; RESET ALL"])
async def test_apply_ef_search_validates_before_touching_the_database(value: Any) -> None:
    with pytest.raises(ValueError, match="ef_search"):
        await apply_ef_search(NO_SESSION, value)

"""Profiler de memória (`app.bench.memleak`): o critério, o diff por linha e o liga/desliga."""

import asyncio
import linecache
import math
import tracemalloc
from contextvars import ContextVar
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from sqlalchemy import text
from sqlmodel.ext.asyncio.session import AsyncSession

import app.ingest
from app.bench import memleak
from app.chunking import count_words
from app.evals.gate import AnyTermSearch
from app.settings import get_settings

SHOWN: dict[str, Any] = {"clients": 3, "seconds": 1.0, "warmup": 0.2}
FAST = SHOWN | {"interval": 0.05}
EMPTY = tracemalloc.Snapshot((), 1)
LIMIT = memleak.MAX_BYTES_PER_CALL
STATE = ContextVar[bytes]("state")  # cada tarefa de cliente tem o seu contexto


def keeping(kept: list[bytes], until: float = math.inf) -> memleak.Call:
    """Cada chamada aloca 1000 B e os guarda em `kept` até o instante `until` do loop."""

    async def call(client: int, n: int) -> None:
        if n == 0:  # 50 kB que vivem com a tarefa do cliente: somem quando ela acaba
            STATE.set(bytes(50_000))
        payload = bytes(1000)
        await asyncio.sleep(0.001)
        if asyncio.get_running_loop().time() < until:
            kept.append(payload)

    return call


async def test_a_leak_fails_and_the_diff_points_at_the_line() -> None:
    result = await memleak.profile(keeping([]), **FAST)

    assert 1000 < result.bytes_per_call() < 1100  # o objeto (1033 B) e a posição na lista
    top = result.growth()[0].traceback[0]
    assert top.filename == __file__ and "bytes(1000)" in linecache.getline(__file__, top.lineno)
    passed, report = memleak.render(result, **SHOWN)
    where = __file__.removeprefix(f"{Path.cwd()}/")  # o relatório mostra o caminho relativo
    assert not passed and f"| `{where}:{top.lineno}` | +" in report


@pytest.mark.parametrize("keep_for", [-math.inf, 0.1])  # nunca; só na 1ª metade do aquecimento
async def test_released_memory_and_growth_in_the_warmup_pass(keep_for: float) -> None:
    kept: list[bytes] = []
    until = asyncio.get_running_loop().time() + keep_for

    result = await memleak.profile(keeping(kept, until), **FAST)

    assert bool(kept) is (keep_for > 0)
    # B/chamada, com ~120 chamadas por amostra: limpo ~0,5; objeto por amostra ~3; sem clientes ~−40
    assert abs(result.bytes_per_call()) < LIMIT / 8 and memleak.render(result, **SHOWN)[0]
    assert all(diff.traceback[0].filename != __file__ for diff in result.growth())


async def test_samples_are_taken_with_no_call_in_flight(monkeypatch: pytest.MonkeyPatch) -> None:
    busy, seen = 0, list[int]()

    async def call(client: int, n: int) -> None:
        nonlocal busy
        busy += 1
        await asyncio.sleep(0.002)
        busy -= 1

    monkeypatch.setattr(memleak, "gc", SimpleNamespace(collect=lambda: seen.append(busy)))
    await memleak.profile(call, clients=4, seconds=0.4, warmup=0.0, interval=0.02)

    assert len(seen) > 3 and set(seen) == {0}


@pytest.mark.parametrize("outside", [False, True])
async def test_tracemalloc_is_on_inside_and_back_as_it_was(outside: bool) -> None:
    inside: list[bool] = []

    async def call(client: int, n: int) -> None:
        inside.append(tracemalloc.is_tracing())
        await asyncio.sleep(0.001)

    assert not tracemalloc.is_tracing()  # desligado por padrão
    if outside:  # PYTHONTRACEMALLOC, por exemplo
        tracemalloc.start()
    try:
        await memleak.profile(call, **FAST)
        assert inside and all(inside) and tracemalloc.is_tracing() is outside
    finally:
        tracemalloc.stop()


async def test_a_failing_call_cancels_the_rest_and_raises() -> None:
    async def call(client: int, n: int) -> None:
        await asyncio.sleep(0.001)
        if n == 5:
            raise RuntimeError("falhou")

    with pytest.raises(ExceptionGroup) as caught:
        await memleak.profile(call, **FAST)

    assert caught.group_contains(RuntimeError, match="falhou")
    assert not tracemalloc.is_tracing() and asyncio.all_tasks() == {asyncio.current_task()}


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"clients": 0}, "precisa de"),
        ({"interval": 0}, "precisa de"),
        ({"interval": math.nan}, "precisa de"),
        ({"interval": math.inf}, "precisa de"),  # o amostrador dormiria para sempre
        ({"warmup": -0.1}, "precisa de"),
        ({"warmup": 1.0}, "precisa de"),
        ({"seconds": math.inf}, "precisa de"),
        # 1ª amostra em 0,6 s, antes do aquecimento; a 2ª já passaria do prazo
        ({"warmup": 0.9, "interval": 0.6}, "nenhuma amostra depois do aquecimento"),
    ],
)
async def test_degenerate_runs_are_errors(
    monkeypatch: pytest.MonkeyPatch, options: dict[str, float], message: str
) -> None:
    monkeypatch.setattr(memleak, "gc", SimpleNamespace(collect=lambda: None))  # amostra sem demora
    with pytest.raises(ValueError, match=message):
        await memleak.profile(keeping([], -math.inf), **(FAST | options))
    assert not tracemalloc.is_tracing()


@pytest.mark.parametrize(
    ("calls", "message"),
    [([], "0 amostras"), ([5, 9], "2 amostras"), ([7, 7, 7], "nenhuma chamada terminou")],
)
def test_too_little_data_is_an_error_not_a_pass(calls: list[int], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        memleak.Profile(calls, [100] * len(calls), EMPTY, EMPTY).bytes_per_call()


@pytest.mark.parametrize(
    ("per_call", "code"), [(LIMIT, 0), (LIMIT + 0.04, 0), (LIMIT + 0.06, 1), (-5.0, 0)]
)
def test_main_prints_the_report_and_exits_by_the_printed_slope(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], per_call: float, code: int
) -> None:
    seen: dict[str, float] = {}

    async def run(**options: float) -> memleak.Profile:
        seen.update(options)
        return memleak.Profile(
            [0, 1000, 2000], [0, round(1000 * per_call), round(2000 * per_call)], EMPTY, EMPTY
        )

    monkeypatch.setattr(memleak, "_run", run)

    assert memleak.main(["--seconds", "90", "--warmup", "30", "--clients", "4"]) == code
    assert seen == {"seconds": 90.0, "warmup": 30.0, "clients": 4}
    out = capsys.readouterr().out
    assert out.startswith(f"### Memória (tracemalloc): {per_call:+.1f} B por chamada")
    assert ("— passou" in out) is (code == 0) and "de 90 s (30 de aquecimento), 4 clientes" in out


def test_refuses_to_run_against_production(monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in [("APP_ENV", "production"), ("DATABASE_URL", "postgresql://u:p@db/b")]:
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-x")
    get_settings.cache_clear()
    try:
        with pytest.raises(SystemExit, match="APP_ENV=production"):
            memleak.main([])
    finally:
        get_settings.cache_clear()


@pytest.mark.integration
async def test_a_leak_in_the_search_is_found_through_the_api_and_postgres(
    monkeypatch: pytest.MonkeyPatch, db_session: AsyncSession
) -> None:
    monkeypatch.setattr(app.ingest, "token_counter", lambda model: count_words)  # job sem tiktoken
    monkeypatch.setattr(memleak, "secrets", SimpleNamespace(token_hex=lambda n: "itest"))
    kept, search_text = list[bytes](), AnyTermSearch.search_text

    async def leaky(self: AnyTermSearch, collection: str, query: str, limit: int = 10) -> Any:
        kept.append(bytes(500))
        return await search_text(self, collection, query, limit)

    monkeypatch.setattr(AnyTermSearch, "search_text", leaky)

    result = await memleak._run(seconds=10, warmup=3, clients=4)

    passed, report = memleak.render(result, seconds=10, warmup=3, clients=4)
    top = result.growth()[0].traceback[0]
    assert not passed and top.filename == __file__, report
    assert "bytes(500)" in linecache.getline(__file__, top.lineno)
    count = text("SELECT count(*) FROM documents WHERE collection_id = 'memleak-itest'")
    assert await (await db_session.connection()).scalar(count) == 0  # apagada no fim

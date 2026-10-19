"""Monitor de saturação sobre um QueuePool de verdade, com conexões falsas (sem banco)."""

import io
import json
import math
from collections.abc import Callable, Iterator
from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError
from sqlalchemy import create_engine as create_sync_engine
from sqlalchemy import event
from sqlalchemy.pool import NullPool, QueuePool

from app.db import create_engine
from app.log import configure_logging, shutdown_logging
from app.pool_monitor import PoolSaturationMonitor, watch_pool
from app.settings import Settings


class Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def _pool(size: int, overflow: int) -> QueuePool:
    # O creator sem argumento: o SQLAlchemy passaria o connection record a quem aceita um.
    return QueuePool(lambda: MagicMock(), pool_size=size, max_overflow=overflow)


def _watch(pool: QueuePool, **options: Any) -> PoolSaturationMonitor:
    """A mesma ligação do `watch_pool`, num pool solto (sem engine nem banco)."""
    monitor = PoolSaturationMonitor(lambda: pool, **options)
    event.listen(pool, "checkout", monitor.on_checkout)
    event.listen(pool, "checkin", monitor.on_checkin)
    return monitor


def _settings(**overrides: Any) -> Settings:
    values = {"database_url": "postgresql://u:p@db:5432/b", "openrouter_api_key": "sk-or-v1-x"}
    return Settings(_env_file=None, **(values | overrides))


@pytest.fixture
def logged() -> Iterator[Callable[[], list[dict[str, Any]]]]:
    stream = io.StringIO()
    configure_logging("INFO", stream=stream)

    def read() -> list[dict[str, Any]]:
        shutdown_logging()  # esvazia a fila de escrita
        lines = [json.loads(line) for line in stream.getvalue().splitlines()]
        return [line for line in lines if line["event"].startswith("db.pool_")]

    yield read
    shutdown_logging()


def test_warns_once_per_episode_and_recovers_at_half_the_threshold(
    logged: Callable[[], list[dict[str, Any]]],
) -> None:
    pool, clock = _pool(5, 5), Clock()
    monitor = _watch(pool, capacity=10, warn_ratio=0.8, clock=clock)  # 8 avisa, 4 encerra
    conns = [pool.connect() for _ in range(7)]
    assert monitor.episodes == 0

    conns += [pool.connect() for _ in range(3)]  # 8 avisa; 9 e 10 só sobem o pico
    clock.now += 2.5
    while len(conns) > 4:  # 5 em uso ainda é o mesmo episódio; 4 encerra
        conns.pop().close()
    conns += [pool.connect() for _ in range(4)]  # 8 de novo: novo episódio

    events = logged()
    assert [(e["event"], e["in_use"]) for e in events] == [
        ("db.pool_saturated", 8),
        ("db.pool_recovered", 4),
        ("db.pool_saturated", 8),
    ]
    assert (events[1]["peak"], events[1]["seconds"], events[0]["capacity"]) == (10, 2.5, 10)
    assert events[0]["level"] == "warning" and events[1]["level"] == "info"
    assert monitor.episodes == 2


def test_oscillating_around_the_threshold_does_not_repeat_the_warning(
    logged: Callable[[], list[dict[str, Any]]],
) -> None:
    pool = _pool(5, 5)
    monitor = _watch(pool, capacity=10, warn_ratio=0.8)
    conns = [pool.connect() for _ in range(8)]
    for _ in range(20):  # 7, 8, 7, 8, ...: acima de 4, o episódio é um só
        conns.pop().close()
        conns.append(pool.connect())

    assert [e["event"] for e in logged()] == ["db.pool_saturated"]
    assert monitor.episodes == 1


def test_capacity_one_opens_and_closes_an_episode_per_use(
    logged: Callable[[], list[dict[str, Any]]],
) -> None:
    pool = _pool(1, 0)
    monitor = _watch(pool, capacity=1, warn_ratio=0.8)  # avisa com 1, encerra com 0
    for _ in range(3):
        pool.connect().close()

    assert [e["event"] for e in logged()] == ["db.pool_saturated", "db.pool_recovered"] * 3
    assert monitor.episodes == 3


def test_failed_reconnect_counts_from_the_pool_not_from_the_events(
    logged: Callable[[], list[dict[str, Any]]],
) -> None:
    # Um reconnect que falha dispara `checkin` sem ter disparado `checkout`: uma contagem
    # própria (+1/-1 por evento) desceria abaixo do real e encerraria o episódio antes da hora.
    outcomes: list[Exception | None] = [None, None, ConnectionRefusedError("banco fora")]

    def creator() -> MagicMock:
        if error := outcomes.pop(0):
            raise error
        return MagicMock()

    pool = QueuePool(creator, pool_size=2, max_overflow=0)
    _watch(pool, capacity=2, warn_ratio=0.5)  # avisa com 1, encerra com 0
    first, second = pool.connect(), pool.connect()
    second.invalidate()  # devolve a vaga sem conexão: o próximo checkout reconecta
    with pytest.raises(ConnectionRefusedError):
        pool.connect()
    first.close()

    assert [(e["event"], e["in_use"], e.get("peak")) for e in logged()] == [
        ("db.pool_saturated", 1, None),
        ("db.pool_recovered", 0, 2),
    ]


@pytest.mark.parametrize(
    ("ratio", "capacity", "warn_at", "clear_at"),
    [
        (0.8, 15, 12, 6),  # o padrão: pool_size 5 + max_overflow 10
        (0.8, 1, 1, 0),
        (1.0, 7, 7, 3),
        (0.28, 25, 7, 3),  # 0.28 * 25 = 7.000000000000001 no float: o ceil daria 8
        (0.07, 100, 7, 3),
        (0.01, 5, 1, 0),  # o limiar nunca fica abaixo de uma conexão
    ],
)
def test_threshold_is_the_ceiling_of_the_ratio_without_float_noise(
    ratio: float, capacity: int, warn_at: int, clear_at: int
) -> None:
    monitor = PoolSaturationMonitor(lambda: _pool(1, 0), capacity=capacity, warn_ratio=ratio)
    assert (monitor.warn_at, monitor.clear_at) == (warn_at, clear_at)


@pytest.mark.parametrize("capacity", [0, -1, True, 1.5, "10"])
def test_refuses_a_capacity_without_room(capacity: Any) -> None:
    with pytest.raises(ValueError, match="capacity"):
        PoolSaturationMonitor(lambda: _pool(1, 0), capacity=capacity)


@pytest.mark.parametrize("ratio", [0, -0.1, 1.01, math.nan, math.inf, True, "0.8"])
def test_refuses_a_ratio_outside_zero_to_one(ratio: Any) -> None:
    with pytest.raises(ValueError, match="warn_ratio"):
        PoolSaturationMonitor(lambda: _pool(1, 0), capacity=10, warn_ratio=ratio)


def test_refuses_a_pool_without_a_limit() -> None:
    engine = create_sync_engine("sqlite://", poolclass=NullPool)  # não conecta ao criar
    with pytest.raises(TypeError, match="NullPool"):
        watch_pool(engine, capacity=1, warn_ratio=0.8)


@pytest.mark.parametrize("ratio", [0, -0.5, 1.5, math.nan, math.inf])
def test_settings_validate_the_warn_ratio(ratio: float) -> None:
    assert _settings().db_pool_warn_ratio == 0.8
    with pytest.raises(ValidationError, match="db_pool_warn_ratio"):
        _settings(db_pool_warn_ratio=ratio)


async def test_create_engine_watches_the_whole_capacity_even_after_dispose(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monitors: list[PoolSaturationMonitor] = []

    def recording(engine: Any, **options: Any) -> PoolSaturationMonitor:
        monitors.append(watch_pool(engine, **options))
        return monitors[-1]

    monkeypatch.setattr("app.db.watch_pool", recording)
    engine = create_engine(_settings(db_pool_size=7, db_max_overflow=3, db_pool_warn_ratio=0.5))
    [monitor] = monitors
    first = engine.sync_engine.pool
    await engine.dispose()  # troca o pool por um novo (nunca conectou: não precisa de banco)

    assert (monitor.capacity, monitor.warn_at, monitor.clear_at) == (10, 5, 2)
    assert engine.sync_engine.pool is not first and monitor.pool is engine.sync_engine.pool
    # O recreate do dispose copia os ouvintes para o pool novo (o `event.contains`, que olha o
    # registro por alvo, não os vê); o monitor lê o pool novo pelo engine.
    dispatch: Any = engine.sync_engine.pool.dispatch
    assert monitor.on_checkout in list(dispatch.checkout)
    assert monitor.on_checkin in list(dispatch.checkin)

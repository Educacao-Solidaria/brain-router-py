import asyncio
import io
import json
import math
from collections.abc import Iterator
from typing import Any

import pytest

from app.log import configure_logging, shutdown_logging
from app.perf import CallStats, get_stats, reset_stats, timed

SECRET = "sk-or-v1-nao-pode-vazar"


@pytest.fixture
def output() -> Iterator[io.StringIO]:
    stream = io.StringIO()
    reset_stats()
    configure_logging("DEBUG", stream=stream)
    yield stream
    shutdown_logging()
    reset_stats()


def timed_lines(stream: io.StringIO) -> list[dict[str, Any]]:
    shutdown_logging()  # esvazia a fila antes de ler
    lines = [json.loads(line) for line in stream.getvalue().splitlines()]
    return [line for line in lines if line["event"] == "perf.timed"]


def test_sync_function_is_timed_and_counted(output: io.StringIO) -> None:
    @timed("unit.add")
    def add(a: int, b: int) -> int:
        return a + b

    assert add(1, 2) == 3
    assert add(3, 4) == 7

    stats = get_stats("unit.add")
    assert (stats.calls, stats.errors) == (2, 0)
    assert stats.total_seconds >= stats.last_seconds > 0
    assert math.isclose(stats.mean_seconds, stats.total_seconds / 2)
    first, second = timed_lines(output)
    assert (first["name"], first["calls"], second["calls"]) == ("unit.add", 1, 2)
    assert first["level"] == "debug" and first["error"] is None
    assert isinstance(first["elapsed_ms"], float)


async def test_async_function_measures_awaited_time(output: io.StringIO) -> None:
    @timed("unit.sleep")
    async def nap(seconds: float) -> str:
        await asyncio.sleep(seconds)
        return "ok"

    assert await nap(0.02) == "ok"

    assert get_stats("unit.sleep").last_seconds >= 0.02
    (line,) = timed_lines(output)
    assert line["elapsed_ms"] >= 20


def test_errors_are_counted_and_reraised_without_message(output: io.StringIO) -> None:
    @timed("unit.boom")
    def boom(api_key: str) -> None:
        raise ValueError(f"chave recusada: {api_key}")

    with pytest.raises(ValueError, match="chave recusada"):
        boom(SECRET)

    stats = get_stats("unit.boom")
    assert (stats.calls, stats.errors) == (1, 1)
    (line,) = timed_lines(output)
    assert line["error"] == "ValueError"
    assert SECRET not in output.getvalue()


async def test_arguments_and_return_values_never_reach_the_log(output: io.StringIO) -> None:
    @timed()
    async def ask(question: str, *, token: str) -> str:
        return f"{question}:{token}"

    assert await ask("qual a nota de corte?", token=SECRET) == f"qual a nota de corte?:{SECRET}"

    (line,) = timed_lines(output)  # esvazia a fila; depois disso o stream está completo
    raw = output.getvalue()
    assert line["name"].endswith("ask")
    assert SECRET not in raw
    assert "nota de corte" not in raw


async def test_cancellation_counts_as_error(output: io.StringIO) -> None:
    @timed("unit.slow")
    async def slow() -> None:
        await asyncio.sleep(10)

    task = asyncio.create_task(slow())
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert get_stats("unit.slow").errors == 1
    (line,) = timed_lines(output)
    assert line["error"] == "CancelledError"


def test_default_name_and_metadata_are_preserved() -> None:
    def documented() -> None:
        """Docstring preservada."""

    wrapped = timed()(documented)
    wrapped()

    assert wrapped.__doc__ == "Docstring preservada."
    assert wrapped.__name__ == "documented"
    expected = f"{__name__}.test_default_name_and_metadata_are_preserved.<locals>.documented"
    assert get_stats(expected).calls == 1
    reset_stats()


def test_unknown_name_returns_empty_stats() -> None:
    assert get_stats("nunca.chamado") == CallStats()
    assert CallStats().mean_seconds == 0.0


def test_generators_are_rejected() -> None:
    def gen() -> Iterator[int]:
        yield 1

    async def agen() -> Any:
        yield 1

    with pytest.raises(TypeError, match="geradores"):
        timed()(gen)
    with pytest.raises(TypeError, match="geradores"):
        timed()(agen)

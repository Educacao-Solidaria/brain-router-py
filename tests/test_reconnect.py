"""Repetição na queda de conexão, sem banco: classificação, espera e limites."""

import asyncio
import io
import json
import math
import socket
from typing import Any

import asyncpg
import pytest
from sqlalchemy.exc import DBAPIError, InterfaceError
from sqlalchemy.exc import TimeoutError as PoolTimeoutError

from app.log import configure_logging, shutdown_logging
from app.reconnect import backoff_delay, is_disconnect, retry_on_disconnect

LOST = DBAPIError("SELECT 1", None, Exception("conexão caiu"), connection_invalidated=True)
SQL_ERROR = DBAPIError("SELEC 1", None, Exception("erro de sintaxe"), connection_invalidated=False)


class Script:
    """Operação que falha com cada erro da lista, nesta ordem, e depois devolve 'ok'."""

    def __init__(self, *errors: BaseException) -> None:
        self.errors = list(errors)
        self.calls = 0

    async def __call__(self) -> str:
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return "ok"


class Sleeps:
    def __init__(self) -> None:
        self.waits: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.waits.append(seconds)


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (LOST, True),
        (InterfaceError("SELECT 1", None, Exception("closed"), connection_invalidated=True), True),
        (ConnectionResetError("reset by peer"), True),
        (ConnectionRefusedError("refused"), True),
        (BrokenPipeError("pipe"), True),
        (OSError("Multiple exceptions: [Errno 111] Connect call failed"), True),  # IPv4 + IPv6
        (socket.gaierror(-3, "Temporary failure in name resolution"), True),
        (asyncpg.CannotConnectNowError("the database system is starting up"), True),
        (SQL_ERROR, False),  # erro da consulta: repetir daria o mesmo erro
        (TimeoutError(), False),  # é OSError, mas repetir só somaria espera
        (PoolTimeoutError("QueuePool limit"), False),
        (asyncpg.InvalidPasswordError("senha"), False),
        (ValueError("x"), False),
    ],
)
def test_only_lost_connections_are_worth_a_retry(exc: BaseException, expected: bool) -> None:
    assert is_disconnect(exc) is expected


def test_backoff_is_full_jitter_over_a_capped_exponential() -> None:
    ceilings = [backoff_delay(n, base=0.05, cap=1.0, rand=lambda: 1.0) for n in range(1, 8)]
    assert ceilings == pytest.approx([0.05, 0.1, 0.2, 0.4, 0.8, 1.0, 1.0])
    assert backoff_delay(3, base=0.05, cap=1.0, rand=lambda: 0.0) == 0.0
    assert backoff_delay(3, base=0.05, cap=1.0, rand=lambda: 0.5) == pytest.approx(0.1)
    assert backoff_delay(10_000, base=0.05, cap=1.0, rand=lambda: 1.0) == 1.0  # sem overflow
    assert backoff_delay(1, base=0.0, cap=1.0) == 0.0


async def test_retries_lost_connections_with_backoff_until_it_works() -> None:
    operation, sleeps = Script(LOST, ConnectionResetError()), Sleeps()
    stream = io.StringIO()
    configure_logging("INFO", stream=stream)
    try:
        result = await retry_on_disconnect(operation, attempts=3, sleep=sleeps, rand=lambda: 1.0)
    finally:
        shutdown_logging()

    assert (result, operation.calls) == ("ok", 3)
    assert sleeps.waits == pytest.approx([0.05, 0.1])
    logged = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert [(e["event"], e["attempt"], e["error"]) for e in logged] == [
        ("db.retry_after_disconnect", 1, "DBAPIError"),
        ("db.retry_after_disconnect", 2, "ConnectionResetError"),
    ]
    assert "conexão caiu" not in stream.getvalue()  # mensagem do driver fica fora do log


async def test_gives_up_after_the_last_attempt_with_its_own_error() -> None:
    last = ConnectionRefusedError("ainda fora")
    operation, sleeps = Script(LOST, LOST, last), Sleeps()
    with pytest.raises(ConnectionRefusedError) as caught:
        await retry_on_disconnect(operation, attempts=3, sleep=sleeps)
    assert caught.value is last and operation.calls == 3 and len(sleeps.waits) == 2


@pytest.mark.parametrize("error", [SQL_ERROR, TimeoutError(), KeyError("x")])
async def test_errors_that_are_not_disconnects_raise_at_once(error: BaseException) -> None:
    operation, sleeps = Script(error), Sleeps()
    with pytest.raises(type(error)):
        await retry_on_disconnect(operation, attempts=5, sleep=sleeps)
    assert (operation.calls, sleeps.waits) == (1, [])


async def test_one_attempt_means_no_retry() -> None:
    operation = Script(LOST)
    with pytest.raises(DBAPIError):
        await retry_on_disconnect(operation, attempts=1, sleep=Sleeps())
    assert operation.calls == 1


async def test_cancellation_is_never_retried() -> None:
    operation, sleeps = Script(asyncio.CancelledError()), Sleeps()
    with pytest.raises(asyncio.CancelledError):
        await retry_on_disconnect(operation, attempts=3, sleep=sleeps)
    assert (operation.calls, sleeps.waits) == (1, [])


async def test_cancellation_during_the_wait_stops_the_retries() -> None:
    operation = Script(LOST)
    retrying = retry_on_disconnect(
        operation, attempts=3, base_delay=30.0, max_delay=30.0, rand=lambda: 1.0
    )
    task = asyncio.create_task(retrying)
    await asyncio.sleep(0)  # a 1ª tentativa falha e a tarefa dorme 30 s no backoff
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert operation.calls == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"attempts": 0},
        {"attempts": -1},
        {"attempts": 11},
        {"attempts": True},
        {"attempts": 2.0},
        {"base_delay": -0.1},
        {"base_delay": math.nan},
        {"max_delay": math.inf},
    ],
)
async def test_refuses_degenerate_settings_before_running(kwargs: dict[str, Any]) -> None:
    operation = Script()
    with pytest.raises(ValueError):
        await retry_on_disconnect(operation, **kwargs)
    assert operation.calls == 0

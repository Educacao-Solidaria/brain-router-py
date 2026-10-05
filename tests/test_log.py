import asyncio
import io
import json
import logging
from collections.abc import Iterator
from typing import Any

import pytest
from pydantic import SecretStr
from structlog.contextvars import bound_contextvars, clear_contextvars

from app.log import REDACTED, configure_logging, get_logger, shutdown_logging


@pytest.fixture
def output() -> Iterator[io.StringIO]:
    stream = io.StringIO()
    clear_contextvars()
    configure_logging("DEBUG", stream=stream)
    yield stream
    shutdown_logging()
    clear_contextvars()


def read_lines(stream: io.StringIO) -> list[dict[str, Any]]:
    shutdown_logging()  # esvazia a fila antes de ler
    return [json.loads(line) for line in stream.getvalue().splitlines()]


def test_emits_one_json_object_per_event(output: io.StringIO) -> None:
    get_logger("app.test").info("search.done", top_k=5)

    (line,) = read_lines(output)
    assert line["event"] == "search.done"
    assert line["top_k"] == 5
    assert line["level"] == "info"
    assert line["logger"] == "app.test"
    assert line["timestamp"].endswith("Z")


def test_injects_bound_context_automatically(output: io.StringIO) -> None:
    log = get_logger()
    with bound_contextvars(request_id="req-1", tenant_id="t-9"):
        log.info("inside")
    log.info("outside")

    inside, outside = read_lines(output)
    assert (inside["request_id"], inside["tenant_id"]) == ("req-1", "t-9")
    assert "request_id" not in outside


async def test_async_tasks_keep_their_own_context(output: io.StringIO) -> None:
    async def handle(request_id: str) -> None:
        with bound_contextvars(request_id=request_id):
            await asyncio.sleep(0)  # força intercalar as tasks
            await get_logger().ainfo("handled", expected=request_id)

    await asyncio.gather(*(handle(f"req-{i}") for i in range(5)))

    lines = [line for line in read_lines(output) if line["event"] == "handled"]
    assert len(lines) == 5
    assert all(line["request_id"] == line["expected"] for line in lines)


def test_stdlib_loggers_get_context_and_json(output: io.StringIO) -> None:
    with bound_contextvars(request_id="req-2"):
        logging.getLogger("sqlalchemy.engine").warning("pool %s exhausted", "main")

    (line,) = read_lines(output)
    assert line["event"] == "pool main exhausted"
    assert line["logger"] == "sqlalchemy.engine"
    assert line["request_id"] == "req-2"


def test_redacts_sensitive_keys(output: io.StringIO) -> None:
    get_logger().info(
        "config.loaded",
        password="hunter2",
        openrouter_api_key="sk-or-v1-xyz",
        database_url="postgresql+asyncpg://u:p@db/x",
        Authorization="Bearer abc",
        db_pool_size=5,
    )

    (line,) = read_lines(output)
    assert "hunter2" not in json.dumps(line)
    for key in ("password", "openrouter_api_key", "database_url", "Authorization"):
        assert line[key] == REDACTED
    assert line["db_pool_size"] == 5


def test_secret_str_values_stay_masked(output: io.StringIO) -> None:
    get_logger().info("connect", credential=SecretStr("s3nha"))

    (line,) = read_lines(output)
    assert "s3nha" not in line["credential"]


def test_level_filter(output: io.StringIO) -> None:
    configure_logging("WARNING", stream=output)
    log = get_logger()
    log.info("dropped")
    log.warning("kept")

    assert [line["event"] for line in read_lines(output)] == ["kept"]


def test_exceptions_are_structured(output: io.StringIO) -> None:
    try:
        raise ValueError("boom")
    except ValueError:
        get_logger().exception("failed")

    (line,) = read_lines(output)
    assert line["level"] == "error"
    assert line["exception"][0]["exc_type"] == "ValueError"


def test_exception_traceback_does_not_leak_locals(output: io.StringIO) -> None:
    def connect() -> None:
        dsn = "postgresql://app:s3nh4-vazada@db/app"
        raise ConnectionError(f"falha ao conectar em {dsn.split('@')[1]}")

    try:
        connect()
    except ConnectionError:
        get_logger().exception("db_down")

    (line,) = read_lines(output)
    assert "s3nh4-vazada" not in json.dumps(line)
    assert all("locals" not in frame for exc in line["exception"] for frame in exc["frames"])


def test_console_renderer_for_local_development() -> None:
    stream = io.StringIO()
    configure_logging("INFO", json=False, stream=stream)
    get_logger().info("hello", user="ana")
    shutdown_logging()

    text = stream.getvalue()
    assert "hello" in text
    assert "user" in text
    with pytest.raises(json.JSONDecodeError):
        json.loads(text)

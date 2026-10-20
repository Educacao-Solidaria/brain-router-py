"""Tarefas em segundo plano: limite de concorrência, estado, falha registrada e desligamento."""

import asyncio
import gc
import io
import json
from functools import partial

import pytest

from app.api import create_app
from app.errors.exceptions import EmbeddingServiceError
from app.log import configure_logging, shutdown_logging
from app.tasks import TaskInfo, TaskManager


async def _settle() -> None:
    for _ in range(5):  # deixa as tasks agendadas andarem até o próximo ponto de espera
        await asyncio.sleep(0)


async def _finish(manager: TaskManager) -> list[TaskInfo]:
    async with asyncio.timeout(5):
        while any(info.finished_at is None for info in manager.list()):
            await asyncio.sleep(0)
    return manager.list()


async def test_runs_at_most_max_concurrent_in_order_and_keeps_each_result() -> None:
    manager, gate = TaskManager(max_concurrent=2), asyncio.Event()

    async def job(n: int) -> int:
        await gate.wait()
        return n * 10

    first = manager.submit("job-0", partial(job, 0))
    for n in range(1, 4):
        manager.submit(f"job-{n}", partial(job, n))
    await _settle()
    assert [info.state for info in manager.list()] == ["running", "running", "pending", "pending"]
    copy = manager.get(first.id)
    assert copy is not None and copy.started_at is not None
    copy.state = "failed"  # uma cópia: quem consulta não mexe no estado
    assert [info.state for info in manager.list()][0] == "running"
    gate.set()
    finished = await _finish(manager)
    assert [(info.state, info.result) for info in finished] == [
        ("done", 0),
        ("done", 10),
        ("done", 20),
        ("done", 30),
    ]


async def test_a_failing_job_is_recorded_and_logged_and_never_left_unretrieved() -> None:
    stream, manager = io.StringIO(), TaskManager()

    async def boom() -> None:
        raise ValueError("dado do usuário")

    async def provider() -> None:
        raise EmbeddingServiceError("HTTP 503")

    configure_logging("INFO", stream=stream)
    try:
        manager.submit("boom", boom)
        manager.submit("provedor", provider)
        finished = await _finish(manager)
        gc.collect()  # task com exceção sem dono avisaria aqui, no logger "asyncio"
    finally:
        shutdown_logging()
    assert [(info.state, info.error) for info in finished] == [
        ("failed", "ValueError"),  # a mensagem de erro qualquer pode ter dado: só o tipo
        ("failed", "EmbeddingServiceError: Falha na geracao de embeddings (openrouter): HTTP 503"),
    ]
    lines = [json.loads(raw) for raw in stream.getvalue().splitlines()]
    assert [line["task"] for line in lines if line["event"] == "task.failed"] == [
        "boom",
        "provedor",
    ]
    assert "never retrieved" not in stream.getvalue()


async def test_cancel_stops_a_pending_or_running_job_and_says_false_for_the_rest() -> None:
    manager, gate = TaskManager(max_concurrent=1), asyncio.Event()
    running, pending = manager.submit("a", gate.wait), manager.submit("b", gate.wait)
    await _settle()
    assert manager.cancel(pending.id) and manager.cancel(running.id)
    finished = await _finish(manager)
    assert [(info.state, info.started_at is None) for info in finished] == [
        ("cancelled", False),
        ("cancelled", True),  # o da fila nunca chegou a rodar
    ]
    assert not manager.cancel(running.id) and not manager.cancel("nada")


async def test_the_app_lifespan_cancels_and_awaits_the_jobs_on_shutdown() -> None:
    manager = TaskManager()
    app = create_app(tasks=manager)
    async with app.router.lifespan_context(app):
        job = manager.submit("reindexação", asyncio.Event().wait)
        await _settle()
        assert [info.state for info in manager.list()] == ["running"]
    info = manager.get(job.id)
    assert info is not None and info.state == "cancelled" and info.finished_at is not None
    with pytest.raises(RuntimeError, match="encerrado"):
        manager.submit("depois", asyncio.Event().wait)


async def test_keeps_only_the_latest_finished_and_refuses_degenerate_settings() -> None:
    manager = TaskManager(keep_finished=1)

    async def quick() -> str:
        return "ok"

    for n in range(3):
        manager.submit(f"q{n}", quick)
    assert [info.name for info in await _finish(manager)] == ["q2"]
    for bad in ({"max_concurrent": 0}, {"keep_finished": -1}):
        with pytest.raises(ValueError):
            TaskManager(**bad)

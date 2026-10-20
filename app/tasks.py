"""Tarefas em segundo plano (re-indexação, ingestão em lote) sem travar as requisições.

`TaskManager(max_concurrent=2, keep_finished=100)`, no event loop do servidor:
- `submit(name, job)` agenda a corrotina `job()` numa task do asyncio e devolve o `TaskInfo`.
  No máximo `max_concurrent` rodam ao mesmo tempo; as outras esperam na fila (`pending`).
- `get(id)` e `list()` dão uma cópia do estado: `pending` → `running` → `done` (com o
  `result`), `failed` ou `cancelled`, com os horários. Na falha, `error` traz o tipo (e a
  mensagem, se for `BrainRouterError`, que já vem sem segredo) e o traceback vai para o log
  (`task.failed`). A task nunca termina com exceção pendente: nada de "Task exception was
  never retrieved".
- `cancel(id)`; `shutdown(timeout)` cancela todas e espera por elas. `create_app(tasks=...)`
  liga isso ao `lifespan` do FastAPI: o servidor não desliga com job pela metade solto.
- Guarda as `keep_finished` terminadas mais recentes (memória limitada); as em curso ficam.

O job tem de ceder o event loop: arquivo e CPU síncronos vão por `asyncio.to_thread` (o
`ingest_directory` faz isso). ponytail: em memória e por processo; fila durável (tabela,
Redis) só quando uma tarefa precisar sobreviver a restart.
"""

import asyncio
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Literal

from app.errors import BrainRouterError
from app.log import get_logger

State = Literal["pending", "running", "done", "failed", "cancelled"]

log = get_logger(__name__)


@dataclass(slots=True)
class TaskInfo:
    id: str
    name: str
    created_at: datetime
    state: State = "pending"
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None
    result: object = None


class TaskManager:
    def __init__(self, *, max_concurrent: int = 2, keep_finished: int = 100) -> None:
        if max_concurrent < 1:
            raise ValueError(f"max_concurrent deve ser >= 1, veio {max_concurrent}")
        if keep_finished < 0:
            raise ValueError(f"keep_finished deve ser >= 0, veio {keep_finished}")
        self._slots = asyncio.Semaphore(max_concurrent)
        self._keep = keep_finished
        self._info: dict[str, TaskInfo] = {}  # na ordem de chegada
        # Referência forte: o loop só guarda referência fraca, e a task sumiria no meio.
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._closed = False

    def submit(self, name: str, job: Callable[[], Awaitable[object]]) -> TaskInfo:
        if self._closed:
            raise RuntimeError("TaskManager encerrado: não aceita tarefa nova")
        loop = asyncio.get_running_loop()  # fora do loop, falha aqui: antes de registrar nada
        info = TaskInfo(id=uuid.uuid4().hex, name=name, created_at=datetime.now(UTC))
        self._tasks[info.id] = loop.create_task(self._run(info, job), name=f"task:{name}")
        self._info[info.id] = info
        return replace(info)

    def get(self, task_id: str) -> TaskInfo | None:
        info = self._info.get(task_id)
        return None if info is None else replace(info)

    def list(self) -> list[TaskInfo]:
        return [replace(info) for info in self._info.values()]

    def cancel(self, task_id: str) -> bool:
        """Pede o cancelamento; False se a tarefa não existe ou já terminou."""
        task = self._tasks.get(task_id)
        return task is not None and task.cancel()

    async def shutdown(self, timeout: float = 10.0) -> None:
        """Recusa tarefa nova, cancela as em curso e espera até `timeout` segundos por elas."""
        self._closed = True
        tasks = list(self._tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            _, pending = await asyncio.wait(tasks, timeout=timeout)
            if pending:  # job que engole o cancelamento: fica registrado, não trava o desligamento
                log.warning("task.shutdown_timeout", pending=len(pending))

    @asynccontextmanager
    async def lifespan(self, _app: object) -> AsyncIterator[None]:
        """Para `FastAPI(lifespan=...)`: no desligamento do servidor, `shutdown()`."""
        try:
            yield
        finally:
            await self.shutdown()

    async def _run(self, info: TaskInfo, job: Callable[[], Awaitable[object]]) -> None:
        try:
            async with self._slots:
                info.state, info.started_at = "running", datetime.now(UTC)
                info.result = await job()
            info.state = "done"
        except asyncio.CancelledError:
            info.state = "cancelled"
            log.info("task.cancelled", task=info.name, id=info.id)
            raise  # a task termina cancelada, como pediram
        except Exception as exc:  # o job é de quem chamou: qualquer erro dele é registrado aqui
            error = type(exc).__name__
            if isinstance(exc, BrainRouterError):  # mensagem já sem segredo; a de outro, não
                error += f": {exc.message}"
            info.state, info.error = "failed", error
            log.exception("task.failed", task=info.name, id=info.id)
        finally:
            info.finished_at = datetime.now(UTC)
            self._tasks.pop(info.id, None)
            self._trim()

    def _trim(self) -> None:
        finished = [info for info in self._info.values() if info.finished_at is not None]
        for info in finished[: max(0, len(finished) - self._keep)]:
            del self._info[info.id]

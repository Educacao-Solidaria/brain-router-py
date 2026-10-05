"""Medição de tempo e contagem de chamadas com `@timed()`, para funções sync e async.

    @timed("rag.search")
    async def search(query: str) -> list[SearchResult]: ...

Cada chamada gera um log `perf.timed` (nível DEBUG) com nome, duração em ms, número de
chamadas e o tipo da exceção, se houver. Argumentos, retorno e mensagem de erro **nunca**
entram no log: podem carregar texto do usuário ou segredo. Os acumulados ficam em
`get_stats(name)`.
"""

import functools
import inspect
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

from app.log import get_logger

log = get_logger(__name__)


@dataclass
class CallStats:
    calls: int = 0
    errors: int = 0
    total_seconds: float = 0.0
    last_seconds: float = 0.0

    @property
    def mean_seconds(self) -> float:
        return self.total_seconds / self.calls if self.calls else 0.0


# ponytail: acumulado em memória por processo, sem lock (o asyncio roda numa thread só).
# Exportar para Prometheus/OTel é trabalho da Fase 4.
_stats: dict[str, CallStats] = {}


def get_stats(name: str) -> CallStats:
    """Acumulado de `name` (zerado se nunca foi chamado)."""
    return _stats.get(name, CallStats())


def reset_stats() -> None:
    _stats.clear()


def _record(name: str, started: float, error: str | None) -> None:
    elapsed = time.perf_counter() - started
    stats = _stats.setdefault(name, CallStats())
    stats.calls += 1
    stats.total_seconds += elapsed
    stats.last_seconds = elapsed
    if error is not None:
        stats.errors += 1
    log.debug(
        "perf.timed",
        name=name,
        elapsed_ms=round(elapsed * 1000, 3),
        calls=stats.calls,
        error=error,
    )


def timed[F: Callable[..., Any]](name: str | None = None) -> Callable[[F], F]:
    """Decorator que mede cada chamada com `time.perf_counter`.

    `name` padrão: `<módulo>.<qualname>`. Erros são contados e relançados; cancelamento
    (`CancelledError`) também conta como erro, com o tipo no log.
    """

    def decorate(func: F) -> F:
        if inspect.isgeneratorfunction(func) or inspect.isasyncgenfunction(func):
            # Mediria só a criação do gerador, não o consumo.
            raise TypeError("@timed não suporta geradores")
        label = name or f"{func.__module__}.{func.__qualname__}"

        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                started, error = time.perf_counter(), None
                try:
                    return await func(*args, **kwargs)
                except BaseException as exc:
                    error = type(exc).__name__
                    raise
                finally:
                    _record(label, started, error)

            return cast(F, async_wrapper)

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            started, error = time.perf_counter(), None
            try:
                return func(*args, **kwargs)
            except BaseException as exc:
                error = type(exc).__name__
                raise
            finally:
                _record(label, started, error)

        return cast(F, sync_wrapper)

    return decorate

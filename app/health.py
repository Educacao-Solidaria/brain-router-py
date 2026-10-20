"""Saúde detalhada para o `GET /healthz` da API v1: cada dependência checada de verdade.

- `CachedCheck(check, ttl_seconds)`: guarda o resultado por alguns segundos e junta as sondas
  simultâneas numa chamada só. Assim a sonda frequente (balanceador, Kubernetes) não vira uma
  consulta ao banco nem uma chamada à OpenRouter por pedido, e a rota pode ficar fora do rate
  limit. Checagem que levanta conta como `down` (o tipo do erro vai para o log).
- `Dependency(check, critical=True)`: dependência crítica caída → 503 `unavailable`; a não
  crítica → 200 `degraded`. A OpenRouter não é crítica quando a busca tem o fallback léxico.
- O corpo diz só `ok`/`down` por dependência: nada de host, versão, mensagem de erro ou segredo.
"""

import asyncio
import math
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel

from app.log import get_logger

HEALTH_PATH = "/healthz"

Check = Callable[[], Awaitable[bool]]

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class Dependency:
    check: Check
    critical: bool = True


class HealthReport(BaseModel):
    status: Literal["ok", "degraded", "unavailable"]
    checks: dict[str, Literal["ok", "down"]]


class CachedCheck:
    def __init__(
        self, check: Check, *, ttl_seconds: float, clock: Callable[[], float] = time.monotonic
    ) -> None:
        if not (math.isfinite(ttl_seconds) and ttl_seconds >= 0):
            raise ValueError(f"ttl_seconds deve ser finito e >= 0, veio {ttl_seconds}")
        self._check, self._ttl, self._clock = check, ttl_seconds, clock
        self._value: bool | None = None
        self._checked_at = 0.0
        self._pending: asyncio.Task[bool] | None = None

    async def __call__(self) -> bool:
        if self._value is not None and self._clock() - self._checked_at < self._ttl:
            return self._value
        if self._pending is None or self._pending.done():
            self._pending = asyncio.create_task(self._refresh())
        # shield: a sonda que desiste (cliente caiu) não cancela a checagem das outras.
        return await asyncio.shield(self._pending)

    async def _refresh(self) -> bool:
        try:
            value = await self._check()
        except Exception as exc:  # checagem com defeito ou dependência em estado estranho: down
            log.warning("health.check_error", error=type(exc).__name__)
            value = False
        self._value, self._checked_at = value, self._clock()
        return value


async def report(dependencies: Mapping[str, Dependency]) -> tuple[int, HealthReport]:
    """Roda as checagens em paralelo e devolve (status HTTP, corpo). Checagem crua que levanta
    (sem o `CachedCheck`) também conta como down, em vez de virar 500."""
    names = list(dependencies)
    raw = await asyncio.gather(*(dependencies[n].check() for n in names), return_exceptions=True)
    for name, outcome in zip(names, raw, strict=True):
        if isinstance(outcome, BaseException):
            log.warning("health.check_error", dependency=name, error=type(outcome).__name__)
    results = [outcome is True for outcome in raw]
    pairs = list(zip(names, results, strict=True))
    checks: dict[str, Literal["ok", "down"]] = {n: "ok" if ok else "down" for n, ok in pairs}
    if any(dependencies[name].critical and not ok for name, ok in pairs):
        return 503, HealthReport(status="unavailable", checks=checks)
    degraded = not all(results)
    return 200, HealthReport(status="degraded" if degraded else "ok", checks=checks)

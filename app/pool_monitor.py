"""Aviso de saturação do pool de conexões com o Postgres, pelos eventos do próprio pool.

`watch_pool(engine, capacity=..., warn_ratio=0.8)` liga ouvintes nos eventos `checkout` e
`checkin` do pool do SQLAlchemy: nada de varredura periódica, e o custo por evento é ler um
contador do pool e comparar dois inteiros (medido: ~1 µs por checkout + checkin). O
`create_engine` já liga, com a capacidade `db_pool_size + db_max_overflow` e o
`db_pool_warn_ratio` do `Settings`.

- `db.pool_saturated` (WARNING): as conexões em uso chegaram a `ceil(warn_ratio * capacity)`.
  É a hora de olhar: com o pool cheio, a próxima requisição espera até `db_pool_timeout`.
- Histerese: um aviso por episódio, não um por checkout. O episódio só termina quando o uso
  cai à metade do limiar (ou menos), com `db.pool_recovered` (INFO) trazendo o pico e a
  duração; oscilar em volta do limiar não repete o aviso.

O número em uso vem do pool (`checkedout()`), não de uma contagem paralela que se perderia
com conexão invalidada ou checkout que falhou no meio. No `checkin` o evento dispara antes de
a conexão voltar ao pool, então o uso depois da devolução é `checkedout() - 1`. E o pool é
lido do engine a cada evento: o `engine.dispose()` troca o pool por um novo (os ouvintes vão
junto), e o antigo, fechado, diria 0 para sempre.

ponytail: estado sem lock, porque o pool do engine assíncrono vive na thread do event loop.
Um engine síncrono com várias threads pode repetir ou perder um aviso numa corrida; se isso
passar a existir, proteger `_saturated` com um `threading.Lock`.
"""

import math
import time
from collections.abc import Callable
from typing import cast

from sqlalchemy import Engine, event
from sqlalchemy.pool import QueuePool

from app.log import get_logger

log = get_logger(__name__)


class PoolSaturationMonitor:
    def __init__(
        self,
        current_pool: Callable[[], QueuePool],
        *,
        capacity: int,
        warn_ratio: float = 0.8,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if type(capacity) is not int or capacity < 1:
            raise ValueError(f"capacity deve ser um inteiro >= 1, veio {capacity!r}")
        if isinstance(warn_ratio, bool) or not isinstance(warn_ratio, int | float):
            raise ValueError(f"warn_ratio deve ser um número, veio {warn_ratio!r}")
        if not 0 < warn_ratio <= 1:  # NaN também cai aqui
            raise ValueError(f"warn_ratio deve estar em (0, 1], veio {warn_ratio!r}")
        self.capacity = capacity
        # O round tira o ruído do float antes do ceil: 0.28 * 25 dá 7.000000000000001.
        self.warn_at = math.ceil(round(warn_ratio * capacity, 6))  # >= 1: warn_ratio > 0
        self.clear_at = self.warn_at // 2
        self.episodes = 0
        self._current_pool = current_pool
        self._clock = clock
        self._saturated = False
        self._peak = 0
        self._since = 0.0

    @property
    def pool(self) -> QueuePool:
        return self._current_pool()

    def on_checkout(self, *_: object) -> None:
        in_use = self.pool.checkedout()
        if self._saturated:
            self._peak = max(self._peak, in_use)
        elif in_use >= self.warn_at:
            self._saturated, self._peak, self._since = True, in_use, self._clock()
            self.episodes += 1
            log.warning(
                "db.pool_saturated", in_use=in_use, capacity=self.capacity, warn_at=self.warn_at
            )

    def on_checkin(self, *_: object) -> None:
        if not self._saturated:
            return
        in_use = self.pool.checkedout() - 1  # a conexão ainda não voltou ao pool
        if in_use <= self.clear_at:
            self._saturated = False
            log.info(
                "db.pool_recovered",
                in_use=in_use,
                peak=self._peak,
                capacity=self.capacity,
                seconds=round(self._clock() - self._since, 3),
            )


def watch_pool(
    engine: Engine,
    *,
    capacity: int,
    warn_ratio: float,
    clock: Callable[[], float] = time.monotonic,
) -> PoolSaturationMonitor:
    """Liga o monitor aos eventos do pool de `engine` (síncrono: `AsyncEngine.sync_engine`)."""
    pool = engine.pool
    if not isinstance(pool, QueuePool):
        raise TypeError(f"só pools com limite (QueuePool) saturam; veio {type(pool).__name__}")
    monitor = PoolSaturationMonitor(
        lambda: cast(QueuePool, engine.pool), capacity=capacity, warn_ratio=warn_ratio, clock=clock
    )
    event.listen(pool, "checkout", monitor.on_checkout)
    event.listen(pool, "checkin", monitor.on_checkin)
    return monitor

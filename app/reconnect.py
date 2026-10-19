"""Repetição de leitura quando a conexão com o Postgres cai (servidor reiniciado, failover, rede).

O primeiro degrau já está no `create_engine`: `pool_pre_ping` testa a conexão ao tirá-la do
pool e troca a que morreu ociosa, sem erro para ninguém. Sobra a queda entre o ping e a
consulta, ou no meio dela: aí a consulta falha, e só uma nova tentativa, com outra conexão,
resolve. Repetir é seguro só para o que não escreve: um COMMIT perdido no meio da queda pode
ter gravado, e repetir a escrita duplicaria o dado. Por isso a repetição mora no
`Database.read()`, que abre cada tentativa numa transação nova e READ ONLY (o Postgres recusa
escrita ali), e nunca no `Database.session()`.

- `is_disconnect(exc)`: o que vale repetir. A conexão que o SQLAlchemy invalidou ao ver a
  queda (`DBAPIError.connection_invalidated`), erro de rede cru (`OSError`: reset, recusa,
  DNS; o asyncpg não os traduz ao conectar, e host com mais de um endereço dá um `OSError`
  genérico, "Multiple exceptions") e o servidor subindo ou em recuperação
  (`CannotConnectNowError`, SQLSTATE 57P03). `TimeoutError` não: repetir só somaria espera.
- `backoff_delay(n)`: jitter cheio sobre o exponencial, `uniform(0, min(teto, base * 2^(n-1)))`,
  para as réplicas de um serviço não voltarem todas no mesmo instante.
"""

import asyncio
import math
import random
from collections.abc import Awaitable, Callable

import asyncpg
from sqlalchemy.exc import DBAPIError

from app.log import get_logger

log = get_logger(__name__)

MAX_ATTEMPTS = 10


def is_disconnect(exc: BaseException) -> bool:
    """Queda de conexão (outra conexão resolve), e não erro da própria consulta."""
    if isinstance(exc, DBAPIError):
        return exc.connection_invalidated
    if isinstance(exc, TimeoutError):  # é OSError desde o Python 3.3
        return False
    return isinstance(exc, OSError | asyncpg.CannotConnectNowError)


def backoff_delay(
    attempt: int, *, base: float, cap: float, rand: Callable[[], float] = random.random
) -> float:
    """Espera antes da tentativa `attempt + 1`, em segundos: entre 0 e o teto exponencial."""
    ceiling = min(cap, base * 2.0 ** min(attempt - 1, 64))  # sem overflow com attempt enorme
    return rand() * ceiling


async def retry_on_disconnect[T](
    operation: Callable[[], Awaitable[T]],
    *,
    attempts: int = 3,
    base_delay: float = 0.05,
    max_delay: float = 1.0,
    sleep: Callable[[float], Awaitable[object]] = asyncio.sleep,
    rand: Callable[[], float] = random.random,
) -> T:
    """Roda `operation` e, se a conexão cair, de novo, até `attempts` vezes ao todo.

    Erro que não é queda sobe na hora, sem espera; na última tentativa sobe o erro dela.
    Cancelamento (BaseException) nunca é repetido, nem durante a espera.
    """
    if type(attempts) is not int or not 1 <= attempts <= MAX_ATTEMPTS:
        raise ValueError(f"attempts deve ser um inteiro em [1, {MAX_ATTEMPTS}], veio {attempts!r}")
    for name, value in (("base_delay", base_delay), ("max_delay", max_delay)):
        if not (math.isfinite(value) and value >= 0):
            raise ValueError(f"{name} deve ser finito e >= 0, veio {value!r}")
    for attempt in range(1, attempts):
        try:
            return await operation()
        except Exception as exc:
            if not is_disconnect(exc):
                raise
            delay = backoff_delay(attempt, base=base_delay, cap=max_delay, rand=rand)
            # Só o tipo: a mensagem do driver pode trazer host e usuário.
            log.warning(
                "db.retry_after_disconnect",
                attempt=attempt,
                attempts=attempts,
                error=type(exc).__name__,
                delay_ms=round(delay * 1000, 1),
            )
            await sleep(delay)
    return await operation()

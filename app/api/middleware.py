"""Middlewares da API v1: X-Request-Id ligado ao log e rate limit por IP (CORS e GZip no app).

Ordem no `create_app`, de fora para dentro: CORS → `RequestIdMiddleware` → `RateLimitMiddleware`
→ GZip → teto de corpo. O id sai em toda resposta que passa por ele (429 e 413 inclusos), e o
CORS por fora põe os headers dele até no 500.
"""

import ipaddress
import math
import re
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass

import structlog
from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.api.errors import INTERNAL_ERROR, error_response
from app.health import HEALTH_PATH
from app.log import get_logger
from app.metrics import METRICS_PATH

log = get_logger(__name__)

REQUEST_ID_HEADER = "X-Request-Id"
# O mesmo do flow-gateway-go: nada de espaço, aspas ou quebra de linha que suje o log.
_VALID_REQUEST_ID = re.compile(r"[A-Za-z0-9._:-]{1,128}")

IPAddress = ipaddress.IPv4Address | ipaddress.IPv6Address


class RequestIdMiddleware:
    """Aceita o `X-Request-Id` bem formado (senão gera um), ecoa na resposta e liga ao structlog.

    Exceção que escapa do app vira aqui o 500 do envelope, com o id na resposta e no log (o
    `ServerErrorMiddleware` do Starlette fica por fora e não veria nem um nem outro).
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        received = Headers(scope=scope).get(REQUEST_ID_HEADER, "")
        request_id = received if _VALID_REQUEST_ID.fullmatch(received) else uuid.uuid4().hex
        started = False

        async def send_with_id(message: Message) -> None:
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
                message.setdefault("headers", [])  # opcional no ASGI
                MutableHeaders(scope=message)[REQUEST_ID_HEADER] = request_id
            await send(message)

        with structlog.contextvars.bound_contextvars(request_id=request_id):
            try:
                await self.app(scope, receive, send_with_id)
            except Exception:
                log.exception("api.unhandled", path=scope["path"])
                if started:  # o status já foi: só resta derrubar a conexão
                    raise
                await error_response(500, *INTERNAL_ERROR)(scope, receive, send_with_id)


@dataclass(frozen=True)
class RateLimit:
    """Token bucket por cliente: `per_second` pedidos por segundo em média, rajadas de `burst`.

    `max_clients` é o teto de chaves em memória; acima dele, sai o cliente parado há mais tempo.
    `trusted_proxies` (IPs ou redes) são os únicos de quem o `X-Forwarded-For` vale.

    ponytail: contagem em memória, por processo; com vários workers, cada um conta o seu (o
    limite efetivo multiplica). Se precisar de um limite global, o lugar é o gateway ou o Redis.
    """

    per_second: float
    burst: int
    max_clients: int = 10_000
    trusted_proxies: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        # Valida já aqui: o Starlette só monta os middlewares no primeiro pedido.
        if not (math.isfinite(self.per_second) and self.per_second > 0):
            raise ValueError(f"per_second deve ser finito e > 0, veio {self.per_second}")
        if self.burst < 1 or self.max_clients < 1:
            raise ValueError("burst e max_clients devem ser >= 1")
        for proxy in self.trusted_proxies:
            ipaddress.ip_network(proxy)  # ValueError com o valor ruim


def _address(text: str) -> IPAddress | None:
    try:
        address = ipaddress.ip_address(text.strip())
    except ValueError:
        return None
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        return address.ipv4_mapped
    return address


class RateLimitMiddleware:
    """429 `rate_limited` com `Retry-After` para o cliente acima do `RateLimit`.

    O cliente é o IP da conexão. O `X-Forwarded-For` só vale quando a conexão vem de um dos
    `trusted_proxies`: percorrido da direita para a esquerda, o cliente é o primeiro salto que
    não é proxy confiável (e o salto mal formado encerra a busca). Quem fala direto com o
    serviço não escolhe a própria chave. IPv6 conta por /64, o bloco que um cliente costuma ter.
    """

    def __init__(
        self, app: ASGIApp, limit: RateLimit, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self.app = app
        self._limit = limit
        self._trusted = [ipaddress.ip_network(proxy) for proxy in limit.trusted_proxies]
        self._clock = clock
        self._buckets: OrderedDict[str, tuple[float, float]] = OrderedDict()  # fichas, quando

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        # A sonda de saúde (cache curto, app.health) e a raspagem do Prometheus não contam.
        if scope["type"] != "http" or scope.get("path") in (HEALTH_PATH, METRICS_PATH):
            await self.app(scope, receive, send)
            return
        wait = self._take(self._client(scope))
        if wait > 0:
            message = "pedidos demais deste cliente; tente de novo depois do Retry-After"
            headers = {"Retry-After": str(max(1, math.ceil(wait)))}
            await error_response(429, "rate_limited", message, headers)(scope, receive, send)
            return
        await self.app(scope, receive, send)

    def _is_trusted(self, address: IPAddress) -> bool:
        return any(address in network for network in self._trusted)

    def _client(self, scope: Scope) -> str:
        peer = scope.get("client")
        address = _address(peer[0]) if peer else None
        if address is None:
            return "desconhecido"
        if self._is_trusted(address):
            hops = ",".join(Headers(scope=scope).getlist("x-forwarded-for")).split(",")
            for hop in reversed(hops):
                candidate = _address(hop)
                if candidate is None:
                    break
                address = candidate
                if not self._is_trusted(candidate):
                    break
        if isinstance(address, ipaddress.IPv6Address):
            return str(ipaddress.ip_network(f"{address}/64", strict=False))
        return str(address)

    def _take(self, key: str) -> float:
        """Gasta uma ficha de `key`; devolve 0 ou os segundos até a próxima ficha."""
        now, limit = self._clock(), self._limit
        tokens, at = self._buckets.pop(key, (float(limit.burst), now))
        tokens = min(float(limit.burst), tokens + (now - at) * limit.per_second)
        wait = 0.0 if tokens >= 1 else (1 - tokens) / limit.per_second
        self._buckets[key] = (tokens - 1 if wait == 0 else tokens, now)  # vai para o fim
        if len(self._buckets) > limit.max_clients:
            self._buckets.popitem(last=False)
        return wait

"""Saúde detalhada: checagem com cache, dependência crítica, sonda da OpenRouter e o /healthz."""

import asyncio
import io
import json
from typing import Any

import httpx
import pytest
from pydantic import SecretStr

from app.api import RateLimit, create_app
from app.clients.openrouter import OpenRouterEmbeddingClient
from app.health import CachedCheck, Dependency, report
from app.log import configure_logging, shutdown_logging

KEY = "sk-or-v1-chave-que-nao-pode-vazar"


class Clock:
    now = 0.0

    def __call__(self) -> float:
        return self.now


class Probe:
    """Checagem que conta as chamadas; `gate` segura a resposta, `error` faz levantar."""

    def __init__(self, value: bool = True) -> None:
        self.value, self.calls, self.gate, self.error = value, 0, asyncio.Event(), False
        self.gate.set()

    async def __call__(self) -> bool:
        self.calls += 1
        await self.gate.wait()
        if self.error:
            raise ConnectionRefusedError("host interno 10.0.0.5:5432")
        return self.value


async def test_cached_check_reuses_the_result_within_the_ttl_and_joins_concurrent_probes() -> None:
    clock, probe = Clock(), Probe()
    check = CachedCheck(probe, ttl_seconds=10, clock=clock)
    assert await check() and await check() and probe.calls == 1
    clock.now, probe.value = 10.0, False
    assert not await check() and probe.calls == 2  # venceu: checa de novo
    clock.now, probe.value = 20.0, True
    probe.gate.clear()
    waiting = [asyncio.create_task(check()) for _ in range(3)]
    await asyncio.sleep(0)
    waiting[0].cancel()  # a sonda que desiste não cancela a checagem das outras
    probe.gate.set()
    assert await asyncio.gather(*waiting[1:]) == [True, True] and probe.calls == 3
    with pytest.raises(ValueError):
        CachedCheck(probe, ttl_seconds=float("nan"))


async def test_a_check_that_raises_counts_as_down_and_logs_only_the_error_type() -> None:
    stream, probe = io.StringIO(), Probe()
    probe.error = True
    configure_logging("INFO", stream=stream)
    try:
        assert not await CachedCheck(probe, ttl_seconds=0)()
    finally:
        shutdown_logging()
    (line,) = [json.loads(raw) for raw in stream.getvalue().splitlines()]
    assert (line["event"], line["error"]) == ("health.check_error", "ConnectionRefusedError")
    assert "10.0.0.5" not in stream.getvalue()


@pytest.mark.parametrize(
    ("postgres", "openrouter", "expected"),
    [
        (True, True, (200, "ok")),
        (True, False, (200, "degraded")),  # a OpenRouter tem o fallback léxico: não é crítica
        (False, True, (503, "unavailable")),
    ],
)
async def test_report_is_503_only_when_a_critical_dependency_is_down(
    postgres: bool, openrouter: bool, expected: tuple[int, str]
) -> None:
    deps = {
        "postgres": Dependency(Probe(postgres)),
        "openrouter": Dependency(Probe(openrouter), False),
    }
    status, body = await report(deps)
    assert (status, body.status) == expected
    assert body.checks == {
        "postgres": "ok" if postgres else "down",
        "openrouter": "ok" if openrouter else "down",
    }
    empty_status, empty = await report({})
    assert (empty_status, empty.status, empty.checks) == (200, "ok", {})  # sem dependência: de pé


async def test_a_raw_check_that_raises_is_down_in_the_report_not_a_500() -> None:
    probe = Probe()
    probe.error = True  # sem o CachedCheck por fora
    status, body = await report({"postgres": Dependency(probe)})
    assert (status, body.checks) == (503, {"postgres": "down"})


@pytest.mark.parametrize(
    ("response", "healthy"),
    [
        (httpx.Response(200, json={"data": {"limit_remaining": 7.5}}), True),
        (httpx.Response(200, json={"data": {"limit_remaining": None}}), True),  # sem teto
        (httpx.Response(200, json={"data": {"limit_remaining": 0}}), False),  # crédito acabou
        (httpx.Response(401, json={"error": {"message": "chave inválida"}}), False),
    ],
)
async def test_openrouter_probe_hits_the_free_key_endpoint_and_reads_the_credit(
    response: httpx.Response, healthy: bool
) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return response

    transport = httpx.MockTransport(handler)
    async with OpenRouterEmbeddingClient(SecretStr(KEY), transport=transport) as client:
        assert await client.check_key() is healthy
    (request,) = seen
    assert (request.method, request.url.path) == ("GET", "/api/v1/key")
    assert request.headers["authorization"] == f"Bearer {KEY}"
    assert request.extensions["timeout"]["read"] == 5.0  # não o de 60 s do embedding


async def test_healthz_route_reports_each_dependency_and_skips_the_rate_limit() -> None:
    probes: dict[str, Any] = {"postgres": Probe(False), "openrouter": Probe(True)}
    deps = {
        "postgres": Dependency(probes["postgres"]),
        "openrouter": Dependency(probes["openrouter"], False),
    }
    app = create_app(health=deps, rate_limit=RateLimit(per_second=0.001, burst=1))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://api") as http:
        replies = [await http.get("/healthz") for _ in range(3)]  # além do burst de 1
        limited = [
            (await http.post("/api/v1/search", json={"query": "q"})).status_code for _ in range(2)
        ]
    assert [reply.status_code for reply in replies] == [503, 503, 503]
    assert replies[0].json() == {
        "status": "unavailable",
        "checks": {"postgres": "down", "openrouter": "ok"},
    }
    assert limited == [503, 429]  # sem busca configurada: 503; o 2º já esbarra no rate limit

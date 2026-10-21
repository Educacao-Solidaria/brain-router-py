"""Rota `/metrics`: raspagem de verdade num uvicorn, fora do rate limit e do GZip."""

import asyncio
import subprocess
import sys
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest
import uvicorn
from prometheus_client.parser import text_string_to_metric_families

from app.api import RateLimit, create_app
from app.metrics import CONTENT_TYPE
from app.schemas.mcp import HybridSearchInput, HybridSearchOutput


async def _answer(body: HybridSearchInput) -> HybridSearchOutput:
    return HybridSearchOutput(query=body.query, results=[], total_found=0, execution_time_ms=1)


@pytest.fixture
async def base_url() -> AsyncIterator[str]:
    """A API num uvicorn de verdade (porta livre), com um pedido só no rate limit."""
    app = create_app(search=_answer, rate_limit=RateLimit(per_second=0.001, burst=1))
    server = uvicorn.Server(uvicorn.Config(app, port=0, log_level="warning"))
    task = asyncio.create_task(server.serve())
    try:
        async with asyncio.timeout(10):
            while not server.started:
                await asyncio.sleep(0.01)
        yield f"http://127.0.0.1:{server.servers[0].sockets[0].getsockname()[1]}"
    finally:
        server.should_exit = True
        await task


async def test_a_real_scrape_gets_the_text_format_without_limit_or_gzip(base_url: str) -> None:
    async with httpx.AsyncClient(base_url=base_url) as client:  # pede gzip, como o Prometheus
        statuses = [(await client.post("/api/v1/search", json={"query": "q"})).status_code]
        statuses.append((await client.post("/api/v1/search", json={"query": "q"})).status_code)
        scrapes = [await client.get("/metrics") for _ in range(5)]

    assert statuses == [200, 429]  # o limite vale para a API...
    assert [scrape.status_code for scrape in scrapes] == [200] * 5  # ...e não para a raspagem
    last = scrapes[-1]
    assert last.request.headers["accept-encoding"].startswith("gzip")
    assert "content-encoding" not in last.headers and len(last.content) > 1000  # o GZip pularia
    assert last.headers["content-type"] == "text/plain; version=0.0.4; charset=utf-8"
    families = {family.name: family for family in text_string_to_metric_families(last.text)}
    searches = families["brain_router_search_requests"].samples
    assert [s.value for s in searches if s.labels == {"mode": "hybrid", "outcome": "ok"}][0] >= 1
    assert "process_resident_memory_bytes" in families  # as de processo vêm junto


async def test_a_prometheus_accept_header_gets_the_text_format(base_url: str) -> None:
    # Como o Prometheus 3: OpenMetrics primeiro, o texto depois. Sai o texto, que está na lista.
    accept = (
        "application/openmetrics-text;version=1.0.0;q=0.5,"
        "text/plain;version=1.0.0;escaping=allow-utf-8;q=0.3,text/plain;version=0.0.4;q=0.2"
    )
    async with httpx.AsyncClient(base_url=base_url) as client:
        scrape = await client.get("/metrics", headers={"Accept": accept})
    assert scrape.status_code == 200 and scrape.headers["content-type"] == CONTENT_TYPE
    assert scrape.text.endswith("\n") and "# EOF" not in scrape.text  # texto, não OpenMetrics


def test_in_multiprocess_mode_the_route_serves_what_every_worker_counted(tmp_path: Path) -> None:
    # Três processos: dois buscam e saem; o terceiro, que nunca buscou, serve a soma no /metrics.
    script = """if True:
        import asyncio, sys, httpx
        from app.api import create_app
        from app.schemas.mcp import HybridSearchOutput

        async def answer(body):
            return HybridSearchOutput(query="q", results=[], total_found=0, execution_time_ms=1)

        async def main(step):
            transport = httpx.ASGITransport(app=create_app(search=answer))
            async with httpx.AsyncClient(transport=transport, base_url="http://brain") as client:
                if step == "search":
                    assert (await client.post("/api/v1/search", json={"query": "q"})).is_success
                else:  # num arquivo: o stdout pode levar log
                    open(step, "w").write((await client.get("/metrics")).text)

        asyncio.run(main(sys.argv[1]))
    """
    env = {"PROMETHEUS_MULTIPROC_DIR": str(tmp_path), "PATH": "/usr/bin:/bin"}
    scraped = tmp_path.parent / "scrape.txt"
    for step in ("search", "search", str(scraped)):
        run = subprocess.run(
            [sys.executable, "-c", script, step], env=env, capture_output=True, text=True
        )
        assert run.returncode == 0, run.stderr
    families = {f.name: f for f in text_string_to_metric_families(scraped.read_text())}
    searches = families["brain_router_search_requests"].samples
    assert [s.value for s in searches if s.labels == {"mode": "hybrid", "outcome": "ok"}] == [2.0]
    assert "process_resident_memory_bytes" not in families  # nesse modo, só as do serviço


def test_the_route_stays_out_of_the_openapi_schema() -> None:
    assert "/metrics" not in create_app().openapi()["paths"]

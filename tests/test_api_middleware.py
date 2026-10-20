"""Middlewares da API v1: X-Request-Id, rate limit por IP, CORS e GZip (sem rede nem banco)."""

import io
import json
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from starlette.types import Message, Receive, Scope, Send

from app.api import RateLimit, create_app
from app.api.middleware import RateLimitMiddleware
from app.log import configure_logging, get_logger, shutdown_logging
from app.schemas.domain import Chunk, SearchResult
from app.schemas.mcp import HybridSearchInput, HybridSearchOutput

SEARCH = "/api/v1/search"
ORIGIN = "https://studio.exemplo.org"
log = get_logger("tests.api")


async def search(body: HybridSearchInput) -> HybridSearchOutput:
    log.info("busca.feita")  # tem de sair com o request_id do pedido
    if body.query == "explode":
        raise RuntimeError("falha inesperada")
    chunk = Chunk(id="doc-1:0", document_id="doc-1", content="x" * (400 * body.top_k))
    result = SearchResult(chunk=chunk, rrf_score=0.03, rank=1)
    return HybridSearchOutput(
        query=body.query, results=[result], total_found=1, execution_time_ms=1.0
    )


def client(app: FastAPI, ip: str = "203.0.113.9") -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=app, client=(ip, 4321))
    return httpx.AsyncClient(transport=transport, base_url="http://api")


@pytest.mark.parametrize(
    ("sent", "echoed"),
    [
        ("pedido-42._:AZ", True),
        ("a" * 128, True),
        ("a" * 129, False),
        ("id com espaco", False),
        (b"id-\xe7", False),  # latin-1, como o h11 entrega byte fora do ASCII
        ('id"aspas', False),
        ("", False),
        (None, False),
    ],
)
async def test_request_id_is_echoed_when_well_formed_and_replaced_when_not(
    sent: str | bytes | None, echoed: bool
) -> None:
    raw = sent.encode() if isinstance(sent, str) else sent
    headers = [] if raw is None else [(b"x-request-id", raw)]
    async with client(create_app(search=search)) as http:
        response = await http.post(SEARCH, json={"query": "x"}, headers=headers)

    request_id = response.headers["x-request-id"]
    if echoed:
        assert request_id == sent
    else:
        assert request_id != sent and len(request_id) == 32 and int(request_id, 16) >= 0


async def test_request_id_goes_on_every_response_even_errors_and_the_stream() -> None:
    app = create_app(search=search, max_body_bytes=200, rate_limit=RateLimit(0.001, burst=6))
    headers = {"x-request-id": "pedido-7"}
    async with client(app) as http:
        responses = [
            await http.get("/nada", headers=headers),
            await http.post(SEARCH, json={"query": "x", "top_k": 99}, headers=headers),
            await http.post(SEARCH, json={"query": "x" * 300}, headers=headers),
            await http.post(SEARCH, json={"query": "explode"}, headers=headers),
            await http.post(f"{SEARCH}/stream", json={"query": "x"}, headers=headers),
            await http.get("/nada", headers=headers),
            await http.get("/nada", headers=headers),  # a 7ª passa da rajada de 6
        ]

    assert [r.status_code for r in responses] == [404, 422, 413, 500, 200, 404, 429]
    assert all(r.headers["x-request-id"] == "pedido-7" for r in responses)
    assert responses[3].json() == {
        "error": {"code": "internal_error", "message": "erro interno do servidor"}
    }


async def test_request_id_is_bound_to_every_log_line_of_the_request() -> None:
    stream = io.StringIO()
    configure_logging("INFO", stream=stream)
    try:
        async with client(create_app(search=search)) as http:
            await http.post(SEARCH, json={"query": "x"}, headers={"x-request-id": "pedido-1"})
            await http.post(SEARCH, json={"query": "explode"}, headers={"x-request-id": "pedido-2"})
    finally:
        shutdown_logging()

    lines = [json.loads(raw) for raw in stream.getvalue().splitlines()]
    done = [(line["event"], line["request_id"]) for line in lines if line["event"] == "busca.feita"]
    assert done == [("busca.feita", "pedido-1"), ("busca.feita", "pedido-2")]
    (crash,) = [line for line in lines if line["event"] == "api.unhandled"]
    assert crash["request_id"] == "pedido-2" and crash["level"] == "error" and "exception" in crash


async def test_rate_limit_answers_429_with_retry_after_per_client() -> None:
    app = create_app(search=search, rate_limit=RateLimit(per_second=0.001, burst=2))
    async with client(app) as http, client(app, ip="198.51.100.4") as other:
        codes = [(await http.post(SEARCH, json={"query": "x"})).status_code for _ in range(2)]
        limited = await http.post(SEARCH, json={"query": "x"})
        elsewhere = await other.post(SEARCH, json={"query": "x"})

    assert codes == [200, 200]
    assert (limited.status_code, limited.headers["retry-after"]) == (429, "1000")
    assert limited.json()["error"]["code"] == "rate_limited"
    assert elsewhere.status_code == 200


class Clock:
    now = 0.0

    def __call__(self) -> float:
        return self.now


async def _ok(scope: Scope, receive: Receive, send: Send) -> None:
    await send({"type": "http.response.start", "status": 200, "headers": []})
    await send({"type": "http.response.body", "body": b""})


async def hit(limiter: RateLimitMiddleware, peer: str, xff: str | None = None) -> tuple[int, str]:
    """Status e Retry-After de um pedido de `peer` (com `X-Forwarded-For`, se vier)."""
    headers = [(b"x-forwarded-for", xff.encode())] if xff is not None else []
    scope = {"type": "http", "method": "GET", "client": (peer, 1), "headers": headers}
    sent: list[Message] = []

    async def receive() -> Message:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: Message) -> None:
        sent.append(message)

    await limiter(scope, receive, send)
    found = dict(sent[0]["headers"])
    return sent[0]["status"], found.get(b"retry-after", b"").decode()


async def test_bucket_refills_at_the_configured_rate() -> None:
    clock = Clock()
    limiter = RateLimitMiddleware(_ok, RateLimit(per_second=2, burst=1), clock=clock)
    assert await hit(limiter, "203.0.113.1") == (200, "")
    clock.now = 0.1
    assert await hit(limiter, "203.0.113.1") == (429, "1")  # faltam 0,4 s → 1
    clock.now = 0.5
    assert await hit(limiter, "203.0.113.1") == (200, "")


async def test_forwarded_for_only_counts_from_trusted_proxies() -> None:
    limit = RateLimit(per_second=0.001, burst=1, trusted_proxies=("10.0.0.0/8",))
    limiter = RateLimitMiddleware(_ok, limit, clock=Clock())
    # Direto, sem proxy: o header não troca a chave.
    assert (await hit(limiter, "198.51.100.7", "1.1.1.1"))[0] == 200
    assert (await hit(limiter, "198.51.100.7", "2.2.2.2"))[0] == 429
    # Pelo proxy: o cliente é o 1º salto não confiável, da direita para a esquerda.
    assert (await hit(limiter, "10.0.0.1", "203.0.113.5, 10.1.2.3"))[0] == 200
    assert (await hit(limiter, "10.0.0.1", "203.0.113.6, 10.1.2.3"))[0] == 200
    assert (await hit(limiter, "10.0.0.1", "6.6.6.6, 203.0.113.5"))[0] == 429  # forjar à esquerda
    # Salto mal formado: fica a chave do proxy, que o cliente não escolhe.
    assert (await hit(limiter, "10.0.0.1", "lixo"))[0] == 200
    assert (await hit(limiter, "10.0.0.1", "outro lixo"))[0] == 429


async def test_ipv6_counts_per_64_and_mapped_ipv4_is_ipv4() -> None:
    limiter = RateLimitMiddleware(_ok, RateLimit(per_second=0.001, burst=1), clock=Clock())
    assert (await hit(limiter, "2001:db8::1"))[0] == 200
    assert (await hit(limiter, "2001:db8::ffff"))[0] == 429  # mesmo /64
    assert (await hit(limiter, "2001:db8:0:1::1"))[0] == 200
    assert (await hit(limiter, "203.0.113.9"))[0] == 200
    assert (await hit(limiter, "::ffff:203.0.113.9"))[0] == 429


async def test_key_cap_drops_the_client_idle_the_longest() -> None:
    limit = RateLimit(per_second=0.001, burst=1, max_clients=2)
    limiter = RateLimitMiddleware(_ok, limit, clock=Clock())
    for peer in ("203.0.113.1", "203.0.113.2", "203.0.113.3"):
        assert (await hit(limiter, peer))[0] == 200
    assert len(limiter._buckets) == 2
    assert (await hit(limiter, "203.0.113.1"))[0] == 200  # saiu do teto: começou de novo
    assert (await hit(limiter, "203.0.113.3"))[0] == 429


@pytest.mark.parametrize(
    "options",
    [
        {"per_second": 0.0, "burst": 1},
        {"per_second": float("nan"), "burst": 1},
        {"per_second": float("inf"), "burst": 1},
        {"per_second": 1.0, "burst": 0},
        {"per_second": 1.0, "burst": 1, "max_clients": 0},
        {"per_second": 1.0, "burst": 1, "trusted_proxies": ("proxy.interno",)},
    ],
)
def test_invalid_rate_limit_is_refused(options: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        RateLimit(**options)


async def test_cors_allows_only_the_listed_origins_without_credentials() -> None:
    app = create_app(search=search, cors_origins=[ORIGIN])
    preflight = {"origin": ORIGIN, "access-control-request-method": "POST"}
    async with client(app) as http:
        allowed = await http.options(SEARCH, headers=preflight)
        refused = await http.options(SEARCH, headers=preflight | {"origin": "https://outro.org"})
        simple = await http.post(SEARCH, json={"query": "x"}, headers={"origin": ORIGIN})
        crash = await http.post(SEARCH, json={"query": "explode"}, headers={"origin": ORIGIN})
        stranger = await http.post(SEARCH, json={"query": "x"}, headers={"origin": "https://a.b"})

    assert (allowed.status_code, refused.status_code) == (200, 400)
    assert allowed.headers["access-control-allow-origin"] == ORIGIN
    assert "access-control-allow-credentials" not in allowed.headers
    assert simple.headers["access-control-allow-origin"] == ORIGIN
    assert "x-request-id" in simple.headers["access-control-expose-headers"].lower()
    assert crash.headers["access-control-allow-origin"] == ORIGIN  # até no 500
    assert "access-control-allow-origin" not in stranger.headers
    async with client(create_app(search=search)) as http:  # sem lista, sem CORS
        plain = await http.post(SEARCH, json={"query": "x"}, headers={"origin": ORIGIN})
    assert "access-control-allow-origin" not in plain.headers


@pytest.mark.parametrize("origin", ["*", "https://x.org/", "https://x.org/app", "x.org", ""])
def test_cors_needs_an_explicit_origin(origin: str) -> None:
    with pytest.raises(ValueError, match="origem de CORS"):
        create_app(cors_origins=[origin])


async def test_gzip_compresses_big_json_but_never_the_stream() -> None:
    async with client(create_app(search=search)) as http:
        big = await http.post(SEARCH, json={"query": "x", "top_k": 5})
        small = await http.post(SEARCH, json={"query": "x", "top_k": 1})
        stream = await http.post(f"{SEARCH}/stream", json={"query": "x", "top_k": 5})

    assert big.request.headers["accept-encoding"].startswith("gzip")
    assert big.headers["content-encoding"] == "gzip" and big.json()["total_found"] == 1
    assert "content-encoding" not in small.headers  # abaixo de 1000 bytes
    assert "content-encoding" not in stream.headers
    assert stream.headers["content-type"].startswith("text/event-stream")
    assert len(stream.content) > 2000

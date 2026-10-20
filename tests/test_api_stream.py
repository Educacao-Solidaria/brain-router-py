"""`POST /api/v1/search/stream` (SSE) com handlers falsos, sem rede nem banco."""

import asyncio
import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from starlette.types import Message
from support import events

from app.api import SearchHandler, create_app
from app.errors.exceptions import EmbeddingServiceError
from app.schemas.domain import Chunk, SearchResult
from app.schemas.mcp import HybridSearchInput, HybridSearchOutput

STREAM = "/api/v1/search/stream"


def _result(rank: int, content: str = "") -> SearchResult:
    chunk = Chunk(id=f"doc-1:{rank}", document_id="doc-1", content=content or f"trecho {rank}")
    return SearchResult(chunk=chunk, dense_score=0.5, rrf_score=1 / (60 + rank), rank=rank)


def _search(*results: SearchResult) -> SearchHandler:
    async def search(body: HybridSearchInput) -> HybridSearchOutput:
        return HybridSearchOutput(
            query=body.query, results=list(results), total_found=7, execution_time_ms=2.5
        )

    return search


async def post(app: FastAPI, body: dict[str, Any]) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)  # junta a resposta inteira: serve ao stream que acaba
    async with httpx.AsyncClient(transport=transport, base_url="http://api") as http:
        return await http.post(STREAM, json=body)


async def drive(
    app: FastAPI, on_body: Callable[[bytes], None], disconnect: asyncio.Event
) -> list[bytes]:
    """Chama o app ASGI direto, vendo cada pedaço do corpo na hora em que sai (como o uvicorn)."""
    raw = json.dumps({"query": "matrícula"}).encode()
    pieces: list[bytes] = []
    asked = False

    async def receive() -> Message:
        nonlocal asked
        if not asked:
            asked = True
            return {"type": "http.request", "body": raw, "more_body": False}
        await disconnect.wait()
        return {"type": "http.disconnect"}

    async def send(message: Message) -> None:
        if message["type"] == "http.response.body" and message.get("body"):
            pieces.append(message["body"])
            on_body(message["body"])

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},  # o que o uvicorn 0.54 declara
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": STREAM,
        "raw_path": STREAM.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": [(b"content-type", b"application/json")],
        "client": ("127.0.0.1", 1),
        "server": ("api", 80),
    }
    await asyncio.wait_for(app(scope, receive, send), timeout=10)
    return pieces


async def test_one_result_event_per_result_in_rank_order_then_done() -> None:
    response = await post(
        create_app(search=_search(_result(2), _result(1), _result(3))), {"query": "x"}
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache"
    assert response.headers["x-accel-buffering"] == "no"  # o nginx não segura os eventos
    names = [name for name, _ in events(response.content)]
    assert names == ["result", "result", "result", "done"]
    results = [data for name, data in events(response.content) if name == "result"]
    assert [r["rank"] for r in results] == [1, 2, 3]
    assert results[0] == _result(1).model_dump(mode="json")
    assert events(response.content)[-1] == ("done", {"total_found": 7, "execution_time_ms": 2.5})


async def test_no_results_is_just_done() -> None:
    response = await post(create_app(search=_search()), {"query": "x"})
    assert events(response.content) == [("done", {"total_found": 7, "execution_time_ms": 2.5})]


async def test_line_breaks_inside_a_chunk_cannot_forge_events() -> None:
    forged = "a\n\nevent: done\ndata: {}\r\n\r\nb\rc"
    response = await post(create_app(search=_search(_result(1, forged))), {"query": "x"})

    parsed = events(response.content)
    assert [name for name, _ in parsed] == ["result", "done"]
    assert parsed[0][1]["chunk"]["content"] == forged
    assert all(
        line.startswith(("event: ", "data: ", ": ")) for line in response.text.split("\n") if line
    )


@pytest.mark.parametrize(
    ("error", "code", "shown"),
    [
        (EmbeddingServiceError("HTTP 429 senha=hunter2"), "embedding_service_error", True),
        (RuntimeError("HTTP 429 senha=hunter2"), "internal_error", False),
    ],
)
async def test_a_failure_after_the_200_is_an_error_event(
    error: Exception, code: str, shown: bool
) -> None:
    async def failing(body: HybridSearchInput) -> HybridSearchOutput:
        raise error

    response = await post(create_app(search=failing), {"query": "x"})

    assert response.status_code == 200  # o stream já tinha começado
    ((name, data),) = events(response.content)
    assert (name, list(data), data["error"]["code"]) == ("error", ["error"], code)
    assert sorted(data["error"]) == ["code", "message"]
    assert ("hunter2" in data["error"]["message"]) is shown


async def test_missing_backend_and_invalid_body_are_still_plain_http_errors() -> None:
    unavailable = await post(create_app(), {"query": "x"})
    invalid = await post(create_app(search=_search()), {"query": "x", "top_k": 99})

    assert (unavailable.status_code, invalid.status_code) == (503, 422)
    assert unavailable.headers["content-type"] == "application/json"
    assert unavailable.json()["error"]["code"] == "backend_unavailable"
    assert invalid.json()["error"]["code"] == "validation_error"


async def test_heartbeat_comment_while_the_search_is_slow(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("fastapi.routing._PING_INTERVAL", 0.01)  # 15 s em produção
    pinged = asyncio.Event()

    async def slow(body: HybridSearchInput) -> HybridSearchOutput:
        await pinged.wait()  # só responde depois de o proxy ter visto o heartbeat
        return await _search(_result(1))(body)

    def watch(piece: bytes) -> None:
        if piece == b": ping\n\n":
            pinged.set()

    body = b"".join(await drive(create_app(search=slow), watch, asyncio.Event()))
    assert body.startswith(b": ping\n\n")
    assert [name for name, _ in events(body)] == ["result", "done"]


async def test_client_disconnect_cancels_the_search() -> None:
    started, disconnect, cancelled = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def endless(body: HybridSearchInput) -> HybridSearchOutput:
        started.set()
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        raise AssertionError("não devia acordar")

    async def leave_after_start() -> None:
        await started.wait()
        disconnect.set()

    leaving = asyncio.create_task(leave_after_start())
    pieces = await drive(create_app(search=endless), lambda piece: None, disconnect)
    await leaving

    assert cancelled.is_set()
    assert events(b"".join(pieces)) == []  # nenhum resultado depois da saída

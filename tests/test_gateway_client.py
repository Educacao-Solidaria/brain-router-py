"""Cliente do flow-gateway-go contra um transporte falso do httpx (sem rede)."""

import asyncio
import io
import json
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import SecretStr

from app.clients.gateway import GatewayClient, GatewayError
from app.log import configure_logging, shutdown_logging
from app.settings import Settings

KEY = "gw-chave-que-nao-pode-vazar"
BASE = "http://gateway:8080"
MESSAGES = [{"role": "user", "content": "Quando abre a matrícula?"}]

Step = Callable[[httpx.Request], httpx.Response]


def _completion(content: str | None = "Em março.", **extra: Any) -> dict[str, Any]:
    message = {"role": "assistant", "content": content}
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "model": "openai/gpt-x",
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15},
        **extra,
    }


def _ok(request: httpx.Request) -> httpx.Response:
    chat = request.url.path.endswith("/chat/completions")
    return httpx.Response(200, json=_completion() if chat else {})


class Gateway:
    """Gateway falso: responde com `steps` na ordem (depois, `_ok`) e guarda os pedidos.

    Com `gate`, os pedidos de evento esperam o teste soltar.
    """

    def __init__(self, *steps: Step, api_key: str | None = KEY, **options: Any) -> None:
        self.steps = list(steps)
        self.requests: list[httpx.Request] = []
        self.gate: asyncio.Event | None = None
        self.client = GatewayClient(
            BASE,
            api_key=SecretStr(api_key) if api_key is not None else None,
            transport=httpx.MockTransport(self._handle),
            **options,
        )

    async def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.gate is not None and request.url.path == "/v1/events":
            await self.gate.wait()
        return (self.steps.pop(0) if self.steps else _ok)(request)

    def events(self) -> list[dict[str, Any]]:
        return [json.loads(r.content) for r in self.requests if r.url.path == "/v1/events"]


def logs() -> tuple[io.StringIO, Callable[[str], list[dict[str, Any]]]]:
    stream = io.StringIO()
    configure_logging("DEBUG", stream=stream)

    def lines(event: str) -> list[dict[str, Any]]:
        shutdown_logging()
        parsed = [json.loads(raw) for raw in stream.getvalue().splitlines()]
        return [line for line in parsed if line["event"] == event]

    return stream, lines


async def test_chat_posts_the_openai_format_and_reads_the_first_choice() -> None:
    gateway = Gateway()
    reply = await gateway.client.chat(MESSAGES, model="openai/gpt-x", temperature=0.2, stream=True)

    (request,) = gateway.requests
    assert request.url == f"{BASE}/v1/chat/completions"
    assert request.headers["authorization"] == f"Bearer {KEY}"
    assert json.loads(request.content) == {
        "temperature": 0.2,
        "model": "openai/gpt-x",
        "messages": MESSAGES,
        "stream": False,  # este cliente não faz stream, nem se pedirem
    }
    assert (reply.content, reply.model, reply.finish_reason) == (
        "Em março.",
        "openai/gpt-x",
        "stop",
    )
    assert (reply.usage.prompt_tokens, reply.usage.total_tokens) == (12, 15)


@pytest.mark.parametrize("api_key", [None, "", "  "])
async def test_without_a_key_there_is_no_authorization_header(api_key: str | None) -> None:
    gateway = Gateway(api_key=api_key)
    await gateway.client.chat(MESSAGES, model="m")
    assert "authorization" not in gateway.requests[0].headers


def _raise(exc: type[httpx.TransportError]) -> Step:
    def step(request: httpx.Request) -> httpx.Response:
        raise exc("falhou", request=request)

    return step


def _status(code: int) -> Step:
    return lambda request: httpx.Response(code, json={"error": {"message": f"erro {code}"}})


def _body(body: Any) -> Step:
    return lambda request: httpx.Response(200, json=body)


@pytest.mark.parametrize(
    ("step", "message"),
    [
        (_status(500), "HTTP 500: erro 500"),
        (_status(429), "HTTP 429: erro 429"),
        (_status(401), "HTTP 401: erro 401"),
        (_raise(httpx.ConnectError), "ConnectError"),
        (_raise(httpx.ReadTimeout), "ReadTimeout"),
        (lambda request: httpx.Response(200, content=b"<html>"), "fora do formato"),
        (_body(_completion() | {"choices": []}), "fora do formato em choices"),
        (_body({k: v for k, v in _completion().items() if k != "model"}), "em model"),
        (_body(_completion(content=None)), "sem texto"),
    ],
)
async def test_chat_failures_raise_gateway_error_once(step: Step, message: str) -> None:
    gateway = Gateway(step)
    with pytest.raises(GatewayError, match=message) as caught:
        await gateway.client.chat(MESSAGES, model="m")

    assert (caught.value.http_status, caught.value.code) == (502, "GATEWAY_ERROR")
    assert len(gateway.requests) == 1  # sem retry: o 5xx pode já ter custado


async def test_chat_needs_messages() -> None:
    gateway = Gateway()
    with pytest.raises(ValueError, match="messages"):
        await gateway.client.chat([], model="m")
    assert gateway.requests == []


async def test_notify_sends_the_event_in_the_background() -> None:
    gateway = Gateway()
    assert gateway.client.notify("index.completed", {"document_id": "doc-1", "chunks": 3})
    assert gateway.requests == []  # quem chamou não esperou nada

    await gateway.client.aclose()  # esvazia a fila
    (event,) = gateway.events()
    assert (event["event"], event["data"]) == (
        "index.completed",
        {"document_id": "doc-1", "chunks": 3},
    )
    assert datetime.fromisoformat(event["at"]).tzinfo is not None
    assert gateway.requests[0].headers["content-type"] == "application/json"


async def test_a_full_queue_drops_without_waiting_and_warns_once() -> None:
    gateway = Gateway(queue_size=2)
    gateway.gate = asyncio.Event()  # gateway lento: o primeiro envio fica preso
    _, lines = logs()
    sent = [gateway.client.notify("busca", {"n": n}) for n in range(5)]

    assert sent == [True, True, False, False, False]
    assert gateway.client.dropped == 3
    gateway.gate.set()
    await gateway.client.aclose()
    assert [e["data"]["n"] for e in gateway.events()] == [0, 1]
    assert len(lines("gateway.events_dropped")) == 1


async def test_send_failures_are_logged_once_and_never_reach_the_caller() -> None:
    gateway = Gateway(_status(500), _raise(httpx.ConnectError), _status(503))
    _, lines = logs()
    for n in range(4):
        assert gateway.client.notify("busca", {"n": n})
    await gateway.client.aclose()

    assert len(gateway.events()) == 4  # cada um tentado uma vez; o 4º passou
    (failed,) = lines("gateway.events_failed")
    assert failed["reason"] == "HTTP 500"


async def test_close_gives_up_on_a_stuck_gateway_after_drain_seconds() -> None:
    gateway = Gateway(drain_seconds=0.05)
    gateway.gate = asyncio.Event()  # nunca solto
    _, lines = logs()
    assert gateway.client.notify("busca", {"n": 1})
    assert gateway.client.notify("busca", {"n": 2})

    await asyncio.wait_for(gateway.client.aclose(), timeout=10)
    await gateway.client.aclose()  # de novo: não espera nem avisa outra vez
    (abandoned,) = lines("gateway.events_abandoned")
    assert abandoned["pending"] == 1  # o 1º estava no ar; o 2º nem saiu
    assert not gateway.client.notify("busca", {"n": 3})  # fechado


def test_notify_outside_the_event_loop_fails_without_enqueueing() -> None:
    gateway = Gateway()
    with pytest.raises(RuntimeError):
        gateway.client.notify("busca", {"n": 1})
    assert gateway.client._events.empty()


async def test_bad_event_fails_at_the_caller_and_enqueues_nothing() -> None:
    gateway = Gateway()
    with pytest.raises(TypeError):
        gateway.client.notify("x", {"v": object()})
    with pytest.raises(ValueError):
        gateway.client.notify("x", {"v": float("nan")})
    with pytest.raises(ValueError, match="event"):
        gateway.client.notify("", {})

    await gateway.client.aclose()
    assert gateway.requests == []


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("queue_size", 0),
        ("timeout", 0.0),
        ("timeout", float("nan")),
        ("drain_seconds", 0.0),
        ("drain_seconds", float("nan")),
    ],
)
def test_invalid_configuration_is_refused(option: str, value: float) -> None:
    options: dict[str, Any] = {option: value}
    with pytest.raises(ValueError, match=option):
        GatewayClient(BASE, **options)


async def test_from_settings_reads_the_gateway_url_and_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)  # nenhum .env do repositório
    monkeypatch.setenv("DATABASE_URL", "postgresql://brain:x@db:5432/brain_router")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-x")
    with pytest.raises(ValueError, match="GATEWAY_URL"):
        GatewayClient.from_settings(Settings())

    monkeypatch.setenv("GATEWAY_URL", "http://gw.interno:9000/flow")
    monkeypatch.setenv("GATEWAY_API_KEY", KEY)
    seen: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _ok(request)

    client = GatewayClient.from_settings(Settings(), transport=httpx.MockTransport(handle))
    await client.chat(MESSAGES, model="m")

    (request,) = seen
    assert request.url == "http://gw.interno:9000/flow/v1/chat/completions"
    assert request.headers["authorization"] == f"Bearer {KEY}"
    assert KEY not in repr(client) and KEY not in repr(Settings())
    await client.aclose()

"""Harness do streaming da OpenRouter: o formato SSE de ponta a ponta, lido como um cliente."""

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest

from app.testing.openrouter_stream import Gate, StreamMock, StreamScript

URL = "https://openrouter.ai/api/v1/chat/completions"
BODY = {
    "model": "openai/gpt-4o-mini",
    "stream": True,
    "messages": [{"role": "user", "content": "oi"}],
}


async def _read(mock: StreamMock) -> tuple[int, list[str]]:
    """Status e linhas não vazias do corpo, como um cliente httpx em streaming lê."""
    async with (
        httpx.AsyncClient(transport=mock.transport) as http,
        http.stream("POST", URL, json=BODY) as response,
    ):
        return response.status_code, [line async for line in response.aiter_lines() if line]


def _chunk(line: str) -> Any:
    assert line.startswith("data: ")
    return json.loads(line.removeprefix("data: "))


async def _next_data(lines: AsyncIterator[str]) -> str:
    async for line in lines:
        if line:
            return line
    raise AssertionError("o stream acabou antes")


async def test_a_full_answer_has_keepalive_chunks_usage_and_done_in_the_openrouter_shape() -> None:
    mock = StreamMock(StreamScript(tokens=["Olá", ", mundo", "\nfim"]))
    status, lines = await _read(mock)
    assert status == 200 and lines[0] == ": OPENROUTER PROCESSING" and lines[-1] == "data: [DONE]"
    chunks = [_chunk(line) for line in lines[1:-1]]
    choices = [chunk["choices"][0] for chunk in chunks]
    # A quebra de linha do texto chega inteira, escapada no JSON de uma linha `data:` só.
    assert "".join(choice["delta"]["content"] for choice in choices) == "Olá, mundo\nfim"
    assert [choice["finish_reason"] for choice in choices] == [None, None, "stop", "stop"]
    assert choices[0]["delta"]["role"] == "assistant" and "role" not in choices[1]["delta"]
    assert chunks[-1]["usage"] == {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13}
    assert {chunk["object"] for chunk in chunks} == {"chat.completion.chunk"}
    assert json.loads(mock.requests[0].content)["stream"] is True  # o pedido fica guardado
    assert mock.streams[0].sent == 6 and mock.streams[0].closed
    empty = StreamScript(tokens=[], keepalives=0).frames()  # sem texto: só usage e [DONE]
    assert len(empty) == 2 and _chunk(empty[0].decode().strip())["usage"]["completion_tokens"] == 0


@pytest.mark.parametrize(("fail_after", "contents"), [(1, ["Olá"]), (0, [])])
async def test_a_mid_stream_error_ends_the_stream_with_no_usage_and_no_done(
    fail_after: int, contents: list[str]
) -> None:
    script = StreamScript(tokens=["Olá", "mundo"], fail_after=fail_after, keepalives=0)
    status, lines = await _read(StreamMock(script))
    *answer, last = [_chunk(line) for line in lines]
    assert status == 200  # o 200 já tinha saído: o erro vem como evento
    assert [chunk["choices"][0]["delta"]["content"] for chunk in answer] == contents
    assert last["error"] == {
        "code": "server_error",
        "message": "Provider disconnected unexpectedly",
    }
    assert last["choices"][0]["finish_reason"] == "error" and "usage" not in last


def test_line_separators_stay_raw_in_one_data_line_as_in_the_real_api() -> None:
    separator = chr(0x2028)
    (frame,) = [
        f for f in StreamScript(tokens=[f"a{separator}b"]).frames() if separator.encode() in f
    ]
    assert frame.decode().count("\n") == 2  # para o SSE, uma linha `data:` só (quebra em CR/LF)
    assert len(frame.decode().splitlines()) == 3  # o splitlines (e o aiter_lines) parte no U+2028


async def test_an_error_before_the_commit_is_a_json_status_and_no_stream() -> None:
    mock = StreamMock(StreamScript(tokens=["x"], status=402, error_message="Insufficient credits"))
    async with httpx.AsyncClient(transport=mock.transport) as http:
        response = await http.post(URL, json=BODY)
    assert response.status_code == 402 and mock.streams == []
    assert response.json() == {"error": {"code": 402, "message": "Insufficient credits"}}
    with pytest.raises(ValueError):
        StreamScript(tokens=["x"], fail_after=-1)


async def test_the_gate_releases_event_by_event_and_a_client_that_leaves_closes_the_stream() -> (
    None
):
    gate = Gate()
    mock = StreamMock(StreamScript(tokens=["a", "b", "c"], keepalives=0), pace=gate)
    async with httpx.AsyncClient(transport=mock.transport) as http:
        async with http.stream("POST", URL, json=BODY) as response:
            lines = response.aiter_lines()
            gate.release()
            assert _chunk(await _next_data(lines))["choices"][0]["delta"]["content"] == "a"
            waiting = asyncio.create_task(_next_data(lines))
            for _ in range(5):
                await asyncio.sleep(0)
            assert not waiting.done() and mock.streams[0].sent == 1  # o "b" espera o teste
            gate.release()
            assert _chunk(await waiting)["choices"][0]["delta"]["content"] == "b"
        # saiu no meio: o stream foi fechado e o resto nunca saiu
        assert mock.streams[0].closed and mock.streams[0].sent == 2

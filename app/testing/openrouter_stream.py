"""Simulador do streaming SSE da OpenRouter (Chat Completions), para teste sem chamada paga.

`StreamMock(StreamScript(tokens=[...]))` dá um `httpx.MockTransport` (`.transport`) que
responde todo pedido com o roteiro, e guarda os pedidos (`.requests`) e os streams entregues
(`.streams`: quantos eventos saíram e se o cliente fechou antes do fim). Serve a qualquer
cliente `httpx` (o de geração do pipeline, o e2e do lote C).

O formato segue a referência atual (https://openrouter.ai/docs/api_reference/streaming):
- 200 `text/event-stream`; comentários `: OPENROUTER PROCESSING` (keep-alive), que o cliente
  tem de ignorar;
- um `data: {chunk}` por pedaço de texto (`chat.completion.chunk`, texto em
  `choices[0].delta.content`; o primeiro traz `role`); o último pedaço traz o `finish_reason`;
- depois, o chunk de `usage` (delta sem conteúdo, que repete o `finish_reason`) e `data: [DONE]`;
- erro antes do commit: status HTTP ≠ 200 com `{"error": {"code", "message"}}`, sem stream;
- erro no meio: `data:` com `error` no topo e `finish_reason: "error"`, e o stream acaba ali
  (sem `usage` nem `[DONE]`). Com `fail_after=0`, ele é o único evento.

Tempo controlável e determinístico: antes de cada evento o stream espera `pace(n)`. O padrão
(`no_wait`) só cede o event loop; `Gate` solta os eventos quando o teste mandar; qualquer
`async def pace(n)` serve (`asyncio.sleep(0.05)` para latência de verdade).
Para quem consome: o texto vai cru no JSON, como na API, inclusive U+2028/U+2029. O SSE só
quebra linha em CR/LF, mas o `aiter_lines()` do httpx quebra como o `str.splitlines` (nesses
também): leia o corpo com `aiter_text()` e parta só no LF.
ponytail: um token = um pedaço de texto (o `completion_tokens` do usage conta pedaços). Se um
teste precisar da contagem real, passe os pedaços já tokenizados.
"""

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass

import httpx

KEEPALIVE = b": OPENROUTER PROCESSING\n\n"
DONE = b"data: [DONE]\n\n"

Pace = Callable[[int], Awaitable[None]]


async def no_wait(step: int) -> None:
    await asyncio.sleep(0)  # cede o loop, sem relógio: o mesmo resultado em qualquer máquina


class Gate:
    """`pace` que só deixa passar os eventos liberados por `release(n)`."""

    def __init__(self) -> None:
        self._allowed = 0
        self._moved = asyncio.Event()

    def release(self, count: int = 1) -> None:
        self._allowed += count
        self._moved.set()

    async def __call__(self, step: int) -> None:
        while step >= self._allowed:  # acordou e ainda não é a vez: espera o próximo release
            self._moved.clear()
            await self._moved.wait()


@dataclass(frozen=True, slots=True)
class StreamScript:
    """Uma resposta: os pedaços de texto e, se for o caso, o erro (antes ou no meio)."""

    tokens: Sequence[str]
    model: str = "openai/gpt-4o-mini"
    keepalives: int = 1  # comentários antes do primeiro pedaço
    fail_after: int | None = None  # erro no meio depois de tantos pedaços
    status: int = 200  # ≠ 200: erro antes do commit
    error_code: str = "server_error"
    error_message: str = "Provider disconnected unexpectedly"
    finish_reason: str = "stop"
    prompt_tokens: int = 10

    def __post_init__(self) -> None:
        if self.fail_after is not None and self.fail_after < 0:  # fatia negativa mentiria
            raise ValueError(f"fail_after deve ser >= 0, veio {self.fail_after}")

    def frames(self) -> list[bytes]:
        """Os bytes do stream, evento por evento, como chegam pela rede."""
        frames = [KEEPALIVE] * self.keepalives
        sent = list(self.tokens if self.fail_after is None else self.tokens[: self.fail_after])
        for n, token in enumerate(sent):
            last = self.fail_after is None and n == len(sent) - 1
            finish = self.finish_reason if last else None
            frames.append(self._data({"choices": [_choice(token, n == 0, finish)]}))
        if self.fail_after is not None:
            choice = {"index": 0, "delta": {"content": ""}, "finish_reason": "error"}
            error = {"code": self.error_code, "message": self.error_message}
            return [*frames, self._data({"error": error, "choices": [choice]})]
        tokens = {"prompt_tokens": self.prompt_tokens, "completion_tokens": len(sent)}
        usage = {**tokens, "total_tokens": self.prompt_tokens + len(sent)}
        frames.append(
            self._data({"choices": [_choice("", True, self.finish_reason)], "usage": usage})
        )
        return [*frames, DONE]

    def _data(self, payload: dict[str, object]) -> bytes:
        chunk: dict[str, object] = {"id": "gen-mock", "provider": "mock", "model": self.model}
        chunk |= {"object": "chat.completion.chunk", "created": 0, **payload}
        # json.dumps escapa a quebra de linha do texto: cada evento é uma linha `data:` só.
        return b"data: " + json.dumps(chunk, ensure_ascii=False).encode() + b"\n\n"


def _choice(content: str, role: bool, finish: str | None) -> dict[str, object]:
    delta = {"role": "assistant", "content": content} if role else {"content": content}
    return {"index": 0, "delta": delta, "finish_reason": finish, "native_finish_reason": finish}


class ScriptedStream(httpx.AsyncByteStream):
    """O corpo da resposta: entrega os eventos um a um, esperando `pace(n)` antes de cada."""

    def __init__(self, frames: Sequence[bytes], pace: Pace) -> None:
        self.frames, self.pace = list(frames), pace
        self.sent, self.closed = 0, False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for n, frame in enumerate(self.frames):
            await self.pace(n)
            self.sent += 1
            yield frame

    async def aclose(self) -> None:
        self.closed = True  # o cliente fechou a resposta (no fim ou no meio)


class StreamMock:
    """`httpx.MockTransport` (`.transport`) que responde todo pedido com o `script`."""

    def __init__(self, script: StreamScript, *, pace: Pace = no_wait) -> None:
        self.script, self.pace = script, pace
        self.requests: list[httpx.Request] = []
        self.streams: list[ScriptedStream] = []
        self.transport = httpx.MockTransport(self._handle)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.script.status != 200:
            error = {"code": self.script.status, "message": self.script.error_message}
            return httpx.Response(self.script.status, json={"error": error})
        stream = ScriptedStream(self.script.frames(), self.pace)
        self.streams.append(stream)
        headers = {"content-type": "text/event-stream", "cache-control": "no-cache"}
        return httpx.Response(200, headers=headers, stream=stream)

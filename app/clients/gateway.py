"""Cliente assíncrono do flow-gateway-go: chat completions (formato da OpenAI) e eventos.

- `chat(messages, model=...)`: `POST /v1/chat/completions` sem stream, no formato da OpenAI (a
  rota é o PR 66 do gateway, ainda fora da main). Devolve o texto da primeira escolha, o modelo,
  o motivo de parada e o uso de tokens. **Sem retry**: um 5xx depois de o gateway ter chamado o
  modelo pode já ter custado, e repetir cobraria de novo. Falha vira `GatewayError` (502) com o
  status e a mensagem curta do gateway.
- `notify(event, data)`: fire-and-forget. Serializa na hora (dado fora do JSON falha em quem
  chamou), enfileira sem esperar numa fila limitada e devolve False se ela estiver cheia (o
  evento é descartado e contado em `dropped`). Uma tarefa em segundo plano manda os eventos, um
  por vez, em `POST /v1/events`; falha de envio só vai para o log, uma vez por sequência de
  falhas, para não inundá-lo com o gateway fora. Chame de dentro do event loop.
- `aclose()`: espera a fila esvaziar por até `drain_seconds` e fecha a tarefa e o cliente HTTP.
"""

import asyncio
import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any, Self

import httpx
from pydantic import BaseModel, Field, SecretStr, ValidationError

from app.clients.openrouter import _error_detail
from app.errors.exceptions import BrainRouterError
from app.log import get_logger
from app.settings import Settings

log = get_logger(__name__)

CHAT_PATH = "/v1/chat/completions"
# ponytail: nenhum item do gateway define ainda a rota que recebe eventos; este caminho é a
# proposta. Quando o gateway tiver a dele, troque aqui (ou vire configuração).
EVENTS_PATH = "/v1/events"
EVENT_TIMEOUT = 5.0


class GatewayError(BrainRouterError):
    def __init__(self, message: str) -> None:
        super().__init__(
            message=f"Falha no flow-gateway-go: {message}", code="GATEWAY_ERROR", http_status=502
        )


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatReply(BaseModel):
    """O que interessa da resposta: o texto da primeira escolha e os números dela."""

    content: str
    model: str
    finish_reason: str | None
    usage: Usage


class _Message(BaseModel):
    content: str | None = None


class _Choice(BaseModel):
    message: _Message
    finish_reason: str | None = None


class _Completion(BaseModel):
    model: str
    choices: list[_Choice] = Field(min_length=1)
    usage: Usage = Field(default_factory=Usage)


class GatewayClient:
    """Cliente do gateway. Use com `async with` ou chame `aclose()` no shutdown."""

    def __init__(
        self,
        base_url: str,
        *,
        api_key: SecretStr | None = None,
        timeout: float = 120.0,
        queue_size: int = 1000,
        drain_seconds: float = 5.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not timeout > 0:  # recusa também NaN
            raise ValueError(f"timeout deve ser > 0, veio {timeout}")
        if queue_size < 1:
            raise ValueError(f"queue_size deve ser >= 1, veio {queue_size}")
        if not drain_seconds > 0:  # com 0, o wait_for do 3.12 estoura até com a fila vazia
            raise ValueError(f"drain_seconds deve ser > 0, veio {drain_seconds}")
        token = api_key.get_secret_value() if api_key is not None else ""
        self._http = httpx.AsyncClient(
            base_url=base_url,
            headers={"Authorization": f"Bearer {token}"} if token.strip() else {},
            timeout=timeout,
            transport=transport,
        )
        self._events: asyncio.Queue[bytes] = asyncio.Queue(maxsize=queue_size)
        self._drain_seconds = drain_seconds
        self._sender: asyncio.Task[None] | None = None
        self._closed = False
        self._dropping = False  # já avisou do descarte nesta sequência de fila cheia
        self._failing = False  # já avisou da falha nesta sequência de envios falhos
        self.dropped = 0

    @classmethod
    def from_settings(cls, settings: Settings, **options: Any) -> Self:
        """URL e chave do `Settings` (`GATEWAY_URL`, `GATEWAY_API_KEY`); o resto por `options`."""
        if settings.gateway_url is None:
            raise ValueError("GATEWAY_URL não está configurada")
        return cls(str(settings.gateway_url), api_key=settings.gateway_api_key, **options)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(base_url={str(self._http.base_url)!r})"

    async def chat(
        self, messages: list[dict[str, str]], *, model: str, **options: Any
    ) -> ChatReply:
        """Uma resposta do modelo pelo gateway; `options` vão no corpo (ex.: `temperature`)."""
        if not messages:
            raise ValueError("messages não pode ser vazia")
        payload = {**options, "model": model, "messages": messages, "stream": False}
        try:
            response = await self._http.post(CHAT_PATH, json=payload)
        except httpx.HTTPError as exc:  # timeout, conexão recusada, DNS
            raise GatewayError(type(exc).__name__) from exc
        if not response.is_success:
            raise GatewayError(f"HTTP {response.status_code}: {_error_detail(response)}")
        try:
            completion = _Completion.model_validate_json(response.content)
        except ValidationError as exc:
            # Só o primeiro erro, sem o valor: a resposta traz texto do modelo.
            first = exc.errors(include_input=False)[0]
            where = ".".join(str(part) for part in first["loc"]) or "(corpo)"
            raise GatewayError(f"resposta fora do formato em {where}: {first['msg']}") from None
        choice = completion.choices[0]
        if choice.message.content is None:
            raise GatewayError("resposta sem texto na primeira escolha")
        return ChatReply(
            content=choice.message.content,
            model=completion.model,
            finish_reason=choice.finish_reason,
            usage=completion.usage,
        )

    def notify(self, event: str, data: Mapping[str, Any]) -> bool:
        """Enfileira o evento sem esperar. False se a fila está cheia ou o cliente fechou."""
        if not event:
            raise ValueError("event não pode ser vazio")
        loop = asyncio.get_running_loop()  # fora do loop, falha antes de mexer na fila
        moment = datetime.now(UTC).isoformat()
        record = {"event": event, "data": dict(data), "at": moment}
        body = json.dumps(record, ensure_ascii=False, allow_nan=False).encode()
        if self._closed:
            return False
        try:
            self._events.put_nowait(body)
        except asyncio.QueueFull:
            self.dropped += 1
            if not self._dropping:
                self._dropping = True
                log.warning("gateway.events_dropped", name=event, queue_size=self._events.maxsize)
            return False
        self._dropping = False
        if self._sender is None:
            self._sender = loop.create_task(self._send_events())
        return True

    async def aclose(self) -> None:
        self._closed = True
        if self._sender is not None:
            try:
                await asyncio.wait_for(self._events.join(), self._drain_seconds)
            except TimeoutError:
                log.warning("gateway.events_abandoned", pending=self._events.qsize())
            self._sender.cancel()
            await asyncio.wait({self._sender})  # não engole o cancelamento de quem fecha
            self._sender = None  # fechar de novo não espera nem avisa outra vez
        await self._http.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    async def _send_events(self) -> None:
        while True:
            body = await self._events.get()
            try:
                await self._send(body)
            except Exception:  # bug, não falha de rede: registra e segue com a fila
                log.exception("gateway.events_bug")
            finally:
                self._events.task_done()

    async def _send(self, body: bytes) -> None:
        headers = {"content-type": "application/json"}
        try:
            response = await self._http.post(
                EVENTS_PATH, content=body, headers=headers, timeout=EVENT_TIMEOUT
            )
            reason = None if response.is_success else f"HTTP {response.status_code}"
        except httpx.HTTPError as exc:
            reason = type(exc).__name__
        if reason is None:
            self._failing = False
        elif not self._failing:
            self._failing = True
            log.warning("gateway.events_failed", reason=reason)

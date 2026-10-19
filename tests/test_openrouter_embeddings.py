"""Cliente de embeddings da OpenRouter contra um transporte falso do httpx (sem rede)."""

import email.utils
import io
import json
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import SecretStr

from app.clients.interfaces import BaseEmbeddingClient
from app.clients.openrouter import OpenRouterEmbeddingClient
from app.errors.exceptions import EmbeddingServiceError
from app.log import configure_logging, shutdown_logging
from app.settings import Settings

KEY = "sk-or-v1-chave-que-nao-pode-vazar"
DIMENSION = 3
TEXTS = [f"texto {n}" for n in range(5)]

Step = Callable[[httpx.Request], httpx.Response]


def _vector(text: str) -> list[float]:
    """Vetor que identifica o texto: "texto 3" → [3.0, 3.0, 3.0]."""
    return [float(text.split()[-1])] * DIMENSION


def _ok(request: httpx.Request) -> httpx.Response:
    inputs = json.loads(request.content)["input"]
    data = [
        {"object": "embedding", "index": i, "embedding": _vector(t)} for i, t in enumerate(inputs)
    ]
    # Ordem invertida de propósito: quem garante a ordem é o `index`, não a posição.
    return httpx.Response(200, json={"object": "list", "data": data[::-1]})


def _status(code: int, headers: dict[str, str] | None = None, body: Any = None) -> Step:
    return lambda request: httpx.Response(code, headers=headers, json=body)


def _timeout(request: httpx.Request) -> httpx.Response:
    raise httpx.ReadTimeout("lento demais", request=request)


def _raw(content: bytes) -> Step:
    return lambda request: httpx.Response(200, content=content)


class Fake:
    """Responde com `steps` na ordem (depois, sempre `_ok`) e guarda pedidos e esperas."""

    def __init__(self, *steps: Step, batch_size: int = 2) -> None:
        self.steps = list(steps)
        self.requests: list[httpx.Request] = []
        self.sleeps: list[float] = []
        self.client = OpenRouterEmbeddingClient(
            SecretStr(KEY),
            dimension=DIMENSION,
            batch_size=batch_size,
            transport=httpx.MockTransport(self._handle),
            sleep=self._sleep,
        )

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return (self.steps.pop(0) if self.steps else _ok)(request)

    async def _sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)

    def inputs(self) -> list[list[str]]:
        return [json.loads(request.content)["input"] for request in self.requests]


async def test_batches_the_input_and_keeps_the_input_order() -> None:
    fake = Fake(batch_size=2)
    assert await fake.client.embed_documents(TEXTS) == [_vector(t) for t in TEXTS]
    assert fake.inputs() == [TEXTS[0:2], TEXTS[2:4], TEXTS[4:5]]


async def test_every_request_carries_model_key_and_the_no_data_collection_policy() -> None:
    fake = Fake()
    await fake.client.embed_query("texto 1")

    (request,) = fake.requests
    assert request.url == "https://openrouter.ai/api/v1/embeddings"
    assert request.headers["authorization"] == f"Bearer {KEY}"
    assert json.loads(request.content) == {
        "model": "openai/text-embedding-3-small",
        "input": ["texto 1"],
        "encoding_format": "float",
        "provider": {"data_collection": "deny"},
    }


async def test_implements_the_embedding_client_contract() -> None:
    client = Fake().client
    assert isinstance(client, BaseEmbeddingClient)
    assert client.dimension == DIMENSION
    assert await client.embed_query("texto 7") == _vector("texto 7")


async def test_empty_input_makes_no_request_and_blank_text_is_refused() -> None:
    fake = Fake()
    assert await fake.client.embed_documents([]) == []
    with pytest.raises(EmbeddingServiceError, match="texto vazio"):
        await fake.client.embed_documents(["texto 1", "   "])
    assert fake.requests == []


async def test_429_waits_the_retry_after_seconds_and_tries_again() -> None:
    fake = Fake(_status(429, {"Retry-After": "2"}))
    assert await fake.client.embed_query("texto 1") == _vector("texto 1")
    assert fake.sleeps == [2.0]
    assert len(fake.requests) == 2


@pytest.mark.parametrize("gmt", [True, False], ids=["GMT", "sem-fuso"])
async def test_retry_after_also_accepts_an_http_date(gmt: bool) -> None:
    later = datetime.now(UTC) + timedelta(seconds=30)
    # "-0000" é data sem fuso: o parser devolve datetime ingênuo, lido como UTC.
    value = email.utils.format_datetime(later, usegmt=True)
    if not gmt:
        value = value.replace("GMT", "-0000")
    fake = Fake(_status(429, {"Retry-After": value}))
    await fake.client.embed_query("texto 1")
    (waited,) = fake.sleeps
    assert 25 <= waited <= 30


async def test_retry_after_beyond_the_cap_fails_without_waiting() -> None:
    fake = Fake(_status(429, {"Retry-After": "3600"}))
    with pytest.raises(EmbeddingServiceError, match="Retry-After de 3600s"):
        await fake.client.embed_query("texto 1")
    assert fake.sleeps == []
    assert len(fake.requests) == 1


async def test_unusable_retry_after_falls_back_to_exponential_backoff() -> None:
    fake = Fake(
        _status(503, {"Retry-After": "nan"}),
        _status(503, {"Retry-After": "depois"}),
        _status(503, {"Retry-After": "-5"}),
    )
    await fake.client.embed_query("texto 1")
    assert fake.sleeps == [0.5, 1.0, 0.0]


async def test_server_errors_and_timeouts_are_retried_then_give_up() -> None:
    fake = Fake(_status(502), _timeout, _status(500), _timeout)
    with pytest.raises(EmbeddingServiceError, match="ReadTimeout depois de 4 tentativas"):
        await fake.client.embed_query("texto 1")
    assert fake.sleeps == [0.5, 1.0, 2.0]
    assert len(fake.requests) == 4


@pytest.mark.parametrize("code", [400, 401, 402, 403, 413])
async def test_client_errors_fail_at_once_with_the_api_message(code: int) -> None:
    fake = Fake(
        _status(code, body={"error": {"code": code, "message": "Sem créditos suficientes"}})
    )
    with pytest.raises(EmbeddingServiceError, match=f"HTTP {code}: Sem créditos suficientes"):
        await fake.client.embed_query("texto 1")
    assert len(fake.requests) == 1
    assert fake.sleeps == []


async def test_an_error_body_that_is_not_json_falls_back_to_the_http_reason() -> None:
    fake = Fake(lambda request: httpx.Response(400, content=b"<html>proxy</html>"))
    with pytest.raises(EmbeddingServiceError, match="HTTP 400: Bad Request"):
        await fake.client.embed_query("texto 1")


def _embedding(index: int, values: list[Any]) -> dict[str, Any]:
    return {"index": index, "embedding": values}


@pytest.mark.parametrize(
    ("step", "message"),
    [
        (_status(200, body={"data": [_embedding(0, [1.0, 2.0])]}), "vetor com 2 dimensões"),
        (_status(200, body={"data": [_embedding(1, [1.0] * 3)]}), "índice faltando"),
        (_status(200, body={"data": [_embedding(0, [1.0] * 3)] * 2}), "2 vetores para 1"),
        (_raw(b'{"data": [{"index": 0, "embedding": [NaN, 1, 2]}]}'), "data.0.embedding.0"),
        (_raw(b'{"data": [{"index": 0, "embedding": [Infinity, 1, 2]}]}'), "finite"),
        (_raw(b"<html>Bad gateway</html>"), "resposta fora do formato"),
        (_status(200, body={"error": {"message": "falhou no meio"}}), "data"),
    ],
    ids=["dimensao", "indice", "repetido", "nan", "infinito", "nao-json", "sem-data"],
)
async def test_malformed_responses_are_refused(step: Step, message: str) -> None:
    with pytest.raises(EmbeddingServiceError, match=message):
        await Fake(step).client.embed_query("texto 1")


async def test_the_api_key_stays_out_of_repr_errors_and_logs() -> None:
    stream = io.StringIO()
    configure_logging("DEBUG", stream=stream)
    try:
        fake = Fake(_status(503), _status(401, body={"error": {"message": "chave inválida"}}))
        with pytest.raises(EmbeddingServiceError) as caught:
            await fake.client.embed_query("texto 1")
    finally:
        shutdown_logging()  # esvazia a fila antes de ler

    logs = stream.getvalue()
    assert "embeddings.retry" in logs  # o log existiu; só não leva a chave
    for place in (logs, repr(fake.client), str(caught.value)):
        assert KEY not in place


async def test_async_with_closes_the_http_client() -> None:
    fake = Fake()
    async with fake.client as client:
        await client.embed_query("texto 1")
    with pytest.raises(RuntimeError):
        await fake.client.embed_query("texto 2")


@pytest.mark.parametrize(
    ("option", "value"), [("batch_size", 0), ("batch_size", 2049), ("dimension", 0)]
)
def test_invalid_configuration_is_rejected(option: str, value: int) -> None:
    options: dict[str, Any] = {option: value}
    with pytest.raises(ValueError, match=option):
        OpenRouterEmbeddingClient(SecretStr(KEY), **options)


async def test_from_settings_uses_the_configured_key_url_and_timeout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)  # nenhum .env do repositório
    monkeypatch.setenv("DATABASE_URL", "postgresql://brain:x@db:5432/brain_router")
    monkeypatch.setenv("OPENROUTER_API_KEY", KEY)
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://proxy.interno/openrouter/v1")
    monkeypatch.setenv("OPENROUTER_TIMEOUT", "7.5")
    seen: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _ok(request)

    client = OpenRouterEmbeddingClient.from_settings(
        Settings(), dimension=DIMENSION, transport=httpx.MockTransport(handle)
    )
    await client.embed_query("texto 1")

    (request,) = seen
    assert request.url == "https://proxy.interno/openrouter/v1/embeddings"
    assert request.headers["authorization"] == f"Bearer {KEY}"
    assert request.extensions["timeout"] == {"connect": 7.5, "read": 7.5, "write": 7.5, "pool": 7.5}

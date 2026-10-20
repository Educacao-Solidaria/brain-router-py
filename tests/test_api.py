"""API HTTP v1 (FastAPI) com handlers falsos, sem rede nem banco (contrato em docs/api-v1.md)."""

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from starlette.types import Message

from app.api import create_app
from app.api.errors import BodyLimitMiddleware
from app.errors.exceptions import EmbeddingServiceError
from app.schemas.domain import Chunk, SearchResult
from app.schemas.mcp import (
    HybridSearchInput,
    HybridSearchOutput,
    IndexDocumentInput,
    IndexDocumentOutput,
)

SEARCH = "/api/v1/search"
INDEX = "/api/v1/index"
JSON = {"content-type": "application/json"}
DOCUMENT = {"title": "Regimento", "content": "Matrícula em março.", "source_uri": "file://r.md"}


def client(app: FastAPI, *, raise_app_exceptions: bool = True) -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=raise_app_exceptions)
    return httpx.AsyncClient(transport=transport, base_url="http://api")


class Backend:
    """Handlers falsos que guardam o que receberam (e levantam `error`, se houver)."""

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.searches: list[HybridSearchInput] = []
        self.indexed: list[IndexDocumentInput] = []

    async def search(self, body: HybridSearchInput) -> HybridSearchOutput:
        self.searches.append(body)
        if self.error is not None:
            raise self.error
        chunk = Chunk(id="doc-1:0", document_id="doc-1", content="Matrícula em março.")
        result = SearchResult(chunk=chunk, dense_score=0.9, rrf_score=0.0325, rank=1)
        return HybridSearchOutput(
            query=body.query, results=[result], total_found=1, execution_time_ms=1.5
        )

    async def index(self, body: IndexDocumentInput) -> IndexDocumentOutput:
        self.indexed.append(body)
        return IndexDocumentOutput(document_id="doc-9", chunks_created=3, success=True, message="")

    def app(self, **options: Any) -> FastAPI:
        return create_app(search=self.search, index=self.index, **options)


def error_of(response: httpx.Response) -> dict[str, str]:
    """O envelope `{"error": {"code", "message"}}`, sem nenhuma outra chave."""
    body = response.json()
    assert list(body) == ["error"] and sorted(body["error"]) == ["code", "message"]
    error: dict[str, str] = body["error"]
    return error


async def test_search_passes_the_parsed_input_and_returns_the_output() -> None:
    backend = Backend()
    async with client(backend.app()) as http:
        response = await http.post(SEARCH, json={"query": "matrícula", "top_k": 3})

    assert response.status_code == 200
    body = response.json()
    assert (body["query"], body["total_found"], body["execution_time_ms"]) == ("matrícula", 1, 1.5)
    (result,) = body["results"]
    assert (result["chunk"]["id"], result["chunk"]["content"]) == ("doc-1:0", "Matrícula em março.")
    scores = [result[k] for k in ("dense_score", "sparse_score", "rrf_score", "rank")]
    assert scores == [0.9, None, 0.0325, 1]
    (received,) = backend.searches
    assert (received.query, received.top_k, received.alpha) == ("matrícula", 3, 0.5)
    assert (received.tenant_id, received.filter_metadata) == (None, None)


async def test_index_answers_201_with_the_schema_defaults_filled() -> None:
    backend = Backend()
    async with client(backend.app()) as http:
        response = await http.post(INDEX, json=DOCUMENT)

    assert response.status_code == 201
    body = response.json()
    assert (body["document_id"], body["chunks_created"], body["success"]) == ("doc-9", 3, True)
    (received,) = backend.indexed
    assert (received.tenant_id, received.chunk_size, received.chunk_overlap) == ("default", 500, 50)


@pytest.mark.parametrize(("path", "body"), [(SEARCH, {"query": "x"}), (INDEX, DOCUMENT)])
async def test_missing_backend_is_503_not_an_empty_result(path: str, body: dict[str, Any]) -> None:
    async with client(create_app()) as http:
        response = await http.post(path, json=body)

    assert response.status_code == 503
    assert error_of(response)["code"] == "backend_unavailable"


@pytest.mark.parametrize(
    ("body", "field"),
    [
        ({"query": ""}, "query"),
        ({"top_k": 3}, "query"),
        ({"query": "cpf 123", "top_k": 0}, "top_k"),
        ({"query": "cpf 123", "top_k": 51}, "top_k"),
        ({"query": "cpf 123", "top_k": "muitos"}, "top_k"),
        ({"query": "cpf 123", "alpha": 1.01}, "alpha"),
        ({"query": "cpf 123", "alpha": "NaN"}, "alpha"),
        ({"query": "cpf 123", "filter_metadata": []}, "filter_metadata"),
    ],
)
async def test_invalid_search_is_422_validation_error(body: dict[str, Any], field: str) -> None:
    backend = Backend()
    async with client(backend.app()) as http:
        response = await http.post(SEARCH, json=body)

    assert response.status_code == 422
    error = error_of(response)
    assert error["code"] == "validation_error"
    assert error["message"].startswith(f"{field}: ")
    assert "cpf" not in error["message"] and "muitos" not in error["message"]  # sem o valor
    assert backend.searches == []


def test_openapi_documents_the_envelope_instead_of_the_fastapi_422() -> None:
    responses = create_app().openapi()["paths"][SEARCH]["post"]["responses"]
    refs = {responses[k]["content"]["application/json"]["schema"]["$ref"] for k in ("4XX", "5XX")}
    assert "422" not in responses and refs == {"#/components/schemas/ErrorBody"}


async def test_validation_message_lists_at_most_five_errors() -> None:
    body = {"title": "", "content": "", "tenant_id": 7, "chunk_size": 1, "chunk_overlap": -1}
    async with client(Backend().app()) as http:
        response = await http.post(INDEX, json=body)  # 6 erros: os 5 acima e o source_uri

    parts = error_of(response)["message"].split("; ")
    assert len(parts) == 6 and parts[-1] == "(+1 erros)"


@pytest.mark.parametrize(
    ("raw", "headers", "status", "code", "text"),
    [
        (b'{"query": ', JSON, 422, "validation_error", "JSON inválido"),
        (b"[1, 2", JSON, 422, "validation_error", "JSON inválido"),
        (b'{"query": "x", "alpha": NaN}', JSON, 422, "validation_error", "alpha: "),  # json.loads
        (b"\xff", JSON, 400, "bad_request", "parsing the body"),  # UTF-8 inválido: FastAPI dá 400
        (b'{"query": "x"}', {}, 422, "validation_error", "corpo: "),  # sem Content-Type: não lê
    ],
)
async def test_malformed_body_uses_the_envelope(
    raw: bytes, headers: dict[str, str], status: int, code: str, text: str
) -> None:
    async with client(Backend().app()) as http:
        response = await http.post(SEARCH, content=raw, headers=headers)

    assert response.status_code == status
    error = error_of(response)
    assert error["code"] == code and text in error["message"]


@pytest.mark.parametrize("chunked", [False, True])
@pytest.mark.parametrize(("size", "status"), [(64, 200), (65, 413)])
async def test_body_limit_at_the_boundary(size: int, status: int, chunked: bool) -> None:
    raw = json.dumps({"query": "a" * (size - 12)}, separators=(",", ":")).encode()
    assert len(raw) == size

    async def pieces() -> AsyncIterator[bytes]:  # sem Content-Length: Transfer-Encoding chunked
        yield raw[:40]
        yield raw[40:]

    backend = Backend()
    async with client(backend.app(max_body_bytes=64)) as http:
        response = await http.post(SEARCH, content=pieces() if chunked else raw, headers=JSON)

    assert ("content-length" in response.request.headers) is not chunked
    assert response.status_code == status
    if status == 413:
        assert error_of(response)["code"] == "payload_too_large"
        assert backend.searches == []  # nada chega ao handler


async def test_declared_oversize_is_refused_without_reading_the_body() -> None:
    sent: list[Message] = []

    async def never(*args: Any) -> Any:
        raise AssertionError("nem o app nem o corpo deviam ser tocados")

    async def send(message: Message) -> None:
        sent.append(message)

    scope = {"type": "http", "headers": [(b"content-length", b"65")]}
    await BodyLimitMiddleware(never, max_bytes=64)(scope, never, send)

    assert sent[0]["status"] == 413
    assert json.loads(sent[1]["body"])["error"]["code"] == "payload_too_large"


def test_body_limit_must_be_positive() -> None:
    with pytest.raises(ValueError, match="max_body_bytes"):
        create_app(max_body_bytes=0)


async def test_unknown_route_and_wrong_method_use_the_envelope() -> None:
    async with client(Backend().app()) as http:
        missing = await http.get("/api/v1/nada")
        wrong = await http.get(SEARCH)

    assert (missing.status_code, error_of(missing)["code"]) == (404, "not_found")
    assert (wrong.status_code, error_of(wrong)["code"]) == (405, "method_not_allowed")
    assert wrong.headers["allow"] == "POST"


@pytest.mark.parametrize(
    ("error", "status", "code", "shown"),
    [
        (EmbeddingServiceError("HTTP 429 senha=hunter2"), 502, "embedding_service_error", True),
        (RuntimeError("HTTP 429 senha=hunter2"), 500, "internal_error", False),
    ],
)
async def test_handler_errors(error: Exception, status: int, code: str, shown: bool) -> None:
    # O erro de domínio leva status, código e mensagem; o inesperado não mostra o texto.
    async with client(Backend(error).app(), raise_app_exceptions=False) as http:
        response = await http.post(SEARCH, json={"query": "x"})

    assert (response.status_code, error_of(response)["code"]) == (status, code)
    assert ("hunter2" in error_of(response)["message"]) is shown

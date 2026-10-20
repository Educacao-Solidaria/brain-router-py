"""Ponta a ponta do serviço RAG (`pytest -m integration`): Postgres real e um uvicorn de verdade.

Ingestão pelo `POST /api/v1/index` (o `index_handler`, com o cliente da OpenRouter num
`MockTransport`) e pelo `ingest_directory` → busca pelo `/api/v1/search` e pelo SSE do
`/search/stream` → geração com citação. A busca no Postgres e a geração são do Luan (PRs 22, 24
e 30; 54, 62 e 64, em Todo): até lá, dois dublês daqui. O `PgTextSearch` é o `BaseTextSearch`
dele sobre o `content_tsv` da 0003. A geração manda as fontes numeradas ao harness do PR 69
(`StreamMock`, o SSE da OpenRouter) e cita pelo `metadata["source_uri"]` que a ingestão grava em
cada chunk (o contrato da API v1).
"""

import asyncio
import json
import re
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pytest
import uvicorn
from alembic import command
from alembic.config import Config
from pydantic import SecretStr
from sqlalchemy import text
from sqlmodel.ext.asyncio.session import AsyncSession
from support import events, fake_embedding

from app import ingest
from app.api import create_app
from app.chunking import count_words
from app.clients.openrouter import OpenRouterEmbeddingClient
from app.db import Database
from app.ingest import Options, ingest_directory
from app.schemas.domain import Chunk, SearchResult
from app.search.fallback import text_search_handler
from app.settings import get_settings
from app.storage.interfaces import BaseTextSearch
from app.testing.openrouter_stream import StreamMock, StreamScript

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[1]
MODEL = "openai/text-embedding-3-small"
DOCS = {  # source_uri → (título, texto)
    "e2e://guias/matricula.md": (
        "Matrícula",
        "A matrícula da escola abre em fevereiro. Leve o histórico escolar à secretaria.",
    ),
    "e2e://guias/biblioteca.md": (
        "Biblioteca",
        "A biblioteca empresta livros por duas semanas, com renovação pela internet.",
    ),
    "e2e://guias/transporte.md": ("Transporte", "O transporte escolar passa às seis e meia."),
}
SEARCH = text(
    "SELECT c.id, c.document_id, c.content, c.token_count, c.metadata, "
    "ts_rank(c.content_tsv, q) AS score FROM chunks AS c JOIN documents AS d "
    "ON d.id = c.document_id, websearch_to_tsquery('portuguese', :query) AS q "
    "WHERE d.collection_id = :collection AND c.content_tsv @@ q "
    "ORDER BY score DESC, c.id LIMIT :limit"
)


class PgTextSearch(BaseTextSearch):
    """Dublê do motor textual do Luan (PR 24): a config `portuguese` nos dois lados, como a 0003."""

    def __init__(self, db: Database) -> None:
        self.db = db

    async def search_text(
        self, collection_name: str, query: str, limit: int = 10
    ) -> list[SearchResult]:
        async def run(session: AsyncSession) -> list[SearchResult]:
            params = {"collection": collection_name, "query": query, "limit": limit}
            rows = await (await session.connection()).execute(SEARCH, params)
            return [
                SearchResult(
                    chunk=Chunk(
                        id=row.id,
                        document_id=row.document_id,
                        content=row.content,
                        token_count=row.token_count,
                        metadata=row.metadata,
                    ),
                    sparse_score=row.score,
                    rrf_score=row.score,
                    rank=rank,
                )
                for rank, row in enumerate(rows, 1)
            ]

        return await self.db.read(run)


@dataclass
class Service:
    url: str
    db: Database
    collection: str
    embedder: OpenRouterEmbeddingClient
    sent: list[dict[str, Any]]  # os pedidos à "OpenRouter"


@pytest.fixture(scope="module", autouse=True)
def _at_head() -> None:  # síncrono: o env.py do Alembic chama asyncio.run()
    command.upgrade(Config(str(ROOT / "alembic.ini")), "head")


@pytest.fixture
async def service(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[Service]:
    """A API v1 num uvicorn (porta livre) com a ingestão e a busca no Postgres, numa coleção só
    do teste; o cliente de embeddings é o de verdade, e a "OpenRouter", um `MockTransport`."""
    monkeypatch.setattr(ingest, "token_counter", lambda model: count_words)  # sem tiktoken no job
    sent: list[dict[str, Any]] = []

    def openrouter(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        sent.append(body)
        data = [
            {"index": n, "embedding": fake_embedding(t, 1536)} for n, t in enumerate(body["input"])
        ]
        return httpx.Response(200, json={"object": "list", "data": data})

    get_settings.cache_clear()
    db, collection = Database.from_settings(get_settings()), f"e2e-{uuid.uuid4().hex[:8]}"
    key, transport = SecretStr("sk-or-v1-e2e"), httpx.MockTransport(openrouter)
    embedder = OpenRouterEmbeddingClient(key, model=MODEL, transport=transport)
    app = create_app(
        index=ingest.index_handler(db, embedder, model=MODEL, collection_id=collection),
        search=text_search_handler(PgTextSearch(db), collection),
    )
    server = uvicorn.Server(uvicorn.Config(app, port=0, log_level="warning"))
    task = asyncio.create_task(server.serve())
    try:
        async with asyncio.timeout(10):
            while not server.started:
                await asyncio.sleep(0.01)
        port = server.servers[0].sockets[0].getsockname()[1]
        yield Service(f"http://127.0.0.1:{port}", db, collection, embedder, sent)
    finally:
        server.should_exit = True
        await task
        async with db.session() as session:
            delete = text("DELETE FROM documents WHERE collection_id = :collection")
            await (await session.connection()).execute(delete, {"collection": collection})
        await embedder.aclose()
        await db.dispose()


async def _search(client: httpx.AsyncClient, query: str) -> list[dict[str, Any]]:
    response = await client.post("/api/v1/search", json={"query": query})
    assert response.status_code == 200, response.text
    results: list[dict[str, Any]] = response.json()["results"]
    return results


async def _answer(mock: StreamMock, question: str, results: list[dict[str, Any]]) -> str:
    """Dublê da geração do Luan (PRs 54/62): as fontes numeradas no prompt e o SSE da
    OpenRouter lido pelo LF (o `aiter_lines` do httpx partiria também em U+2028)."""
    sources = "\n".join(f"[{n}] {r['chunk']['content']}" for n, r in enumerate(results, 1))
    messages = [{"role": "user", "content": f"Fontes:\n{sources}\n\nPergunta: {question}"}]
    payload = {"model": "openai/gpt-4o-mini", "messages": messages, "stream": True}
    parts, buffer = [], ""
    async with (
        httpx.AsyncClient(transport=mock.transport, base_url="https://openrouter.ai") as llm,
        llm.stream("POST", "/api/v1/chat/completions", json=payload) as response,
    ):
        async for piece in response.aiter_text():
            *lines, buffer = (buffer + piece).split("\n")
            for line in lines:
                if line.startswith("data: {"):
                    chunk = json.loads(line.removeprefix("data: "))
                    if "error" in chunk:  # erro no meio: não há resposta, nem parcial
                        raise RuntimeError(chunk["error"]["message"])
                    parts.append(chunk["choices"][0]["delta"].get("content", ""))
    return "".join(parts)


def _cited(answer: str, results: list[dict[str, Any]]) -> list[str]:
    """Dublê das citações do Luan (PR 64): `[n]` da resposta → `source_uri` da n-ésima fonte."""
    return [
        results[int(n) - 1]["chunk"]["metadata"]["source_uri"]
        for n in re.findall(r"\[(\d+)\]", answer)
    ]


async def test_uploads_are_searched_streamed_and_cited_by_source_uri(service: Service) -> None:
    async with httpx.AsyncClient(base_url=service.url) as client:
        for uri, (title, content) in DOCS.items():
            body = {"title": title, "content": content, "source_uri": uri}
            response = await client.post("/api/v1/index", json=body)
            assert response.status_code == 201 and response.json()["chunks_created"] == 1
        # Um pedido por documento, o texto do chunk e nenhum dado para provedor que retém.
        assert [request["input"] for request in service.sent] == [[c] for _, c in DOCS.values()]
        assert all(request["provider"] == {"data_collection": "deny"} for request in service.sent)

        question = "quando abre a matrícula"
        headers = {"X-Request-Id": "e2e-busca-1"}
        response = await client.post("/api/v1/search", json={"query": question}, headers=headers)
        assert response.headers["x-request-id"] == "e2e-busca-1"
        results = response.json()["results"]
        assert [r["chunk"]["metadata"]["source_uri"] for r in results] == [
            "e2e://guias/matricula.md"
        ]
        assert results[0]["rank"] == 1 and results[0]["sparse_score"] > 0

        streamed = await client.post("/api/v1/search/stream", json={"query": question})
        assert streamed.headers["content-type"].startswith("text/event-stream")
        *chunks, (done, summary) = events(streamed.content)
        assert chunks == [("result", r) for r in results]  # os mesmos, na ordem do rank
        assert (done, summary["total_found"]) == ("done", 1)

    tokens = ["A matrícula abre em fevereiro ", "[1]."]
    mock = StreamMock(StreamScript(tokens=tokens))
    answer = await _answer(mock, question, results)
    assert answer == "A matrícula abre em fevereiro [1]."
    assert _cited(answer, results) == ["e2e://guias/matricula.md"]
    sent = json.loads(mock.requests[0].content)
    assert sent["stream"] is True and DOCS["e2e://guias/matricula.md"][1] in str(sent["messages"])
    with pytest.raises(RuntimeError, match="Provider disconnected"):
        await _answer(StreamMock(StreamScript(tokens=tokens, fail_after=1)), question, results)


async def test_reupload_and_batch_ingest_change_what_search_finds(
    service: Service, tmp_path: Path
) -> None:
    uri = "e2e://guias/calendario.md"
    async with httpx.AsyncClient(base_url=service.url) as client:
        old, new = "As provas começam em junho.", "As provas começam em julho."
        for content in (old, new):
            body = {"title": "Calendário", "content": content, "source_uri": uri}
            assert (await client.post("/api/v1/index", json=body)).status_code == 201
        assert await _search(client, "junho") == []  # a versão velha saiu, com os chunks
        assert [r["chunk"]["content"] for r in await _search(client, "julho")] == [new]

        (tmp_path / "bolsas.md").write_text("A bolsa de estudo cobre o material didático.")
        options = Options(collection_id=service.collection)
        report = await ingest_directory(
            tmp_path, db=service.db, embedder=service.embedder, model=MODEL, options=options
        )
        assert [item.path for item in report.indexed] == ["bolsas.md"]
        found = await _search(client, "bolsa de estudo")
        expected = (tmp_path / "bolsas.md").resolve().as_uri()
        assert [r["chunk"]["metadata"]["source_uri"] for r in found] == [expected]


async def test_search_refuses_tenant_filters_and_uploads_get_the_error_envelope(
    service: Service,
) -> None:
    async with httpx.AsyncClient(base_url=service.url) as client:
        # A busca léxica não filtra por tenant (a interface do Luan não tem o filtro): recusa
        # em vez de devolver documento de outro tenant.
        response = await client.post("/api/v1/search", json={"query": "x", "tenant_id": "outro"})
        assert response.status_code == 502
        assert response.json()["error"]["code"] == "embedding_service_error"
        for body, code in [
            ({"title": "x", "content": "y", "source_uri": " "}, "validation_error"),  # o handler
            ({"title": "x", "content": "y"}, "validation_error"),  # o schema (sem source_uri)
        ]:
            response = await client.post("/api/v1/index", json=body)
            assert (response.status_code, response.json()["error"]["code"]) == (422, code)

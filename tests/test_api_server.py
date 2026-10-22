"""A fábrica do container com banco (`app.api.server`): lê `Settings` do ambiente, põe o Postgres
no `/healthz` e fecha o pool ao desligar. O 200 com o banco de pé é o teste `integration`."""

import socket
from collections.abc import Iterator

import httpx
import pytest
from fastapi import FastAPI

from app.api import server
from app.db import Database
from app.settings import get_settings

Calls = list[tuple[object, ...]]


@pytest.fixture(autouse=True)
def logging_calls(monkeypatch: pytest.MonkeyPatch) -> Iterator[Calls]:
    # O log de verdade trocaria o handler da raiz, e o shutdown pararia a thread de escrita que
    # outro teste ligou: os dois só ficam registrados.
    calls: Calls = []
    monkeypatch.setattr(
        server, "configure_logging", lambda level, *, json: calls.append(("configure", level, json))
    )
    monkeypatch.setattr(server, "shutdown_logging", lambda: calls.append(("shutdown",)))
    get_settings.cache_clear()
    yield calls
    get_settings.cache_clear()  # o Settings deste teste não fica no cache para os seguintes


def _closed_port() -> int:
    with socket.socket() as sock:  # livre agora e fechada depois: conexão recusada na hora
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


async def _healthz(app: FastAPI) -> httpx.Response:
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://rag") as http:
            return await http.get("/healthz")


async def test_factory_reads_settings_reports_the_database_down_and_closes_the_pool(
    monkeypatch: pytest.MonkeyPatch, logging_calls: Calls
) -> None:
    monkeypatch.setenv("DATABASE_URL", f"postgresql://brain:x@127.0.0.1:{_closed_port()}/brain")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-teste")
    monkeypatch.setenv("LOG_LEVEL", "warning")
    monkeypatch.setenv("LOG_JSON", "false")
    disposed: list[Database] = []
    dispose = Database.dispose

    async def spy(db: Database) -> None:
        disposed.append(db)
        await dispose(db)

    monkeypatch.setattr(Database, "dispose", spy)

    response = await _healthz(server.create_server())  # como o uvicorn chama: sem argumentos

    assert response.status_code == 503
    assert response.json() == {"status": "unavailable", "checks": {"postgres": "down"}}
    assert logging_calls == [("configure", "WARNING", False), ("shutdown",)]
    assert len(disposed) == 1  # o lifespan fechou o pool


def test_factory_without_database_url_does_not_start(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir("/")  # nenhum .env do repositório é lido
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-teste")

    with pytest.raises(ValueError, match="database_url"):
        server.create_server()


@pytest.mark.integration
async def test_healthz_is_ok_with_the_database_up() -> None:
    response = await _healthz(server.create_server())  # o banco de DATABASE_URL

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "checks": {"postgres": "ok"}}

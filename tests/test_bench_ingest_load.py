"""Peças do teste de carga sem o locust e sem banco; com Postgres, test_integration_ingest.py."""

import json

import pytest

from app.bench import ingest_server
from app.bench.ingest_load import Rejected, Totals, accepted, document, payload
from app.schemas.mcp import IndexDocumentInput
from app.settings import get_settings


def test_document_and_payload_are_deterministic_and_unique_per_upload() -> None:
    assert document(7, 30) == document(7, 30) != document(8, 30)
    assert len(document(7, 30).split()) == 36  # 3 frases de 12: arredonda para cima
    paragraphs = document(3, 12 * 11).split("\n\n")
    assert [paragraph.count(".") for paragraph in paragraphs] == [5, 5, 1]
    with pytest.raises(ValueError, match="words deve ser >= 1"):
        document(1, 0)
    IndexDocumentInput.model_validate(payload("abc", 1, 50))  # o corpo passa no schema
    uris = {payload(run, n, 50)["source_uri"] for run, n in [("abc", 1), ("abc", 2), ("x", 1)]}
    assert len(uris) == 3 and "loadtest://abc/1" in uris


def test_accepted_counts_new_chunks_and_rejects_the_rest() -> None:
    assert str(Totals(2, 7)) == "locust: documentos=2 chunks=7"  # o formato do ingest_server
    ok = {"document_id": "d", "chunks_created": 3, "success": True, "message": "indexado"}
    assert accepted(201, json.dumps(ok)) == 3
    for status, body, reason in [
        (0, "", "HTTP 0"),  # sem conexão, no FastHttpUser
        (503, '{"error": {"code": "backend_unavailable", "message": "x"}}', "HTTP 503"),
        (201, "{}", "201 com corpo fora do contrato"),
        (201, json.dumps(ok | {"chunks_created": 0}), "201 sem chunk novo"),
        (201, json.dumps(ok | {"success": False}), "201 sem chunk novo"),
    ]:
        with pytest.raises(Rejected) as info:
            accepted(status, body)
        assert str(info.value) == reason


def test_server_refuses_to_run_against_production(monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in [("APP_ENV", "production"), ("DATABASE_URL", "postgresql://u:p@db/b")]:
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-x")
    get_settings.cache_clear()
    try:
        with pytest.raises(SystemExit, match="APP_ENV=production"):
            ingest_server.main([])
    finally:
        get_settings.cache_clear()

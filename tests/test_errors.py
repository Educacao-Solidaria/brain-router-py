import pytest
from app.errors.exceptions import (
    BrainRouterError,
    DocumentNotFoundError,
    CollectionNotFoundError,
    EmbeddingServiceError,
    DatabaseQueryError,
    RerankServiceError,
)


def test_base_brain_router_error() -> None:
    err = BrainRouterError(
        message="Algo falhou",
        code="CUSTOM_ERR",
        http_status=400,
        details={"field": "query"},
    )
    assert str(err) == "Algo falhou"
    assert err.code == "CUSTOM_ERR"
    assert err.http_status == 400
    assert err.to_dict() == {
        "error": {
            "code": "CUSTOM_ERR",
            "message": "Algo falhou",
            "details": {"field": "query"},
        }
    }


def test_document_not_found_error() -> None:
    err = DocumentNotFoundError("doc-123")
    assert err.code == "DOCUMENT_NOT_FOUND"
    assert err.http_status == 404
    assert err.details == {"document_id": "doc-123"}
    assert "doc-123" in err.message


def test_collection_not_found_error() -> None:
    err = CollectionNotFoundError("kb-finance")
    assert err.code == "COLLECTION_NOT_FOUND"
    assert err.http_status == 404
    assert err.details == {"collection_id": "kb-finance"}
    assert "kb-finance" in err.message


def test_embedding_service_error() -> None:
    err = EmbeddingServiceError("Timeout ao conectar", provider="openrouter")
    assert err.code == "EMBEDDING_SERVICE_ERROR"
    assert err.http_status == 502
    assert err.details == {"provider": "openrouter"}
    assert "Timeout" in err.message


def test_database_query_error() -> None:
    err = DatabaseQueryError("Connection refused")
    assert err.code == "DATABASE_QUERY_ERROR"
    assert err.http_status == 500
    assert "Connection refused" in err.message


def test_rerank_service_error() -> None:
    err = RerankServiceError("Modelo sobrecarregado")
    assert err.code == "RERANK_SERVICE_ERROR"
    assert err.http_status == 502
    assert "Modelo sobrecarregado" in err.message

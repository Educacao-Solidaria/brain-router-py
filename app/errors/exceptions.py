from typing import Any


class BrainRouterError(Exception):
    """Exceção base de domínio para o serviço Brain Router."""

    def __init__(
        self,
        message: str,
        code: str = "INTERNAL_ERROR",
        http_status: int = 500,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.http_status = http_status
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "details": self.details,
            }
        }


class DocumentNotFoundError(BrainRouterError):
    def __init__(self, document_id: str) -> None:
        super().__init__(
            message=f"Documento '{document_id}' nao encontrado",
            code="DOCUMENT_NOT_FOUND",
            http_status=404,
            details={"document_id": document_id},
        )


class CollectionNotFoundError(BrainRouterError):
    def __init__(self, collection_id: str) -> None:
        super().__init__(
            message=f"Colecao '{collection_id}' nao encontrada",
            code="COLLECTION_NOT_FOUND",
            http_status=404,
            details={"collection_id": collection_id},
        )


class EmbeddingServiceError(BrainRouterError):
    def __init__(self, message: str, provider: str = "openrouter") -> None:
        super().__init__(
            message=f"Falha na geracao de embeddings ({provider}): {message}",
            code="EMBEDDING_SERVICE_ERROR",
            http_status=502,
            details={"provider": provider},
        )


class DatabaseQueryError(BrainRouterError):
    def __init__(self, message: str) -> None:
        super().__init__(
            message=f"Falha na execucao de consulta no banco de dados: {message}",
            code="DATABASE_QUERY_ERROR",
            http_status=500,
        )


class RerankServiceError(BrainRouterError):
    def __init__(self, message: str) -> None:
        super().__init__(
            message=f"Falha na reordenacao de chunks (reranker): {message}",
            code="RERANK_SERVICE_ERROR",
            http_status=502,
        )

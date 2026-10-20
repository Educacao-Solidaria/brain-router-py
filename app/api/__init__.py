"""API HTTP v1 do RAG (FastAPI), alternativa ao MCP. Contrato em `docs/api-v1.md`."""

from app.api.app import DEFAULT_MAX_BODY_BYTES, IndexHandler, SearchHandler, create_app

__all__ = ["DEFAULT_MAX_BODY_BYTES", "IndexHandler", "SearchHandler", "create_app"]

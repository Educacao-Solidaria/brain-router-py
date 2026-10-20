"""API HTTP v1 do RAG (FastAPI), alternativa ao MCP. Contrato em `docs/api-v1.md`."""

from app.api.app import DEFAULT_MAX_BODY_BYTES, IndexHandler, SearchHandler, create_app
from app.api.middleware import RateLimit

__all__ = ["DEFAULT_MAX_BODY_BYTES", "IndexHandler", "RateLimit", "SearchHandler", "create_app"]

"""Cria as tabelas documents e chunks, com o embedding em vector(1536).

Revision ID: 0002_documents_and_chunks
Revises: 0001_create_extensions
Create Date: 2026-10-19

As colunas têm os nomes e os tipos de `app.schemas.domain.Document` e `Chunk`, e os
CHECKs repetem as restrições dos modelos (`min_length=1`, `ge=0`), para que um INSERT
que não passe pelo pydantic também não grave lixo. Fora daqui, de propósito:
- `Document.chunks_count` não vira coluna: é `count(*)` em chunks por `document_id`.
  Um contador gravado à parte diverge do real na primeira falha parcial.
- O índice HNSW (cosine) do embedding fica com o PR 22 (cliente pgvector).
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0002_documents_and_chunks"
down_revision: str | None = "0001_create_extensions"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # collection_id é a partição do BaseVectorStore (`collection_name`), com o nome de
    # ChunkFilter.collection_id. Fica no documento: os chunks herdam pela FK, e
    # `DELETE FROM documents WHERE collection_id = ...` limpa a coleção inteira.
    op.execute(
        """
        CREATE TABLE documents (
            id text PRIMARY KEY,
            collection_id text NOT NULL DEFAULT 'default',
            title text NOT NULL CHECK (length(title) > 0),
            source_uri text NOT NULL,
            tenant_id text NOT NULL DEFAULT 'default',
            created_at timestamptz NOT NULL DEFAULT now()
        )
        """
    )
    # ON DELETE CASCADE, de propósito: o chunk é parte do documento. Apagar o documento
    # (reindexação, delete_document do PR 40) leva os chunks na mesma instrução, sem
    # vetor órfão respondendo busca; com RESTRICT, todo chamador teria de apagar os
    # chunks antes. O id do documento não muda, então ON UPDATE fica no padrão (falha).
    # embedding NULL = chunk gravado antes de ganhar vetor (Chunk.embedding é opcional).
    # 1536 = dimensão do openai/text-embedding-3-small, o modelo padrão de Collection.
    op.execute(
        """
        CREATE TABLE chunks (
            id text PRIMARY KEY,
            document_id text NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
            content text NOT NULL CHECK (length(content) > 0),
            embedding vector(1536),
            token_count integer NOT NULL DEFAULT 0 CHECK (token_count >= 0),
            metadata jsonb NOT NULL DEFAULT '{}' CHECK (jsonb_typeof(metadata) = 'object')
        )
        """
    )
    # A FK não indexa o lado filho: sem isto, cada documento apagado varre chunks
    # inteira para achar os filhos do CASCADE (e o count(*) por documento também).
    op.execute("CREATE INDEX ix_chunks_document_id ON chunks (document_id)")


def downgrade() -> None:
    # Filho antes do pai. Sem CASCADE: se algo de fora ainda depende destas tabelas
    # (uma view, uma FK nova), o DROP falha em vez de levar junto em silêncio.
    op.execute("DROP TABLE chunks")
    op.execute("DROP TABLE documents")

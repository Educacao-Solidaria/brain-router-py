"""Índices das consultas de relatório e de manutenção (`app.reports`, `check-integrity`, ingest).

Revision ID: 0004_report_indexes
Revises: 0003_fulltext_portuguese
Create Date: 2026-10-20

Medido com 100 mil documentos e chunks (vetores de 1536, 1% sem vetor), antes → depois:
- `ix_chunks_pending_embedding` (parcial, `embedding IS NULL`): só a fila de embedding. O
  `chunks_sem_embedding` cai de 5,1 para 0,1 ms; a fila por coleção, de 10,7 para 6,0 ms (o
  resto é o join com `documents`).
- `ix_chunks_zero_embedding` (parcial, `vector_norm(embedding) = 0`): quase sempre vazio. O
  `embeddings_zerados` deixa de calcular a norma de cada vetor da tabela: de 426 para 0,08 ms.
  O custo vai para cada INSERT, uma norma por linha (`vector_norm` é IMMUTABLE).
- `ix_documents_collection` (`collection_id, tenant_id, source_uri`): a busca por `source_uri`
  que a ingestão faz a cada documento, de 5,9 para 0,05 ms (e o teste de carga do PR 71 deixa
  de perder 25% de vazão com a tabela cheia). Coleção na frente: é a partição do vector store.
- Documentos por coleção não ganha índice: a contagem lê a tabela toda (13 ms nos dois casos).

CONCURRENTLY, decidido: um CREATE INDEX comum trava INSERT/UPDATE/DELETE da tabela enquanto o
índice é construído, e a ingestão parava junto. O CONCURRENTLY não roda em transação, então
cada comando vai num `autocommit_block` (o Alembic faz COMMIT antes e BEGIN depois; no
`--sql`, as mesmas linhas). Preço: sem rollback no meio, e uma falha deixa índice INVALID. Por
isso o DROP ... IF EXISTS antes de cada CREATE: rodar a migração de novo refaz o que ficou.
Ele também espera toda transação do banco mais antiga que ele: uma `idle in transaction`
esquecida trava a migração (o `pg_stat_activity` mostra qual).
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0004_report_indexes"
down_revision: str | None = "0003_fulltext_portuguese"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

INDEXES = {  # nome → tabela e definição
    "ix_chunks_pending_embedding": "chunks (document_id, id) WHERE embedding IS NULL",
    "ix_chunks_zero_embedding": "chunks (id) WHERE vector_norm(embedding) = 0",
    "ix_documents_collection": "documents (collection_id, tenant_id, source_uri)",
}


def upgrade() -> None:
    with op.get_context().autocommit_block():
        for name, definition in INDEXES.items():
            op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {name}")  # o INVALID de uma falha
            op.execute(f"CREATE INDEX CONCURRENTLY {name} ON {definition}")


def downgrade() -> None:
    with op.get_context().autocommit_block():
        for name in reversed(INDEXES):
            op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {name}")

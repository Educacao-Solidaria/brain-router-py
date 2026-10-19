"""Busca textual em português: colunas tsvector geradas e índices GIN.

Revision ID: 0003_fulltext_portuguese
Revises: 0002_documents_and_chunks
Create Date: 2026-10-19

`chunks.content_tsv` e `documents.title_tsv` são `GENERATED ... STORED` com a config
`portuguese` (stopwords + stemmer Snowball): o Postgres as recalcula a cada INSERT/UPDATE,
então nunca ficam velhas, e o `ts_rank` lê o vetor pronto em vez de refazê-lo por linha.
`to_tsvector(regconfig, text)` com a config fixa é IMMUTABLE, o que a coluna gerada exige;
a forma de um argumento depende de `default_text_search_config` e seria recusada.

Consultas precisam usar a mesma config (`websearch_to_tsquery('portuguese', ...)`), senão
o stemming dos dois lados não bate e o índice não serve.

Sem `unaccent`, de propósito. Ele é STABLE (não entra em coluna gerada), e o jeito certo de
usá-lo, como dicionário de filtro antes do `portuguese_stem`, piora o stemming: o Snowball
português reconhece os sufixos com acento, e sem eles "educação"/"educações" deixam de cair
no mesmo radical ("educ" vira "educaca"/"educaco"), além de stopwords acentuadas ("não",
"você") passarem a ser indexadas. Busca sem acento fica para quando houver demanda (por
exemplo, uma segunda coluna `simple` + unaccent mantida por trigger).
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0003_fulltext_portuguese"
down_revision: str | None = "0002_documents_and_chunks"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE chunks ADD COLUMN content_tsv tsvector "
        "GENERATED ALWAYS AS (to_tsvector('portuguese', content)) STORED"
    )
    # O título entra à parte para o PR 24 poder pesá-lo diferente do conteúdo (setweight).
    op.execute(
        "ALTER TABLE documents ADD COLUMN title_tsv tsvector "
        "GENERATED ALWAYS AS (to_tsvector('portuguese', title)) STORED"
    )
    op.execute("CREATE INDEX ix_chunks_content_tsv ON chunks USING gin (content_tsv)")
    op.execute("CREATE INDEX ix_documents_title_tsv ON documents USING gin (title_tsv)")


def downgrade() -> None:
    op.execute("DROP INDEX ix_documents_title_tsv")
    op.execute("DROP INDEX ix_chunks_content_tsv")
    op.execute("ALTER TABLE documents DROP COLUMN title_tsv")
    op.execute("ALTER TABLE chunks DROP COLUMN content_tsv")

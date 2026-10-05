"""Cria as extensões vector (pgvector) e pg_trgm.

Revision ID: 0001_create_extensions
Revises:
Create Date: 2026-10-05
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0001_create_extensions"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Ordem de criação; o downgrade remove na ordem inversa.
EXTENSIONS = ("vector", "pg_trgm")


def upgrade() -> None:
    for name in EXTENSIONS:
        op.execute(f"CREATE EXTENSION IF NOT EXISTS {name}")


def downgrade() -> None:
    # Sem CASCADE: se uma coluna/índice ainda usa a extensão, o DROP falha em vez de
    # apagar dado junto.
    for name in reversed(EXTENSIONS):
        op.execute(f"DROP EXTENSION IF EXISTS {name}")

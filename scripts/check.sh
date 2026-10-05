#!/usr/bin/env bash
# Roda localmente a mesma sequência do CI (.github/workflows/ci.yml).
# Uso: scripts/check.sh   (a partir de qualquer pasta do repositório)
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

uv sync --locked --extra dev
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest

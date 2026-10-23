# Imagem do RAG: o serviço `rag` do compose do ecossistema (contrato da Fase 5, §2), com a raiz do
# repositório como contexto. O estágio `builder` monta o .venv com o uv a partir do uv.lock; a
# imagem final leva só esse .venv (sem o uv e sem o código-fonte) e roda como 10001:10001 na 8000.
#   docker build -t educacao-solidaria/brain-router:dev .
#   docker compose up -d --wait   # com o Postgres (DATABASE_URL e OPENROUTER_API_KEY)

# Bases fixadas pelo digest do índice multi-arquitetura; a tag de origem fica no comentário.
# python:3.12.15-slim-trixie (a mesma nos dois estágios: o .venv aponta para o Python dela)
FROM python:3.12.15-slim-trixie@sha256:a0f57064cbdcd6285fa45f51237ab0b35ea8ebaed06a1f5973e89e19dfb86704 AS base
# ghcr.io/astral-sh/uv:0.12.24
FROM ghcr.io/astral-sh/uv:0.12.24@sha256:3af4716e991d6956a41e573eab705d0ee08500cd829ed30293eb8472f372c65a AS uv

FROM base AS builder
COPY --from=uv /uv /usr/local/bin/uv
# Bytecode compilado aqui (a imagem final não precisa escrever __pycache__), cópia em vez de
# hardlink (o cache montado é outro sistema de arquivos) e só o Python da imagem base.
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
# 1) Só as dependências do uv.lock (`--locked` recusa lock desatualizado em relação ao
#    pyproject.toml; sem extras, nada de dev nem de carga): a camada só refaz quando um dos dois
#    muda, e o cache do uv montado (fora da imagem) poupa o download num rebuild.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-dev --no-install-project
# 2) O pacote `app` como wheel dentro do .venv (sem editable): o .venv fica autossuficiente.
COPY pyproject.toml uv.lock README.md ./
COPY app ./app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable

FROM base AS runtime
RUN groupadd --system --gid 10001 brain \
    && useradd --system --no-log-init --uid 10001 --gid 10001 \
        --home-dir /nonexistent --no-create-home --shell /usr/sbin/nologin brain
# Arquivos de root: o processo lê e executa o .venv, mas não consegue alterá-lo.
COPY --from=builder /app/.venv /app/.venv
# APP_ENV=production: os benchmarks e o quality gate (app/bench, app/evals/gate.py) recusam rodar.
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1 APP_ENV=production
WORKDIR /app
USER 10001:10001
EXPOSE 8000
# Sem curl na imagem: o próprio Python sonda o /healthz (503 ou conexão recusada = exceção = 1). O
# start-period cobre a partida, que espera o banco por até 60 s antes de migrar.
HEALTHCHECK --interval=10s --timeout=3s --start-period=60s --retries=3 \
    CMD ["python", "-c", "import urllib.request as r; r.urlopen('http://127.0.0.1:8000/healthz', timeout=2)"]
# A partida (`brain-router migrate`, app/startup.py) espera o Postgres, aplica as migrações sob a
# trava entre réplicas e vira o uvicorn com exec: ele herda o PID 1 e o SIGTERM o encerra com 0. A
# fábrica `create_server` lê o Settings e põe o banco no /healthz.
CMD ["brain-router", "migrate", "--", \
     "uvicorn", "app.api.server:create_server", "--factory", "--host", "0.0.0.0", "--port", "8000"]

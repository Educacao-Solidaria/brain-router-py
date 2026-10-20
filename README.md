# Brain Router Python

> RAG híbrido dual (código e documentação) com pgvector, tsvector e Reciprocal Rank Fusion (RRF), integrado ao OpenRouter via servidor MCP nativo em Python.

## Visão Geral

O **Brain Router Python** é o núcleo de inteligência e recuperação de contexto do ecossistema, derivado da arquitetura provada no Kimi-Hybrid. Ele unifica busca vetorial densa com busca lexical de texto completo para responder com precisão sobre regras de negócio e bases de código.

## Como Funciona

```
[ Pergunta do Usuário ] ──▶ [ Brain Router Python ]
                                   │
                   ┌───────────────┴───────────────┐
                   ▼                               ▼
       [ pgvector (HNSW) ]               [ tsvector (Full-text) ]
       Busca Semântica Densa             Busca Lexical / Exata
                   │                               │
                   └───────────────┬───────────────┘
                                   ▼
                   [ Reciprocal Rank Fusion (RRF) ]
                                   ▼
                    [ Cross-Encoder / Reranker ]
                                   ▼
                 [ Prompt Aumentado ──▶ OpenRouter ]
```

1. **Dual-Index:**
   - **Índice de Conhecimento:** Responde o *porquê* (ADRs, runbooks, regras de negócio, documentação).
   - **Índice de Código:** Responde o *onde* (funções, módulos, dependências e linhas exatas).
2. **RRF & Reranking:** Fusão recíproca de posições para evitar pontos cegos de busca puramente vetorial.
3. **Servidor MCP:** Expõe ferramentas para busca semântica, chunking de repositórios e avaliação contínua de alucinações (evals).

## Arquitetura & Módulos

O desenvolvimento é guiado pelo roadmap de **100 PRs** no [Plane da in100tiva (RAGPY)](https://plane.in100tiva.com/in100tiva/):

- **Fase 1:** Fundação, CI/CD e Contratos (PRs 01-20)
- **Fase 2:** Core Engine e Protocolo MCP (PRs 21-50)
- **Fase 3:** Adaptadores, Conectores e Streaming (PRs 51-75)
- **Fase 4:** Observabilidade OTel, Evals e Benchmarks (PRs 76-90)
- **Fase 5:** Release v1.0, Docker e Documentação (PRs 91-100+)

## Desenvolvimento

Requisitos: [uv](https://docs.astral.sh/uv/) (Python 3.12 é instalado por ele, se faltar).

```bash
uv sync --extra dev              # cria .venv e instala dependências do uv.lock
uv run pre-commit install        # liga os hooks no git commit (uma vez por clone)
scripts/check.sh                 # mesma sequência do CI, na mesma ordem
```

`scripts/check.sh` roda, em ordem: `uv sync --locked`, `ruff check`,
`ruff format --check`, `mypy` (modo strict) e `pytest` (cobertura mínima de
90%). O CI (`.github/workflows/ci.yml`) roda exatamente os mesmos comandos — um
teste (`tests/test_tooling.py`) falha se os dois divergirem.

| Comando | O que faz |
|---|---|
| `uv run brain-router --version` | CLI do pacote (entry point `app.__main__:main`) |
| `uv run brain-router check-integrity [--fix]` | audita documentos e vetores do banco de `DATABASE_URL`; só leitura sem `--fix` (`app/integrity.py`) |
| `uv run brain-router ingest <pasta> [--dry-run]` | indexa os `.md`/`.txt`/`.json` da pasta, idempotente por `source_uri`; `--dry-run` só relata (`app/ingest.py`) |
| `uv run brain-router report` | documentos e chunks sem embedding por coleção e tenant, só leitura (`app/reports.py`) |
| `uv run pre-commit run --all-files` | todos os hooks sobre o repositório inteiro |
| `uv add <pacote>` / `uv add --optional dev <pacote>` | nova dependência, já atualizando o `uv.lock` |
| `uv run alembic upgrade head` | aplica as migrações (lê `DATABASE_URL`; nada de URL no `alembic.ini`); a 0004 cria índices com `CONCURRENTLY`, fora de transação, e espera as transações abertas do banco |
| `uv run alembic upgrade head --sql` | só imprime o SQL das migrações, sem conectar nem pedir credencial |

### Testes de integração (PostgreSQL + pgvector)

Testes marcados com `@pytest.mark.integration` precisam de banco e ficam fora do
`pytest` padrão (e do `scripts/check.sh`). No CI, o job `integration` sobe o
service `pgvector/pgvector:pg17`, espera o healthcheck, roda `alembic upgrade head`
e então `pytest -m integration`. Localmente, com Docker:

```bash
docker compose up -d --wait db
export DATABASE_URL=postgresql://brain:brain-local@localhost:5432/brain_router_test
export OPENROUTER_API_KEY=sk-or-v1-local
uv run pytest -m integration
docker compose down -v           # descarta o volume
```

Sem `DATABASE_URL` os testes de integração falham em vez de pular — de propósito.

### API HTTP v1 (FastAPI)

Alternativa ao MCP para o gateway e o studio (`POST /api/v1/search`, `/search/stream` em SSE e
`/index`).
Contrato, códigos de erro e teto de corpo em [`docs/api-v1.md`](docs/api-v1.md). A busca e a
indexação entram por `create_app(search=..., index=...)`; sem elas, as rotas dão 503
`backend_unavailable`: `uv run uvicorn --factory app.api:create_app --port 8000`.

### Teste de carga da ingestão (locust)

`loadtests/locustfile.py` (comando no docstring; extra `load`) manda uploads simultâneos ao
`/api/v1/index` de `app/bench/ingest_server.py`. No CI, o workflow `Carga` roda sob demanda e nos
PRs da ingestão. Com mais uploads simultâneos que `DB_POOL_SIZE`, as conexões de overflow abrem e
fecham a cada pedido (SCRAM e processo novo no Postgres): dimensione o pool pela concorrência.

### Contagem de tokens (tiktoken)

`app.tokens` conta tokens com o tiktoken, que baixa o arquivo de cada codificação no
primeiro uso (cl100k ~1,7 MB, o200k ~3,6 MB, de `openaipublic.blob.core.windows.net`)
e o guarda em `TIKTOKEN_CACHE_DIR` (sem a variável, numa pasta do diretório
temporário). O primeiro `pytest` precisa de rede; depois, não. Sem acesso a esse host
(CI fechado, imagem de produção), deixe os arquivos prontos numa pasta e aponte a variável.

Os hooks de pre-commit usam o ambiente do projeto (`uv run`), então ruff e mypy
rodam nas versões travadas no `uv.lock` — as mesmas do CI. Se o
`pyproject.toml` mudar sem `uv lock`, o hook `uv-lock` e o CI barram.

## Regras de Engenharia

- **Tamanho dos PRs:** Mínimo 100 linhas, máximo 500 linhas de código.
- **Commits:** Atômicos seguindo padrão Conventional Commits.
- **Contract-First:** Schemas e interfaces definidos primeiro para trabalho paralelo sem bloqueios mútuos.

## Mantenedores

- **Luan Oliveira** ([@in100tiva](https://github.com/in100tiva)) — Arquiteto de Software & Tech Lead
- **Victor Nascimento** ([@VictorNascimento14](https://github.com/VictorNascimento14)) — Tech Lead & Engenheiro de Software

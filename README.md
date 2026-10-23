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
e então `pytest -m integration`. Localmente, com Docker, o serviço `db-test` do perfil `test`
(dados em tmpfs, porta só no loopback; nada a ver com o banco do ambiente abaixo):

```bash
docker compose --profile test up -d --wait db-test
export DATABASE_URL=postgresql://brain:brain-local@localhost:5432/brain_router_test
export OPENROUTER_API_KEY=sk-or-v1-local
uv run pytest -m integration
docker compose --profile test rm -sf db-test   # para e apaga só o banco de teste
```

Sem `DATABASE_URL` os testes de integração falham em vez de pular — de propósito.

### Quality gate do RAG (CI)

O job `quality-gate` roda `uv run python -m app.evals.gate` no Postgres do job e falha com o
score médio abaixo de 0,85; a tabela por tema vai para o resumo do job. O score é o
`FaithfulnessEvaluator` da resposta de referência contra o 1º resultado da busca léxica, em 200
perguntas do dataset sintético, cada uma numa coleção com o trecho que a responde e 4 de outros
temas: mede a recuperação (ingestão, `tsvector` e consulta), sem LLM e sem chamada paga, e não a
fidelidade de uma resposta gerada. Localmente: as variáveis da seção acima, `uv run alembic
upgrade head` e o mesmo comando.

### Auditoria de segurança (CI)

O workflow `Auditoria` (todo PR, a main e todo dia: aviso novo aparece sem mudar o código) roda o
`pip-audit` no `uv.lock` inteiro, no OSV, e o `bandit` no `app/`, por `uvx` com versão fixada. A
política (`app/security_audit.py`) reprova HIGH/CRITICAL com correção publicada (severidade do GHSA;
sem ela, conta como HIGH) e pacote malicioso (`MAL-`), e põe a tabela no resumo do job. Exceção só
no `audit-exceptions.toml`, com motivo e data de revisão (vencida, o achado volta a reprovar);
`# nosec` não vale. Local: os passos do workflow, com `RUNNER_TEMP` e `GITHUB_STEP_SUMMARY` no env.

### API HTTP v1 (FastAPI)

Alternativa ao MCP para o gateway e o studio (`POST /api/v1/search`, `/search/stream` em SSE e
`/index`).
Contrato, códigos de erro e teto de corpo em [`docs/api-v1.md`](docs/api-v1.md). A busca e a
indexação entram por `create_app(search=..., index=...)`; sem elas, as rotas dão 503
`backend_unavailable`: `uv run uvicorn --factory app.api:create_app --port 8000`. Traces
(OpenTelemetry, correlacionados com o gateway): [`docs/observability.md`](docs/observability.md).

### Imagem Docker

O `Dockerfile` tem dois estágios: o uv monta o `.venv` só a partir do `uv.lock` (`--locked`, sem os
extras `dev` e `load`, com o cache do uv montado no build) e a imagem final, sobre
`python:3.12.15-slim-trixie` fixada por digest, leva só esse `.venv`, roda como `10001:10001` e
escuta em `0.0.0.0:8000`, com `HEALTHCHECK` no `/healthz`. O CMD é a partida (abaixo), que termina
virando o uvicorn com a fábrica `app.api.server:create_server` (o Postgres no `/healthz`): a imagem
precisa de `DATABASE_URL` e `OPENROUTER_API_KEY`, como no compose. O `.dockerignore` fecha o
contexto e reabre só o que o build lê.

A partida, `brain-router migrate [--timeout 60] [--seed PASTA] [-- COMANDO]` (`app/startup.py`):

1. espera o Postgres aceitar conexão por até `--timeout` s (o container pode subir antes do banco);
2. `alembic upgrade head` sob uma trava consultiva (também no `alembic` da linha de comando): com
   várias réplicas, uma migra e as outras esperam e encontram a head (no-op);
3. com `--seed PASTA`, indexa a pasta como o `brain-router ingest`, sob a mesma trava;
4. vira o COMANDO com `exec`: o uvicorn herda o PID 1 e o SIGTERM o encerra com 0.

Um passo que não termina sai com 1, sem servir; repetir a partida é seguro.

No CI, o workflow `Docker` passa o hadolint, confere que o contexto do build não leva `.env`,
`.git` nem ambiente local, constrói a imagem (cache do Buildx no GitHub, nada publicado) e sobe o
container, com a raiz somente leitura e sem capabilities, antes de um Postgres: a partida espera o
banco, migra e serve o `/healthz` com ele pela porta publicada; confere ainda o uid do processo e
que o SIGTERM o encerra com 0. Sem Docker, `tests/test_docker.py` confere os arquivos.

### Compose (RAG + Postgres com pgvector)

O `docker-compose.yml` sobe o `rag` (a imagem acima) e o `db` (`pgvector/pgvector:pg17`, fixada por
digest), com os nomes, as variáveis e o volume do compose do ecossistema. Na 1ª partida, o Postgres
cria o papel e o banco `brain` com a senha do `.env` e roda `deploy/postgres/initdb/` (as extensões
da migração 0001); os dados ficam no volume `pg-data`. O `rag` sobe depois do banco saudável e
migra na partida; o `/healthz` dá 503 com o banco fora. O banco não é publicado no host; o `rag`,
só no loopback.

```bash
cp .env.example .env              # preencha POSTGRES_PASSWORD (só caracteres de URL)
docker compose up -d --wait --build   # imagem do código atual; o rag espera o banco e migra
curl -fsS http://127.0.0.1:8000/healthz   # {"status":"ok","checks":{"postgres":"ok"}}
docker compose down               # os dados ficam no volume; `down -v` os apaga
```

O workflow `Compose` do CI sobe tudo e confere as extensões, a migração na partida, o `/healthz`
com e sem o banco, os dados depois de `down`/`up` e o `rag` saindo com 0; `tests/test_compose.py`
confere o arquivo.

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

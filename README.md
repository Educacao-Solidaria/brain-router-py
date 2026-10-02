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

## Regras de Engenharia

- **Tamanho dos PRs:** Mínimo 100 linhas, máximo 500 linhas de código.
- **Commits:** Atômicos seguindo padrão Conventional Commits.
- **Contract-First:** Schemas e interfaces definidos primeiro para trabalho paralelo sem bloqueios mútuos.

## Mantenedores

- **Luan Oliveira** ([@in100tiva](https://github.com/in100tiva)) — Arquiteto de Software & Tech Lead
- **Victor Nascimento** ([@VictorNascimento14](https://github.com/VictorNascimento14)) — Tech Lead & Engenheiro de Software

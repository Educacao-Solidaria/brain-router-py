# API HTTP v1 do RAG (brain-router-py)

> Contrato fixado em 19/10/2026 entre as frentes da Fase 3 (servidor Python, cliente Go do
> gateway e upload do studio). A seção 2 abaixo é o texto do contrato, sem alteração: mudança
> nela é mudança de contrato e vale para os três lados.

## 2. API HTTP v1 do RAG — brain-router-py (Python) ⇄ gateway (Go) e studio (TS)

**Fonte da verdade: os schemas pydantic do Luan em `app/schemas/mcp.py`** (não os tipos Go).

| método e rota | corpo | resposta |
|---|---|---|
| `POST /api/v1/search` | `HybridSearchInput` (`query`, `top_k` 1–50, `alpha` 0–1, `tenant_id?`, `filter_metadata?`) | 200 `HybridSearchOutput` (`query`, `results: [SearchResult{chunk, dense_score, sparse_score, rrf_score, rank}]`, `total_found`, `execution_time_ms`) |
| `POST /api/v1/index` | `IndexDocumentInput` (`title`, `content`, `source_uri`, `tenant_id`, `chunk_size`, `chunk_overlap`) | 201 `IndexDocumentOutput` (`document_id`, `chunks_created`, `success`, `message`) |
| `POST /api/v1/search/stream` (RAGPY 55) | `HybridSearchInput` | `text/event-stream`: `event: result` com `data:` = um `SearchResult` por evento, na ordem do rank; depois `event: done` com `{"total_found", "execution_time_ms"}`; falha no meio = `event: error` com o envelope de erro |
| `GET /healthz` (RAGPY 67) | — | 200/503 com o detalhe por dependência, sem segredo |

- Erro: status 4xx/5xx com `{"error": {"code": str, "message": str}}` — inclusive a validação
  (422 do FastAPI mapeado para esse envelope, com `code: "validation_error"`).
- `X-Request-Id` aceito e ecoado (RAGPY 59).
- **Cliente Go (GATEGO 55)** implementa `rag.Client` do Luan e MAPEIA: `results[]` → `hits[]`;
  `chunk.id`/`chunk.content` → `ChunkHit.ID`/`Content`; a origem do chunk → `SourceURI`;
  `rrf_score` → `Score`; `rank` → `Rank`; `IndexDocument(title, content, uri)` → `source_uri = uri`.
  Confira os campos de `Chunk` em `app/schemas/domain.py`.
- **Upload do studio (STUDTS 71)** usa `POST /api/v1/index` com o texto do arquivo. O backend não
  extrai texto de PDF (os parsers do Luan cobrem Markdown, texto e JSON): no v1 o studio aceita
  `.md`/`.txt`/`.json` e recusa PDF com mensagem clara, em vez de mandar binário.
- Enquanto o lado servidor não estiver na main, o cliente testa contra um fake (`httptest`, MSW)
  que segue esta tabela.

---

## Detalhes desta implementação (não mudam o contrato)

- Código em `app/api/`: busca e indexação entram por `create_app(search=..., index=...)`, com a
  forma dos handlers das tools MCP; rota sem handler responde **503 `backend_unavailable`**.
- **`Content-Type: application/json` é obrigatório**: sem ele (ou com o form do `curl -d`), o
  FastAPI não lê o corpo como JSON (proteção contra CSRF) e a resposta é 422.
- Corpo limitado a **2 MiB** (`DEFAULT_MAX_BODY_BYTES`): acima, **413 `payload_too_large`**, pelo
  `Content-Length` antes de ler o corpo ou, em chunked, assim que passa do teto.
- O OpenAPI gerado (`/docs`) declara o envelope abaixo como resposta `4XX`/`5XX` de toda rota.
- `POST /api/v1/search/stream`: 422 e 503 saem **antes** do stream, como erro HTTP comum. Depois
  do 200, falha vira `event: error` (mesmo envelope) e o stream acaba sem `done`. Os `result` vêm
  ordenados por `rank`. Sem evento por 15 s, sai o comentário `: ping` (o cliente ignora linha que
  começa com `:`), com `Cache-Control: no-cache` e `X-Accel-Buffering: no` (o nginx não segura os
  eventos). O `data` é JSON numa linha só. Cliente que desconecta cancela a busca no servidor.
- `X-Request-Id`: aceito se casar `[A-Za-z0-9._:-]{1,128}` (a regra do flow-gateway-go); senão,
  um novo (32 hex). Volta em toda resposta, inclusive 413, 429 e 500, e vai como `request_id` em
  toda linha de log do pedido.
- Rate limit (desligado por padrão: atrás do gateway, todo pedido vem do IP dele): token bucket
  por IP (IPv6 por /64), **429 `rate_limited`** com `Retry-After` em segundos. O
  `X-Forwarded-For` só conta quando a conexão vem de um proxy configurado como confiável.
- CORS só para origens listadas (`create_app(cors_origins=[...])`), sem credenciais; preflight de
  origem fora da lista é o 400 em texto do Starlette (o navegador não mostra o corpo).
- GZip em resposta de 1000 bytes ou mais, quando o cliente aceita; nunca no `text/event-stream`.
- Códigos de erro que esta API emite:

| status | `code` | quando |
|---|---|---|
| 400 | `bad_request` | corpo que nem chega a ser JSON (ex.: UTF-8 inválido) |
| 404 | `not_found` | rota inexistente |
| 405 | `method_not_allowed` | método errado na rota (com o header `Allow`) |
| 413 | `payload_too_large` | corpo acima do teto |
| 422 | `validation_error` | JSON inválido ou fora do schema; a mensagem diz campo e regra (até 5), nunca o valor recebido |
| 429 | `rate_limited` | cliente acima do limite (com `Retry-After`) |
| 503 | `backend_unavailable` | busca ou indexação não configurada neste servidor |
| 500 | `internal_error` | erro inesperado; o texto da exceção fica só no log do servidor |
| do erro | `code` do `BrainRouterError` em minúsculas | erro de domínio (ex.: 502 `embedding_service_error`) |

- Sem autenticação: serviço interno, atrás do gateway. Não exponha a porta fora da rede privada.

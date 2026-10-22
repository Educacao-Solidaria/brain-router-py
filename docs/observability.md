# Observabilidade do brain-router

A parte deste serviço nas convenções da Fase 4, fixas entre gateway, cache, brain-router e
studio (nomes, labels, buckets e privacidade): traces, métricas Prometheus e o dashboard.

## Traces (OpenTelemetry)

**Ligar:** `app.tracing.configure_tracing()` na subida (o `brain-router ingest` já chama); o
`FASTAPI_OTEL_AUTO_CONFIGURE` é ignorado. As variáveis são do **processo**: o `.env` só alimenta o
`Settings`.

| Variável | Efeito |
|---|---|
| `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` | URL do coletor, como veio (OTLP http/protobuf, porta 4318) |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | base, sem a do sinal: soma-se `/v1/traces` |
| `OTEL_EXPORTER_OTLP_HEADERS`, `..._TIMEOUT`, `..._COMPRESSION`, `..._CERTIFICATE` | lidos pelo exportador |
| `OTEL_TRACES_SAMPLER`, `OTEL_TRACES_SAMPLER_ARG` | padrão `parentbased_always_on`: segue a decisão de quem chamou |
| `OTEL_RESOURCE_ATTRIBUTES` | extras (ex.: `deployment.environment`); `service.name` é sempre `brain-router` |
| `OTEL_TRACES_EXPORTER=none` ou `OTEL_SDK_DISABLED=true` | desligam |

Sem endpoint, nada é configurado e os spans nem são criados. Protocolo diferente de
`http/protobuf`, exportador diferente de `otlp`/`none` ou URL que não é http(s): `ValueError` na
subida.

| Span | Kind | Origem | Atributos (nenhum com texto de usuário) |
|---|---|---|---|
| `POST /api/v1/search` (e as outras rotas) | SERVER | telemetria nativa do FastAPI 0.150 | `http.request.method`, `http.route`, `http.response.status_code`, `server.address` |
| `fastapi.dependencies`, `fastapi.endpoint`, `fastapi.serialization` | INTERNAL | idem | `code.function.name` |
| `embeddings` | EMBEDDING | `OpenRouterEmbeddingClient`, um por lote | `embedding.model_name`, `brain_router.embedding.texts`, `brain_router.embedding.chars` |
| `vector_search` | RETRIEVER | `TracedVectorStore(store)` | `brain_router.search.dimension`, `.limit`, `.results`; `retrieval.documents.N.document.id` e `.score` |
| `rerank` | RERANKER | `OpenRouterRerankerClient.rank` | `reranker.model_name`, `reranker.top_k`, `brain_router.rerank.documents` |

Kinds `EMBEDDING`, `RETRIEVER` e `RERANKER` são da OpenInference (`openinference.span.kind`).
Na falha, todo span ganha `error.type` e status ERROR. Até 25 itens (textos, documentos) por
span; a contagem total sempre vai.

**Propagação (W3C Trace Context).** O gateway em Go manda `traceparent` e `tracestate`
(`internal/tracing` e o cliente RAG, #39); o span do pedido continua o trace dele (filho do salto
do gateway, mesmo trace-id, `tracestate` repassado). `traceparent` inválido (id zerado, versão
`ff`, hexadecimal maiúsculo, lixo) começa um trace novo; flag `00` (não amostrado) é respeitada.
Na volta, `GatewayClient.chat` manda o `traceparent` do span corrente. Num trace começado aqui,
o SDK escreve a flag `03` (amostrado + o bit "random" do nível 2), que o `parseParent` do gateway
lê como amostrado. O `baggage` não é gravado nem repassado. O `/healthz` não gera span.

**Privacidade.** Query, chunk e texto de embedding só entram nos atributos que o `TraceConfig` da
OpenInference oculta, com a ocultação ligada: sai `__REDACTED__`. Só
`OPENINFERENCE_HIDE_{INPUTS,OUTPUTS,EMBEDDINGS_TEXT,EMBEDDINGS_VECTORS,RETRIEVAL_DOCUMENTS}=false`
revela, para depurar fora de produção, e a subida avisa no log (`tracing.content_revealed`). Vetor
nunca entra. Exceção, só pelo tipo (`error.type`). A URL crua (`url.query`, `url.path`) sai como
`REDACTED`: rota desconhecida é texto livre. Os logs da telemetria nativa do FastAPI ficam
desligados: com um provedor de logs do OTel, a exceção que escapasse do `RequestIdMiddleware`
iria com mensagem e traceback.

## Métricas (Prometheus)

Definidas em `app/metrics.py`, no registro padrão do `prometheus_client`, que já traz as de
processo e runtime (`process_*`, `python_*`). Latência em segundos, buckets `0.005 0.01 0.025
0.05 0.1 0.25 0.5 1 2.5 5`; sem as séries `*_created`.

**Rota.** `GET /metrics` (fora do OpenAPI) serve o `metrics.render()` com `text/plain;
version=0.0.4; charset=utf-8`: o `prometheus_client` 0.26 rotula o padrão como 1.0.0, mas com
nomes do formato antigo os bytes são os do 0.0.4, que todo raspador lê. Fica fora do rate limit e
do GZip e não gera span. **Sem autenticação, por contrato: a exposição é só a rede interna.** O
brain-router inteiro já é interno (a API v1 também não tem autenticação; quem a chama é o gateway,
que só usa `/api/v1/*`); o uvicorn sobe em `127.0.0.1` por padrão e, em contêiner, a porta fica na
rede dos serviços, sem publicar no host. O Prometheus da mesma rede raspa direto
(`http://<brain-router>:8000/metrics`). Se a porta um dia ficar atrás de um proxy público,
bloqueie `/metrics` nele.

```yaml
scrape_configs:
  - job_name: brain-router
    scrape_interval: 15s
    metrics_path: /metrics
    static_configs:
      - targets: ["brain-router:8000"]  # o nome do serviço na rede interna
```

| Métrica | Tipo | Labels | O que conta |
|---|---|---|---|
| `brain_router_search_requests_total` | counter | `mode`, `outcome` | cada busca do `/api/v1/search` e do `/search/stream` |
| `brain_router_search_duration_seconds` | histogram | `mode` | a duração dela |
| `brain_router_embedding_requests_total` | counter | `outcome` | cada pedido à OpenRouter (um lote; o acerto do cache não conta) |
| `brain_router_embedding_duration_seconds` | histogram | — | a duração dele, com os retries |
| `brain_router_rerank_duration_seconds` | histogram | — | cada rerank, inclusive o que falha e cai no fallback |
| `brain_router_result_score` | histogram (`0.1` … `1.0`) | — | a relevância que o reranker deu a cada resultado devolvido |
| `brain_router_eval_score` | gauge | `metric` | o último score de cada eval, por `metrics.record_eval_score` |

- `mode`: `lexical` quando o fallback léxico respondeu (`X-Search-Degraded`) ou com `alpha` 0;
  `dense` com `alpha` 1; senão `hybrid`. `outcome`: `ok`, `error`, `timeout` (`TimeoutError`) ou
  `canceled` (cliente desconectou, prazo de quem chamou).
- `brain_router_result_score` usa o score do reranker porque é o único de 0 a 1 no caminho: o RRF
  dá `peso / (60 + rank)`, perto de 0,016. O fallback do reranker (ordem de entrada, score 0) não
  entra. Sem reranker no pipeline, a métrica fica vazia.
- `record_eval_score(metric, score)`: o nome vira label, então só snake_case de até 64 caracteres
  (`faithfulness`, `context_relevance`); score fora de [0, 1] ou NaN é recusado.

**Processos.** O modo suportado é um processo por réplica (uvicorn sem `--workers`; escale por
réplicas): com vários workers, cada um tem o seu registro e a raspagem cairia num deles ao acaso.
Para vários workers, aponte `PROMETHEUS_MULTIPROC_DIR` para uma pasta vazia a cada subida, antes
de o processo importar `app.metrics`: cada worker grava em arquivo, o `render()` soma todos (o
gauge de eval fica com o valor mais recente) e as métricas de processo e runtime saem.

## Dashboard (Grafana)

`deploy/grafana/brain-router.json`, com `uid` fixo `brain-router`. A fonte de dados é a variável
`DS_PROMETHEUS` (tipo datasource, Prometheus): todo painel e toda query usam `${DS_PROMETHEUS}`,
então o mesmo JSON entra pelo *Import* da interface ou por provisionamento de arquivo, sem editar
uid de fonte.

Painéis: scores de eval ao longo do tempo e o atual de cada um (`brain_router_eval_score`; com
várias réplicas, a pior; a linha em 0,85 é o mesmo piso do quality gate do CI); relevância do
reranker, média e fração até 0,5 (**vazio sem reranker no pipeline**); latência da busca
p50/p95/p99 e p95 por modo; buscas por segundo por modo; buscas que não terminaram ok; latência e
desfecho dos pedidos de embedding; latência do rerank.

`tests/test_dashboard.py` confere no CI que toda métrica citada nas queries sai no texto do
`/metrics` (nome errado não passa), que o `uid` é o fixo e que toda fonte é `${DS_PROMETHEUS}`.
O teste não valida a sintaxe da PromQL, só os nomes: query nova se confere no Explore do Grafana
(ou no parser do Prometheus) antes do commit.

## Memória (tracemalloc)

`python -m app.bench.memleak [--seconds 60] [--warmup 20] [--clients 8]` monta a API v1 no próprio
processo, com o `index_handler` e a busca léxica no Postgres de `DATABASE_URL` (coleção própria,
apagada no fim; recusa `APP_ENV=production`), e os clientes alternam `/api/v1/index` (texto novo
em 4 `source_uri` por cliente: o banco não cresce) e `/api/v1/search`. A cada segundo, as chamadas
novas esperam, as em voo terminam, roda o `gc.collect()` e sai uma amostra: só o que ficou vivo.
**Critério:** a inclinação (mínimos quadrados) da memória contra as chamadas, depois do
aquecimento, até 16 B por chamada (um objeto de 40 B retido a cada busca já reprova); sai 1 acima.
O relatório em Markdown lista as 10 linhas que mais cresceram (`Snapshot.compare_to`, por linha).
**Desligado por padrão:** a carga faz ~135 chamadas/s com o tracemalloc e ~450 sem ele (medido
local); só o `profile()` o liga. Só vê os alocadores do Python (memória nativa, como o Rust do
tiktoken, pede RSS). CI: o workflow `Memória` roda 60 s em todo PR e na main; o *Run workflow*
aceita a duração.

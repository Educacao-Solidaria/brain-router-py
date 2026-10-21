# Observabilidade do brain-router

A parte deste serviço nas convenções da Fase 4, fixas entre gateway, cache, brain-router e
studio (nomes, labels, buckets e privacidade). Métricas Prometheus: seção a entrar com a rota
`/metrics`.

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

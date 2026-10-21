"""Tracing OpenTelemetry nas convenções da OpenInference: embeddings, busca vetorial e rerank.

`configure_tracing()` liga o envio na subida, pelas variáveis padrão do OTel (do processo, não do
`.env`): `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` (ou `OTEL_EXPORTER_OTLP_ENDPOINT` + `/v1/traces`),
`..._HEADERS`, `OTEL_TRACES_SAMPLER`, `OTEL_BSP_*`. Só OTLP http/protobuf, `service.name` sempre
`brain-router`. Sem endpoint, com `OTEL_TRACES_EXPORTER=none` ou com `OTEL_SDK_DISABLED=true`,
nada é configurado e os spans daqui nem são criados.

Privacidade: query, chunk e texto de embedding só entram nos atributos que o `TraceConfig` da
OpenInference oculta, e aqui a ocultação vem LIGADA (na biblioteca vem desligada): o texto sai como
`__REDACTED__` ou nem sai. Só `OPENINFERENCE_HIDE_<X>=false` explícito mostra o campo, para depurar
fora de produção, e a subida avisa no log. Vetor nunca entra. Sempre visíveis: contagens, tamanhos,
modelo e id do chunk. A exceção entra só pelo tipo (`error.type`): o `str()` de um erro do
SQLAlchemy traz o SQL e os parâmetros, que aqui são o vetor e a query.
"""

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from os import environ as os_environ
from urllib.parse import urlsplit

from openinference.instrumentation import OITracer, TraceConfig
from openinference.semconv.trace import DocumentAttributes as Doc
from openinference.semconv.trace import EmbeddingAttributes, RerankerAttributes, SpanAttributes
from openinference.semconv.trace import OpenInferenceSpanKindValues as Kind
from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.sdk.resources import SERVICE_NAME, SERVICE_VERSION, Resource
from opentelemetry.sdk.trace import Span as SdkSpan
from opentelemetry.sdk.trace import SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import INVALID_SPAN, Span, StatusCode
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

from app import __version__
from app.log import get_logger
from app.schemas.domain import Chunk, SearchResult
from app.storage.interfaces import BaseVectorStore

log = get_logger(__name__)

SERVICE = "brain-router"
# Itens por span (textos, documentos). O SDK guarda até 128 atributos por span e, cheio, descarta
# os mais antigos: o modelo e as contagens. 25 documentos x 3 atributos cabem com folga.
MAX_SPAN_ITEMS = 25
# A ocultação da OpenInference que vale aqui; cada variável só mostra o campo com "false".
HIDE_VARIABLES = (
    "OPENINFERENCE_HIDE_INPUTS",  # input.value, reranker.query e reranker.input_documents
    "OPENINFERENCE_HIDE_OUTPUTS",  # output.value, reranker.output_documents e documentos
    "OPENINFERENCE_HIDE_EMBEDDINGS_TEXT",  # embedding.embeddings.N.embedding.text
    "OPENINFERENCE_HIDE_EMBEDDINGS_VECTORS",  # embedding.embeddings.N.embedding.vector
    "OPENINFERENCE_HIDE_RETRIEVAL_DOCUMENTS",  # retrieval.documents.N.document.content
)

_tracer: OITracer | None = None  # None: tracing desligado, `_span` devolve um span no-op
_W3C = TraceContextTextMapPropagator()  # só traceparent e tracestate: o baggage não sai daqui
# O que o cliente escreve na URL e a telemetria do FastAPI grava cru (da query, só 5 chaves
# conhecidas viram REDACTED): a query string inteira e o path de rota desconhecida.
_URL_ATTRIBUTES = ("url.query", "url.path")


class _ScrubUrl(SpanProcessor):
    """Troca a URL crua do span de cada pedido por REDACTED antes de qualquer exportação. Fica o
    `http.route`, o template da rota."""

    def on_start(self, span: SdkSpan, parent_context: Context | None = None) -> None:
        for key in _URL_ATTRIBUTES:
            if key in (span.attributes or {}):
                span.set_attribute(key, "REDACTED")


def trace_headers() -> dict[str, str]:
    """`traceparent` e `tracestate` do span corrente, para o serviço chamado continuar o trace
    (vazio sem span). Só W3C: o `baggage` que chegou não é repassado."""
    headers: dict[str, str] = {}
    _W3C.inject(headers)
    return headers


def _revealed(environ: Mapping[str, str]) -> list[str]:
    """As variáveis com "false" explícito; qualquer outro valor (inclusive inválido) oculta."""
    return [name for name in HIDE_VARIABLES if environ.get(name, "").strip().lower() == "false"]


def trace_config(environ: Mapping[str, str] = os_environ) -> TraceConfig:
    revealed = _revealed(environ)
    inputs, outputs, text, vectors, documents = (name not in revealed for name in HIDE_VARIABLES)
    return TraceConfig(
        hide_inputs=inputs,
        hide_outputs=outputs,
        hide_embeddings_text=text,
        hide_embeddings_vectors=vectors,
        hide_retrieval_documents=documents,
    )


def traces_endpoint(environ: Mapping[str, str] = os_environ) -> str | None:
    """O endpoint OTLP de traces pedido pelo ambiente (None = desligado). ValueError para o pedido
    que não seria cumprido: outro exportador, outro protocolo (gRPC exigiria o grpcio), URL ruim."""
    if environ.get("OTEL_SDK_DISABLED", "").strip().lower() == "true":
        return None
    exporter = environ.get("OTEL_TRACES_EXPORTER", "").strip().lower() or "otlp"
    endpoint = environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "").strip()
    if not endpoint and (base := environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()):
        endpoint = f"{base.removesuffix('/')}/v1/traces"
    if not endpoint or exporter == "none":
        return None
    if exporter != "otlp":
        raise ValueError(f"OTEL_TRACES_EXPORTER={exporter!r} não é suportado: use otlp ou none")
    protocol = (
        environ.get("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL")
        or environ.get("OTEL_EXPORTER_OTLP_PROTOCOL")
        or "http/protobuf"
    ).strip()
    if protocol != "http/protobuf":
        raise ValueError(f"protocolo OTLP {protocol!r} não é suportado: use http/protobuf")
    parts = urlsplit(endpoint)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        # Sem ecoar o valor: a URL pode carregar credencial.
        raise ValueError("endpoint OTLP de traces inválido: use uma URL http(s) absoluta")
    return endpoint


def configure_tracing(environ: Mapping[str, str] = os_environ) -> TracerProvider | None:
    """Liga o envio dos spans se o ambiente pedir (uma vez, na subida); None se ficou desligado.

    O provedor vira o global do OTel: com ele, a telemetria nativa do FastAPI 0.150 cria o span
    de cada pedido HTTP e continua o `traceparent` W3C de quem chamou (o gateway em Go).
    """
    global _tracer
    endpoint = traces_endpoint(environ)
    if endpoint is None:
        return None
    # Aqui dentro: o exportador traz o protobuf, que só vale a pena com o tracing ligado.
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    resource = Resource.create({SERVICE_NAME: SERVICE, SERVICE_VERSION: __version__})
    provider = TracerProvider(resource=resource)  # amostrador do OTEL_TRACES_SAMPLER
    provider.add_span_processor(_ScrubUrl())  # no início do span: antes de qualquer exportação
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
    trace.set_tracer_provider(provider)
    _tracer = OITracer(provider.get_tracer(__name__, __version__), trace_config(environ))
    if revealed := _revealed(environ):  # texto de usuário vai ao coletor: tem de aparecer
        log.warning("tracing.content_revealed", variables=revealed)
    return provider


@contextmanager
def _span(name: str, kind: Kind) -> Iterator[Span]:
    if _tracer is None:
        yield INVALID_SPAN
        return
    with _tracer.start_as_current_span(
        name, openinference_span_kind=kind, record_exception=False, set_status_on_exception=False
    ) as span:
        try:
            yield span
        except Exception as exc:  # o tipo e não a mensagem (ver o docstring do módulo)
            span.set_attribute("error.type", type(exc).__qualname__)
            span.set_status(StatusCode.ERROR)
            raise


@contextmanager
def embedding_span(model: str, texts: Sequence[str]) -> Iterator[Span]:
    """Span EMBEDDING de um pedido ao provedor (um lote de `texts`)."""
    with _span("embeddings", Kind.EMBEDDING) as span:
        if span.is_recording():
            span.set_attribute(SpanAttributes.EMBEDDING_MODEL_NAME, model)
            span.set_attribute("brain_router.embedding.texts", len(texts))
            span.set_attribute("brain_router.embedding.chars", sum(map(len, texts)))
            for i, text in enumerate(texts[:MAX_SPAN_ITEMS]):
                prefix = f"{SpanAttributes.EMBEDDING_EMBEDDINGS}.{i}."
                span.set_attribute(prefix + EmbeddingAttributes.EMBEDDING_TEXT, text)
        yield span


@contextmanager
def reranker_span(model: str, query: str, documents: Sequence[str], top_n: int) -> Iterator[Span]:
    """Span RERANKER de um pedido ao provedor (`top_n` já limitado ao número de documentos)."""
    with _span("rerank", Kind.RERANKER) as span:
        if span.is_recording():
            span.set_attribute(RerankerAttributes.RERANKER_MODEL_NAME, model)
            span.set_attribute(RerankerAttributes.RERANKER_TOP_K, top_n)
            span.set_attribute(RerankerAttributes.RERANKER_QUERY, query)
            # Os documentos são os da busca, que o span RETRIEVER já descreve: só a contagem.
            span.set_attribute("brain_router.rerank.documents", len(documents))
        yield span


class TracedVectorStore(BaseVectorStore):
    """`BaseVectorStore` com span RETRIEVER em `search_vector` (o resto passa direto): o encaixe
    para a busca no pgvector, cuja implementação é do Luan. Entram a dimensão, o limite e, de cada
    documento, id e score; o conteúdo, só com a ocultação desligada. O vetor da query não entra."""

    def __init__(self, inner: BaseVectorStore) -> None:
        self._inner = inner

    async def add_chunks(self, collection_name: str, chunks: list[Chunk]) -> int:
        return await self._inner.add_chunks(collection_name, chunks)

    async def search_vector(
        self,
        collection_name: str,
        query_vector: list[float],
        limit: int = 10,
        min_score: float = 0.0,
    ) -> list[SearchResult]:
        with _span("vector_search", Kind.RETRIEVER) as span:
            span.set_attribute("brain_router.search.dimension", len(query_vector))
            span.set_attribute("brain_router.search.limit", limit)
            results = await self._inner.search_vector(
                collection_name, query_vector, limit, min_score
            )
            if span.is_recording():
                span.set_attribute("brain_router.search.results", len(results))
                for i, result in enumerate(results[:MAX_SPAN_ITEMS]):
                    prefix = f"{SpanAttributes.RETRIEVAL_DOCUMENTS}.{i}."
                    span.set_attribute(prefix + Doc.DOCUMENT_ID, result.chunk.id)
                    span.set_attribute(prefix + Doc.DOCUMENT_SCORE, result.rrf_score)
                    span.set_attribute(prefix + Doc.DOCUMENT_CONTENT, result.chunk.content)
        return results

    async def delete_collection(self, collection_name: str) -> bool:
        return await self._inner.delete_collection(collection_name)

    async def count_chunks(self, collection_name: str) -> int:
        return await self._inner.count_chunks(collection_name)

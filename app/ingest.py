"""Ingestão em lote de uma pasta local de documentação: `brain-router ingest <dir>`.

Cada `.md`/`.txt`/`.json` da árvore, um arquivo por transação: `sanitize_text` → `chunk_text`
(tokens do modelo) → `embed_documents` → COPY binário, com `metadata["source_uri"]`
(`file:///` + caminho absoluto) em cada chunk, como pede o contrato da API v1. Idempotente
por `source_uri`: o id do documento é o SHA-256 do texto e do que muda os chunks; sem
mudança, não paga embedding (o `CachedEmbeddingClient` não serve: só guarda `embed_query`),
e o arquivo alterado troca a versão velha, com os chunks pelo CASCADE, na mesma transação.
Symlink nunca é seguido (laço, arquivo fora da pasta). Falha de embedding ou de banco para a
execução, e rodar de novo retoma; `dry_run` só lê o banco. `index_handler`: o `/api/v1/index`.
ponytail: um arquivo por vez, texto cru, sem `--prune` (o apagado continua indexado) nem
`--follow-symlinks` (laço por dispositivo + inode); paralelismo e parsers por extensão depois.
"""

import asyncio
import hashlib
import json
import os
import stat
import unicodedata
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlmodel.ext.asyncio.session import AsyncSession

from app.chunking import TokenCounter, chunk_text
from app.clients.interfaces import BaseEmbeddingClient
from app.db import Database
from app.errors import BrainRouterError
from app.log import get_logger
from app.sanitize import sanitize_text
from app.schemas import Chunk
from app.schemas.mcp import IndexDocumentInput, IndexDocumentOutput
from app.storage.bulk import insert_chunks
from app.tokens import token_counter

SUFFIXES = frozenset({".md", ".txt", ".json"})
DEFAULT_MAX_BYTES = 2 * 1024 * 1024  # o mesmo teto do corpo do `POST /api/v1/index`
FORMAT_VERSION = 1  # suba se o mesmo arquivo passar a gerar outros chunks: reindexa tudo

_EXISTING = text(
    "SELECT source_uri, id FROM documents WHERE tenant_id = :tenant AND collection_id = :collection"
)
_VERSIONS = text(
    "SELECT id FROM documents WHERE tenant_id = :tenant AND collection_id = :collection "
    "AND source_uri = :source_uri"
)
# Gravações simultâneas do mesmo `source_uri` deixavam todas as versões (o DELETE de uma não vê
# o INSERT sem commit da outra); com a trava da transação, a seguinte apaga a anterior.
_LOCK = text("SELECT pg_advisory_xact_lock(hashtextextended(:key, 0))")
_DELETE_OLD = text(
    "DELETE FROM documents WHERE tenant_id = :tenant AND collection_id = :collection "
    "AND source_uri = :source_uri AND id <> :id"
)
_INSERT = text(
    "INSERT INTO documents (id, collection_id, title, source_uri, tenant_id) VALUES (:id, "
    ":collection, :title, :source_uri, :tenant) ON CONFLICT (id) DO NOTHING RETURNING id"
)

log = get_logger(__name__)


class Skip(Exception):
    """O arquivo fica de fora; a mensagem é o motivo que vai para o relatório."""


class InvalidDocument(BrainRouterError):
    """Pedido do `POST /api/v1/index` que passa no schema e não indexa: 422 `validation_error`."""

    def __init__(self, message: str) -> None:
        super().__init__(message, code="validation_error", http_status=422)


@dataclass(frozen=True, slots=True)
class Options:
    tenant_id: str = "default"
    collection_id: str = "default"
    chunk_size: int = 500
    chunk_overlap: int = 50
    max_bytes: int = DEFAULT_MAX_BYTES
    dry_run: bool = False

    def __post_init__(self) -> None:  # os limites do `IndexDocumentInput` (a API)
        if not 50 <= self.chunk_size <= 4000:
            raise ValueError(f"chunk_size deve estar em [50, 4000], veio {self.chunk_size}")
        if not 0 <= self.chunk_overlap <= 500:
            raise ValueError(f"chunk_overlap deve estar em [0, 500], veio {self.chunk_overlap}")
        if self.max_bytes < 1:  # negativo viraria read(-1): o arquivo inteiro na memória
            raise ValueError(f"max_bytes deve ser >= 1, veio {self.max_bytes}")


@dataclass(frozen=True, slots=True)
class Indexed:
    path: str
    document_id: str
    chunks: int
    tokens: int


@dataclass(slots=True)
class Report:
    dry_run: bool
    indexed: list[Indexed] = field(default_factory=list)
    unchanged: int = 0
    skipped: list[tuple[str, str]] = field(default_factory=list)  # (caminho, motivo)
    failure: tuple[str, str] | None = None  # (caminho, erro): a execução parou nele


def display(name: str) -> str:
    """Nome de uma linha só e sem controle de terminal: o não imprimível vira escape."""
    return "".join(c if c.isprintable() else repr(c)[1:-1] for c in name)


def scan(root: Path, report: Report) -> list[Path]:
    """Arquivos a ler sob `root`, em ordem estável. Ocultos (`.git`, `.obsidian`) e outras
    extensões ficam de fora em silêncio; symlink e pasta ilegível vão para o `report`."""
    found: list[Path] = []

    def skip(path: str | Path, reason: str) -> None:
        report.skipped.append((display(Path(path).relative_to(root).as_posix()), reason))

    def unreadable(exc: OSError) -> None:
        skip(exc.filename or root, f"erro de leitura ({type(exc).__name__})")

    for current, dirs, names in os.walk(root, onerror=unreadable):
        here = Path(current)
        for name in sorted(dirs):
            if (here / name).is_symlink():  # o os.walk não entra, mas lista
                skip(here / name, "symlink")
        dirs[:] = [d for d in sorted(dirs) if not (d.startswith(".") or (here / d).is_symlink())]
        for name in sorted(names):
            path = here / name
            if name.startswith(".") or path.suffix.lower() not in SUFFIXES:
                continue
            if path.is_symlink():
                skip(path, "symlink")
            else:  # tipo e tamanho, o `read_text` confere no arquivo já aberto
                found.append(path)
    return found


def read_text(path: Path, options: Options) -> str:
    """O texto UTF-8 de um arquivo regular de até `max_bytes`; senão `Skip` com o motivo."""
    # No arquivo aberto, valendo para o que for trocado depois da varredura: O_NONBLOCK (FIFO
    # não trava o open; no regular não muda nada) e O_NOFOLLOW (symlink não é seguido).
    flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        with open(os.open(path, flags), "rb") as file:
            if not stat.S_ISREG(os.fstat(file.fileno()).st_mode):
                raise Skip("não é arquivo regular")
            data = file.read(options.max_bytes + 1)
    except OSError as exc:
        raise Skip(f"erro de leitura ({type(exc).__name__})") from None
    if len(data) > options.max_bytes:
        raise Skip(f"maior que {options.max_bytes} bytes")
    try:
        return data.decode("utf-8")  # o BOM vira U+FEFF, que o sanitize_text tira
    except UnicodeDecodeError as exc:
        raise Skip(f"UTF-8 inválido (byte {exc.start})") from None


def _load(path: Path, options: Options) -> str:
    return sanitize_text(read_text(path, options))


def document_id(content: str, source_uri: str, options: Options, *, model: str, dim: int) -> str:
    """Id da versão: o mesmo enquanto o texto e o que muda os chunks forem os mesmos."""
    key = [FORMAT_VERSION, options.tenant_id, options.collection_id, source_uri]
    key += [options.chunk_size, options.chunk_overlap, model, dim]
    digest = hashlib.sha256(json.dumps(key).encode())
    digest.update(b"\n" + content.encode())  # o JSON (ASCII) não tem \n: sem ambiguidade
    return digest.hexdigest()[:32]


async def _existing(session: AsyncSession, options: Options) -> dict[str, set[str]]:
    params = {"tenant": options.tenant_id, "collection": options.collection_id}
    versions: dict[str, set[str]] = {}
    for uri, id in await (await session.connection()).execute(_EXISTING, params):
        versions.setdefault(uri, set()).add(id)
    return versions


async def _versions(session: AsyncSession, params: dict[str, str]) -> set[str]:
    return set((await (await session.connection()).execute(_VERSIONS, params)).scalars())


async def _store(
    db: Database, embedder: BaseEmbeddingClient, chunks: list[Chunk], params: dict[str, str]
) -> bool:
    """Vetoriza (fora da transação) e grava; False se a mesma versão já estava gravada."""
    vectors = await embedder.embed_documents([chunk.content for chunk in chunks])
    rows = [c.model_copy(update={"embedding": v}) for c, v in zip(chunks, vectors, strict=True)]
    key = json.dumps([params["tenant"], params["collection"], params["source_uri"]])
    async with db.session() as session:
        conn = await session.connection()
        await conn.execute(_LOCK, {"key": key})  # colisão do hash só enfileira à toa
        await conn.execute(_DELETE_OLD, params)
        if (await conn.execute(_INSERT, params)).first() is None:
            return False  # outra execução gravou esta versão depois da leitura inicial
        await insert_chunks(session, rows)
    return True


async def ingest_directory(
    root: Path,
    *,
    db: Database,
    embedder: BaseEmbeddingClient,
    model: str,
    options: Options = Options(),  # noqa: B008 (imutável)
) -> Report:
    """Indexa os arquivos de `root` conforme `options` e devolve o relatório."""
    # Arquivo e CPU síncronos vão por to_thread: rodando como tarefa no servidor (app.tasks),
    # a ingestão não trava o event loop das requisições.
    root = root.resolve(strict=True)
    count = await asyncio.to_thread(token_counter, model)  # pode ler (ou baixar) o tiktoken
    size, overlap = options.chunk_size, options.chunk_overlap
    chunk = partial(chunk_text, max_tokens=size, overlap_tokens=overlap, count_tokens=count)
    report = Report(options.dry_run)
    files = await asyncio.to_thread(scan, root, report)
    existing = await db.read(lambda session: _existing(session, options))
    for path in files:
        name = display(path.relative_to(root).as_posix())
        try:
            content = await asyncio.to_thread(_load, path, options)
        except Skip as exc:
            report.skipped.append((name, str(exc)))
            continue
        uri = path.as_uri()
        doc_id = document_id(content, uri, options, model=model, dim=embedder.dimension)
        if existing.get(uri) == {doc_id}:
            report.unchanged += 1
            continue
        if not (chunks := await asyncio.to_thread(chunk, content, doc_id)):
            report.skipped.append((name, "vazio"))
            continue
        for item in chunks:
            item.metadata["source_uri"] = uri
        if not options.dry_run:
            params = {"id": doc_id, "title": name, "source_uri": uri}
            params |= {"tenant": options.tenant_id, "collection": options.collection_id}
            try:
                stored = await _store(db, embedder, chunks, params)
            except (BrainRouterError, SQLAlchemyError, OSError) as exc:
                # O BrainRouterError já vem sem segredo (o DatabaseQueryError tira o DETAIL);
                # do resto, só o tipo: a mensagem pode trazer SQL, parâmetros, host, usuário.
                safe = exc.message if isinstance(exc, BrainRouterError) else type(exc).__name__
                report.failure = (name, safe)
                break
            if not stored:
                report.unchanged += 1
                continue
            log.info("ingest.stored", path=name, chunks=len(chunks))
        tokens = sum(item.token_count for item in chunks)
        report.indexed.append(Indexed(name, doc_id, len(chunks), tokens))
    return report


def _identifier(name: str, value: str) -> str:
    # Faz parte da identidade da versão: limpar trocaria o documento a que o pedido se refere.
    # Só o que a limpeza tira é recusado; NFD (nome de arquivo do macOS) passa como veio.
    if not value.strip() or sanitize_text(value) != unicodedata.normalize("NFC", value):
        raise InvalidDocument(f"{name} vazio ou com caractere de controle ou invisível")
    return value


def _prepare(
    content: str, uri: str, options: Options, count: TokenCounter, *, model: str, dim: int
) -> tuple[str, list[Chunk]]:
    content = sanitize_text(content)
    doc_id = document_id(content, uri, options, model=model, dim=dim)
    size, overlap = options.chunk_size, options.chunk_overlap
    chunks = chunk_text(
        content, doc_id, max_tokens=size, overlap_tokens=overlap, count_tokens=count
    )
    for chunk in chunks:
        chunk.metadata["source_uri"] = uri
    return doc_id, chunks


def index_handler(
    db: Database, embedder: BaseEmbeddingClient, *, model: str, collection_id: str = "default"
) -> Callable[[IndexDocumentInput], Awaitable[IndexDocumentOutput]]:
    """O `POST /api/v1/index` (`create_app(index=...)`): o caminho de um arquivo do ingest, com o
    texto do pedido e o `source_uri` como veio (também no `metadata` dos chunks). A versão já
    gravada (mesmo texto, tenant, `source_uri` e chunking) não paga embedding: 0 chunks novos."""
    count = token_counter(model)  # na montagem, uma vez: lê (ou baixa) o arquivo do tiktoken
    prepare = partial(_prepare, count=count, model=model, dim=embedder.dimension)

    async def index(body: IndexDocumentInput) -> IndexDocumentOutput:
        uri = _identifier("source_uri", body.source_uri)
        tenant = _identifier("tenant_id", body.tenant_id)
        if not (title := sanitize_text(body.title).strip()):
            raise InvalidDocument("title vazio depois da limpeza")
        options = Options(tenant, collection_id, body.chunk_size, body.chunk_overlap)
        doc_id, chunks = await asyncio.to_thread(prepare, body.content, uri, options)
        if not chunks:
            raise InvalidDocument("content vazio depois da limpeza")
        params = {"id": doc_id, "title": title, "source_uri": uri}
        params |= {"tenant": tenant, "collection": collection_id}
        same = await db.read(lambda session: _versions(session, params)) == {doc_id}
        created = 0 if same or not await _store(db, embedder, chunks, params) else len(chunks)
        if created:
            log.info("index.stored", document_id=doc_id, chunks=created)
        message = "indexado" if created else "sem mudança: esta versão já estava indexada"
        return IndexDocumentOutput(
            document_id=doc_id, chunks_created=created, success=True, message=message
        )

    return index


def render(report: Report) -> str:
    """Uma linha por arquivo indexado (+) ou pulado (-), a falha (!) e o resumo."""
    lines = [f"+ {i.path}: {i.chunks} chunks, {i.tokens} tokens" for i in report.indexed]
    lines += [f"- {path}: {reason}" for path, reason in report.skipped]
    if report.failure is not None:
        lines.append(f"! {report.failure[0]}: {report.failure[1]} (execução interrompida)")
    chunks = sum(item.chunks for item in report.indexed)
    tokens = sum(item.tokens for item in report.indexed)
    verb = "dry-run, nada gravado; seriam indexados" if report.dry_run else "indexados"
    lines.append(
        f"{verb}: {len(report.indexed)} ({chunks} chunks, {tokens} tokens); sem mudança: "
        f"{report.unchanged}; pulados: {len(report.skipped)}"
    )
    return "\n".join(lines) + "\n"

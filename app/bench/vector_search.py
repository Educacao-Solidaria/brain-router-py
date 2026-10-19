"""Benchmark de latência da busca vetorial no índice HNSW do pgvector.

    uv run python -m app.bench.vector_search --vectors 50000 --queries 300

Numa transação que sempre termina em ROLLBACK (nada fica no banco, nem se o processo morrer),
uma tabela UNLOGGED própria (a TEMP viveria nos 8 MB de buffers locais da sessão) recebe N
vetores de 1536 dimensões gerados no servidor, sempre os mesmos (`setseed`): 32 fatores
aleatórios repetidos ao longo das dimensões, mais ruído, para ter a dimensão intrínseca baixa
dos embeddings de texto (com vetores uniformes, o caso patológico do HNSW, o recall@10 medido
foi 0,05). Mede a busca exata (gabarito), constrói o HNSW (cosine) do perfil `balanced` e mede
a busca com o `ef_search` de cada perfil, uma consulta por vez com o vetor como parâmetro.
"""

import argparse
import asyncio
import random
import secrets
import statistics
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection
from sqlalchemy.sql.elements import TextClause

from app.db import Database
from app.log import configure_logging, shutdown_logging
from app.settings import get_settings
from app.storage.hnsw import EF_SEARCH_RANGE, HnswBuildParams, Recall, ef_search_for

DIMENSION = 1536  # a do `chunks.embedding`
WARMUP = 20  # consultas por linha rodadas antes da medida, fora da conta
SEED = 0.42  # setseed() dos vetores da tabela; as consultas usam o Random(42)
LATENT, NOISE = 32, 0.1  # fatores repetidos por vetor e ruído uniforme por coordenada
PROFILES: tuple[Recall, ...] = ("fast", "balanced", "high")
# Sem memória para o grafo inteiro (~400 MB com 50 mil vetores), o pgvector constrói em disco.
BUILD_MEMORY = "1GB"  # maintenance_work_mem

# O SET LOCAL do `apply_ef_search`: some no fim da transação, valor só como parâmetro.
_SET_LOCAL = text("SELECT set_config(:name, :value, true)")
_VERSIONS = text(
    "SELECT split_part(current_setting('server_version'), ' ', 1), "
    "(SELECT extversion FROM pg_extension WHERE extname = 'vector')"
)
# As subconsultas citam a linha de fora para rodar por linha; sem isso o Postgres as calcularia
# uma vez só (InitPlan) e todas as linhas teriam o mesmo vetor.
_GENERATE = (
    f"INSERT INTO {{table}} (id, embedding) SELECT i, (SELECT array_agg(z[(j - 1) % {LATENT} + 1]"
    f" + (random() * 2 - 1) * {NOISE}) FROM generate_series(1, {DIMENSION}) AS j)::real[]::vector"
    " FROM generate_series(1, :n) AS i, LATERAL (SELECT array_agg(random() * 2 - 1) AS z"
    f" FROM generate_series(1, {LATENT}) WHERE i > 0) AS latent"
)
# Os nomes entram no texto do DDL, que não aceita parâmetro: só hex gerado aqui.
_CREATE = f"CREATE UNLOGGED TABLE {{table}} (id integer PRIMARY KEY, embedding vector({DIMENSION}))"
_INDEX = "CREATE INDEX {index} ON {table} USING hnsw (embedding vector_cosine_ops) {options}"
_NEAREST = "SELECT id FROM {table} ORDER BY embedding <=> CAST(CAST(:q AS text) AS vector) LIMIT :k"


@dataclass(frozen=True)
class Latency:  # segundos
    p50: float
    p95: float
    p99: float
    mean: float


@dataclass(frozen=True)
class Row:
    label: str
    ef_search: int | None  # None = busca exata
    latency: Latency
    recall: float


@dataclass(frozen=True)
class Report:
    vectors: int
    queries: int
    k: int
    server: str
    build: HnswBuildParams
    load_seconds: float
    build_seconds: float
    rows: list[Row]


def summarize(seconds: Sequence[float]) -> Latency:
    """p50/p95/p99 (interpolação linear entre as amostras, `inclusive`) e média."""
    if not seconds:
        raise ValueError("sem medições")
    if len(seconds) == 1:  # o statistics.quantiles do 3.12 exige duas amostras
        (only,) = seconds
        return Latency(only, only, only, only)
    cuts = statistics.quantiles(seconds, n=100, method="inclusive")
    return Latency(cuts[49], cuts[94], cuts[98], statistics.fmean(seconds))


def recall_at_k(found: Sequence[Sequence[int]], truth: Sequence[Sequence[int]]) -> float:
    """Fração média do gabarito (busca exata) que a busca aproximada devolveu."""
    if not truth or len(found) != len(truth) or not all(truth):
        raise ValueError("found e truth precisam do mesmo número de consultas, sem gabarito vazio")
    hits = [len(set(f) & set(t)) / len(t) for f, t in zip(found, truth, strict=True)]
    return statistics.fmean(hits)


def query_vectors(count: int, *, seed: int = 42) -> list[str]:
    """Consultas da mesma distribuição da tabela (fora dela), no formato de texto do pgvector."""
    rng = random.Random(seed)
    vectors = []
    for _ in range(count):
        latent = [rng.uniform(-1, 1) for _ in range(LATENT)]
        values = (latent[j % LATENT] + rng.uniform(-NOISE, NOISE) for j in range(DIMENSION))
        vectors.append("[" + ",".join(f"{value:.6f}" for value in values) + "]")
    return vectors


async def _elapsed(conn: AsyncConnection, statement: TextClause, **params: object) -> float:
    started = time.perf_counter()
    await conn.execute(statement, params)
    return time.perf_counter() - started


async def _measure(
    conn: AsyncConnection, nearest: TextClause, probes: list[str], k: int
) -> tuple[list[float], list[list[int]]]:
    for probe in probes[:WARMUP]:
        await conn.execute(nearest, {"q": probe, "k": k})
    latencies: list[float] = []
    found: list[list[int]] = []
    for probe in probes:
        started = time.perf_counter()
        ids = (await conn.execute(nearest, {"q": probe, "k": k})).scalars().all()
        latencies.append(time.perf_counter() - started)
        found.append(list(ids))
    return latencies, found


def check_sizes(vectors: int, queries: int, k: int) -> None:
    """k em [1, min(vectors, 1000)] (o teto do ef_search) e ao menos uma consulta."""
    if not (1 <= k <= min(vectors, EF_SEARCH_RANGE[1]) and queries >= 1):
        raise ValueError(
            f"precisa de 1 <= k <= min(vectors, {EF_SEARCH_RANGE[1]}) e queries >= 1; "
            f"veio vectors={vectors}, queries={queries}, k={k}"
        )


async def run_benchmark(db: Database, *, vectors: int, queries: int, k: int = 10) -> Report:
    check_sizes(vectors, queries, k)
    table = f"bench_vectors_{secrets.token_hex(4)}"
    index, build = f"{table}_hnsw", HnswBuildParams.for_recall("balanced")
    nearest = text(_NEAREST.format(table=table))
    probes = query_vectors(queries)
    async with db.engine.connect() as conn:
        transaction = await conn.begin()
        try:
            version, extension = (await conn.execute(_VERSIONS)).one()
            await conn.execute(text("SELECT setseed(:seed)"), {"seed": SEED})
            await conn.execute(text(_CREATE.format(table=table)))
            load_seconds = await _elapsed(conn, text(_GENERATE.format(table=table)), n=vectors)
            exact, truth = await _measure(conn, nearest, probes, k)

            await conn.execute(_SET_LOCAL, {"name": "maintenance_work_mem", "value": BUILD_MEMORY})
            create_index = text(
                _INDEX.format(index=index, table=table, options=build.with_clause())
            )
            build_seconds = await _elapsed(conn, create_index)
            # Com poucos vetores o planejador preferiria a varredura; a medida é a do índice.
            await conn.execute(_SET_LOCAL, {"name": "enable_seqscan", "value": "off"})
            plan = await conn.execute(text(f"EXPLAIN {nearest.text}"), {"q": probes[0], "k": k})
            if index not in "\n".join(line for (line,) in plan):
                raise RuntimeError("o plano da busca não usa o índice HNSW; medida descartada")

            rows = [Row("exata (sem índice)", None, summarize(exact), 1.0)]
            for profile in PROFILES:
                ef_search = ef_search_for(k, recall=profile)
                await conn.execute(_SET_LOCAL, {"name": "hnsw.ef_search", "value": str(ef_search)})
                latencies, found = await _measure(conn, nearest, probes, k)
                recall = recall_at_k(found, truth)
                rows.append(Row(f"HNSW {profile}", ef_search, summarize(latencies), recall))
        finally:
            await transaction.rollback()
    server = f"PostgreSQL {version}, pgvector {extension}"
    return Report(vectors, queries, k, server, build, load_seconds, build_seconds, rows)


def render_markdown(report: Report) -> str:
    lines = [
        f"### Busca vetorial: {report.vectors:,} vetores × {DIMENSION} dimensões".replace(",", "."),
        "",
        f"{report.server}. HNSW `m = {report.build.m}, ef_construction = "
        f"{report.build.ef_construction}` (perfil balanced): carga em {report.load_seconds:.1f} s, "
        f"índice em {report.build_seconds:.1f} s. {report.queries} consultas por linha, uma por "
        f"vez na mesma conexão, depois de {WARMUP} de aquecimento; k = {report.k}.",
        "",
        f"Recall contra a busca exata, com vetores sintéticos ({LATENT} fatores repetidos nas "
        f"{DIMENSION} dimensões, mais ruído): compara os perfis entre si; não é o da produção.",
        "",
        f"| Busca | ef_search | p50 (ms) | p95 (ms) | p99 (ms) | média (ms) | recall@{report.k} |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report.rows:
        latency = (row.latency.p50, row.latency.p95, row.latency.p99, row.latency.mean)
        cells = " | ".join(f"{seconds * 1000:.2f}" for seconds in latency)
        ef_search = "—" if row.ef_search is None else row.ef_search
        lines.append(f"| {row.label} | {ef_search} | {cells} | {row.recall:.3f} |")
    return "\n".join(lines) + "\n"


async def _run(args: argparse.Namespace) -> Report:
    settings = get_settings()
    if settings.is_production:  # CPU, memória e uma transação longa no banco de verdade
        raise SystemExit("benchmark recusado com APP_ENV=production")
    db = Database.from_settings(settings)
    try:
        return await run_benchmark(db, vectors=args.vectors, queries=args.queries, k=args.k)
    finally:
        await db.dispose()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m app.bench.vector_search",
        description="Latência da busca vetorial (exata e HNSW) no PostgreSQL de DATABASE_URL.",
    )
    parser.add_argument("--vectors", type=int, default=50_000)
    parser.add_argument("--queries", type=int, default=300)
    parser.add_argument("--k", type=int, default=10, help="vizinhos por consulta")
    args = parser.parse_args(argv)
    try:
        check_sizes(args.vectors, args.queries, args.k)
    except ValueError as exc:
        parser.error(str(exc))
    configure_logging("WARNING", json=False, stream=sys.stderr)  # o stdout é do relatório
    try:
        print(render_markdown(asyncio.run(_run(args))), end="")
    finally:
        shutdown_logging()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

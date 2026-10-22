"""Profiler de memória assíncrono (tracemalloc): vazamento em chamadas concorrentes prolongadas.

`python -m app.bench.memleak`: a API v1 no processo, index e search alternados no Postgres; a
inclinação da memória viva contra as chamadas depois do aquecimento vai até `MAX_BYTES_PER_CALL`
(sai 1 acima dela). Desligado por padrão. Regras, custo e limites: `docs/observability.md`.
"""

import argparse
import asyncio
import gc
import math
import re
import secrets
import statistics
import sys
import tracemalloc
from array import array
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import httpx
from sqlalchemy import text

from app.api import create_app
from app.bench.ingest_load import VOCABULARY, document
from app.clients import MockEmbeddingClient
from app.db import Database
from app.evals.gate import AnyTermSearch
from app.ingest import index_handler
from app.log import configure_logging, shutdown_logging
from app.search.fallback import text_search_handler
from app.settings import get_settings
from app.tokens import DEFAULT_MODEL

MAX_BYTES_PER_CALL = 16.0  # um objeto de 40 B retido a cada busca (metade das chamadas) reprova
# Fora do diff: os retratos (memória do próprio tracemalloc) e as amostras deste módulo.
_IGNORED = [tracemalloc.Filter(False, name) for name in (tracemalloc.__file__, __file__)]
_LIBRARY = re.compile(r".*/(?:site-packages|lib/python3\.\d+)/")  # o caminho a partir do pacote
_TERMS = [word for word in VOCABULARY if len(word) > 4]  # sem stopword: toda busca usa o índice
_DROP = text("DELETE FROM documents WHERE collection_id = :collection")  # chunks no CASCADE

Call = Callable[[int, int], Awaitable[None]]  # (cliente, n-ésima chamada dele)


@dataclass(frozen=True)
class Profile:
    calls: Sequence[int]  # concluídas a cada amostra, do fim do aquecimento em diante
    traced: Sequence[int]  # bytes vivos no tracemalloc a cada amostra, depois do gc.collect()
    warm: tracemalloc.Snapshot  # no fim do aquecimento
    end: tracemalloc.Snapshot  # depois da última chamada

    def bytes_per_call(self) -> float:
        """A inclinação da memória contra as chamadas (mínimos quadrados): o critério."""
        if len(self.calls) < 3:
            raise ValueError(f"{len(self.calls)} amostras depois do aquecimento; o mínimo é 3")
        if self.calls[0] == self.calls[-1]:
            raise ValueError("nenhuma chamada terminou depois do aquecimento")
        return statistics.linear_regression(self.calls, self.traced).slope

    def growth(self) -> list[tracemalloc.StatisticDiff]:
        """As 10 linhas que mais cresceram (bytes) do fim do aquecimento ao fim."""
        diffs = self.end.compare_to(self.warm, "lineno")
        return [diff for diff in diffs if diff.size_diff > 0][:10]


async def profile(
    call: Call, *, clients: int, seconds: float, warmup: float, interval: float = 1.0
) -> Profile:
    """`clients` tarefas chamam `call` em sequência por `seconds`; a cada `interval`, com as
    chamadas pausadas, uma amostra. Erro numa chamada cancela as outras e sobe no
    `ExceptionGroup` do `TaskGroup`."""
    # interval < seconds: o amostrador dorme `interval` antes de ver o prazo (infinito = nunca sai)
    if not (clients >= 1 and 0 < interval < seconds and 0 <= warmup < seconds < math.inf):
        raise ValueError(
            "precisa de clients >= 1, 0 < interval < seconds e 0 <= warmup < seconds < inf; veio "
            f"{clients=}, {interval=}, {warmup=}, {seconds=}"
        )
    loop = asyncio.get_running_loop()
    warm_at, deadline = loop.time() + warmup, loop.time() + seconds
    # 16 B por amostra: num objeto por amostra (~320 B), o próprio profiler pareceria vazamento.
    calls, traced = array("q"), array("q")
    warm: list[tracemalloc.Snapshot] = []  # o retrato do fim do aquecimento
    done, busy, opened, idle = 0, 0, asyncio.Event(), asyncio.Event()
    opened.set()

    async def client(n: int) -> None:
        nonlocal done, busy
        i = 0
        while loop.time() < deadline:
            busy += 1
            idle.clear()
            try:
                await call(n, i)
            finally:
                busy -= 1
                done += 1
                if not busy:
                    idle.set()
            i += 1
            await opened.wait()  # a amostra fecha a porta; o prazo só se confere com ela aberta

    async def sampler() -> None:
        # Só antes do prazo: depois dele os clientes saem, e a memória deles sumiria da amostra.
        while True:
            await asyncio.sleep(interval)
            if loop.time() >= deadline:
                return
            opened.clear()
            try:
                if busy:
                    await idle.wait()
                gc.collect()
                if not warm and loop.time() >= warm_at:
                    warm.append(tracemalloc.take_snapshot().filter_traces(_IGNORED))
                if warm:
                    calls.append(done)
                    traced.append(tracemalloc.get_traced_memory()[0])
            finally:
                opened.set()

    # Os padrões dos filtros compilam (fnmatch, em cache) no 1º uso: antes do retrato do
    # aquecimento, senão o lixo do `re` sairia como crescimento no relatório.
    tracemalloc.Snapshot([(0, 0, (("", 0),), 1)], 1).filter_traces(_IGNORED)
    started = not tracemalloc.is_tracing()  # ligado de fora (PYTHONTRACEMALLOC): fica ligado
    if started:
        tracemalloc.start()
    try:
        async with asyncio.TaskGroup() as group:
            group.create_task(sampler())
            for n in range(clients):
                group.create_task(client(n))
        gc.collect()
        end = tracemalloc.take_snapshot().filter_traces(_IGNORED)
    finally:
        if started:
            tracemalloc.stop()
    if not warm:
        raise ValueError("nenhuma amostra depois do aquecimento: aumente seconds")
    return Profile(calls, traced, warm[0], end)


def render(result: Profile, *, seconds: float, warmup: float, clients: int) -> tuple[bool, str]:
    """Se passou (a inclinação impressa é a que se compara) e o relatório em Markdown."""
    slope = round(result.bytes_per_call(), 1)
    passed = slope <= MAX_BYTES_PER_CALL
    lines = [
        f"### Memória (tracemalloc): {slope:+.1f} B por chamada depois do aquecimento "
        f"(máximo {MAX_BYTES_PER_CALL:g}) — {'passou' if passed else '**vazamento**'}",
        "",
        f"Rodada de {seconds:g} s ({warmup:g} de aquecimento), {clients} clientes alternando "
        f"index e search: {result.calls[-1]} chamadas até a última amostra; {len(result.calls)} "
        f"amostras sem chamada em voo, de {result.traced[0] / 2**20:.2f} MiB a "
        f"{result.traced[-1] / 2**20:.2f} MiB.",
        "",
        "| Linha que mais cresceu do fim do aquecimento ao fim | KiB | Blocos |",
        "|---|---:|---:|",
    ]
    for diff in result.growth():
        frame = diff.traceback[0]
        where = _LIBRARY.sub("", frame.filename, count=1).removeprefix(f"{Path.cwd()}/")
        size, count = diff.size_diff / 1024, diff.count_diff
        lines.append(f"| `{where}:{frame.lineno}` | {size:+.1f} | {count:+d} |")
    return passed, "\n".join(lines) + "\n"


async def _run(*, seconds: float, warmup: float, clients: int) -> Profile:
    settings = get_settings()
    if settings.is_production:  # grava e apaga no banco
        raise SystemExit("profiler de memória recusado com APP_ENV=production")
    db, collection = Database.from_settings(settings), f"memleak-{secrets.token_hex(4)}"
    try:
        embedder = MockEmbeddingClient(dimension=1536)  # a do `chunks.embedding`
        app = create_app(
            search=text_search_handler(AnyTermSearch(db), collection),
            index=index_handler(db, embedder, model=DEFAULT_MODEL, collection_id=collection),
        )
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://memleak") as http:

            async def call(client: int, n: int) -> None:
                if n % 2:
                    query = " ".join(_TERMS[(client + n + k) % len(_TERMS)] for k in (0, 3))
                    response = await http.post("/api/v1/search", json={"query": query})
                else:  # 4 `source_uri` por cliente: cada gravação troca a versão de um deles
                    body = {"title": "carga", "content": document(client * 10**6 + n, 300)}
                    body["source_uri"] = f"memleak://{client}/{n // 2 % 4}"
                    response = await http.post("/api/v1/index", json=body)
                response.raise_for_status()

            return await profile(call, clients=clients, seconds=seconds, warmup=warmup)
    finally:
        try:
            async with db.session() as session:
                await (await session.connection()).execute(_DROP, {"collection": collection})
        finally:
            await db.dispose()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.bench.memleak")
    parser.add_argument("--seconds", type=float, default=60.0, help="duração, com o aquecimento")
    parser.add_argument("--warmup", type=float, default=20.0, help="segundos fora do critério")
    parser.add_argument("--clients", type=int, default=8, help="chamadas simultâneas")
    args = parser.parse_args(argv)
    configure_logging("WARNING", json=False, stream=sys.stderr)  # o stdout é do relatório
    try:
        result = asyncio.run(_run(**vars(args)))
    finally:
        shutdown_logging()
    passed, report = render(result, **vars(args))
    print(report, end="")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Uploads simultâneos ao `POST /api/v1/index`: cada usuário manda documentos novos, sem pausa.

    uv sync --extra dev --extra load
    uv run python -m app.bench.ingest_server &   # ao desligar (kill %1), conta e apaga
    uv run locust -f loadtests/locustfile.py --headless -H http://127.0.0.1:8000 \
        -u 20 -r 20 -t 60s --stop-timeout 30 --only-summary; kill %1

Com `--stop-timeout`, o upload em andamento termina, e a linha `locust: documentos=N chunks=M`
tem de bater com a do servidor ao desligar. Um processo só (sem `--processes`)."""

import itertools
import secrets
from typing import Any

from locust import FastHttpUser, constant, events, task

from app.bench.ingest_load import Rejected, Totals, accepted, payload

WORDS = 1500  # ~9,4 KB, ~2500 tokens no cl100k: 6 ou 7 chunks de 500
RUN = secrets.token_hex(4)  # outra execução contra o mesmo servidor não repete source_uri
uploads = itertools.count()
totals = Totals()


class Uploader(FastHttpUser):
    wait_time = constant(0)  # sem pausa: a vazão com N uploads simultâneos

    @task
    def upload(self) -> None:
        body = payload(RUN, next(uploads), WORDS)
        with self.client.post("/api/v1/index", json=body, catch_response=True) as response:
            try:
                chunks = accepted(response.status_code, response.text or "")
            except Rejected as exc:
                response.failure(str(exc))
            else:
                totals.documents += 1
                totals.chunks += chunks


# O `EventHook.add_listener` do locust não tem anotação (o mypy strict recusa o decorador).
@events.test_stop.add_listener  # type: ignore[untyped-decorator]
def _summary(**_: Any) -> None:
    print(totals, flush=True)

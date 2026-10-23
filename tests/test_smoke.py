"""O smoke da imagem (`app/smoke.py`) contra um Docker de mentira: cada defeito que ele promete
pegar reprova a sua conferência, sem derrubar as outras, e a limpeza roda sempre. O pedido HTTP, o
/proc, o script da head e o runner rodam de verdade, sem Docker; o Docker de verdade é o do job
`image` do workflow Docker.
"""

import io
import itertools
import json
import os
import subprocess
import sys
import threading
from collections.abc import Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
import yaml
from alembic.config import Config
from alembic.script import ScriptDirectory

from app import smoke
from app.smoke import Reply

ROOT = Path(__file__).resolve().parents[1]
HEAD = ScriptDirectory.from_config(Config(str(ROOT / "alembic.ini"))).get_current_head()
RAG, PORT, PID, SEARCH = "rag:teste", 49153, 4242, "busca sem provedor de embeddings"
HEALTH = ("com o banco: /healthz 200", "sem o banco: /healthz 503", "banco de volta: /healthz 200")
CHECKS = [
    *("usuário da imagem", "porta da imagem", "a partida espera o banco", HEALTH[0]),
    *("migração na partida", "HEALTHCHECK da imagem", "uid e gid do processo", SEARCH),
    *(HEALTH[1], HEALTH[2], "SIGTERM encerra com 0", "segredos fora do log"),
]
OK, DOWN = {"status": "ok", "checks": {"postgres": "ok"}}, {"checks": {"postgres": "down"}}


class Stack:
    """O docker, o HTTP e o /proc que o smoke vê: um RAG e um Postgres de mentira, com defeitos."""

    def __init__(self, **defects: Any) -> None:
        self.user, self.version, self.health = "10001:10001", HEAD, "healthy"
        self.uid = self.gid = "10001"  # a 1ª coluna do Uid (real) e a última do Gid (arquivos)
        self.exposed: dict[str, Any] = {"8000/tcp": {}}
        self.waits = self.serves = True
        self.down_status, self.exit_code, self.leak, self.broken = 503, 0, "", ""
        self.starting = 0  # pedidos sem resposta antes de o servidor subir
        self.checks = True  # False: a fábrica sem o banco (`create_app()`: 200 e `checks` vazio)
        self.search = Reply(503, {}, {"error": {"code": "backend_unavailable"}})
        vars(self).update(defects)  # `broken`: o subcomando do docker que falha
        self.calls: list[list[str]] = []
        self.runs: dict[str, tuple[list[str], dict[str, str]]] = {}  # imagem → (opções, env)
        self.db_up = False

    def docker(self, args: Sequence[str]) -> subprocess.CompletedProcess[str]:
        call, out, err = list(args), "", ""
        self.calls.append(call)
        match call:
            case [command, *_] if command == self.broken:
                return subprocess.CompletedProcess(call, 1, "", "Error response from daemon: x")
            case ["image", "inspect", *_]:
                out = json.dumps({"User": self.user, "ExposedPorts": self.exposed})
            case ["run", *options, image]:
                env = [options[i + 1].partition("=") for i, o in enumerate(options) if o == "--env"]
                self.runs[image] = (options, {name: value for name, _, value in env})
                self.db_up = self.db_up or image == smoke.PGVECTOR
            case ["logs", _]:  # `leak`: a variável do RAG que vaza, no stderr do container
                out = '{"event": "db.ping_failed"}\n' * self.waits
                err = self.runs[RAG][1][self.leak] if self.leak else ""
            case ["port", _, "8000/tcp"]:
                out = f"127.0.0.1:{PORT}\n"
            case ["exec", _, "python", "-I", "-c", smoke.HEAD]:
                out = f"{HEAD}\n"
            case ["exec", _, "psql", *_, smoke.VERSION]:
                out = f"{self.version}\n"
            case ["inspect", *_]:
                state = {"Pid": PID, "ExitCode": self.exit_code, "Health": {"Status": self.health}}
                out = json.dumps(state)
            case ["stop", _] | ["start", _]:  # o banco; o RAG para com `-t 10`
                self.db_up = call[0] == "start"
        return subprocess.CompletedProcess(call, 0, out, err)

    def request(self, port: int, method: str, path: str, body: bytes | None = None) -> Reply | None:
        if not self.serves or port != PORT or self.starting:
            self.starting = max(self.starting - 1, 0)
            return None
        if (method, path, body) == ("POST", "/api/v1/search", b'{"query": "smoke"}'):
            return self.search
        assert (method, path) == ("GET", "/healthz")
        if not self.checks:
            return Reply(200, {}, {"status": "ok", "checks": {}})
        return Reply(200, {}, OK) if self.db_up else Reply(self.down_status, {}, DOWN)

    def ids(self, pid: int) -> list[str]:
        return [self.uid, *["10001"] * 6, self.gid] if pid == PID else []


def run(stack: Stack, *, annotate: bool = False) -> tuple[int, list[str], str]:
    out = io.StringIO()
    tested = smoke.Smoke(
        *(RAG, smoke.PGVECTOR, stack.docker, stack.request, stack.ids),
        *(lambda _: None, itertools.count().__next__, out, annotate),  # cada leitura: +1 s
    )
    return tested.run(), tested.failed, out.getvalue()


def _option(options: list[str], flag: str) -> str:
    return options[options.index(flag) + 1]


def test_a_sound_image_passes_every_check_and_the_smoke_removes_what_it_created() -> None:
    stack = Stack(starting=5)  # o servidor demora alguns segundos: o smoke espera
    code, failed, out = run(stack)

    assert (code, failed) == (0, [])
    lines = out.splitlines()  # cada conferência, na ordem, e o resumo
    assert len(lines) == len(CHECKS) + 1
    assert all(line.startswith(f"ok    {c}: ") for line, c in zip(lines, CHECKS, strict=False))
    (rag, rag_env), (db, db_env) = stack.runs[RAG], stack.runs[smoke.PGVECTOR]
    # Como produção: raiz só leitura, sem capabilities nem escalada, o usuário da imagem e a
    # porta 8000 numa porta livre do loopback (nada publicado para fora do host).
    assert {"--read-only", "--tmpfs=/tmp"} <= set(rag) and "--user" not in rag
    assert _option(rag, "--cap-drop") == "ALL"
    assert _option(rag, "--security-opt") == "no-new-privileges"
    assert _option(rag, "--publish") == "127.0.0.1::8000"
    password, key = db_env["POSTGRES_PASSWORD"], rag_env["OPENROUTER_API_KEY"]
    assert len(password) == 32 and key.startswith("sk-or-v1-smoke-") and len(key) == 31
    assert rag_env == {
        "DATABASE_URL": f"postgresql+asyncpg://brain:{password}@db:5432/brain",
        "OPENROUTER_API_KEY": key,
        "OPENROUTER_BASE_URL": "http://127.0.0.1:9/api/v1",  # nenhuma chamada à OpenRouter
    }
    assert db_env == {
        "POSTGRES_USER": "brain",
        "POSTGRES_DB": "brain",
        "POSTGRES_PASSWORD": password,
    }
    # O RAG sobe antes do banco (já baixado), na mesma rede, onde o banco atende pelo nome `db`.
    pulled = stack.calls.index(["pull", "--quiet", smoke.PGVECTOR])
    assert pulled < min(i for i, call in enumerate(stack.calls) if call[0] == "run")
    assert [call[-1] for call in stack.calls if call[0] == "run"] == [RAG, smoke.PGVECTOR]
    net, name = _option(rag, "--network"), _option(rag, "--name")
    assert (_option(db, "--network"), _option(db, "--network-alias")) == (net, "db")
    assert ["stop", "-t", "10", name] in stack.calls  # o SIGTERM, no RAG
    assert stack.calls[-2:] == [
        ["rm", "--force", "--volumes", name, _option(db, "--name")],
        ["network", "rm", net],
    ]


def test_the_degraded_search_of_the_fallback_passes() -> None:
    reply = Reply(200, {"x-search-degraded": "lexical"}, {"results": [], "total_found": 0})

    code, failed, out = run(Stack(search=reply))

    assert (code, failed) == (0, [])
    assert f"ok    {SEARCH}: 200" in out and "(X-Search-Degraded: lexical)" in out


@pytest.mark.parametrize(
    ("defect", "failed"),
    [
        ({"user": ""}, ["usuário da imagem"]),  # sem USER: root
        ({"user": "0:0"}, ["usuário da imagem"]),
        ({"exposed": {"8000/tcp": {}, "8080/tcp": {}}}, ["porta da imagem"]),
        ({"waits": False}, ["a partida espera o banco"]),
        ({"version": "0003_fulltext_search"}, ["migração na partida"]),
        ({"broken": "exec"}, ["migração na partida"]),  # sem head nem versão: "" == "" não passa
        ({"health": "unhealthy"}, ["HEALTHCHECK da imagem"]),
        ({"uid": "0"}, ["uid e gid do processo"]),
        ({"gid": "0"}, ["uid e gid do processo"]),
        ({"search": Reply(502, {}, {"error": {"code": "embedding_service_error"}})}, [SEARCH]),
        ({"search": Reply(200, {}, {"results": [], "total_found": 0})}, [SEARCH]),  # sem a marca
        ({"search": Reply(503, {}, {"error": {"code": "unavailable"}})}, [SEARCH]),
        ({"down_status": 200}, [HEALTH[1]]),
        ({"checks": False}, [HEALTH[0], HEALTH[1], HEALTH[2]]),
        ({"exit_code": 137}, ["SIGTERM encerra com 0"]),
        ({"leak": "OPENROUTER_API_KEY"}, ["segredos fora do log"]),
        ({"leak": "DATABASE_URL"}, ["segredos fora do log"]),  # a senha do banco, dentro da URL
        ({"serves": False}, [HEALTH[0], SEARCH, *HEALTH[1:]]),  # nada responde na porta publicada
    ],
)
def test_each_defect_fails_its_own_check(defect: dict[str, Any], failed: list[str]) -> None:
    stack = Stack(**defect)

    code, found, out = run(stack, annotate=True)

    assert (code, found) == (1, failed)
    assert f"::error::smoke: {failed[0]}: " in out
    assert out.endswith(f"smoke: {len(failed)} conferência(s) reprovada(s)\n")
    # O fim do log do RAG sai na falha, sem a chave nem a senha.
    password = stack.runs[smoke.PGVECTOR][1]["POSTGRES_PASSWORD"]
    assert stack.runs[RAG][1]["OPENROUTER_API_KEY"] not in out and password not in out


def test_a_refused_docker_step_fails_and_the_cleanup_still_runs() -> None:
    stack = Stack(broken="run")

    code, failed, out = run(stack)

    assert (code, failed) == (1, ["docker"])
    assert "FALHA docker: docker run --detach: Error response from daemon: x" in out
    assert "--env" not in out and "DATABASE_URL" not in out  # os argumentos levam a senha
    assert [call[:2] for call in stack.calls[-2:]] == [["rm", "--force"], ["network", "rm"]]


def test_main_runs_the_given_images_and_annotates_on_github(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[tuple[str, str, bool]] = []

    def fake_run(self: smoke.Smoke) -> int:
        seen.append((self.image, self.pgvector, self.annotate))
        return 0

    monkeypatch.setattr(smoke.Smoke, "run", fake_run)
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    assert smoke.main(["--image", "rag:x", "--pgvector", "pg:y"]) == 0
    monkeypatch.delenv("GITHUB_ACTIONS")
    assert smoke.main([]) == 0

    assert seen == [("rag:x", "pg:y", True), (smoke.IMAGE, smoke.PGVECTOR, False)]
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    assert compose["services"]["rag"]["image"] == smoke.IMAGE  # as do docker-compose.yml
    assert compose["services"]["db"]["image"] == smoke.PGVECTOR


def test_head_script_prints_the_head_of_the_installed_migrations(tmp_path: Path) -> None:
    # Sem o alembic.ini, que a imagem não leva: fora do repositório e com -I, como no container.
    command = [sys.executable, "-I", "-c", smoke.HEAD]
    done = subprocess.run(command, capture_output=True, text=True, timeout=60, cwd=tmp_path)

    assert (done.returncode, done.stdout) == (0, f"{HEAD}\n")


def test_request_reads_status_headers_and_json_and_gives_none_when_nothing_answers() -> None:
    received: list[tuple[str, str, bytes]] = []

    class Api(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            body = self.rfile.read(int(self.headers["Content-Length"]))
            received.append((self.path, self.headers["Content-Type"], body))
            self.send_response(200)
            self.send_header("X-Search-Degraded", "lexical")
            self.end_headers()
            self.wfile.write(b'{"results": []}')

        def do_GET(self) -> None:
            self.send_response(503)
            self.end_headers()
            self.wfile.write(b"fora do ar")

        def log_message(self, format: str, *args: Any) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Api)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]
    try:
        searched = smoke.request(port, "POST", "/api/v1/search", b'{"query": "smoke"}')
        probed = smoke.request(port, "GET", "/healthz")
    finally:
        server.shutdown()
        server.server_close()

    assert received == [("/api/v1/search", "application/json", b'{"query": "smoke"}')]
    assert searched is not None and (searched.status, searched.body) == (200, {"results": []})
    assert searched.headers["x-search-degraded"] == "lexical"  # nomes em minúsculas
    assert probed is not None and (probed.status, probed.body) == (503, None)  # não é JSON
    assert smoke.request(port, "GET", "/healthz") is None  # ninguém mais escuta na porta


def test_ids_and_the_docker_runner_work_on_this_machine(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    assert smoke.ids(os.getpid()) == [str(os.getuid())] * 4 + [str(os.getgid())] * 4
    assert smoke.ids(0) == []  # sem processo: vazio, não exceção
    (tmp_path / "7").mkdir()
    (tmp_path / "7" / "status").write_text(
        "Name:\tuvicorn\nUid:\t10001\t10001\t10001\t10001\nGid:\t0\t0\t0\t0\n"
    )
    assert smoke.ids(7, tmp_path) == ["10001"] * 4 + ["0"] * 4  # gid root não passa por uid

    monkeypatch.setattr(smoke, "DOCKER", sys.executable)
    done = smoke.docker(["-c", "import sys; sys.stdout.buffer.write(b'fim\\xff'); sys.exit(3)"])
    assert (done.returncode, done.stdout) == (3, "fim\ufffd")

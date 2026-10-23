"""Smoke da imagem final do RAG antes do deploy: `python -m app.smoke [--image TAG]`. Só stdlib e o
CLI `docker`: roda com o python3 do runner, sem instalar o projeto.

Sobe a imagem como produção a sobe (raiz só leitura, sem capabilities nem escalada, com o usuário
dela), numa rede própria e antes de um Postgres com pgvector, e confere pela porta 8000 publicada
no loopback: `USER 10001:10001` e `EXPOSE 8000` na imagem e no processo; a partida esperando o
banco e migrando até a head da própria imagem; o `/healthz` 200 com o banco, 503 sem ele e 200 de
novo, e o HEALTHCHECK; a busca sem provedor de embeddings (`OPENROUTER_BASE_URL` numa porta fechada
do container: nada sai para a OpenRouter), que responde pelo fallback de `app/search/fallback.py`
(`X-Search-Degraded: lexical`) ou, enquanto `create_server` não compõe a busca, com o 503
`backend_unavailable`; a senha e a chave fora do log; e o SIGTERM saindo com 0.

Toda conferência roda (uma falha não esconde a outra) e o que o smoke cria sai no fim; sai 1 com
alguma reprovada. ponytail: o uid vem do /proc do host (Docker no Linux, como o CI).
"""

import argparse
import http.client
import json
import os
import secrets
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

IMAGE = "educacao-solidaria/brain-router:dev"  # a tag local do contrato, nunca publicada
# O banco do docker-compose.yml, pelo mesmo digest (pgvector 0.8.7 sobre o PostgreSQL 17.11).
PGVECTOR = (
    "pgvector/pgvector:pg17@sha256:ac08538c6f8b9904c33c8224c5e5706dbe760aca29db1d096972b4052c22a75d"
)
USER, PORT, UID = "10001:10001", "8000/tcp", "10001"
NO_PROVIDER = "http://127.0.0.1:9/api/v1"  # porta fechada no próprio container do RAG
# A head das migrações instaladas na imagem, que não leva o alembic.ini (ver app/startup.py).
HEAD = (
    "from alembic.config import Config; from alembic.script import ScriptDirectory; "
    "c = Config(); c.set_main_option('script_location', 'app:migrations'); "
    "print(ScriptDirectory.from_config(c).get_current_head())"
)
VERSION = "SELECT version_num FROM alembic_version"
DOCKER, PROC = "docker", Path("/proc")


class SmokeError(Exception):
    """O docker recusou um passo de que o smoke depende (a limpeza roda assim mesmo)."""


@dataclass(frozen=True)
class Reply:
    status: int
    headers: dict[str, str]  # nomes em minúsculas
    body: Any  # o JSON; None se não for JSON


def docker(args: Sequence[str]) -> subprocess.CompletedProcess[str]:
    command = [DOCKER, *args]  # byte inválido no log do container vira U+FFFD, não exceção
    return subprocess.run(command, capture_output=True, text=True, errors="replace", timeout=300)


def request(port: int, method: str, path: str, body: bytes | None = None) -> Reply | None:
    """Pelo loopback do host; None sem resposta (porta fechada, prazo de 10 s, conexão caída)."""
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        connection.request(method, path, body, {"Content-Type": "application/json"})
        response = connection.getresponse()
        raw = response.read()
    except (OSError, http.client.HTTPException):
        return None
    finally:
        connection.close()
    try:
        parsed = json.loads(raw)
    except ValueError:
        parsed = None
    headers = {name.lower(): value for name, value in response.getheaders()}
    return Reply(response.status, headers, parsed)


def ids(pid: int, proc: Path = PROC) -> list[str]:
    """uid e gid (real, efetivo, salvo e de arquivos) do processo, pelo /proc do host."""
    try:
        status = (proc / str(pid) / "status").read_text(encoding="utf-8")
    except OSError:  # o processo não existe
        return []
    fields = {key: value for key, _, value in (line.partition(":") for line in status.splitlines())}
    return fields.get("Uid", "").split() + fields.get("Gid", "").split()


def healthz(status: int, postgres: str) -> Callable[[Reply | None], bool]:
    def matches(reply: Reply | None) -> bool:
        if reply is None or reply.status != status or not isinstance(reply.body, dict):
            return False
        checks = reply.body.get("checks")
        return isinstance(checks, dict) and checks.get("postgres") == postgres

    return matches


def degraded_or_absent(reply: Reply | None) -> bool:
    """A busca sem provedor: degradada pelo fallback, ou ainda sem handler no servidor.

    ponytail: aceita o 503 enquanto o `create_server` não compõe a busca; quando compuser, o 503
    sai daqui e só o 200 degradado passa."""
    if reply is None or reply.status == 200:
        return reply is not None and reply.headers.get("x-search-degraded") == "lexical"
    error = reply.body.get("error") if isinstance(reply.body, dict) else None
    code = error.get("code") if isinstance(error, dict) else None
    return reply.status == 503 and code == "backend_unavailable"


def describe(reply: Reply | None) -> str:
    if reply is None:
        return "sem resposta"
    mark = reply.headers.get("x-search-degraded")
    text = f"{reply.status} {json.dumps(reply.body, ensure_ascii=False)[:200]}"
    return f"{text} (X-Search-Degraded: {mark})" if mark else text


def health(state: dict[str, Any]) -> Any:
    return (state.get("Health") or {}).get("Status")


@dataclass
class Smoke:
    image: str = IMAGE
    pgvector: str = PGVECTOR
    docker: Callable[[Sequence[str]], subprocess.CompletedProcess[str]] = docker
    request: Callable[..., Reply | None] = request
    ids: Callable[[int], list[str]] = ids
    sleep: Callable[[float], None] = time.sleep
    clock: Callable[[], float] = time.monotonic
    out: TextIO = sys.stdout
    annotate: bool = False  # anotação `::error::` do GitHub Actions em cada falha
    failed: list[str] = field(default_factory=list)

    def check(self, name: str, ok: bool, detail: str) -> None:
        print(f"{'ok' if ok else 'FALHA':5} {name}: {detail}", file=self.out)
        if not ok:
            self.failed.append(name)
            if self.annotate:
                print(f"::error::smoke: {name}: {detail}", file=self.out)

    def sh(self, *args: str, check: bool = True) -> str:
        done = self.docker(args)
        if check and done.returncode != 0:  # só o subcomando: os argumentos do `run` têm a senha
            raise SmokeError(f"docker {' '.join(args[:2])}: {done.stderr.strip()[-300:]}")
        return done.stdout

    def until[T](self, probe: Callable[[], T], done: Callable[[T], bool], timeout: float) -> T:
        """Repete `probe` a cada 1 s até `done` ou o prazo; devolve o último valor."""
        deadline = self.clock() + timeout
        while not done(value := probe()) and self.clock() < deadline:
            self.sleep(1.0)
        return value

    def logs(self, container: str) -> str:
        done = self.docker(["logs", container])
        return done.stdout + done.stderr

    def state(self, container: str) -> dict[str, Any]:
        found: dict[str, Any] = json.loads(self.sh("inspect", "-f", "{{json .State}}", container))
        return found

    def run(self) -> int:
        tag = secrets.token_hex(4)
        net, rag, db = f"smoke-{tag}", f"smoke-rag-{tag}", f"smoke-db-{tag}"
        password, key = secrets.token_hex(16), f"sk-or-v1-smoke-{secrets.token_hex(8)}"
        try:
            self._smoke(net, rag, db, password, key)
        except SmokeError as exc:
            self.check("docker", False, str(exc))
        finally:
            if self.failed:  # o fim do log do RAG, sem os segredos, para entender a falha
                tail = self.logs(rag).replace(password, "***").replace(key, "***")
                print(*tail.splitlines()[-20:], sep="\n", file=self.out)
            self.docker(["rm", "--force", "--volumes", rag, db])
            self.docker(["network", "rm", net])
        print(f"smoke: {len(self.failed)} conferência(s) reprovada(s)", file=self.out)
        return 1 if self.failed else 0

    def _smoke(self, net: str, rag: str, db: str, password: str, key: str) -> None:
        config = json.loads(self.sh("image", "inspect", "-f", "{{json .Config}}", self.image))
        user, exposed = config.get("User"), sorted(config.get("ExposedPorts") or {})
        self.check("usuário da imagem", user == USER, f"USER {user!r} (contrato: {USER})")
        self.check("porta da imagem", exposed == [PORT], f"EXPOSE {exposed} (contrato: {PORT})")
        self.sh("pull", "--quiet", self.pgvector)  # o download não come o prazo da partida
        self.sh("network", "create", net)
        url = f"postgresql+asyncpg://brain:{password}@db:5432/brain"  # a do compose (contrato §4)
        self.sh(
            *("run", "--detach", "--name", rag, "--network", net, "--publish", "127.0.0.1::8000"),
            *("--read-only", "--tmpfs=/tmp", "--cap-drop", "ALL"),
            *("--security-opt", "no-new-privileges", "--env", f"DATABASE_URL={url}"),
            *("--env", f"OPENROUTER_API_KEY={key}", "--env", f"OPENROUTER_BASE_URL={NO_PROVIDER}"),
            self.image,
        )
        # O RAG sobe antes do banco: a partida segue tentando (db.ping_failed), migra e serve.
        logs = self.until(lambda: self.logs(rag), lambda text: "db.ping_failed" in text, 30)
        waited = "db.ping_failed" in logs
        self.check("a partida espera o banco", waited, f"db.ping_failed no log: {waited}")
        self.sh(
            *("run", "--detach", "--name", db, "--network", net, "--network-alias", "db"),
            *("--env", "POSTGRES_USER=brain", "--env", "POSTGRES_DB=brain"),
            *("--env", f"POSTGRES_PASSWORD={password}", self.pgvector),
        )
        published = self.sh("port", rag, PORT, check=False).partition("\n")[0].rpartition(":")[2]
        port = int(published) if published.isdigit() else 0
        up, down = healthz(200, "ok"), healthz(503, "down")

        def probe() -> Reply | None:
            return self.request(port, "GET", "/healthz")

        reply = self.until(probe, up, 90)
        self.check("com o banco: /healthz 200", up(reply), describe(reply))
        head = self.sh("exec", rag, "python", "-I", "-c", HEAD, check=False).strip()
        psql = ("exec", db, "psql", "-U", "brain", "-d", "brain", "-tAc", VERSION)
        version = self.sh(*psql, check=False).strip()
        migrated = bool(head) and version == head
        self.check("migração na partida", migrated, f"banco em {version!r}, imagem em {head!r}")
        state = self.until(lambda: self.state(rag), lambda now: health(now) == "healthy", 40)
        self.check("HEALTHCHECK da imagem", health(state) == "healthy", f"{health(state)}")
        found = self.ids(state.get("Pid") or 0)
        self.check("uid e gid do processo", found == [UID] * 8, " ".join(found) or "fora do ar")
        reply = self.request(port, "POST", "/api/v1/search", b'{"query": "smoke"}')
        self.check("busca sem provedor de embeddings", degraded_or_absent(reply), describe(reply))

        self.sh("stop", db)
        reply = self.until(probe, down, 30)
        self.check("sem o banco: /healthz 503", down(reply), describe(reply))
        self.sh("start", db)
        reply = self.until(probe, up, 60)
        self.check("banco de volta: /healthz 200", up(reply), describe(reply))

        self.sh("stop", "-t", "10", rag)  # SIGTERM; depois de 10 s, SIGKILL (saída 137)
        code = self.state(rag).get("ExitCode")
        self.check("SIGTERM encerra com 0", code == 0, f"saída {code}")
        logs = self.logs(rag)
        secret = {"a senha do banco": password, "a chave da OpenRouter": key}
        leaked = [name for name, value in secret.items() if value in logs]
        self.check("segredos fora do log", not leaked, ", ".join(leaked) or "nenhum")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.smoke", description=__doc__)
    parser.add_argument("--image", default=IMAGE, help=f"a imagem construída (padrão: {IMAGE})")
    parser.add_argument("--pgvector", default=PGVECTOR, help="o Postgres (padrão: o do compose)")
    args = parser.parse_args(argv)
    annotate = os.environ.get("GITHUB_ACTIONS") == "true"
    return Smoke(args.image, args.pgvector, annotate=annotate).run()


if __name__ == "__main__":
    sys.exit(main())

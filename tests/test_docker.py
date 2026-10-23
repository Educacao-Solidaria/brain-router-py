"""Contrato da imagem do RAG (Fase 5, §2): o Dockerfile, o .dockerignore e o workflow `Docker`.

Sem Docker na máquina de desenvolvimento, estes testes leem os arquivos e conferem o que o contrato
fixa: bases por digest, só o uv.lock instalado, usuário 10001:10001, porta 8000, contexto sem
segredo e nada publicado. O build e o container são provados no CI (`.github/workflows/docker.yml`),
pelo smoke de antes do deploy (`app/smoke.py`, testado em tests/test_smoke.py).
"""

import importlib
import json
import re
import subprocess
import sys
import threading
import tomllib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
import yaml

from app.__main__ import build_parser
from app.startup import DEFAULT_TIMEOUT

ROOT = Path(__file__).resolve().parents[1]
IMAGE = "educacao-solidaria/brain-router:dev"  # a tag local do contrato, nunca publicada
PINNED = re.compile(r"[a-z0-9./-]+:[\w.-]+@sha256:[0-9a-f]{64}")  # a tag fica para quem lê
# A partida (espera o banco e migra) e, com exec, a fábrica com o Settings no uvicorn, em todas as
# interfaces do container, na porta do contrato.
FACTORY = "app.api.server:create_server"
SERVER = ["uvicorn", FACTORY, "--factory", "--host", "0.0.0.0", "--port", "8000"]
COMMAND = ["brain-router", "migrate", "--", *SERVER]

Instruction = tuple[str, str]


def _instructions() -> list[Instruction]:
    """(INSTRUÇÃO, argumentos) do Dockerfile, com as continuações juntadas e sem comentários."""
    parsed: list[Instruction] = []
    pending = ""
    for raw in (ROOT / "Dockerfile").read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.endswith("\\"):
            pending += line[:-1].strip() + " "
            continue
        keyword, _, args = (pending + line).partition(" ")
        parsed.append((keyword.upper(), args.strip()))
        pending = ""
    return parsed


def _stages() -> dict[str, list[Instruction]]:
    """Instruções por estágio (o nome do `AS`), na ordem do arquivo: o último é a imagem final."""
    stages: dict[str, list[Instruction]] = {}
    current: list[Instruction] = []
    for keyword, args in _instructions():
        if keyword == "FROM":
            current = stages.setdefault(args.split()[-1], [])
        current.append((keyword, args))
    return stages


def _args(instructions: list[Instruction], keyword: str) -> list[str]:
    return [args for kw, args in instructions if kw == keyword]


def _final() -> list[Instruction]:
    return list(_stages().values())[-1]


def test_external_bases_are_pinned_by_digest_and_python_matches_the_repo() -> None:
    stages = _stages()
    images = [args.split()[0] for args in _args(_instructions(), "FROM")]
    external = [image for image in images if image not in stages]

    assert external and all(PINNED.fullmatch(image) for image in external)
    python = (ROOT / ".python-version").read_text(encoding="utf-8").strip()
    (base,) = [image for image in external if image.startswith("python:")]
    assert base.startswith(f"python:{python}.") and "-slim" in base  # python slim, do contrato


def test_builder_and_final_image_share_the_python_the_venv_points_to() -> None:
    assert _stages()["builder"][0] == ("FROM", "base AS builder")
    assert _final()[0][1].split()[0] == "base"


def test_dependencies_come_only_from_the_lock_with_the_uv_cache_mounted() -> None:
    syncs = [run for run in _args(_stages()["builder"], "RUN") if "uv sync" in run]

    assert len(syncs) == 2  # 1) só as dependências (camada estável); 2) o pacote `app`
    for run in syncs:
        assert "--mount=type=cache,target=/root/.cache/uv" in run
        flags = run.split("uv sync", 1)[1].split()
        assert "--locked" in flags and "--no-dev" in flags
        assert not {"--extra", "--all-extras"} & set(flags)  # dev e load ficam fora da imagem
    assert "--no-install-project" in syncs[0] and "--no-editable" in syncs[1]


def test_final_image_runs_as_10001_on_port_8000() -> None:
    final = _final()
    (adduser,) = [run for run in _args(final, "RUN") if "useradd" in run]

    assert _args(final, "USER")[-1] == "10001:10001"  # numérico: runAsNonRoot confere sem passwd
    assert "--uid 10001" in adduser and "--gid 10001" in adduser
    assert _args(final, "EXPOSE") == ["8000"]


def test_final_image_gets_only_the_venv_and_no_secret() -> None:
    final = _final()
    declared = " ".join(_args(_instructions(), "ARG") + _args(_instructions(), "ENV"))

    assert _args(final, "COPY") + _args(final, "ADD") == ["--from=builder /app/.venv /app/.venv"]
    assert not re.search(r"KEY|SECRET|TOKEN|PASSWORD|DATABASE_URL", declared, re.IGNORECASE)


def test_command_is_the_startup_that_becomes_the_settings_factory() -> None:
    (command,) = [json.loads(args) for args in _args(_final(), "CMD")]  # sem shell: exec no fim
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    module, _, factory = FACTORY.partition(":")

    assert command == COMMAND
    assert pyproject["project"]["scripts"][command[0]] == "app.__main__:main"
    assert build_parser().parse_args(command[1:]).command == "migrate"  # a linha passa no CLI
    assert callable(getattr(importlib.import_module(module), factory))


def _serve(status: int) -> ThreadingHTTPServer:
    class Healthz(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(status if self.path == "/healthz" else 404)
            self.end_headers()

        def log_message(self, format: str, *args: Any) -> None:  # sem ruído no stderr do pytest
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Healthz)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


@pytest.mark.parametrize(("status", "exit_code"), [(200, 0), (503, 1)])
def test_healthcheck_probe_fails_unless_healthz_answers_200(status: int, exit_code: int) -> None:
    (healthcheck,) = _args(_final(), "HEALTHCHECK")
    options, _, command = healthcheck.partition(" CMD ")
    probe = json.loads(command)
    assert probe[:2] == ["python", "-c"] and "--timeout=" in options  # sem curl na imagem
    # A espera do banco na partida não conta como falha da sonda.
    start = re.search(r"--start-period=(\d+)s", options)
    assert start is not None and int(start.group(1)) >= DEFAULT_TIMEOUT

    server = _serve(status)
    try:
        code = probe[2].replace("127.0.0.1:8000", f"127.0.0.1:{server.server_address[1]}")
        assert code != probe[2]  # a sonda mira a porta do contrato
        done = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, timeout=30)
    finally:
        server.shutdown()
        server.server_close()

    assert done.returncode == exit_code


def test_dockerignore_starts_closed_and_reopens_exactly_what_the_build_reads() -> None:
    lines = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
    rules = [line.strip() for line in lines if line.strip() and not line.startswith("#")]
    reopened = {rule[1:] for rule in rules if rule.startswith("!")}
    last = max(index for index, rule in enumerate(rules) if rule.startswith("!"))

    assert rules[0] == "*"  # na raiz: .env, .git, .venv, target/ e node_modules/ ficam fora
    assert {"**/.env", "**/.env.*", "**/.git", "**/.venv"} <= set(rules[last + 1 :])  # nem em app/
    # Reaberto = o que o Dockerfile lê do contexto (COPY sem --from e os bind mounts), nada além.
    read: set[str] = set()
    for keyword, args in _instructions():
        if keyword == "COPY" and "--from=" not in args:
            read.update(token for token in args.split()[:-1] if not token.startswith("--"))
        read.update(re.findall(r"type=bind,source=([^,\s]+)", args))
    assert read == reopened


def _workflow() -> Any:
    return yaml.safe_load((ROOT / ".github/workflows/docker.yml").read_text(encoding="utf-8"))


def test_docker_workflow_builds_and_runs_the_image_without_publishing() -> None:
    workflow = _workflow()
    jobs = workflow["jobs"]
    steps = [step for job in jobs.values() for step in job["steps"]]
    (build,) = [step for step in steps if step.get("uses", "").startswith("docker/build-push")]
    runs = [step for step in steps if "run" in step]

    assert workflow["permissions"] == {"contents": "read"}
    assert not [job for job in jobs.values() if "permissions" in job]  # nenhum job amplia
    assert "paths" not in (workflow[True]["pull_request"] or {})  # `on:` vira True no YAML 1.1
    assert all("timeout-minutes" in job for job in jobs.values())
    assert build["with"]["push"] is False and build["with"]["load"] is True  # só no runner
    assert build["env"]["DOCKER_BUILD_RECORD_UPLOAD"] is False  # nem artefato no run
    assert not [step for step in steps if "login" in step.get("uses", "")]
    assert jobs["image"]["env"]["IMAGE"] == IMAGE and build["with"]["tags"] == "${{ env.IMAGE }}"
    assert build["with"]["cache-from"] == "type=gha"
    assert all(step["shell"] == "bash" and "${{" not in step["run"] for step in runs)
    smoke = jobs["image"]["steps"][-1]["run"].splitlines()
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    assert jobs["image"]["env"]["PGVECTOR"] == compose["services"]["db"]["image"]
    # O último comando do passo é o smoke, na imagem construída: a saída dele é a do passo.
    assert smoke[-1] == 'python3 -m app.smoke --image "$IMAGE" --pgvector "$PGVECTOR"'

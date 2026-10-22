"""Contrato do docker-compose.yml (Fase 5, §4): banco e RAG com os nomes, as variáveis e o volume
do compose do ecossistema, o pgvector criado na inicialização do banco e nada de segredo no arquivo.

Sem Docker na máquina de desenvolvimento, estes testes leem os arquivos. Subir os containers, o
/healthz com o banco (e sem ele), os dados no volume depois de `down`/`up` e o desligamento são
provados no CI (`.github/workflows/compose.yml`).
"""

import importlib
import re
from pathlib import Path
from typing import Any

import yaml
from alembic.config import Config
from alembic.script import ScriptDirectory

from app.settings import Settings

ROOT = Path(__file__).resolve().parents[1]
PGDATA = "/var/lib/postgresql/data"  # o PGDATA e o VOLUME da imagem pgvector/pgvector:pg17
PINNED = re.compile(r"pgvector/pgvector:pg17@sha256:[0-9a-f]{64}")  # a tag do contrato, fixada
COMMAND = "uvicorn app.api.server:create_server --factory --host 0.0.0.0 --port 8000"
DATABASE_URL = "postgresql+asyncpg://brain:${POSTGRES_PASSWORD}@db:5432/brain"  # contrato, §4


def _load(path: str) -> Any:
    return yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))


def _services() -> dict[str, Any]:
    services: dict[str, Any] = _load("docker-compose.yml")["services"]
    return services


def _pg_isready(service: dict[str, Any]) -> list[str]:
    # Pelo TCP: o servidor temporário do initdb só escuta no socket unix e não passa por pronto.
    db = service["environment"]["POSTGRES_DB"]
    return ["CMD", "pg_isready", "-h", "127.0.0.1", "-U", "brain", "-d", db]


def test_database_is_pinned_pgvector_with_contract_names_and_no_published_port() -> None:
    services = _services()
    db = services["db"]

    assert [n for n, s in services.items() if "profiles" not in s] == ["db", "rag"]  # up puro
    assert PINNED.fullmatch(db["image"])
    assert db["environment"] == {
        "POSTGRES_USER": "brain",
        "POSTGRES_DB": "brain",
        "POSTGRES_PASSWORD": "${POSTGRES_PASSWORD}",  # do .env; sem valor padrão no arquivo
    }
    assert "ports" not in db  # só a rede do compose alcança o banco
    assert db["healthcheck"]["test"] == _pg_isready(db)


def test_database_keeps_data_in_the_named_volume_and_runs_the_init_scripts() -> None:
    compose = _load("docker-compose.yml")

    assert compose["services"]["db"]["volumes"] == [
        f"pg-data:{PGDATA}",
        "./deploy/postgres/initdb:/docker-entrypoint-initdb.d:ro",
    ]
    assert list(compose["volumes"]) == ["pg-data"]


def test_init_script_creates_the_extensions_of_migration_0001_idempotently() -> None:
    script = ScriptDirectory.from_config(Config(str(ROOT / "alembic.ini")))
    revision = script.get_revision("0001_create_extensions")
    initdb = sorted((ROOT / "deploy/postgres/initdb").iterdir())
    lines = initdb[0].read_text(encoding="utf-8").splitlines()

    assert revision is not None and [path.name for path in initdb] == ["01-extensions.sql"]
    assert [line for line in lines if not line.startswith("--")] == [
        f"CREATE EXTENSION IF NOT EXISTS {name};" for name in revision.module.EXTENSIONS
    ]


def test_rag_serves_the_settings_factory_after_the_database_is_healthy() -> None:
    rag = _services()["rag"]
    module, _, factory = COMMAND.split()[1].partition(":")

    assert rag["build"] == "." and rag["image"] == "educacao-solidaria/brain-router:dev"
    assert " ".join(rag["command"]) == COMMAND
    assert callable(getattr(importlib.import_module(module), factory))
    assert rag["depends_on"] == {"db": {"condition": "service_healthy"}}
    assert rag["ports"] == ["127.0.0.1:8000:8000"]  # API sem autenticação: só o loopback
    assert rag["read_only"] is True and rag["tmpfs"] == ["/tmp"]
    assert rag["cap_drop"] == ["ALL"] and rag["security_opt"] == ["no-new-privileges:true"]
    assert "user" not in rag  # roda como o 10001:10001 da imagem (contrato, §2)


def test_rag_gets_only_the_contract_variables_and_the_url_passes_settings() -> None:
    env = _services()["rag"]["environment"]
    url = env["DATABASE_URL"].replace("${POSTGRES_PASSWORD}", "s3nha")
    settings = Settings(_env_file=None, database_url=url, openrouter_api_key="sk-or-v1-teste")

    assert env == {"DATABASE_URL": DATABASE_URL, "OPENROUTER_API_KEY": "${OPENROUTER_API_KEY}"}
    assert settings.database_url.get_secret_value() == url  # o driver, o host db e o banco brain


def test_test_database_is_disposable_and_only_on_the_loopback() -> None:
    services = _services()
    test_db = services["db-test"]

    assert test_db["profiles"] == ["test"] and test_db["image"] == services["db"]["image"]
    assert test_db["environment"]["POSTGRES_DB"] == "brain_router_test"
    assert test_db["tmpfs"] == [PGDATA] and "volumes" not in test_db  # sem initdb, como no CI
    assert test_db["ports"] == ["127.0.0.1:5432:5432"]
    assert test_db["healthcheck"]["test"] == _pg_isready(test_db)


def test_secrets_come_only_from_the_env_file_listed_in_env_example() -> None:
    text = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    lines = (ROOT / ".env.example").read_text(encoding="utf-8").splitlines()
    example = {key: value for key, _, value in (line.partition("=") for line in lines)}

    assert set(re.findall(r"\$\{(\w+)", text)) == {"POSTGRES_PASSWORD", "OPENROUTER_API_KEY"}
    assert "${POSTGRES_PASSWORD:" not in text and "env_file" not in text  # sem senha padrão
    assert example["POSTGRES_PASSWORD"] == ""  # vazia: o banco se recusa a inicializar sem ela
    assert "OPENROUTER_API_KEY" in example
    assert ".env" in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()


def test_compose_workflow_runs_the_stack_without_publishing() -> None:
    workflow = _load(".github/workflows/compose.yml")
    (job,) = workflow["jobs"].values()
    (build,) = [step for step in job["steps"] if step.get("uses", "").startswith("docker/build")]
    runs = [step for step in job["steps"] if "run" in step]
    contract = "\n".join(step["run"] for step in runs)

    assert workflow["permissions"] == {"contents": "read"} and "permissions" not in job
    assert "paths" not in (workflow[True]["pull_request"] or {})  # `on:` vira True no YAML 1.1
    assert "timeout-minutes" in job
    assert not [step for step in job["steps"] if "login" in step.get("uses", "")]
    assert build["with"]["push"] is False and build["with"]["load"] is True
    assert build["with"]["tags"] == _services()["rag"]["image"]
    assert build["env"]["DOCKER_BUILD_RECORD_UPLOAD"] is False
    assert all(step["shell"] == "bash" for step in runs) and "${{" not in contract
    for probe in ("up -d --wait", '"postgres":"ok"', '"postgres":"down"', "pg_trgm,vector"):
        assert probe in contract

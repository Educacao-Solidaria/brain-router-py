"""A partida (`app/startup.py` e `brain-router migrate`) sem Postgres: espera com prazo, migrações
do pacote, ordem dos passos, erros e exec. Trava e seed no banco: test_integration_startup.py."""

import asyncio
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory

from app import log, startup
from app.__main__ import build_parser, main
from app.db import LockTimeout
from app.ingest import Report
from app.settings import Settings, get_settings

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = Settings(  # porta 1: conexão recusada na hora
    _env_file=None, database_url="postgresql://brain:senha@127.0.0.1:1/b", openrouter_api_key="k"
)
Steps = list[tuple[Any, ...]]


async def test_wait_returns_as_soon_as_the_database_answers() -> None:
    answers = iter([False, False, True, False])
    await startup.wait_for_database(lambda: asyncio.sleep(0, next(answers)), timeout=30, interval=0)
    assert list(answers) == [False]  # parou na primeira resposta


@pytest.mark.parametrize("delay", [0, 3600])  # recusa na hora; host que nem recusa nem responde
async def test_wait_gives_up_at_the_deadline_even_with_a_stuck_ping(delay: float) -> None:
    wait = startup.wait_for_database(lambda: asyncio.sleep(delay, False), timeout=0.05, interval=0)
    waiting = asyncio.create_task(wait)
    done, _ = await asyncio.wait({waiting}, timeout=10)  # sem prazo, a espera não acabaria
    waiting.cancel()
    assert waiting in done and isinstance(waiting.exception(), TimeoutError)


def test_packaged_migrations_are_those_of_alembic_ini_with_the_lock_deadline() -> None:
    config = startup.alembic_config(lock_timeout=7.0)
    packaged = ScriptDirectory.from_config(config)
    repo = ScriptDirectory.from_config(Config(str(ROOT / "alembic.ini")))

    assert config.config_file_name is None  # a imagem não leva o alembic.ini
    assert Path(packaged.dir) == Path(repo.dir) and packaged.get_heads() == repo.get_heads()
    assert config.attributes["lock_timeout"] == 7.0  # lido pelo env.py


def test_prepare_gives_up_on_a_database_that_never_answers() -> None:
    with pytest.raises(startup.StartupError) as error:
        startup.prepare(SETTINGS, timeout=0.3)  # o ping de verdade, contra a porta fechada
    assert str(error.value) == "o Postgres não aceitou conexão em 0.3 s"  # sem URL nem senha


@pytest.mark.parametrize(
    ("seed_dir", "failure"),
    [(None, None), (Path("/seed"), None), (Path("/seed"), "lock"), (Path("/seed"), "seed")],
)
def test_prepare_waits_migrates_seeds_and_turns_failures_into_startup_errors(
    monkeypatch: pytest.MonkeyPatch, seed_dir: Path | None, failure: str | None
) -> None:
    steps: Steps = []
    errors = {"lock": "a trava 'x' seguiu ocupada por 5.0 s", "seed": "o seed parou em a.md: 401"}

    async def wait(ping: object, *, timeout: float) -> None:
        steps.append(("wait", timeout))

    def upgrade(config: Config, revision: str) -> None:
        steps.append(("upgrade", revision, config.attributes["lock_timeout"]))
        if failure == "lock":  # outra réplica migrando além do prazo
            raise LockTimeout(errors["lock"])

    async def seed(directory: Path, **kwargs: Any) -> Report:
        steps.append(("seed", directory, kwargs["model"], kwargs["timeout"]))
        return Report(dry_run=False, failure=("a.md", "401") if failure == "seed" else None)

    monkeypatch.setattr(startup, "wait_for_database", wait)
    monkeypatch.setattr(command, "upgrade", upgrade)
    monkeypatch.setattr(startup, "seed", seed)
    try:
        startup.prepare(SETTINGS, timeout=5.0, seed_dir=seed_dir)
    except startup.StartupError as exc:
        assert str(exc) == errors.get(failure or "")
    else:
        assert failure is None

    expected: Steps = [("wait", 5.0), ("upgrade", "head", 5.0)]
    if seed_dir is not None and failure != "lock":  # com o modelo de embeddings do Settings
        expected.append(("seed", seed_dir, "openai/text-embedding-3-small", 5.0))
    assert steps == expected


class Exec(Exception): ...  # o exec de verdade não volta: o espião levanta esta exceção


@pytest.fixture
def cli(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Steps]:
    calls: Steps = []
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@db/b")  # nada conecta
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-x")
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()
    # O log de verdade trocaria o handler da raiz e pararia a thread de escrita de outro teste.
    monkeypatch.setattr(log, "configure_logging", lambda level, *, json: calls.append(("log",)))
    monkeypatch.setattr(log, "shutdown_logging", lambda: calls.append(("flush",)))

    def prepare(settings: Settings, *, timeout: float, seed_dir: Path | None) -> None:
        calls.append(("prepare", timeout, seed_dir))

    def execvp(file: str, args: list[str]) -> None:
        calls.append(("exec", file, args))
        raise Exec

    monkeypatch.setattr(startup, "prepare", prepare)
    monkeypatch.setattr(os, "execvp", execvp)
    yield calls
    get_settings.cache_clear()


def test_migrate_flushes_the_log_before_becoming_the_command(cli: Steps, tmp_path: Path) -> None:
    server = ["uvicorn", "app.api.server:create_server", "--factory", "--port", "8000"]
    timeout = build_parser().parse_args(["migrate"]).timeout

    with pytest.raises(Exec):
        main(["migrate", "--timeout", "5", "--seed", str(tmp_path), "--", *server])
    assert main(["migrate"]) == 0  # sem comando: só a partida

    assert timeout == startup.DEFAULT_TIMEOUT
    first = [("log",), ("prepare", 5.0, tmp_path), ("flush",), ("exec", "uvicorn", server)]
    assert cli == [*first, ("log",), ("prepare", timeout, None), ("flush",)]


def test_migrate_fails_with_1_and_the_reason_without_serving(
    cli: Steps, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fail(settings: Settings, **kwargs: Any) -> None:
        raise startup.StartupError("o Postgres não aceitou conexão em 5 s")

    monkeypatch.setattr(startup, "prepare", fail)

    assert main(["migrate", "--", "uvicorn"]) == 1
    assert capsys.readouterr().err == "migrate: o Postgres não aceitou conexão em 5 s\n"
    assert cli == [("log",), ("flush",)]  # sem exec


@pytest.mark.parametrize(  # o prazo é um número > 0 de segundos; a semente, uma pasta
    "args", [*(["--timeout", t] for t in ("0", "-1", "nan", "inf")), ["--seed", __file__]]
)
def test_migrate_refuses_a_bad_deadline_or_a_seed_that_is_not_a_folder(
    cli: Steps, args: list[str]
) -> None:
    with pytest.raises(SystemExit) as exit:
        main(["migrate", *args])
    assert exit.value.code == 2 and cli == []

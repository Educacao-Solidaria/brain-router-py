"""Auditoria de integridade sem banco: limites, caminho só de leitura, relatório e CLI."""

import math
from collections.abc import Iterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from app import integrity
from app.__main__ import main
from app.integrity import CHECKS, Finding, Report, TooManyChanges, check_integrity, render
from app.settings import get_settings

NAMES = [check.name for check in CHECKS]


def _findings(**counts: int) -> dict[str, Finding]:
    return {
        name: Finding(counts.get(name, 0), [f"{name}-1"] * min(counts.get(name, 0), 1))
        for name in NAMES
    }


def test_only_the_safe_checks_have_a_fix() -> None:
    # Sem embedding e tenants misturados só se relatam; o zerado vira NULL (UPDATE).
    fixable = {check.name: (check.fix or "").split()[0] for check in CHECKS if check.fix}
    assert fixable == dict(
        chunks_orfaos="DELETE", documentos_sem_chunks="DELETE", embeddings_zerados="UPDATE"
    )


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"min_age_hours": -1.0}, "min_age_hours"),
        ({"min_age_hours": math.nan}, "min_age_hours"),
        ({"min_age_hours": math.inf}, "min_age_hours"),
        ({"max_changes": -1}, "max_changes"),
        ({"max_changes": True}, "max_changes"),
    ],
)
async def test_refuses_degenerate_limits_before_touching_the_database(
    options: dict[str, Any], message: str
) -> None:
    db = MagicMock()
    with pytest.raises(ValueError, match=message):
        await check_integrity(db, fix=True, **options)
    db.read.assert_not_called()
    db.session.assert_not_called()


async def test_default_mode_only_reads() -> None:
    db = MagicMock()
    db.read = AsyncMock(return_value=_findings(chunks_sem_embedding=2))
    db.session = MagicMock(side_effect=AssertionError("o modo padrão não abre sessão de escrita"))

    report = await check_integrity(db)

    db.read.assert_awaited_once()
    assert report.changed is None and report.after is None and report.remaining == 2


async def test_repair_stops_when_more_rows_than_the_limit_arrived_meanwhile() -> None:
    # A contagem de antes passou no limite, mas chegou mais caso entre ela e a correção.
    conn = MagicMock()
    conn.execute = AsyncMock(return_value=MagicMock(rowcount=600))
    session = MagicMock()
    session.connection = AsyncMock(return_value=conn)

    with pytest.raises(TooManyChanges, match="mudaria 1200 linhas"):
        await integrity.repair(session, {"min_age": 3600.0}, max_changes=1000)
    assert conn.execute.await_count == 2  # parou na segunda correção; quem chamou desfaz


def test_read_only_report_lists_counts_and_examples() -> None:
    lines = render(Report(before=_findings(documentos_sem_chunks=3))).splitlines()

    assert lines[0].startswith("Integridade de documentos e vetores (somente leitura")
    assert lines[2].split() == ["verificação", "casos", "exemplos"]
    assert lines[4].split() == ["documentos_sem_chunks", "3", "documentos_sem_chunks-1"]
    assert lines[3].split() == ["chunks_orfaos", "0"]


def test_fix_report_shows_before_changed_and_after() -> None:
    report = Report(
        before=_findings(chunks_orfaos=2, chunks_sem_embedding=1),
        changed={"chunks_orfaos": 2, "documentos_sem_chunks": 0, "embeddings_zerados": 0},
        after=_findings(chunks_sem_embedding=1),
    )
    lines = render(report).splitlines()

    assert "--fix" in lines[0]
    assert lines[3].split() == ["chunks_orfaos", "2", "2", "0"]
    assert lines[6].split() == ["chunks_sem_embedding", "1", "-", "1"]  # só relata
    assert report.remaining == 1


@pytest.mark.parametrize(
    "argv",
    [
        ["check-integrity", "--min-age-hours", "-1"],
        ["check-integrity", "--min-age-hours", "nan"],
        ["check-integrity", "--max-changes", "-1"],
        ["check-integrity", "--max-changes", "mil"],
        ["check-integrity", "--nope"],
    ],
)
def test_cli_refuses_bad_arguments(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(argv)
    assert exit_info.value.code == 2


@pytest.fixture
def settings_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@db/b")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-x")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.mark.parametrize(
    ("argv", "report", "code"),
    [
        (["check-integrity"], Report(before=_findings()), 0),
        (["check-integrity"], Report(before=_findings(chunks_sem_embedding=4)), 1),
        (
            ["check-integrity", "--fix", "--max-changes", "5", "--min-age-hours", "0.5"],
            Report(before=_findings(), changed={}, after=_findings()),
            0,
        ),
    ],
)
def test_cli_prints_the_report_and_exits_by_what_remains(
    argv: list[str],
    report: Report,
    code: int,
    settings_env: None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake = AsyncMock(return_value=report)
    monkeypatch.setattr(integrity, "check_integrity", fake)

    assert main(argv) == code
    assert capsys.readouterr().out == render(report)
    assert fake.await_args is not None
    options = fake.await_args.kwargs
    assert options["fix"] is ("--fix" in argv)
    if "--fix" in argv:
        assert (options["max_changes"], options["min_age_hours"]) == (5, 0.5)


def test_cli_reports_a_refused_fix_without_a_traceback(
    settings_env: None, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    refused = AsyncMock(side_effect=TooManyChanges(1500, 1000))
    monkeypatch.setattr(integrity, "check_integrity", refused)
    assert main(["check-integrity", "--fix"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "mudaria 1500 linhas, acima do limite de 1000; nada mudou" in captured.err

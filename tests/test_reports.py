"""Relatório de uso sem banco; as consultas e o EXPLAIN estão em test_integration_reports.py."""

from unittest.mock import AsyncMock

import pytest

from app import reports
from app.__main__ import main
from app.reports import Report, Usage, render
from app.settings import get_settings


def test_render_has_one_aligned_line_per_collection_and_tenant_without_terminal_controls() -> None:
    report = Report(
        [Usage("default", "default", 120, 3), Usage("c" + chr(27) + "[2J", "t", 1, 0)], 2
    )
    lines = render(report).splitlines()
    assert lines[:2] == ["Uso por coleção e tenant (somente leitura)", ""]
    assert lines[3].split() == ["default", "default", "120", "3"]
    assert lines[4].split() == ["c" + chr(92) + "x1b[2J", "t", "1", "0"]  # o escape, não o ESC
    assert len({len(line) for line in lines[2:5]}) == 1  # colunas alinhadas
    assert lines[-1] == "chunks com embedding zerado: 2"
    assert render(Report([], 0)).splitlines()[3] == "(nenhum documento)"


def test_cli_prints_the_report(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@db/b")  # nada conecta
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-x")
    get_settings.cache_clear()
    report = Report([Usage("default", "t", 2, 1)], 0)
    monkeypatch.setattr(reports, "usage_report", fake := AsyncMock(return_value=report))
    try:
        assert main(["report"]) == 0
    finally:
        get_settings.cache_clear()
    assert capsys.readouterr().out == render(report) and fake.await_count == 1

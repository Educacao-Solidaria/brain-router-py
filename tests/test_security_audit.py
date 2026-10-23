"""Política da auditoria de segurança (app/security_audit.py) e o workflow que a roda no CI."""

import http.client
import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest
import yaml

from app import security_audit as sa

ROOT = Path(__file__).resolve().parents[1]
TODAY = date(2026, 10, 22)


def _deps(*vulns: dict[str, Any]) -> dict[str, Any]:
    return {"dependencies": [{"name": "pkg", "version": "1.0", "vulns": list(vulns)}]}


def _vuln(vuln_id: str, *aliases: str, fixes: tuple[str, ...] = ("2.0",)) -> dict[str, Any]:
    return {"id": vuln_id, "aliases": list(aliases), "fix_versions": list(fixes)}


def _bandit(*issues: tuple[str, str, str], loc: int = 100) -> dict[str, Any]:
    same = {"test_name": "t", "line_number": 7, "issue_confidence": "HIGH", "issue_text": "a|b\nc"}
    results = [
        {**same, "test_id": test, "issue_severity": severity, "filename": filename}
        for test, severity, filename in issues
    ]
    return {"errors": [], "results": results, "metrics": {"_totals": {"loc": loc}}}


@pytest.mark.parametrize(
    ("severity", "fixes", "blocking"),
    [
        ("CRITICAL", ("2.0",), True),
        ("HIGH", ("2.0", "1.9.1"), True),
        ("UNKNOWN", ("2.0",), True),  # sem severidade do GHSA: falha fechada
        ("MEDIUM", ("2.0",), False),
        ("CRITICAL", (), False),  # sem correção publicada não reprova (contrato §6)
    ],
)
def test_fixable_high_or_critical_blocks(severity: str, fixes: Any, blocking: bool) -> None:
    report = _deps(_vuln("PYSEC-1", fixes=fixes))
    (finding,) = sa.audit_dependencies(report, [], TODAY, lambda _: severity)
    assert (finding.blocking, finding.where, finding.severity) == (blocking, "pkg 1.0", severity)


@pytest.mark.parametrize(
    ("waived", "review", "blocking"),
    [
        ("GHSA-a", date(2026, 10, 22), False),  # vale pelo alias, até o dia da revisão inclusive
        ("PYSEC-1", date(2027, 1, 1), False),
        ("GHSA-a", date(2026, 10, 21), True),  # vencida: o achado volta a reprovar
        ("GHSA-b", date(2027, 1, 1), True),  # exceção de outro aviso
    ],
)
def test_dependency_waiver_until_its_review_date(waived: str, review: date, blocking: bool) -> None:
    report, waiver = _deps(_vuln("PYSEC-1", "GHSA-a")), sa.Waiver((waived,), "motivo", review)
    (finding,) = sa.audit_dependencies(report, [waiver], TODAY, lambda _: "HIGH")
    assert finding.blocking is blocking
    assert ("motivo" in finding.note) is (waived != "GHSA-b")


def test_malicious_package_blocks_even_without_a_fix_or_severity() -> None:
    report = _deps(_vuln("MAL-2026-1", fixes=()))
    assert sa.audit_dependencies(report, [], TODAY, lambda _: "LOW")[0].blocking


@pytest.mark.parametrize(
    ("report", "message"),
    [
        ({"dependencies": []}, "nenhuma dependência"),
        ({"dependencies": [{"name": "pkg", "skip_reason": "não encontrado"}]}, "não auditou pkg"),
        ({"dependencies": [{"name": "pkg", "version": "1.0"}]}, "formato inesperado"),
        ([], "formato inesperado"),
    ],
)
def test_dependency_report_not_covering_the_lock_fails(report: Any, message: str) -> None:
    with pytest.raises(sa.AuditError, match=message):
        sa.audit_dependencies(report, [], TODAY, lambda _: "LOW")


def test_ghsa_severity_is_the_highest_ghsa_and_asks_only_for_ghsa_ids() -> None:
    records, asked = {"GHSA-a": "MODERATE", "GHSA-b": "HIGH", "GHSA-c": "LOW"}, []

    def fetch(vuln_id: str) -> Any:
        asked.append(vuln_id)
        return {"database_specific": {"severity": records[vuln_id]}}

    assert sa.ghsa_severity(["PYSEC-1", "GHSA-a", "GHSA-b", "CVE-1", "GHSA-c"], fetch) == "HIGH"
    assert asked == ["GHSA-a", "GHSA-b", "GHSA-c"]
    assert sa.ghsa_severity(["GHSA-a"], fetch) == "MEDIUM"  # o MODERATE do GHSA


@pytest.mark.parametrize(
    "outcome",
    [
        OSError("rede"),
        http.client.IncompleteRead(b""),
        {},
        {"database_specific": None},
        {"database_specific": {"severity": "IMPORTANT"}},
    ],
)
def test_ghsa_severity_is_unknown_without_a_readable_ghsa_record(outcome: Any) -> None:
    def fetch(vuln_id: str) -> Any:
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    assert sa.ghsa_severity(["PYSEC-1", "GHSA-a"], fetch) == "UNKNOWN"


@pytest.mark.parametrize(
    ("level", "test", "path", "blocking"),
    [
        ("HIGH", "B324", "app/x.py", False),  # a exceção vale só para o teste e o caminho dela
        ("HIGH", "B324", "app/y.py", True),
        ("HIGH", "B602", "app/x.py", True),
        ("MEDIUM", "B602", "app/x.py", False),  # o bandit não tem CRITICAL: só HIGH reprova
    ],
)
def test_code_blocks_high_unless_waived(level: str, test: str, path: str, blocking: bool) -> None:
    waiver = sa.Waiver((test, path), "motivo", TODAY)
    (finding,) = sa.audit_code(_bandit(("B324", level, "./app/x.py")), [waiver], TODAY)
    assert (finding.blocking, finding.where) == (blocking, "app/x.py:7")


@pytest.mark.parametrize(
    ("report", "message"),
    [
        ({**_bandit(), "errors": [{"filename": "z.py", "reason": "erro"}]}, "não leu: z.py"),
        (_bandit(loc=0), "nenhuma linha"),  # caminho errado ou vazio
        ({"results": []}, "formato inesperado"),
    ],
)
def test_code_report_with_unread_files_or_nothing_scanned_fails(report: Any, message: str) -> None:
    with pytest.raises(sa.AuditError, match=message):
        sa.audit_code(report, [], TODAY)


@pytest.mark.parametrize(
    "text",
    [
        '[[bandit]]\ntest = "B1"\npath = "app/x.py"\nreview = 2027-01-01',  # sem motivo
        '[[bandit]]\ntest = "B1"\npath = "app/x.py"\nreason = " "\nreview = 2027-01-01',
        '[[pip-audit]]\nid = "GHSA-a"\nreason = "r"\nreview = "2027-01-01"',  # data como texto
        '[[pip-audit]]\nid = "GHSA-a"\nreason = "r"\nreview = 2027-01-01T00:00:00',
        '[[pip-audit]]\nid = "GHSA-a"\nreason = "r"\nreview = 2027-01-01\nall = true',
        "pip-audit = 5",
        '[[gosec]]\nid = "G1"',
        "id = ",
    ],
)
def test_malformed_waiver_is_an_error_not_a_silent_exception(tmp_path: Path, text: str) -> None:
    (tmp_path / "x.toml").write_text(text, encoding="utf-8")
    with pytest.raises(sa.AuditError):
        sa.load_waivers(tmp_path / "x.toml")


def test_main_prints_the_report_and_exits_by_the_verdict(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (exceptions := tmp_path / "e.toml").write_text("", encoding="utf-8")
    report = tmp_path / "r.json"
    args = [str(report), "--exceptions", str(exceptions)]
    # O OSV repete o aviso; o sem correção vem antes na entrada, mas o bloqueante abre a tabela.
    dup = _deps(_vuln("PYSEC-0", fixes=()), _vuln("PYSEC-1", "GHSA-a"), _vuln("PYSEC-1", "GHSA-a"))
    report.write_text(json.dumps(dup), encoding="utf-8")

    def critical(vuln_id: str) -> Any:
        return {"database_specific": {"severity": "CRITICAL"}}

    assert sa.main(["pip-audit", *args], today=TODAY, fetch=critical) == 1
    out = capsys.readouterr().out
    assert "**reprovado** (1 bloqueante(s))" in out
    assert "| PYSEC-1 GHSA-a | pkg 1.0 | CRITICAL | correção: 2.0 | **reprova** |" in out
    assert out.index("**reprova**") < out.index("| PYSEC-0 |")
    report.write_text(json.dumps(_bandit(("B608", "MEDIUM", "app/x.py"))), encoding="utf-8")
    assert sa.main(["bandit", *args], today=TODAY) == 0
    out = capsys.readouterr().out
    assert "passou" in out and "| a\\|b c (confiança HIGH) | não reprova |" in out
    report.write_text("{", encoding="utf-8")  # o relatório cortado no meio
    assert sa.main(["bandit", *args]) == 2
    assert sa.main(["bandit", str(tmp_path / "nada.json"), "--exceptions", str(exceptions)]) == 2
    assert capsys.readouterr().out.count("**não concluída**") == 2
    assert sa.render("bandit", [], "10 linhas").endswith("\n\nNenhum achado.\n")


def test_workflow_runs_pinned_tools_on_every_pr_on_main_and_daily() -> None:
    assert set(sa.load_waivers(ROOT / "audit-exceptions.toml")) == {"pip-audit", "bandit"}
    workflow = yaml.safe_load((ROOT / ".github/workflows/audit.yml").read_text(encoding="utf-8"))
    triggers, (job,) = workflow[True], workflow["jobs"].values()  # `on:` vira True no YAML 1.1
    assert triggers["push"] == {"branches": ["main"]} and triggers["pull_request"] is None
    assert triggers["schedule"] and "workflow_dispatch" in triggers
    assert workflow["permissions"] == {"contents": "read"} and "permissions" not in job
    assert job["timeout-minutes"] <= 10
    assert "github.event_name" in workflow["concurrency"]["group"]  # push não cancela o diário
    (uv,) = [s for s in job["steps"] if s.get("uses", "").startswith("astral-sh/setup-uv@")]
    assert uv["with"]["enable-cache"] is False  # a chave do uv.lock é uma só para todos os jobs
    dependencies, code = [s for s in job["steps"] if "app.security_audit" in s.get("run", "")]
    assert code["if"] == "${{ !cancelled() }}"  # o bandit roda mesmo com o pip-audit reprovado
    for step in (dependencies, code):
        assert step["shell"] == "bash"  # pipefail: sem ele o tee devolveria 0 com o achado
        assert step["run"].rstrip().endswith('| tee -a "$GITHUB_STEP_SUMMARY"')
        # A ferramenta sai 1 com qualquer achado (LOW inclusive): quem reprova é a política.
        assert " ".join(step["run"].split()).count(" || true ") == 1 and "${{" not in step["run"]
    run, uvx = " ".join(dependencies["run"].split()), "uvx --python 3.12 "
    assert "uv export --frozen --all-extras" in run and "--vulnerability-service osv" in run
    assert uvx + "pip-audit==2.10.1 " in run
    assert uvx + "bandit==1.9.4 --recursive app --ignore-nosec --ini /dev/null " in code["run"]

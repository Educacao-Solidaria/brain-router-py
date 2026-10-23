"""Política da auditoria de segurança do CI (workflow `Auditoria`) sobre o JSON do pip-audit e do
bandit: reprova achado HIGH/CRITICAL com correção disponível (contrato §6 da Fase 5). Exceção só no
`audit-exceptions.toml`, com motivo e data de revisão; passada a data, o achado volta a reprovar.
Imprime o relatório em Markdown e sai 0, 1 com bloqueante ou 2 sem conclusão (relatório ausente ou
incompleto, exceção mal escrita). Só stdlib: roda com `uv run --no-project`, sem instalar o projeto.
"""

import argparse
import http.client
import json
import tomllib
import urllib.parse
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path, PurePosixPath
from typing import Any

EXCEPTIONS = Path("audit-exceptions.toml")
BLOCKING = frozenset({"HIGH", "CRITICAL", "UNKNOWN"})  # UNKNOWN = sem severidade do GHSA
_RANK = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3, "UNKNOWN": 4}
# Todos obrigatórios: o alvo da exceção, o motivo e a data de revisão (sempre os dois últimos).
_FIELDS = {"pip-audit": ("id", "reason", "review"), "bandit": ("test", "path", "reason", "review")}
Fetch = Callable[[str], Any]


class AuditError(Exception):
    """A auditoria não conclui: relatório ausente ou incompleto, ou exceção mal escrita."""


@dataclass(frozen=True)
class Waiver:
    target: tuple[str, ...]  # pip-audit: (id ou alias do aviso,); bandit: (teste, caminho)
    reason: str
    review: date


@dataclass(frozen=True)
class Finding:
    rule: str
    where: str
    severity: str
    detail: str
    blocking: bool
    note: str = ""


def load_waivers(path: Path) -> dict[str, list[Waiver]]:
    """As exceções por ferramenta; campo faltando, sobrando ou vazio é erro, não exceção a mais."""
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise AuditError(f"{path}: {exc}") from exc
    if unknown := sorted(set(data) - set(_FIELDS)):
        raise AuditError(f"{path}: seção desconhecida {unknown}")
    waivers: dict[str, list[Waiver]] = {tool: [] for tool in _FIELDS}
    for tool, fields in _FIELDS.items():
        entries = data.get(tool, [])
        for n, entry in enumerate(entries if isinstance(entries, list) else [entries], 1):
            if not (
                isinstance(entry, dict)
                and set(entry) == set(fields)
                and all(isinstance(entry[f], str) and entry[f].strip() for f in fields[:-1])
                and type(entry["review"]) is date  # data e hora (datetime) não vale
            ):
                msg = f"[[{tool}]] nº {n} exige só {fields}, textos não vazios e review = data"
                raise AuditError(f"{path}: {msg} TOML (ex.: review = 2027-01-31)")
            target = tuple(entry[f] for f in fields[:-2])
            waivers[tool].append(Waiver(target, entry["reason"], entry["review"]))
    return waivers


def _decide(blocking: bool, matches: Sequence[Waiver], today: date) -> tuple[bool, str]:
    """Exceção vigente (até o dia da revisão, inclusive) tira o bloqueio; vencida, só anota."""
    for waiver in matches:
        if today <= waiver.review:
            return False, f"exceção até {waiver.review}: {waiver.reason}"
    if matches:
        return blocking, f"exceção VENCIDA em {matches[0].review}: {matches[0].reason}"
    return blocking, ""


def fetch_osv(vuln_id: str) -> Any:
    """O registro do aviso no OSV (`GET https://api.osv.dev/v1/vulns/<id>`, TLS verificado)."""
    conn = http.client.HTTPSConnection("api.osv.dev", timeout=15)
    try:
        conn.request("GET", "/v1/vulns/" + urllib.parse.quote(vuln_id, safe=""))
        response = conn.getresponse()
        if response.status != 200:
            raise OSError(f"OSV respondeu {response.status} para {vuln_id}")
        return json.loads(response.read())
    finally:
        conn.close()


def ghsa_severity(ids: Sequence[str], fetch: Fetch = fetch_osv) -> str:
    """A maior severidade do GHSA (a escala do Dependabot) entre o id e os aliases do aviso.

    ponytail: só o registro GHSA traz a severidade pronta; aviso sem GHSA, ou com o OSV fora do
    ar, fica UNKNOWN e reprova se tiver correção. Upgrade: a nota do vetor CVSS que o OSV publica.
    """
    levels: list[str] = []
    for vuln_id in ids:
        if not vuln_id.startswith("GHSA-"):
            continue
        try:
            level = fetch(vuln_id)["database_specific"]["severity"]
        except (OSError, ValueError, LookupError, TypeError, http.client.HTTPException):
            continue
        level = "MEDIUM" if level == "MODERATE" else level
        if level in ("LOW", "MEDIUM", "HIGH", "CRITICAL"):
            levels.append(level)
    return max(levels, key=_RANK.__getitem__, default="UNKNOWN")


def audit_dependencies(
    report: Any, waivers: Sequence[Waiver], today: date, severity: Callable[[Sequence[str]], str]
) -> list[Finding]:
    """Reprova aviso HIGH/CRITICAL (ou sem severidade conhecida) com correção publicada."""
    findings: list[Finding] = []
    try:
        if not report["dependencies"]:
            raise AuditError("o relatório do pip-audit não tem nenhuma dependência")
        for dep in report["dependencies"]:
            if "skip_reason" in dep:
                raise AuditError(f"o pip-audit não auditou {dep['name']}: {dep['skip_reason']}")
            for vuln in {v["id"]: v for v in dep["vulns"]}.values():  # o OSV repete aviso
                ids, fixes = [vuln["id"], *vuln["aliases"]], vuln["fix_versions"]
                level = severity(ids)
                matches = [w for w in waivers if w.target[0] in ids]
                blocking, note = _decide(bool(fixes) and level in BLOCKING, matches, today)
                fixed = "correção: " + ", ".join(fixes) if fixes else "sem correção publicada"
                where = f"{dep['name']} {dep['version']}"
                findings.append(Finding(" ".join(ids), where, level, fixed, blocking, note))
    except (KeyError, TypeError) as exc:
        raise AuditError(f"relatório do pip-audit em formato inesperado ({exc!r})") from exc
    return findings


def audit_code(report: Any, waivers: Sequence[Waiver], today: date) -> list[Finding]:
    """Reprova severidade HIGH (o bandit não tem CRITICAL) e arquivo que o bandit não leu."""
    findings: list[Finding] = []
    try:
        if report["errors"]:
            unread = "; ".join(f"{e['filename']} ({e['reason']})" for e in report["errors"])
            raise AuditError(f"o bandit não leu: {unread}")
        if not report["metrics"]["_totals"]["loc"]:
            raise AuditError("o bandit não varreu nenhuma linha")
        for issue in report["results"]:
            path = str(PurePosixPath(issue["filename"]))
            matches = [w for w in waivers if w.target == (issue["test_id"], path)]
            blocking, note = _decide(issue["issue_severity"] == "HIGH", matches, today)
            rule = f"{issue['test_id']} {issue['test_name']}"
            detail = f"{issue['issue_text']} (confiança {issue['issue_confidence']})"
            where = f"{path}:{issue['line_number']}"
            findings.append(Finding(rule, where, issue["issue_severity"], detail, blocking, note))
    except (KeyError, TypeError) as exc:
        raise AuditError(f"relatório do bandit em formato inesperado ({exc!r})") from exc
    return findings


def render(tool: str, findings: Sequence[Finding], scope: str) -> str:
    blocking = sum(f.blocking for f in findings)
    verdict = f"**reprovado** ({blocking} bloqueante(s))" if blocking else "passou"
    lines = [f"### Auditoria ({tool}): {verdict}", "", f"{scope}. Exceções: `{EXCEPTIONS}`.", ""]
    if not findings:
        return "\n".join([*lines, "Nenhum achado.", ""])
    lines += ["| Achado | Onde | Severidade | Detalhe | Situação |", "|---|---|---|---|---|"]
    # Bloqueantes primeiro; na mesma severidade, a ordem da ferramenta (pacote; arquivo e linha).
    for f in sorted(findings, key=lambda f: (not f.blocking, -_RANK.get(f.severity, -1))):
        status = " — ".join(filter(None, ("**reprova**" if f.blocking else "não reprova", f.note)))
        row = (f.rule, f.where, f.severity, f.detail, status)
        lines.append("| " + " | ".join(" ".join(c.split()).replace("|", "\\|") for c in row) + " |")
    return "\n".join(lines) + "\n"


def main(
    argv: Sequence[str] | None = None, *, today: date | None = None, fetch: Fetch = fetch_osv
) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.security_audit", description=__doc__)
    parser.add_argument("tool", choices=sorted(_FIELDS))
    parser.add_argument("report", type=Path, help="o relatório JSON do pip-audit ou do bandit")
    parser.add_argument("--exceptions", type=Path, default=EXCEPTIONS)
    args = parser.parse_args(argv)
    day = today or datetime.now(UTC).date()
    try:
        waivers = load_waivers(args.exceptions)[args.tool]
        report = json.loads(args.report.read_text(encoding="utf-8"))
        if args.tool == "pip-audit":
            findings = audit_dependencies(report, waivers, day, lambda i: ghsa_severity(i, fetch))
            n = len(report["dependencies"])
            scope = f"{n} pacotes no OSV; reprova HIGH/CRITICAL/UNKNOWN com correção publicada"
        else:
            findings = audit_code(report, waivers, day)
            scope = f"{report['metrics']['_totals']['loc']} linhas; reprova severidade HIGH"
    except (AuditError, OSError, ValueError) as exc:  # relatório ausente ou cortado também
        print(f"### Auditoria ({args.tool}): **não concluída** — {exc}")
        return 2
    print(render(args.tool, findings, scope), end="")
    return 1 if any(f.blocking for f in findings) else 0


if __name__ == "__main__":
    raise SystemExit(main())

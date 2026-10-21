"""O dashboard do Grafana (deploy/grafana/brain-router.json) contra o que o `/metrics` exporta."""

import json
import re
from pathlib import Path

import pytest
from prometheus_client.parser import text_string_to_metric_families

from app import metrics

ROOT = Path(__file__).resolve().parents[1]
DASHBOARD = ROOT / "deploy" / "grafana" / "brain-router.json"
SOURCE = {"type": "prometheus", "uid": "${DS_PROMETHEUS}"}
# As séries que o texto escreve por tipo de família (o parser devolve o contador sem `_total`).
SERIES = {"counter": ("_total",), "histogram": ("_bucket", "_sum", "_count")}
# Sem strings, seletores de label e de intervalo e cláusulas de agrupamento, todo identificador
# que sobra e não é chamada de função nem palavra-chave é métrica (o mesmo critério do
# TestDashboard do gateway). A sintaxe da PromQL não passa por aqui: o `promqlcheck` (parser
# oficial do Prometheus) confere à parte.
_DROP = re.compile(r'"(?:[^"\\]|\\.)*"|\{[^}]*\}|\[[^\]]*\]')
_GROUP = re.compile(r"\b(?:by|without|on|ignoring|group_left|group_right)\s*\([^)]*\)")
_IDENT = re.compile(r"\b[a-zA-Z_][a-zA-Z0-9_:]*(\s*\()?")
_WORDS = {"bool", "offset", "and", "or", "unless", "inf", "nan", "group_left", "group_right"}


def metric_names(expr: str) -> list[str]:
    """As métricas citadas numa expressão PromQL."""
    expr = _GROUP.sub("", _DROP.sub("", expr))
    return [m[0] for m in _IDENT.finditer(expr) if m[1] is None and m[0].lower() not in _WORDS]


def exported() -> set[str]:
    """As séries que o `/metrics` escreve, tiradas do próprio texto (família sem amostra conta)."""
    return {
        family.name + suffix
        for family in text_string_to_metric_families(metrics.render().decode())
        for suffix in SERIES.get(family.type, ("",))
    }


@pytest.mark.parametrize(
    ("expr", "names"),
    [
        (
            'histogram_quantile(0.95, sum by (le, mode) (rate(a_bucket{x=~"y|z"}[$__range])))',
            ["a_bucket"],
        ),
        (
            'sum(rate(a_total{x!="y"}[5m])) / sum without (code) (b_total offset 1h)',
            ["a_total", "b_total"],
        ),
        (
            "topk(10, c_total) > bool 0 and on (mode) group_left d_total unless e:rate5m",
            ["c_total", "d_total", "e:rate5m"],
        ),
        ("min by (metric) (brain_ruoter_eval_score) * 1e3", ["brain_ruoter_eval_score"]),
    ],
)
def test_metric_names_finds_every_metric_of_an_expression(expr: str, names: list[str]) -> None:
    assert metric_names(expr) == names


def test_only_the_series_the_exposition_writes_count_as_exported() -> None:
    names = exported()  # o controle negativo da conferência do dashboard
    assert "brain_router_search_requests_total" in names
    assert "brain_router_search_requests" not in names, "contador só existe com _total"
    assert "brain_router_result_score_bucket" in names
    assert "brain_router_result_score" not in names, "histograma só tem _bucket, _sum e _count"
    assert "brain_router_eval_score" in names
    assert "brain_ruoter_eval_score" not in names


def test_the_dashboard_cites_only_exported_metrics_through_the_datasource_variable() -> None:
    dashboard = json.loads(DASHBOARD.read_text(encoding="utf-8"))
    assert dashboard["uid"] == "brain-router"
    variables = [(v["name"], v["type"], v["query"]) for v in dashboard["templating"]["list"]]
    assert variables == [("DS_PROMETHEUS", "datasource", "prometheus")]

    names, cited, panels = exported(), set(), dashboard["panels"]
    assert len({panel["id"] for panel in panels}) == len(panels), "id de painel repetido"
    for panel in panels:
        title = panel["title"]
        assert panel["datasource"] == SOURCE, title
        assert panel["targets"], title
        for target in panel["targets"]:
            assert target["datasource"] == SOURCE, title
            found = metric_names(target["expr"])
            assert found, f"{title}: nenhuma métrica em {target['expr']!r}"
            assert set(found) <= names, f"{title}: {set(found) - names} não sai no /metrics"
            cited.update(found)
        if any("brain_router_result_score" in target["expr"] for target in panel["targets"]):
            # A relevância é a do reranker: o painel tem de avisar que sem ele fica vazio.
            assert "vazio sem reranker" in panel["description"], title
    assert "brain_router_eval_score" in cited  # a evolução dos scores de eval

"""Garante que CI, pre-commit e o script local não divergem entre si."""

import tomllib
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _load_yaml(relative: str) -> Any:
    return yaml.safe_load((ROOT / relative).read_text(encoding="utf-8"))


def _ci_commands() -> list[str]:
    steps = _load_yaml(".github/workflows/ci.yml")["jobs"]["check"]["steps"]
    return [step["run"] for step in steps if "run" in step]


def _script_commands() -> list[str]:
    lines = (ROOT / "scripts/check.sh").read_text(encoding="utf-8").splitlines()
    return [line for line in lines if line.startswith("uv ")]


def test_console_script_points_to_cli() -> None:
    (script,) = entry_points(group="console_scripts", name="brain-router")
    assert script.value == "app.__main__:main"


def test_check_script_runs_the_same_gates_as_ci() -> None:
    assert _script_commands() == _ci_commands()


def test_pre_commit_covers_lint_format_and_types() -> None:
    config = _load_yaml(".pre-commit-config.yaml")
    hook_ids = {hook["id"] for repo in config["repos"] for hook in repo["hooks"]}
    assert {"ruff-check", "ruff-format", "mypy", "uv-lock", "detect-private-key"} <= hook_ids


def test_mypy_runs_in_strict_mode() -> None:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert config["tool"]["mypy"]["strict"] is True

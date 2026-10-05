from importlib.metadata import version

import pytest

import app
from app.__main__ import main


def test_version_matches_installed_distribution() -> None:
    # Garante que pyproject.toml e app.__version__ não divergem.
    assert app.__version__ == version("brain-router-py")


def test_cli_prints_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])

    assert exc.value.code == 0
    assert capsys.readouterr().out.strip() == f"brain-router {app.__version__}"


def test_cli_without_arguments_exits_cleanly() -> None:
    assert main([]) == 0


def test_cli_rejects_unknown_flag(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--nope"])

    assert exc.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err

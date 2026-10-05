from pathlib import Path

import pytest
from pydantic import ValidationError

from app.settings import Settings, get_settings

ROOT = Path(__file__).resolve().parents[1]
DB_PASSWORD = "s3nha-do-banco"
API_KEY = "sk-or-v1-chave-secreta"


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Sem .env do dev e sem variáveis herdadas do shell influenciando o teste.
    monkeypatch.chdir(tmp_path)
    for name in Settings.model_fields:
        monkeypatch.delenv(name.upper(), raising=False)
    monkeypatch.setenv("DATABASE_URL", f"postgresql://brain:{DB_PASSWORD}@db:5432/brain_router")
    monkeypatch.setenv("OPENROUTER_API_KEY", API_KEY)
    get_settings.cache_clear()


def test_defaults_and_driver_normalization() -> None:
    settings = Settings()

    assert settings.app_env == "development"
    assert not settings.is_production
    assert settings.db_pool_pre_ping is True
    assert settings.database_url.get_secret_value() == (
        f"postgresql+asyncpg://brain:{DB_PASSWORD}@db:5432/brain_router"
    )
    assert str(settings.openrouter_base_url) == "https://openrouter.ai/api/v1"


def test_secrets_never_show_in_repr_or_dump() -> None:
    settings = Settings()
    rendered = " ".join([repr(settings), str(settings), str(settings.model_dump())])

    assert DB_PASSWORD not in rendered
    assert API_KEY not in rendered
    assert settings.openrouter_api_key.get_secret_value() == API_KEY


def test_reads_pool_tuning_and_case_insensitive_log_level(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DB_POOL_SIZE", "20")
    monkeypatch.setenv("DB_POOL_RECYCLE", "-1")
    monkeypatch.setenv("LOG_LEVEL", "debug")
    monkeypatch.setenv("APP_ENV", "production")

    settings = Settings()

    assert (settings.db_pool_size, settings.db_pool_recycle) == (20, -1)
    assert settings.log_level == "DEBUG"
    assert settings.is_production


@pytest.mark.parametrize(
    "url",
    [
        f"mysql://brain:{DB_PASSWORD}@db/brain_router",
        f"postgresql://brain:{DB_PASSWORD}@db",
        "not-a-url",
        f"postgresql://brain:{DB_PASSWORD}@db:porta/brain_router",
    ],
)
def test_invalid_database_url_is_rejected_without_leaking_it(
    monkeypatch: pytest.MonkeyPatch, url: str
) -> None:
    monkeypatch.setenv("DATABASE_URL", url)

    with pytest.raises(ValidationError) as exc:
        Settings()

    assert DB_PASSWORD not in str(exc.value)


@pytest.mark.parametrize(
    ("name", "value"),
    [("OPENROUTER_API_KEY", "   "), ("DB_POOL_SIZE", "0"), ("APP_ENV", "staging")],
)
def test_invalid_values_are_rejected(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str
) -> None:
    monkeypatch.setenv(name, value)

    with pytest.raises(ValidationError):
        Settings()


def test_missing_api_key_fails_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY")

    with pytest.raises(ValidationError, match="openrouter_api_key"):
        Settings()


def test_env_example_is_loadable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATABASE_URL")
    monkeypatch.delenv("OPENROUTER_API_KEY")

    settings = Settings(_env_file=ROOT / ".env.example")

    assert settings.db_pool_recycle == 1800
    assert settings.database_url.get_secret_value().startswith("postgresql+asyncpg://")


def test_get_settings_is_cached() -> None:
    assert get_settings() is get_settings()

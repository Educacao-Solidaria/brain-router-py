"""Configuração da aplicação lida de variáveis de ambiente (e de `.env`, se existir).

Segredos (`DATABASE_URL`, `OPENROUTER_API_KEY`) são `SecretStr`: aparecem como
`'**********'` em `repr`, `str`, logs e `model_dump()`. O valor real só sai por
`get_secret_value()` — e só no ponto em que é consumido.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, HttpUrl, PostgresDsn, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ASYNC_DRIVER = "postgresql+asyncpg"
_ACCEPTED_SCHEMES = ("postgresql", "postgres", ASYNC_DRIVER)

Environment = Literal["development", "test", "production"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
        # A mensagem de erro do pydantic ecoa o valor recebido; aqui ele pode ser segredo.
        hide_input_in_errors=True,
    )

    app_env: Environment = "development"
    log_level: LogLevel = "INFO"
    log_json: bool = True

    database_url: SecretStr
    db_pool_size: int = Field(default=5, ge=1, le=100)
    db_max_overflow: int = Field(default=10, ge=0, le=100)
    db_pool_timeout: float = Field(default=30.0, gt=0)
    db_connect_timeout: float = Field(
        default=10.0, gt=0, description="Segundos para abrir conexão e para o ping."
    )
    db_pool_recycle: int = Field(default=1800, ge=-1, description="Segundos; -1 desliga.")
    db_pool_pre_ping: bool = True
    db_echo: bool = False

    openrouter_api_key: SecretStr
    openrouter_base_url: HttpUrl = HttpUrl("https://openrouter.ai/api/v1")
    openrouter_timeout: float = Field(default=60.0, gt=0)

    @field_validator("log_level", mode="before")
    @classmethod
    def _upper_log_level(cls, value: object) -> object:
        return value.upper() if isinstance(value, str) else value

    @field_validator("database_url")
    @classmethod
    def _normalize_database_url(cls, value: SecretStr) -> SecretStr:
        # Erros daqui não podem ecoar a URL: ela carrega a senha.
        raw = value.get_secret_value()
        scheme, sep, rest = raw.partition("://")
        if not sep or scheme not in _ACCEPTED_SCHEMES:
            raise ValueError(f"DATABASE_URL deve usar um dos esquemas {_ACCEPTED_SCHEMES}")
        normalized = f"{ASYNC_DRIVER}://{rest}"
        try:
            dsn = PostgresDsn(normalized)
        except ValueError:
            raise ValueError("DATABASE_URL não é uma URL PostgreSQL válida") from None
        if not dsn.hosts() or not dsn.path or dsn.path == "/":
            raise ValueError("DATABASE_URL precisa de host e nome do banco")
        return SecretStr(normalized)

    @field_validator("openrouter_api_key")
    @classmethod
    def _reject_blank_api_key(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value().strip():
            raise ValueError("OPENROUTER_API_KEY não pode ser vazia")
        return value

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"


@lru_cache
def get_settings() -> Settings:
    """Settings carregadas uma vez por processo. Em teste, `get_settings.cache_clear()`."""
    return Settings()

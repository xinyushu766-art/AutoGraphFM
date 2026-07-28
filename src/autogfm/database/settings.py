"""Environment-backed database settings."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseSettings):
    """Runtime settings loaded from `.env` or environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="AUTOGFM_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: str = Field(
        default=(
            "postgresql+psycopg://"
            "autogfm:autogfm_dev_password@localhost:5432/autogfm"
        )
    )
    database_echo: bool = False


@lru_cache(maxsize=1)
def get_database_settings() -> DatabaseSettings:
    """Return one cached settings object per process."""
    return DatabaseSettings()

"""Configuration loaded from environment variables."""
from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from .env file and environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Required
    minimax_api_key: str = Field(..., min_length=1)

    # LLM (with defaults)
    litellm_model: str = "minimax/minimax-m3"
    litellm_api_base: str = "https://api.minimax.chat/v1"
    litellm_timeout: int = 60

    # Optional data sources
    alpha_vantage_api_key: str | None = None
    finnhub_api_key: str | None = None
    news_api_key: str | None = None

    # Logging
    log_level: str = "INFO"
    env: str = "development"


def get_settings() -> Settings:
    """Load settings. Fails fast if MINIMAX_API_KEY missing."""
    return Settings()  # type: ignore[call-arg]
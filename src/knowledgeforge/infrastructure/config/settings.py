"""Application configuration for KnowledgeForge."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for KnowledgeForge."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="KNOWLEDGEFORGE_",
        extra="ignore",
    )

    app_name: str = "KnowledgeForge"
    app_version: str = "0.1.0"
    environment: str = "development"

    vault_path: Path = Field(
        default=Path("./vault"),
        description="Root directory of the KnowledgeForge vault.",
    )

    templates_path: Path = Field(
        default=Path("./templates"),
        description="Directory containing note templates.",
    )

    log_level: str = "INFO"

    ai_enabled: bool = False
    ai_base_url: str = "https://api.openai.com/v1"
    ai_api_key: str = ""
    ai_model: str = "gpt-4o-mini"
    ai_embedding_model: str = "text-embedding-3-small"
    ai_timeout: float = 60.0


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached application settings."""
    return Settings()

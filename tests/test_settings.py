"""Tests for KnowledgeForge application settings."""

from pathlib import Path

from knowledgeforge.infrastructure.config.settings import Settings


def test_settings_have_expected_defaults() -> None:
    """Settings should provide stable development defaults."""
    settings = Settings()

    assert settings.app_name == "KnowledgeForge"
    assert settings.app_version == "0.1.0"
    assert settings.environment == "development"
    assert settings.vault_path == Path("./vault")
    assert settings.templates_path == Path("./templates")
    assert settings.log_level == "INFO"


def test_settings_can_be_loaded_from_environment(
    monkeypatch,
) -> None:
    """Settings should read values using the KnowledgeForge prefix."""
    monkeypatch.setenv("KNOWLEDGEFORGE_ENVIRONMENT", "test")
    monkeypatch.setenv("KNOWLEDGEFORGE_LOG_LEVEL", "DEBUG")

    settings = Settings()

    assert settings.environment == "test"
    assert settings.log_level == "DEBUG"
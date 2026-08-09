"""Tests for the KnowledgeForge vault service."""

import json

from knowledgeforge.domain.vault import VaultService


def test_create_creates_standard_vault_structure(tmp_path) -> None:
    """Vault creation should create all required directories and metadata."""
    vault_path = tmp_path / "vault"

    service = VaultService()
    service.create(vault_path)

    expected_directories = (
        "inbox",
        "notes",
        "sources",
        "projects",
        "templates",
        "assets",
    )

    for directory in expected_directories:
        assert (vault_path / directory).is_dir()

    metadata_path = vault_path / ".knowledgeforge" / "metadata.json"

    assert metadata_path.is_file()

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert metadata["format_version"] == 1
    assert metadata["application"] == "KnowledgeForge"


def test_created_vault_is_valid(tmp_path) -> None:
    """A newly created vault should pass validation."""
    vault_path = tmp_path / "vault"

    service = VaultService()
    service.create(vault_path)

    assert service.is_valid(vault_path)


def test_missing_directory_makes_vault_invalid(tmp_path) -> None:
    """A vault missing a required directory should be invalid."""
    vault_path = tmp_path / "vault"

    service = VaultService()
    service.create(vault_path)

    (vault_path / "notes").rmdir()

    assert not service.is_valid(vault_path)
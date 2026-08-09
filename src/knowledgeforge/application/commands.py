"""Application commands for KnowledgeForge."""

from dataclasses import dataclass
from pathlib import Path

from knowledgeforge.domain.vault import VaultService
from knowledgeforge.infrastructure.config.settings import Settings

INITIALIZATION_SUCCESS_MESSAGE = "KnowledgeForge initialized successfully."


@dataclass(frozen=True)
class CommandResult:
    """Result returned by an application command."""

    message: str
    vault_path: Path


def initialize_knowledgeforge(
    vault_path: Path | None = None,
) -> CommandResult:
    """Initialize a KnowledgeForge vault.

    Args:
        vault_path: Optional root directory for the vault. When omitted,
            the configured default path is used.

    Returns:
        Result containing the success message and initialized vault path.
    """
    resolved_vault_path = vault_path or Path(Settings().vault_path)

    vault_service = VaultService()
    created_path = vault_service.create(resolved_vault_path)

    return CommandResult(
        message=INITIALIZATION_SUCCESS_MESSAGE,
        vault_path=created_path,
    )
"""Tests for KnowledgeForge application commands."""

from knowledgeforge.application.commands import (
    INITIALIZATION_SUCCESS_MESSAGE,
    initialize_knowledgeforge,
)


def test_initialize_knowledgeforge_returns_success_message() -> None:
    """Initialization should return the expected success message."""
    result = initialize_knowledgeforge()

    assert result.message == INITIALIZATION_SUCCESS_MESSAGE
"""Tests for the KnowledgeForge CLI."""

from typer.testing import CliRunner

from knowledgeforge.application.commands import INITIALIZATION_SUCCESS_MESSAGE
from knowledgeforge.cli import app

runner = CliRunner()


def test_init_command_returns_success() -> None:
    """The init command should execute successfully."""
    result = runner.invoke(app, ["init"])

    assert result.exit_code == 0
    assert INITIALIZATION_SUCCESS_MESSAGE in result.stdout
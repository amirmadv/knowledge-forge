"""Tests for the KnowledgeForge CLI."""

from typer.testing import CliRunner

from knowledgeforge.application.commands import INITIALIZATION_SUCCESS_MESSAGE
from knowledgeforge.cli import app
from knowledgeforge.domain.note import NoteService

runner = CliRunner()


def test_init_command_returns_success() -> None:
    """The init command should execute successfully."""
    result = runner.invoke(app, ["init"])

    assert result.exit_code == 0
    assert INITIALIZATION_SUCCESS_MESSAGE in result.stdout


def test_note_search_command_returns_matching_note(
    tmp_path,
) -> None:
    """The note search command should display matching notes."""
    vault_path = tmp_path / "vault"
    service = NoteService(vault_path)

    note = service.create("Machine Learning")
    note.path.write_text(
        note.path.read_text(encoding="utf-8")
        + "\nLinear regression is a supervised learning algorithm.\n",
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        ["note", "search", "regression"],
        env={"KNOWLEDGEFORGE_VAULT_PATH": str(vault_path)},
    )

    assert result.exit_code == 0
    assert "Search results for: regression" in result.stdout
    assert "Machine Learning" in result.stdout
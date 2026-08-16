"""Tests for the KnowledgeForge CLI."""
from pathlib import Path

from typer.testing import CliRunner

from knowledgeforge.application.commands import (
    INITIALIZATION_SUCCESS_MESSAGE,
    create_note,
    initialize_knowledgeforge,
)
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

def test_note_delete_command_removes_note(
    tmp_path: Path,
) -> None:
    """The note delete command should remove an existing note."""
    vault_path = tmp_path / "vault"
    templates_path = tmp_path / "templates"

    initialize_knowledgeforge(vault_path)

    create_note(
        title="Linear Regression",
        vault_path=vault_path,
        templates_path=templates_path,
    )

    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "note",
            "delete",
            "Linear Regression",
        ],
        env={
            "KNOWLEDGEFORGE_VAULT_PATH": str(vault_path),
            "KNOWLEDGEFORGE_TEMPLATES_PATH": str(templates_path),
        },
    )

    assert result.exit_code == 0
    assert "Note deleted successfully." in result.stdout
    assert not (
        vault_path
        / "notes"
        / "linear-regression.md"
    ).exists()    

def test_note_list_command_filters_by_type(
    tmp_path: Path,
) -> None:
    """The note list command should filter by note type."""
    vault_path = tmp_path / "vault"

    create_note(
        title="Linear Regression",
        vault_path=vault_path,
        note_type="research",
    )

    create_note(
        title="Python",
        vault_path=vault_path,
        note_type="concept",
    )

    result = runner.invoke(
        app,
        [
            "note",
            "list",
            "--type",
            "research",
        ],
        env={
            "KNOWLEDGEFORGE_VAULT_PATH": str(vault_path),
        },
    )

    assert result.exit_code == 0
    assert "Linear Regression" in result.stdout
    assert "Python" not in result.stdout


def test_note_list_command_filters_by_status(
    tmp_path: Path,
) -> None:
    """The note list command should filter by status."""
    vault_path = tmp_path / "vault"

    create_note(
        title="Active Project",
        vault_path=vault_path,
        status="active",
    )

    create_note(
        title="Draft Project",
        vault_path=vault_path,
        status="draft",
    )

    result = runner.invoke(
        app,
        [
            "note",
            "list",
            "--status",
            "active",
        ],
        env={
            "KNOWLEDGEFORGE_VAULT_PATH": str(vault_path),
        },
    )

    assert result.exit_code == 0
    assert "Active Project" in result.stdout
    assert "Draft Project" not in result.stdout


def test_note_list_command_filters_by_tag(
    tmp_path: Path,
) -> None:
    """The note list command should filter by tag."""
    vault_path = tmp_path / "vault"

    create_note(
        title="Machine Learning",
        vault_path=vault_path,
        tags=("machine-learning",),
    )

    create_note(
        title="Python",
        vault_path=vault_path,
        tags=("programming",),
    )

    result = runner.invoke(
        app,
        [
            "note",
            "list",
            "--tag",
            "machine-learning",
        ],
        env={
            "KNOWLEDGEFORGE_VAULT_PATH": str(vault_path),
        },
    )

    assert result.exit_code == 0
    assert "Machine Learning" in result.stdout
    assert "Python" not in result.stdout    
"""Tests for the KnowledgeForge CLI."""
from pathlib import Path

from typer.testing import CliRunner

from knowledgeforge.application.commands import (
    INITIALIZATION_SUCCESS_MESSAGE,
    add_note_relationship,
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


def test_note_graph_command_returns_graph(
    tmp_path: Path,
) -> None:
    """The note graph command should display graph nodes and edges."""
    vault_path = tmp_path / "vault"
    templates_path = tmp_path / "templates"

    initialize_knowledgeforge(vault_path)

    create_note(
        title="Machine Learning",
        vault_path=vault_path,
        templates_path=templates_path,
    )

    create_note(
        title="Linear Regression",
        vault_path=vault_path,
        templates_path=templates_path,
    )

    add_note_relationship(
        source_title="Machine Learning",
        target_title="Linear Regression",
        relation_type="related",
        vault_path=vault_path,
    )

    result = runner.invoke(
        app,
        [
            "note",
            "graph",
            "Machine Learning",
            "--depth",
            "1",
        ],
        env={
            "KNOWLEDGEFORGE_VAULT_PATH": str(vault_path),
            "KNOWLEDGEFORGE_TEMPLATES_PATH": str(templates_path),
        },
    )

    assert result.exit_code == 0
    assert "Graph for: Machine Learning" in result.stdout
    assert "Depth: 1" in result.stdout
    assert "machine-learning" in result.stdout
    assert "linear-regression" in result.stdout
    assert "machine-learning --[related]--> linear-regression" in result.stdout


def test_note_ancestors_command_returns_ancestors(
    tmp_path: Path,
) -> None:
    """The ancestors command should display recursive ancestors."""
    vault_path = tmp_path / "vault"
    templates_path = tmp_path / "templates"

    initialize_knowledgeforge(vault_path)

    create_note(
        title="Machine Learning",
        vault_path=vault_path,
        templates_path=templates_path,
    )

    create_note(
        title="Linear Regression",
        vault_path=vault_path,
        templates_path=templates_path,
    )

    add_note_relationship(
        source_title="Machine Learning",
        target_title="Linear Regression",
        relation_type="related",
        vault_path=vault_path,
    )

    result = runner.invoke(
        app,
        [
            "note",
            "ancestors",
            "Linear Regression",
        ],
        env={
            "KNOWLEDGEFORGE_VAULT_PATH": str(vault_path),
            "KNOWLEDGEFORGE_TEMPLATES_PATH": str(templates_path),
        },
    )

    assert result.exit_code == 0
    assert "Ancestors of: Linear Regression" in result.stdout
    assert "machine-learning" in result.stdout


def test_note_descendants_command_returns_descendants(
    tmp_path: Path,
) -> None:
    """The descendants command should display recursive descendants."""
    vault_path = tmp_path / "vault"
    templates_path = tmp_path / "templates"

    initialize_knowledgeforge(vault_path)

    create_note(
        title="Machine Learning",
        vault_path=vault_path,
        templates_path=templates_path,
    )

    create_note(
        title="Linear Regression",
        vault_path=vault_path,
        templates_path=templates_path,
    )

    add_note_relationship(
        source_title="Machine Learning",
        target_title="Linear Regression",
        relation_type="related",
        vault_path=vault_path,
    )

    result = runner.invoke(
        app,
        [
            "note",
            "descendants",
            "Machine Learning",
        ],
        env={
            "KNOWLEDGEFORGE_VAULT_PATH": str(vault_path),
            "KNOWLEDGEFORGE_TEMPLATES_PATH": str(templates_path),
        },
    )

    assert result.exit_code == 0
    assert "Descendants of: Machine Learning" in result.stdout
    assert "linear-regression" in result.stdout

def test_note_stats_command_returns_graph_statistics(
    tmp_path: Path,
) -> None:
    """The note stats command should display graph statistics."""
    vault_path = tmp_path / "vault"
    templates_path = tmp_path / "templates"

    initialize_knowledgeforge(vault_path)

    create_note(
        title="Machine Learning",
        vault_path=vault_path,
        templates_path=templates_path,
    )

    create_note(
        title="Linear Regression",
        vault_path=vault_path,
        templates_path=templates_path,
    )

    add_note_relationship(
        source_title="Machine Learning",
        target_title="Linear Regression",
        relation_type="related",
        vault_path=vault_path,
    )

    result = runner.invoke(
        app,
        [
            "note",
            "stats",
        ],
        env={
            "KNOWLEDGEFORGE_VAULT_PATH": str(vault_path),
            "KNOWLEDGEFORGE_TEMPLATES_PATH": str(templates_path),
        },
    )

    assert result.exit_code == 0
    assert "KnowledgeForge Graph Statistics" in result.stdout
    assert "Nodes: 2" in result.stdout
    assert "Edges: 1" in result.stdout
    assert "Orphan nodes: 0" in result.stdout
    assert "Root nodes: 1" in result.stdout
    assert "Leaf nodes: 1" in result.stdout
    assert "Average degree: 1.00" in result.stdout
    assert "Max degree: 1" in result.stdout
    assert "Density: 0.5000" in result.stdout    
"""Tests for KnowledgeForge application commands."""

from pathlib import Path

from knowledgeforge.application.commands import (
    INITIALIZATION_SUCCESS_MESSAGE,
    add_note_relationship,
    create_note,
    delete_note,
    get_note_ancestors,
    get_note_descendants,
    get_note_graph,
    initialize_knowledgeforge,
    list_note_relationships,
    remove_note_relationship,
)
from knowledgeforge.domain.relationship import RelationshipType


def test_initialize_knowledgeforge_returns_success_message() -> None:
    """Initialization should return the expected success message."""
    result = initialize_knowledgeforge()

    assert result.message == INITIALIZATION_SUCCESS_MESSAGE

def test_delete_note_returns_success_result(
    tmp_path: Path,
) -> None:
    """Deleting a note should return a successful command result."""
    vault_path = tmp_path / "vault"

    create_note(
        title="Linear Regression",
        vault_path=vault_path,
    )

    result = delete_note(
        title="Linear Regression",
        vault_path=vault_path,
    )

    assert result.message == "Note deleted successfully."
    assert result.note is not None
    assert result.note.title == "Linear Regression"
    assert not result.note.path.exists()    

def test_add_note_relationship_returns_relationship(
    tmp_path: Path,
) -> None:
    """Adding a relationship should return the created relation."""
    vault_path = tmp_path / "vault"

    create_note(
        title="Linear Regression",
        vault_path=vault_path,
    )

    create_note(
        title="Gradient Descent",
        vault_path=vault_path,
    )

    relation = add_note_relationship(
        source_title="Linear Regression",
        target_title="Gradient Descent",
        relation_type="prerequisite",
        vault_path=vault_path,
    )

    assert relation.source == "linear-regression"
    assert relation.target == "gradient-descent"
    assert relation.relation_type == RelationshipType.PREREQUISITE


def test_list_note_relationships_returns_relationships(
    tmp_path: Path,
) -> None:
    """Listing relationships should return note relations."""
    vault_path = tmp_path / "vault"

    create_note(
        title="Linear Regression",
        vault_path=vault_path,
    )

    create_note(
        title="Gradient Descent",
        vault_path=vault_path,
    )

    add_note_relationship(
        source_title="Linear Regression",
        target_title="Gradient Descent",
        relation_type="prerequisite",
        vault_path=vault_path,
    )

    relations = list_note_relationships(
        title="Linear Regression",
        vault_path=vault_path,
    )

    assert len(relations) == 1
    assert relations[0].relation_type == (
        RelationshipType.PREREQUISITE
    )


def test_remove_note_relationship_returns_relationship(
    tmp_path: Path,
) -> None:
    """Removing a relationship should return the removed relation."""
    vault_path = tmp_path / "vault"

    create_note(
        title="Linear Regression",
        vault_path=vault_path,
    )

    create_note(
        title="Gradient Descent",
        vault_path=vault_path,
    )

    add_note_relationship(
        source_title="Linear Regression",
        target_title="Gradient Descent",
        relation_type="prerequisite",
        vault_path=vault_path,
    )

    relation = remove_note_relationship(
        source_title="Linear Regression",
        target_title="Gradient Descent",
        relation_type="prerequisite",
        vault_path=vault_path,
    )

    assert relation.relation_type == (
        RelationshipType.PREREQUISITE
    )

    assert list_note_relationships(
        title="Linear Regression",
        vault_path=vault_path,
    ) == []    

def test_get_note_graph_returns_graph(
    tmp_path: Path,
) -> None:
    """Getting a note graph should return a graph around the note."""
    vault_path = tmp_path / "vault"

    create_note(
        title="Machine Learning",
        vault_path=vault_path,
    )

    create_note(
        title="Linear Regression",
        vault_path=vault_path,
    )

    create_note(
        title="Gradient Descent",
        vault_path=vault_path,
    )

    add_note_relationship(
        source_title="Machine Learning",
        target_title="Linear Regression",
        relation_type="prerequisite",
        vault_path=vault_path,
    )

    add_note_relationship(
        source_title="Linear Regression",
        target_title="Gradient Descent",
        relation_type="prerequisite",
        vault_path=vault_path,
    )

    graph = get_note_graph(
        title="Machine Learning",
        depth=2,
        vault_path=vault_path,
    )

    assert {
        node.slug
        for node in graph.nodes
    } == {
        "machine-learning",
        "linear-regression",
        "gradient-descent",
    }

    assert len(graph.edges) == 2

def test_get_note_ancestors_returns_recursive_ancestors(
    tmp_path: Path,
) -> None:
    """Getting ancestors should traverse relationships recursively."""
    vault_path = tmp_path / "vault"

    create_note(
        title="Machine Learning",
        vault_path=vault_path,
    )

    create_note(
        title="Linear Regression",
        vault_path=vault_path,
    )

    create_note(
        title="Gradient Descent",
        vault_path=vault_path,
    )

    add_note_relationship(
        source_title="Machine Learning",
        target_title="Linear Regression",
        relation_type="prerequisite",
        vault_path=vault_path,
    )

    add_note_relationship(
        source_title="Linear Regression",
        target_title="Gradient Descent",
        relation_type="prerequisite",
        vault_path=vault_path,
    )

    ancestors = get_note_ancestors(
        title="Gradient Descent",
        vault_path=vault_path,
    )

    assert ancestors == [
        "linear-regression",
        "machine-learning",
    ]

def test_get_note_descendants_returns_recursive_descendants(
    tmp_path: Path,
) -> None:
    """Getting descendants should traverse relationships recursively."""
    vault_path = tmp_path / "vault"

    create_note(
        title="Machine Learning",
        vault_path=vault_path,
    )

    create_note(
        title="Linear Regression",
        vault_path=vault_path,
    )

    create_note(
        title="Gradient Descent",
        vault_path=vault_path,
    )

    add_note_relationship(
        source_title="Machine Learning",
        target_title="Linear Regression",
        relation_type="prerequisite",
        vault_path=vault_path,
    )

    add_note_relationship(
        source_title="Linear Regression",
        target_title="Gradient Descent",
        relation_type="prerequisite",
        vault_path=vault_path,
    )

    descendants = get_note_descendants(
        title="Machine Learning",
        vault_path=vault_path,
    )

    assert descendants == [
        "gradient-descent",
        "linear-regression",
    ]
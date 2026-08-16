"""Tests for KnowledgeForge application commands."""

from pathlib import Path

from knowledgeforge.application.commands import (
    INITIALIZATION_SUCCESS_MESSAGE,
    add_note_relationship,
    create_note,
    delete_note,
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
"""Tests for the KnowledgeForge relationship service."""

from pathlib import Path

import pytest

from knowledgeforge.domain.note import NoteService
from knowledgeforge.domain.relationship import (
    InvalidRelationshipError,
    RelationshipAlreadyExistsError,
    RelationshipNotFoundError,
    RelationshipService,
    RelationshipType,
)


def _create_notes(vault_path: Path) -> NoteService:
    """Create a note service with test notes."""
    service = NoteService(vault_path)

    service.create("Linear Regression")
    service.create("Gradient Descent")
    service.create("Loss Function")

    return service


def test_add_creates_relationship(
    tmp_path: Path,
) -> None:
    """Adding a relationship should persist it."""
    vault_path = tmp_path / "vault"
    note_service = _create_notes(vault_path)

    service = RelationshipService(
        vault_path,
        note_service,
    )

    relation = service.add(
        "Linear Regression",
        "Gradient Descent",
        "prerequisite",
    )

    assert relation.source == "linear-regression"
    assert relation.target == "gradient-descent"
    assert relation.relation_type == RelationshipType.PREREQUISITE

    assert service.relations_path.exists()


def test_add_normalizes_relationship_type(
    tmp_path: Path,
) -> None:
    """Relationship types should be normalized."""
    vault_path = tmp_path / "vault"
    note_service = _create_notes(vault_path)

    service = RelationshipService(
        vault_path,
        note_service,
    )

    relation = service.add(
        "Linear Regression",
        "Gradient Descent",
        " PREREQUISITE ",
    )

    assert relation.relation_type == RelationshipType.PREREQUISITE


def test_add_rejects_duplicate_relationship(
    tmp_path: Path,
) -> None:
    """Adding the same relationship twice should fail."""
    vault_path = tmp_path / "vault"
    note_service = _create_notes(vault_path)

    service = RelationshipService(
        vault_path,
        note_service,
    )

    service.add(
        "Linear Regression",
        "Gradient Descent",
    )

    with pytest.raises(
        RelationshipAlreadyExistsError,
        match="Relationship already exists",
    ):
        service.add(
            "Linear Regression",
            "Gradient Descent",
        )


def test_add_rejects_self_relationship(
    tmp_path: Path,
) -> None:
    """A note cannot be related to itself."""
    vault_path = tmp_path / "vault"
    note_service = _create_notes(vault_path)

    service = RelationshipService(
        vault_path,
        note_service,
    )

    with pytest.raises(
        InvalidRelationshipError,
        match="cannot have a relationship with itself",
    ):
        service.add(
            "Linear Regression",
            "Linear Regression",
        )


def test_add_rejects_missing_source_note(
    tmp_path: Path,
) -> None:
    """A relationship requires an existing source note."""
    vault_path = tmp_path / "vault"
    note_service = _create_notes(vault_path)

    service = RelationshipService(
        vault_path,
        note_service,
    )

    with pytest.raises(FileNotFoundError):
        service.add(
            "Missing Note",
            "Gradient Descent",
        )


def test_add_rejects_invalid_relationship_type(
    tmp_path: Path,
) -> None:
    """Unsupported relationship types should be rejected."""
    vault_path = tmp_path / "vault"
    note_service = _create_notes(vault_path)

    service = RelationshipService(
        vault_path,
        note_service,
    )

    with pytest.raises(
        InvalidRelationshipError,
        match="Invalid relationship type",
    ):
        service.add(
            "Linear Regression",
            "Gradient Descent",
            "unknown",
        )


def test_list_for_returns_note_relationships(
    tmp_path: Path,
) -> None:
    """Listing a note's relationships should return its edges."""
    vault_path = tmp_path / "vault"
    note_service = _create_notes(vault_path)

    service = RelationshipService(
        vault_path,
        note_service,
    )

    service.add(
        "Linear Regression",
        "Gradient Descent",
        "prerequisite",
    )
    service.add(
        "Loss Function",
        "Linear Regression",
        "related",
    )

    results = service.list_for("Linear Regression")

    assert len(results) == 2

    assert {
        relation.relation_type
        for relation in results
    } == {
        RelationshipType.PREREQUISITE,
        RelationshipType.RELATED,
    }


def test_list_all_returns_all_relationships(
    tmp_path: Path,
) -> None:
    """list_all should return every relationship."""
    vault_path = tmp_path / "vault"
    note_service = _create_notes(vault_path)

    service = RelationshipService(
        vault_path,
        note_service,
    )

    service.add(
        "Linear Regression",
        "Gradient Descent",
        "prerequisite",
    )
    service.add(
        "Linear Regression",
        "Loss Function",
        "related",
    )

    results = service.list_all()

    assert len(results) == 2


def test_remove_deletes_relationship(
    tmp_path: Path,
) -> None:
    """Removing a relationship should delete it from storage."""
    vault_path = tmp_path / "vault"
    note_service = _create_notes(vault_path)

    service = RelationshipService(
        vault_path,
        note_service,
    )

    service.add(
        "Linear Regression",
        "Gradient Descent",
    )

    removed = service.remove(
        "Linear Regression",
        "Gradient Descent",
    )

    assert removed.relation_type == RelationshipType.RELATED
    assert service.list_all() == []


def test_remove_rejects_missing_relationship(
    tmp_path: Path,
) -> None:
    """Removing a missing relationship should fail."""
    vault_path = tmp_path / "vault"
    note_service = _create_notes(vault_path)

    service = RelationshipService(
        vault_path,
        note_service,
    )

    with pytest.raises(
        RelationshipNotFoundError,
        match="Relationship not found",
    ):
        service.remove(
            "Linear Regression",
            "Gradient Descent",
        )


def test_relationships_survive_service_recreation(
    tmp_path: Path,
) -> None:
    """Relationships should persist between service instances."""
    vault_path = tmp_path / "vault"
    note_service = _create_notes(vault_path)

    first_service = RelationshipService(
        vault_path,
        note_service,
    )

    first_service.add(
        "Linear Regression",
        "Gradient Descent",
        "prerequisite",
    )

    second_service = RelationshipService(
        vault_path,
        note_service,
    )

    results = second_service.list_for(
        "Linear Regression",
    )

    assert len(results) == 1
    assert results[0].relation_type == (
        RelationshipType.PREREQUISITE
    )

def test_get_note_delegates_to_note_service(
    tmp_path: Path,
) -> None:
    """get_note should expose note lookup through a public API."""
    note_service = NoteService(tmp_path / "vault")
    note_service.create("Machine Learning")

    relationship_service = RelationshipService(
        tmp_path / "vault",
        note_service=note_service,
    )

    result = relationship_service.get_note("Machine Learning")

    assert result.slug == "machine-learning"
    assert result.title == "Machine Learning"
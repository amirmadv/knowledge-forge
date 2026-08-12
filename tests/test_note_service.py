"""Tests for the KnowledgeForge note service."""

from pathlib import Path

import pytest

from knowledgeforge.domain.note import (
    InvalidNoteTitleError,
    NoteAlreadyExistsError,
    NoteService,
)


def test_create_creates_markdown_note(tmp_path: Path) -> None:
    """Creating a note should create a Markdown file."""
    vault_path = tmp_path / "vault"
    service = NoteService(vault_path)

    note = service.create("Machine Learning")

    assert note.title == "Machine Learning"
    assert note.path == vault_path / "notes" / "machine-learning.md"
    assert note.path.exists()

    content = note.path.read_text(encoding="utf-8")

    assert "title: Machine Learning" in content
    assert "# Machine Learning" in content
    assert "created_at:" in content


def test_create_normalizes_note_title_to_slug(tmp_path: Path) -> None:
    """The note filename should be generated from a normalized slug."""
    service = NoteService(tmp_path / "vault")

    note = service.create("  Python: Machine Learning!  ")

    assert note.path.name == "python-machine-learning.md"


def test_create_rejects_empty_title(tmp_path: Path) -> None:
    """An empty title should be rejected."""
    service = NoteService(tmp_path / "vault")

    with pytest.raises(InvalidNoteTitleError):
        service.create("   ")


def test_create_rejects_duplicate_note(tmp_path: Path) -> None:
    """Creating the same note twice should fail safely."""
    service = NoteService(tmp_path / "vault")

    service.create("Machine Learning")

    with pytest.raises(NoteAlreadyExistsError):
        service.create("Machine Learning")


def test_search_finds_note_by_title(tmp_path: Path) -> None:
    """Search should find notes when the query appears in the title."""
    service = NoteService(tmp_path / "vault")

    service.create("Machine Learning")
    service.create("Python Programming")

    results = service.search("machine")

    assert len(results) == 1
    assert results[0].title == "Machine Learning"


def test_search_finds_note_by_content(tmp_path: Path) -> None:
    """Search should find notes when the query appears in Markdown content."""
    service = NoteService(tmp_path / "vault")

    note = service.create("Machine Learning")

    note.path.write_text(
        "---\n"
        "title: Machine Learning\n"
        "---\n\n"
        "# Machine Learning\n\n"
        "Linear regression is a supervised learning algorithm.\n",
        encoding="utf-8",
    )

    results = service.search("REGRESSION")

    assert len(results) == 1
    assert results[0].title == "Machine Learning"


def test_search_is_case_insensitive(tmp_path: Path) -> None:
    """Search should be case-insensitive."""
    service = NoteService(tmp_path / "vault")

    service.create("Machine Learning")

    results = service.search("MACHINE")

    assert len(results) == 1
    assert results[0].title == "Machine Learning"


def test_search_returns_empty_for_no_match(tmp_path: Path) -> None:
    """Search should return an empty list when nothing matches."""
    service = NoteService(tmp_path / "vault")

    service.create("Machine Learning")

    results = service.search("database")

    assert results == []


def test_search_rejects_empty_query(tmp_path: Path) -> None:
    """Search should reject an empty query."""
    service = NoteService(tmp_path / "vault")

    with pytest.raises(ValueError, match="Search query cannot be empty"):
        service.search("   ")
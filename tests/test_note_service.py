"""Tests for the KnowledgeForge note service."""

from pathlib import Path

import pytest

from knowledgeforge.domain.note import (
    InvalidNoteTitleError,
    NoteAlreadyExistsError,
    NoteNotFoundError,
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


def test_create_stores_note_metadata(
    tmp_path: Path,
) -> None:
    """Creating a note should persist its metadata."""
    service = NoteService(tmp_path / "vault")

    note = service.create(
        "Linear Regression",
        note_type="research",
        status="active",
        tags=("machine-learning", "regression"),
    )

    assert note.metadata.note_type == "research"
    assert note.metadata.status == "active"
    assert note.metadata.tags == (
        "machine-learning",
        "regression",
    )

    content = note.path.read_text(encoding="utf-8")

    assert "type: research" in content
    assert "status: active" in content
    assert "  - machine-learning" in content
    assert "  - regression" in content
    assert "updated_at:" in content


def test_create_normalizes_metadata(
    tmp_path: Path,
) -> None:
    """Metadata should be normalized before persistence."""
    service = NoteService(tmp_path / "vault")

    note = service.create(
        "Python",
        note_type=" RESEARCH ",
        status=" ACTIVE ",
        tags=(
            "Machine Learning",
            "machine-learning",
            "Python",
        ),
    )

    assert note.metadata.note_type == "research"
    assert note.metadata.status == "active"
    assert note.metadata.tags == (
        "machine-learning",
        "python",
    )


def test_get_reads_metadata_from_front_matter(
    tmp_path: Path,
) -> None:
    """Getting a note should reconstruct its metadata."""
    service = NoteService(tmp_path / "vault")

    created = service.create(
        "Linear Regression",
        note_type="research",
        status="review",
        tags=("machine-learning", "regression"),
    )

    loaded = service.get("Linear Regression")

    assert loaded.title == created.title
    assert loaded.metadata.note_type == "research"
    assert loaded.metadata.status == "review"
    assert loaded.metadata.tags == (
        "machine-learning",
        "regression",
    )


def test_search_finds_note_by_metadata(
    tmp_path: Path,
) -> None:
    """Search should find notes through metadata."""
    service = NoteService(tmp_path / "vault")

    service.create(
        "Linear Regression",
        note_type="research",
        tags=("machine-learning",),
    )

    service.create("Python Programming")

    results = service.search("machine-learning")

    assert len(results) == 1
    assert results[0].title == "Linear Regression"


def test_update_preserves_metadata(
    tmp_path: Path,
) -> None:
    """Updating note content should preserve metadata."""
    service = NoteService(tmp_path / "vault")

    service.create(
        "Linear Regression",
        note_type="research",
        status="active",
        tags=("machine-learning", "regression"),
    )

    updated = service.update(
        "Linear Regression",
        "New content.",
    )

    assert updated.metadata.note_type == "research"
    assert updated.metadata.status == "active"
    assert updated.metadata.tags == (
        "machine-learning",
        "regression",
    )

    content = updated.path.read_text(encoding="utf-8")

    assert "type: research" in content
    assert "status: active" in content
    assert "  - machine-learning" in content
    assert "  - regression" in content
    assert "New content." in content


def test_create_rejects_invalid_note_type(
    tmp_path: Path,
) -> None:
    """An unsupported note type should be rejected."""
    service = NoteService(tmp_path / "vault")

    with pytest.raises(
        ValueError,
        match="Invalid note type",
    ):
        service.create(
            "Python",
            note_type="invalid",
        )


def test_create_rejects_invalid_status(
    tmp_path: Path,
) -> None:
    """An unsupported status should be rejected."""
    service = NoteService(tmp_path / "vault")

    with pytest.raises(
        ValueError,
        match="Invalid note status",
    ):
        service.create(
            "Python",
            status="invalid",
        )

def test_delete_removes_existing_note(
    tmp_path: Path,
) -> None:
    """Deleting a note should remove its Markdown file."""
    service = NoteService(tmp_path / "vault")

    created = service.create(
        "Linear Regression",
        note_type="research",
        status="active",
        tags=("machine-learning", "regression"),
    )

    deleted = service.delete("Linear Regression")

    assert deleted.title == "Linear Regression"
    assert deleted.path == created.path
    assert not created.path.exists()


def test_delete_returns_note_metadata(
    tmp_path: Path,
) -> None:
    """Deleting a note should return its metadata."""
    service = NoteService(tmp_path / "vault")

    service.create(
        "Linear Regression",
        note_type="research",
        status="active",
        tags=("machine-learning", "regression"),
    )

    deleted = service.delete("Linear Regression")

    assert deleted.metadata.note_type == "research"
    assert deleted.metadata.status == "active"
    assert deleted.metadata.tags == (
        "machine-learning",
        "regression",
    )


def test_delete_rejects_missing_note(
    tmp_path: Path,
) -> None:
    """Deleting a missing note should raise NoteNotFoundError."""
    service = NoteService(tmp_path / "vault")

    with pytest.raises(
        NoteNotFoundError,
        match="Note not found",
    ):
        service.delete("Linear Regression")        

def test_list_filters_notes_by_type(
    tmp_path: Path,
) -> None:
    """Listing should filter notes by note type."""
    service = NoteService(tmp_path / "vault")

    service.create(
        "Linear Regression",
        note_type="research",
    )
    service.create(
        "Python",
        note_type="concept",
    )

    results = service.list_notes(note_type="research")

    assert len(results) == 1
    assert results[0].title == "Linear Regression"


def test_list_filters_notes_by_status(
    tmp_path: Path,
) -> None:
    """Listing should filter notes by status."""
    service = NoteService(tmp_path / "vault")

    service.create(
        "Linear Regression",
        status="active",
    )
    service.create(
        "Python",
        status="draft",
    )

    results = service.list_notes(status="active")

    assert len(results) == 1
    assert results[0].title == "Linear Regression"


def test_list_filters_notes_by_tag(
    tmp_path: Path,
) -> None:
    """Listing should filter notes by tag."""
    service = NoteService(tmp_path / "vault")

    service.create(
        "Linear Regression",
        tags=("machine-learning", "regression"),
    )
    service.create(
        "Python",
        tags=("programming",),
    )

    results = service.list_notes(
        tag="machine-learning",
    )

    assert len(results) == 1
    assert results[0].title == "Linear Regression"


def test_list_combines_metadata_filters(
    tmp_path: Path,
) -> None:
    """Multiple metadata filters should be combined with AND."""
    service = NoteService(tmp_path / "vault")

    service.create(
        "Linear Regression",
        note_type="research",
        status="active",
        tags=("machine-learning",),
    )

    service.create(
        "Python",
        note_type="research",
        status="draft",
        tags=("machine-learning",),
    )

    service.create(
        "Neural Networks",
        note_type="concept",
        status="active",
        tags=("machine-learning",),
    )

    results = service.list_notes(
        note_type="research",
        status="active",
        tag="machine-learning",
    )

    assert len(results) == 1
    assert results[0].title == "Linear Regression"


def test_list_without_filters_returns_all_notes(
    tmp_path: Path,
) -> None:
    """Listing without filters should return every note."""
    service = NoteService(tmp_path / "vault")

    service.create("Linear Regression")
    service.create("Python")
    service.create("Neural Networks")

    results = service.list_notes()

    assert len(results) == 3
    assert [
        note.title
        for note in results
    ] == [
        "Linear Regression",
        "Neural Networks",
        "Python",
    ]        
"""Tests for the KnowledgeForge note domain model."""

from datetime import datetime
from pathlib import Path

from knowledgeforge.domain.note import Note


def test_note_exposes_basic_properties() -> None:
    """A note should expose its filename and slug."""
    path = Path("notes/machine-learning.md")
    timestamp = datetime(2026, 1, 1, 12, 0, 0)

    note = Note(
        title="Machine Learning",
        path=path,
        created_at=timestamp,
        updated_at=timestamp,
    )

    assert note.title == "Machine Learning"
    assert note.path == path
    assert note.created_at == timestamp
    assert note.updated_at == timestamp
    assert note.filename == "machine-learning.md"
    assert note.slug == "machine-learning"
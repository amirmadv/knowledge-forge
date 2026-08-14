"""Domain model for KnowledgeForge notes."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True, slots=True)
class NoteMetadata:
    """Metadata associated with a KnowledgeForge note."""

    note_type: str = "concept"
    status: str = "draft"
    tags: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class Note:
    """Represent a KnowledgeForge Markdown note."""

    title: str
    path: Path
    created_at: datetime
    updated_at: datetime
    metadata: NoteMetadata = field(default_factory=NoteMetadata)

    @property
    def filename(self) -> str:
        """Return the note filename."""
        return self.path.name

    @property
    def slug(self) -> str:
        """Return the note slug derived from its filename."""
        return self.path.stem
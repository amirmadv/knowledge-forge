"""Domain model for KnowledgeForge notes."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Note:
    """Represent a KnowledgeForge Markdown note."""

    title: str
    path: Path
    created_at: datetime
    updated_at: datetime

    @property
    def filename(self) -> str:
        """Return the note filename."""
        return self.path.name

    @property
    def slug(self) -> str:
        """Return the filename without its extension."""
        return self.path.stem
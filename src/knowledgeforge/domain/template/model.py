"""Domain model for KnowledgeForge templates."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Template:
    """Represent a KnowledgeForge Markdown template."""

    name: str
    path: Path
    content: str

    @property
    def filename(self) -> str:
        """Return the template filename."""
        return self.path.name
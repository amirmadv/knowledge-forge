"""Domain models for KnowledgeForge note relationships."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class RelationshipType(StrEnum):
    """Supported relationships between notes."""

    RELATED = "related"
    PREREQUISITE = "prerequisite"
    DERIVED_FROM = "derived_from"
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"


@dataclass(frozen=True, slots=True)
class NoteRelation:
    """A directed relationship between two notes."""

    source: str
    target: str
    relation_type: RelationshipType
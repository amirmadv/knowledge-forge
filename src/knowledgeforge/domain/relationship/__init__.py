"""KnowledgeForge note relationship domain."""

from knowledgeforge.domain.relationship.model import (
    NoteRelation,
    RelationshipType,
)
from knowledgeforge.domain.relationship.service import (
    InvalidRelationshipError,
    RelationshipAlreadyExistsError,
    RelationshipNotFoundError,
    RelationshipService,
)

__all__ = [
    "InvalidRelationshipError",
    "NoteRelation",
    "RelationshipAlreadyExistsError",
    "RelationshipNotFoundError",
    "RelationshipService",
    "RelationshipType",
]
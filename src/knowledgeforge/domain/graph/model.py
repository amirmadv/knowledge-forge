"""Domain models for KnowledgeForge graph operations."""

from __future__ import annotations

from dataclasses import dataclass

from knowledgeforge.domain.relationship.model import RelationshipType


@dataclass(frozen=True, slots=True)
class GraphEdge:
    """A directed edge in the KnowledgeForge note graph."""

    source: str
    target: str
    relation_type: RelationshipType


@dataclass(frozen=True, slots=True)
class GraphNode:
    """A node in the KnowledgeForge note graph."""

    slug: str


@dataclass(frozen=True, slots=True)
class NoteGraph:
    """A subgraph containing nodes and directed edges."""

    nodes: tuple[GraphNode, ...]
    edges: tuple[GraphEdge, ...]
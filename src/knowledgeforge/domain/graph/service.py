"""Domain service for KnowledgeForge graph operations."""

from __future__ import annotations

from pathlib import Path

from knowledgeforge.domain.graph.model import (
    GraphEdge,
    GraphNode,
    NoteGraph,
)
from knowledgeforge.domain.relationship.model import NoteRelation
from knowledgeforge.domain.relationship.service import RelationshipService


class GraphService:
    """Analyze the directed graph of KnowledgeForge notes."""

    def __init__(
        self,
        vault_path: Path,
        relationship_service: RelationshipService | None = None,
    ) -> None:
        """Initialize the graph service.

        Args:
            vault_path: Root directory of the KnowledgeForge vault.
            relationship_service: Optional relationship service.
        """
        self._relationship_service = (
            relationship_service
            or RelationshipService(vault_path)
        )

    def outgoing_links(
        self,
        title: str,
    ) -> list[NoteRelation]:
        """Return relationships leaving the requested note.

        Args:
            title: Note title.

        Returns:
            Relationships where the requested note is the source.
        """
        note = self._relationship_service.get_note(title)

        return sorted(
            [
                relation
                for relation in self._relationship_service.list_all()
                if relation.source == note.slug
            ],
            key=self._relation_sort_key,
        )

    def incoming_links(
        self,
        title: str,
    ) -> list[NoteRelation]:
        """Return relationships entering the requested note.

        Args:
            title: Note title.

        Returns:
            Relationships where the requested note is the target.
        """
        note = self._relationship_service.get_note(title)

        return sorted(
            [
                relation
                for relation in self._relationship_service.list_all()
                if relation.target == note.slug
            ],
            key=self._relation_sort_key,
        )

    def backlinks(
        self,
        title: str,
    ) -> list[NoteRelation]:
        """Return relationships pointing to the requested note.

        Backlinks are directed relationships whose target is the
        requested note.

        Args:
            title: Note title.

        Returns:
            Incoming relationships.
        """
        return self.incoming_links(title)

    def neighbors(
        self,
        title: str,
    ) -> list[str]:
        """Return direct neighboring note slugs.

        Both outgoing and incoming relationships are considered.

        Args:
            title: Note title.

        Returns:
            Sorted unique neighboring note slugs.
        """
        note = self._relationship_service.get_note(title)
        slug = note.slug

        neighbors: set[str] = set()

        for relation in self._relationship_service.list_all():
            if relation.source == slug:
                neighbors.add(relation.target)

            if relation.target == slug:
                neighbors.add(relation.source)

        return sorted(neighbors)

    def graph(
        self,
        title: str,
        depth: int = 1,
    ) -> NoteGraph:
        """Build a subgraph around a note.

        Traversal follows both incoming and outgoing relationships.

        Args:
            title: Starting note title.
            depth: Maximum traversal depth.

        Returns:
            A NoteGraph containing reachable nodes and edges.

        Raises:
            ValueError: If depth is negative.
        """
        if depth < 0:
            raise ValueError("Graph depth cannot be negative.")

        start_note = self._relationship_service.get_note(title)
        start_slug = start_note.slug

        all_relations = self._relationship_service.list_all()

        visited: set[str] = {start_slug}
        frontier: set[str] = {start_slug}

        for _ in range(depth):
            next_frontier: set[str] = set()

            for relation in all_relations:
                if (
                    relation.source in frontier
                    and relation.target not in visited
                ):
                    visited.add(relation.target)
                    next_frontier.add(relation.target)

                if (
                    relation.target in frontier
                    and relation.source not in visited
                ):
                    visited.add(relation.source)
                    next_frontier.add(relation.source)

            frontier = next_frontier

            if not frontier:
                break

        edges = tuple(
            GraphEdge(
                source=relation.source,
                target=relation.target,
                relation_type=relation.relation_type,
            )
            for relation in all_relations
            if relation.source in visited
            and relation.target in visited
        )

        nodes = tuple(
            GraphNode(slug=slug)
            for slug in sorted(visited)
        )

        return NoteGraph(
            nodes=nodes,
            edges=tuple(
                sorted(
                    edges,
                    key=lambda edge: (
                        edge.source,
                        edge.target,
                        edge.relation_type.value,
                    ),
                )
            ),
        )

    @staticmethod
    def _relation_sort_key(
        relation: NoteRelation,
    ) -> tuple[str, str, str]:
        """Return a deterministic relationship sort key."""
        return (
            relation.source,
            relation.target,
            relation.relation_type.value,
        )
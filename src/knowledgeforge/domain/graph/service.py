"""Domain service for KnowledgeForge graph operations."""

from __future__ import annotations

from pathlib import Path

from knowledgeforge.domain.graph.model import (
    GraphEdge,
    GraphNode,
    GraphStatistics,
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

    def ancestors(self, title: str) -> list[str]:
        """Return all recursive ancestors of a note.

        Args:
            title: Note title.

        Returns:
            Sorted ancestor note slugs.
        """
        note = self._relationship_service.get_note(title)
        start_slug = note.slug

        relations = self._relationship_service.list_all()

        ancestors: set[str] = set()
        frontier: set[str] = {start_slug}

        while frontier:
            next_frontier: set[str] = set()

            for relation in relations:
                if (
                    relation.target in frontier
                    and relation.source not in ancestors
                    and relation.source != start_slug
                ):
                    ancestors.add(relation.source)
                    next_frontier.add(relation.source)

            frontier = next_frontier

        return sorted(ancestors)

    def descendants(self, title: str) -> list[str]:
        """Return all recursive descendants of a note.

        Args:
            title: Note title.

        Returns:
            Sorted descendant note slugs.
        """
        note = self._relationship_service.get_note(title)
        start_slug = note.slug

        relations = self._relationship_service.list_all()

        descendants: set[str] = set()
        frontier: set[str] = {start_slug}

        while frontier:
            next_frontier: set[str] = set()

            for relation in relations:
                if (
                    relation.source in frontier
                    and relation.target not in descendants
                    and relation.target != start_slug
                ):
                    descendants.add(relation.target)
                    next_frontier.add(relation.target)

            frontier = next_frontier

        return sorted(descendants)

    def statistics(self) -> GraphStatistics:
        """Return aggregate statistics for the complete note graph."""
        notes = self._relationship_service.list_notes()
        relations = self._relationship_service.list_all()

        total_nodes = len(notes)
        total_edges = len(relations)

        if total_nodes == 0:
            return GraphStatistics(
                total_nodes=0,
                total_edges=total_edges,
                orphan_nodes=0,
                root_nodes=0,
                leaf_nodes=0,
                average_degree=0.0,
                max_degree=0,
                density=0.0,
            )

        node_slugs = {note.slug for note in notes}

        incoming_degree: dict[str, int] = {
            slug: 0 for slug in node_slugs
        }
        outgoing_degree: dict[str, int] = {
            slug: 0 for slug in node_slugs
        }

        for relation in relations:
            if relation.source in outgoing_degree:
                outgoing_degree[relation.source] += 1

            if relation.target in incoming_degree:
                incoming_degree[relation.target] += 1

        degree = {
            slug: incoming_degree[slug] + outgoing_degree[slug]
            for slug in node_slugs
        }

        orphan_nodes = sum(
            1
            for slug in node_slugs
            if degree[slug] == 0
        )

        root_nodes = sum(
            1
            for slug in node_slugs
            if incoming_degree[slug] == 0
            and outgoing_degree[slug] > 0
        )

        leaf_nodes = sum(
            1
            for slug in node_slugs
            if outgoing_degree[slug] == 0
        )

        total_degree = sum(degree.values())
        average_degree = total_degree / total_nodes

        max_degree = max(degree.values(), default=0)

        density = (
            0.0
            if total_nodes <= 1
            else total_edges / (total_nodes * (total_nodes - 1))
        )

        return GraphStatistics(
            total_nodes=total_nodes,
            total_edges=total_edges,
            orphan_nodes=orphan_nodes,
            root_nodes=root_nodes,
            leaf_nodes=leaf_nodes,
            average_degree=average_degree,
            max_degree=max_degree,
            density=density,
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
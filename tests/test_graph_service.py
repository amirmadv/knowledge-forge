"""Tests for the KnowledgeForge graph service."""

from pathlib import Path

import pytest

from knowledgeforge.domain.graph import GraphService
from knowledgeforge.domain.note import NoteService
from knowledgeforge.domain.relationship import (
    RelationshipService,
    RelationshipType,
)


def create_graph_services(
    tmp_path: Path,
) -> tuple[
    NoteService,
    RelationshipService,
    GraphService,
]:
    """Create the services required for graph tests."""
    vault_path = tmp_path / "vault"

    note_service = NoteService(vault_path)
    relationship_service = RelationshipService(
        vault_path,
        note_service=note_service,
    )
    graph_service = GraphService(
        vault_path,
        relationship_service=relationship_service,
    )

    return (
        note_service,
        relationship_service,
        graph_service,
    )


def test_outgoing_links_returns_relationships(
    tmp_path: Path,
) -> None:
    """Outgoing links should return relationships from the note."""
    note_service, relationship_service, graph_service = (
        create_graph_services(tmp_path)
    )

    note_service.create("Machine Learning")
    note_service.create("Linear Regression")
    note_service.create("Gradient Descent")

    relationship_service.add(
        "Machine Learning",
        "Linear Regression",
    )
    relationship_service.add(
        "Machine Learning",
        "Gradient Descent",
    )

    results = graph_service.outgoing_links("Machine Learning")

    assert len(results) == 2
    assert [result.target for result in results] == [
        "gradient-descent",
        "linear-regression",
    ]


def test_incoming_links_returns_relationships(
    tmp_path: Path,
) -> None:
    """Incoming links should return relationships targeting the note."""
    note_service, relationship_service, graph_service = (
        create_graph_services(tmp_path)
    )

    note_service.create("Machine Learning")
    note_service.create("Linear Regression")
    note_service.create("Statistics")

    relationship_service.add(
        "Machine Learning",
        "Linear Regression",
    )
    relationship_service.add(
        "Statistics",
        "Linear Regression",
    )

    results = graph_service.incoming_links(
        "Linear Regression",
    )

    assert len(results) == 2
    assert [result.source for result in results] == [
        "machine-learning",
        "statistics",
    ]


def test_backlinks_returns_incoming_relationships(
    tmp_path: Path,
) -> None:
    """Backlinks should identify notes linking to the target note."""
    note_service, relationship_service, graph_service = (
        create_graph_services(tmp_path)
    )

    note_service.create("Machine Learning")
    note_service.create("Linear Regression")
    note_service.create("Statistics")

    relationship_service.add(
        "Machine Learning",
        "Linear Regression",
        RelationshipType.RELATED,
    )
    relationship_service.add(
        "Statistics",
        "Linear Regression",
        RelationshipType.SUPPORTS,
    )

    results = graph_service.backlinks(
        "Linear Regression",
    )

    assert len(results) == 2
    assert {result.source for result in results} == {
        "machine-learning",
        "statistics",
    }


def test_neighbors_include_incoming_and_outgoing_notes(
    tmp_path: Path,
) -> None:
    """Neighbors should include both directions."""
    note_service, relationship_service, graph_service = (
        create_graph_services(tmp_path)
    )

    note_service.create("Machine Learning")
    note_service.create("Linear Regression")
    note_service.create("Statistics")

    relationship_service.add(
        "Machine Learning",
        "Linear Regression",
    )
    relationship_service.add(
        "Statistics",
        "Machine Learning",
    )

    neighbors = graph_service.neighbors(
        "Machine Learning",
    )

    assert neighbors == [
        "linear-regression",
        "statistics",
    ]


def test_neighbors_are_unique(
    tmp_path: Path,
) -> None:
    """Neighbors should not contain duplicate note slugs."""
    note_service, relationship_service, graph_service = (
        create_graph_services(tmp_path)
    )

    note_service.create("Machine Learning")
    note_service.create("Linear Regression")

    relationship_service.add(
        "Machine Learning",
        "Linear Regression",
        RelationshipType.RELATED,
    )
    relationship_service.add(
        "Linear Regression",
        "Machine Learning",
        RelationshipType.SUPPORTS,
    )

    neighbors = graph_service.neighbors(
        "Machine Learning",
    )

    assert neighbors == ["linear-regression"]


def test_graph_depth_zero_contains_only_start_node(
    tmp_path: Path,
) -> None:
    """A zero-depth graph should contain only the starting node."""
    note_service, relationship_service, graph_service = (
        create_graph_services(tmp_path)
    )

    note_service.create("Machine Learning")
    note_service.create("Linear Regression")

    relationship_service.add(
        "Machine Learning",
        "Linear Regression",
    )

    result = graph_service.graph(
        "Machine Learning",
        depth=0,
    )

    assert [node.slug for node in result.nodes] == [
        "machine-learning",
    ]
    assert result.edges == ()


def test_graph_depth_one_contains_direct_neighbors(
    tmp_path: Path,
) -> None:
    """Depth one should include directly connected notes."""
    note_service, relationship_service, graph_service = (
        create_graph_services(tmp_path)
    )

    note_service.create("Machine Learning")
    note_service.create("Linear Regression")
    note_service.create("Gradient Descent")
    note_service.create("Python")

    relationship_service.add(
        "Machine Learning",
        "Linear Regression",
    )
    relationship_service.add(
        "Machine Learning",
        "Gradient Descent",
    )
    relationship_service.add(
        "Python",
        "Machine Learning",
    )

    result = graph_service.graph(
        "Machine Learning",
        depth=1,
    )

    assert [node.slug for node in result.nodes] == [
        "gradient-descent",
        "linear-regression",
        "machine-learning",
        "python",
    ]

    assert len(result.edges) == 3


def test_graph_depth_two_traverses_multiple_levels(
    tmp_path: Path,
) -> None:
    """Depth two should traverse two relationship levels."""
    note_service, relationship_service, graph_service = (
        create_graph_services(tmp_path)
    )

    note_service.create("Machine Learning")
    note_service.create("Linear Regression")
    note_service.create("Gradient Descent")
    note_service.create("Optimization")
    note_service.create("Python")

    relationship_service.add(
        "Machine Learning",
        "Linear Regression",
    )
    relationship_service.add(
        "Linear Regression",
        "Gradient Descent",
    )
    relationship_service.add(
        "Gradient Descent",
        "Optimization",
    )
    relationship_service.add(
        "Python",
        "Machine Learning",
    )

    result = graph_service.graph(
        "Machine Learning",
        depth=2,
    )

    assert [node.slug for node in result.nodes] == [
        "gradient-descent",
        "linear-regression",
        "machine-learning",
        "python",
    ]

    assert len(result.edges) == 3


def test_graph_contains_only_edges_between_included_nodes(
    tmp_path: Path,
) -> None:
    """Graph edges should connect nodes included in the subgraph."""
    note_service, relationship_service, graph_service = (
        create_graph_services(tmp_path)
    )

    note_service.create("Machine Learning")
    note_service.create("Linear Regression")
    note_service.create("Gradient Descent")
    note_service.create("Optimization")

    relationship_service.add(
        "Machine Learning",
        "Linear Regression",
    )
    relationship_service.add(
        "Linear Regression",
        "Gradient Descent",
    )
    relationship_service.add(
        "Gradient Descent",
        "Optimization",
    )

    result = graph_service.graph(
        "Machine Learning",
        depth=1,
    )

    assert len(result.nodes) == 2
    assert len(result.edges) == 1

    edge = result.edges[0]

    assert edge.source == "machine-learning"
    assert edge.target == "linear-regression"


def test_graph_rejects_negative_depth(
    tmp_path: Path,
) -> None:
    """Negative graph depth should be rejected."""
    note_service, _, graph_service = create_graph_services(tmp_path)

    note_service.create("Machine Learning")

    with pytest.raises(
        ValueError,
        match="Graph depth cannot be negative",
    ):
        graph_service.graph(
            "Machine Learning",
            depth=-1,
        )
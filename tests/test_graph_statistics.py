"""Tests for KnowledgeForge graph statistics."""

from pathlib import Path

from knowledgeforge.application.commands import (
    add_note_relationship,
    create_note,
    get_graph_statistics,
)
from knowledgeforge.domain.graph import GraphService, GraphStatistics
from knowledgeforge.domain.note import NoteService
from knowledgeforge.domain.relationship import RelationshipService


def create_graph_services(
    tmp_path: Path,
) -> tuple[NoteService, RelationshipService, GraphService]:
    """Create isolated graph-related services."""
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


def test_graph_statistics_exposes_expected_fields() -> None:
    """Graph statistics should expose all expected fields."""
    statistics = GraphStatistics(
        total_nodes=10,
        total_edges=15,
        orphan_nodes=2,
        root_nodes=3,
        leaf_nodes=4,
        average_degree=3.0,
        max_degree=7,
        density=0.1667,
    )

    assert statistics.total_nodes == 10
    assert statistics.total_edges == 15
    assert statistics.orphan_nodes == 2
    assert statistics.root_nodes == 3
    assert statistics.leaf_nodes == 4
    assert statistics.average_degree == 3.0
    assert statistics.max_degree == 7
    assert statistics.density == 0.1667


def test_graph_service_statistics_calculates_graph_metrics(
    tmp_path: Path,
) -> None:
    """GraphService should calculate aggregate graph metrics."""
    (
        note_service,
        relationship_service,
        graph_service,
    ) = create_graph_services(tmp_path)

    note_service.create("Machine Learning")
    note_service.create("Linear Regression")
    note_service.create("Gradient Descent")
    note_service.create("Python")
    note_service.create("Isolated Note")

    relationship_service.add(
        "Machine Learning",
        "Linear Regression",
    )
    relationship_service.add(
        "Linear Regression",
        "Gradient Descent",
    )
    relationship_service.add(
        "Python",
        "Machine Learning",
    )

    statistics = graph_service.statistics()

    assert statistics.total_nodes == 5
    assert statistics.total_edges == 3
    assert statistics.orphan_nodes == 1
    assert statistics.root_nodes == 1
    assert statistics.leaf_nodes == 2
    assert statistics.average_degree == 1.2
    assert statistics.max_degree == 2
    assert statistics.density == 0.15


def test_graph_service_statistics_handles_empty_graph(
    tmp_path: Path,
) -> None:
    """Graph statistics should handle an empty graph."""
    _, _, graph_service = create_graph_services(tmp_path)

    statistics = graph_service.statistics()

    assert statistics.total_nodes == 0
    assert statistics.total_edges == 0
    assert statistics.orphan_nodes == 0
    assert statistics.root_nodes == 0
    assert statistics.leaf_nodes == 0
    assert statistics.average_degree == 0.0
    assert statistics.max_degree == 0
    assert statistics.density == 0.0


def test_application_command_returns_graph_statistics(
    tmp_path: Path,
) -> None:
    """The application command should expose graph statistics."""
    vault_path = tmp_path / "vault"

    create_note(
        title="Machine Learning",
        vault_path=vault_path,
    )

    create_note(
        title="Linear Regression",
        vault_path=vault_path,
    )

    add_note_relationship(
        source_title="Machine Learning",
        target_title="Linear Regression",
        vault_path=vault_path,
    )

    statistics = get_graph_statistics(
        vault_path=vault_path,
    )

    assert statistics.total_nodes == 2
    assert statistics.total_edges == 1
    assert statistics.orphan_nodes == 0
    assert statistics.root_nodes == 1
    assert statistics.leaf_nodes == 1
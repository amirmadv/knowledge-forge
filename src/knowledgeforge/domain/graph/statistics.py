"""Graph statistics model for KnowledgeForge."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GraphStatistics:
    """Summary statistics for a KnowledgeForge note graph."""

    total_nodes: int
    total_edges: int
    orphan_nodes: int
    root_nodes: int
    leaf_nodes: int
    average_degree: float
    max_degree: int
    density: float
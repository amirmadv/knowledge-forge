"""Offline retrieval evaluation metrics for KnowledgeForge."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RetrievalEvaluationCase:
    """One query with its expected relevant note identifiers."""

    query: str
    relevant: frozenset[str]


@dataclass(frozen=True, slots=True)
class RetrievalEvaluationResult:
    """Aggregated retrieval quality metrics."""

    precision_at_k: float
    recall_at_k: float
    mrr: float
    queries_evaluated: int


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Return the fraction of the first k results that are relevant."""
    if k <= 0:
        return 0.0
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    return sum(item in relevant for item in top_k) / len(top_k)


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Return the fraction of relevant items found in the first k results."""
    if k <= 0 or not relevant:
        return 0.0
    top_k = retrieved[:k]
    return sum(item in relevant for item in top_k) / len(relevant)


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    """Return the reciprocal rank of the first relevant result."""
    if not relevant:
        return 0.0
    for rank, item in enumerate(retrieved, start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def mean_reciprocal_rank(
    results: list[list[str]],
    relevant_sets: list[set[str]],
) -> float:
    """Return MRR across aligned retrieval result lists."""
    if not results or len(results) != len(relevant_sets):
        return 0.0
    scores = [
        reciprocal_rank(retrieved, relevant)
        for retrieved, relevant in zip(results, relevant_sets, strict=True)
    ]
    return sum(scores) / len(scores)


def evaluate_retrieval(
    cases: list[RetrievalEvaluationCase],
    ranked_results: list[list[str]],
    *,
    k: int = 5,
) -> RetrievalEvaluationResult:
    """Evaluate ranked retrieval output against a small offline gold set."""
    if not cases or len(cases) != len(ranked_results) or k <= 0:
        return RetrievalEvaluationResult(0.0, 0.0, 0.0, 0)

    relevant_sets = [set(case.relevant) for case in cases]
    precision = sum(
        precision_at_k(results, relevant, k)
        for results, relevant in zip(ranked_results, relevant_sets, strict=True)
    ) / len(cases)
    recall = sum(
        recall_at_k(results, relevant, k)
        for results, relevant in zip(ranked_results, relevant_sets, strict=True)
    ) / len(cases)
    mrr = mean_reciprocal_rank(ranked_results, relevant_sets)

    return RetrievalEvaluationResult(
        precision_at_k=precision,
        recall_at_k=recall,
        mrr=mrr,
        queries_evaluated=len(cases),
    )

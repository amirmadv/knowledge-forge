"""Offline retrieval evaluation metrics and quality gates for KnowledgeForge."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path


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


@dataclass(frozen=True, slots=True)
class RetrievalEvaluationItem:
    """Per-query retrieval metrics used by an evaluation report."""

    query: str
    relevant: frozenset[str]
    retrieved: tuple[str, ...]
    precision_at_k: float
    recall_at_k: float
    reciprocal_rank: float


@dataclass(frozen=True, slots=True)
class RetrievalQualityGate:
    """Pass/fail decision for configured retrieval quality thresholds."""

    passed: bool
    failures: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RetrievalEvaluationReport:
    """Complete retrieval evaluation report with per-query evidence."""

    result: RetrievalEvaluationResult
    items: tuple[RetrievalEvaluationItem, ...]
    k: int
    min_precision_at_k: float
    min_recall_at_k: float
    min_mrr: float
    quality_gate: RetrievalQualityGate


def precision_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Return the fraction of the first k results that are relevant."""
    if k <= 0:
        return 0.0
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    return sum(item in relevant for item in top_k) / len(top_k)


def recall_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Return the fraction of relevant items found in the first k results."""
    if k <= 0 or not relevant:
        return 0.0
    top_k = retrieved[:k]
    return sum(item in relevant for item in top_k) / len(relevant)


def reciprocal_rank(retrieved: Sequence[str], relevant: set[str]) -> float:
    """Return the reciprocal rank of the first relevant result."""
    if not relevant:
        return 0.0
    for rank, item in enumerate(retrieved, start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def mean_reciprocal_rank(
    results: Sequence[Sequence[str]],
    relevant_sets: Sequence[set[str]],
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
    cases: Sequence[RetrievalEvaluationCase],
    ranked_results: Sequence[Sequence[str]],
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


def evaluate_retriever(
    cases: Sequence[RetrievalEvaluationCase],
    retriever: Callable[[str, int], Sequence[str]],
    *,
    k: int = 5,
    min_precision_at_k: float = 0.0,
    min_recall_at_k: float = 0.0,
    min_mrr: float = 0.0,
) -> RetrievalEvaluationReport:
    """Run a retriever against a gold dataset and apply a quality gate."""
    _validate_thresholds(
        k=k,
        min_precision_at_k=min_precision_at_k,
        min_recall_at_k=min_recall_at_k,
        min_mrr=min_mrr,
    )

    ranked_results = [
        tuple(retriever(case.query, k))
        for case in cases
    ]
    result = evaluate_retrieval(cases, ranked_results, k=k)

    items = tuple(
        RetrievalEvaluationItem(
            query=case.query,
            relevant=case.relevant,
            retrieved=tuple(results[:k]),
            precision_at_k=precision_at_k(results, set(case.relevant), k),
            recall_at_k=recall_at_k(results, set(case.relevant), k),
            reciprocal_rank=reciprocal_rank(results, set(case.relevant)),
        )
        for case, results in zip(cases, ranked_results, strict=True)
    )

    failures: list[str] = []
    if result.precision_at_k < min_precision_at_k:
        failures.append(
            f"precision@{k} {result.precision_at_k:.3f} < {min_precision_at_k:.3f}"
        )
    if result.recall_at_k < min_recall_at_k:
        failures.append(
            f"recall@{k} {result.recall_at_k:.3f} < {min_recall_at_k:.3f}"
        )
    if result.mrr < min_mrr:
        failures.append(f"mrr {result.mrr:.3f} < {min_mrr:.3f}")

    return RetrievalEvaluationReport(
        result=result,
        items=items,
        k=k,
        min_precision_at_k=min_precision_at_k,
        min_recall_at_k=min_recall_at_k,
        min_mrr=min_mrr,
        quality_gate=RetrievalQualityGate(
            passed=not failures,
            failures=tuple(failures),
        ),
    )


def load_retrieval_evaluation_cases(path: Path) -> list[RetrievalEvaluationCase]:
    """Load retrieval evaluation cases from a JSON dataset."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Evaluation dataset not found: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid evaluation dataset: {path}") from exc

    if isinstance(payload, dict):
        payload = payload.get("cases")

    if not isinstance(payload, list):
        raise ValueError("Evaluation dataset must be a JSON array or an object with 'cases'.")

    cases: list[RetrievalEvaluationCase] = []
    for index, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Evaluation case {index} must be an object.")

        query = item.get("query")
        relevant = item.get("relevant")
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"Evaluation case {index} has an invalid query.")
        if not isinstance(relevant, list) or not relevant:
            raise ValueError(
                f"Evaluation case {index} must contain a non-empty 'relevant' list."
            )
        if not all(isinstance(value, str) and value.strip() for value in relevant):
            raise ValueError(
                f"Evaluation case {index} contains an invalid relevant identifier."
            )

        cases.append(
            RetrievalEvaluationCase(
                query=query.strip(),
                relevant=frozenset(value.strip() for value in relevant),
            )
        )

    if not cases:
        raise ValueError("Evaluation dataset cannot be empty.")

    return cases


def _validate_thresholds(
    *,
    k: int,
    min_precision_at_k: float,
    min_recall_at_k: float,
    min_mrr: float,
) -> None:
    """Validate evaluation parameters before executing a dataset."""
    if k <= 0:
        raise ValueError("k must be greater than zero.")

    thresholds = {
        "min_precision_at_k": min_precision_at_k,
        "min_recall_at_k": min_recall_at_k,
        "min_mrr": min_mrr,
    }
    for name, value in thresholds.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between 0.0 and 1.0.")

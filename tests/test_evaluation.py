from pathlib import Path

import pytest

from knowledgeforge.application.evaluation import (
    RetrievalEvaluationCase,
    evaluate_retrieval,
    evaluate_retriever,
    load_retrieval_evaluation_cases,
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_precision_at_k_counts_only_top_k() -> None:
    assert precision_at_k(["a", "b", "c"], {"a", "c"}, 2) == 0.5


def test_recall_at_k_counts_relevant_items_found() -> None:
    assert recall_at_k(["a", "b", "c"], {"a", "c"}, 2) == 0.5


def test_reciprocal_rank_uses_first_relevant_result() -> None:
    assert reciprocal_rank(["x", "b", "a"], {"a", "b"}) == 0.5


def test_mean_reciprocal_rank_averages_queries() -> None:
    assert mean_reciprocal_rank(
        [["a", "x"], ["x", "b"]],
        [{"a"}, {"b"}],
    ) == 0.75


def test_evaluate_retrieval_returns_aggregate_metrics() -> None:
    cases = [
        RetrievalEvaluationCase("linear regression", frozenset({"linear"})),
        RetrievalEvaluationCase("gradient descent", frozenset({"gradient"})),
    ]

    result = evaluate_retrieval(
        cases,
        [["linear", "noise"], ["noise", "gradient"]],
        k=2,
    )

    assert result.queries_evaluated == 2
    assert result.precision_at_k == 0.5
    assert result.recall_at_k == 1.0
    assert result.mrr == 0.75


def test_evaluate_retrieval_distinguishes_precision_and_recall() -> None:
    cases = [
        RetrievalEvaluationCase(
            "machine learning",
            frozenset({"ml", "machine-learning"}),
        ),
    ]

    result = evaluate_retrieval(
        cases,
        [["ml", "noise"]],
        k=2,
    )

    assert result.precision_at_k == 0.5
    assert result.recall_at_k == 0.5


def test_evaluate_retrieval_rejects_misaligned_inputs() -> None:
    result = evaluate_retrieval(
        [RetrievalEvaluationCase("q", frozenset({"a"}))],
        [],
    )

    assert result.queries_evaluated == 0
    assert result.precision_at_k == 0.0
    assert result.recall_at_k == 0.0
    assert result.mrr == 0.0


def test_evaluate_retriever_builds_per_query_report_and_passes_gate() -> None:
    cases = [
        RetrievalEvaluationCase("first", frozenset({"a"})),
        RetrievalEvaluationCase("second", frozenset({"b"})),
    ]

    report = evaluate_retriever(
        cases,
        lambda query, limit: {
            "first": ["a", "noise"],
            "second": ["noise", "b"],
        }[query][:limit],
        k=2,
        min_precision_at_k=0.5,
        min_recall_at_k=1.0,
        min_mrr=0.75,
    )

    assert report.quality_gate.passed
    assert report.quality_gate.failures == ()
    assert report.result.precision_at_k == 0.5
    assert report.result.recall_at_k == 1.0
    assert report.result.mrr == 0.75
    assert len(report.items) == 2
    assert report.items[1].reciprocal_rank == 0.5


def test_evaluate_retriever_fails_gate_with_explanations() -> None:
    case = RetrievalEvaluationCase("first", frozenset({"a"}))

    report = evaluate_retriever(
        [case],
        lambda _query, _limit: ["noise"],
        k=1,
        min_precision_at_k=1.0,
        min_recall_at_k=1.0,
        min_mrr=1.0,
    )

    assert not report.quality_gate.passed
    assert len(report.quality_gate.failures) == 3
    assert "precision@1" in report.quality_gate.failures[0]
    assert "recall@1" in report.quality_gate.failures[1]
    assert "mrr" in report.quality_gate.failures[2]


def test_evaluate_retriever_rejects_invalid_thresholds() -> None:
    with pytest.raises(ValueError, match="min_mrr"):
        evaluate_retriever(
            [RetrievalEvaluationCase("q", frozenset({"a"}))],
            lambda _query, _limit: ["a"],
            min_mrr=1.1,
        )


def test_load_retrieval_evaluation_cases_supports_array_dataset(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "retrieval.json"
    dataset.write_text(
        '[{"query": "linear regression", "relevant": ["linear-regression"]}]',
        encoding="utf-8",
    )

    cases = load_retrieval_evaluation_cases(dataset)

    assert cases == [
        RetrievalEvaluationCase(
            "linear regression",
            frozenset({"linear-regression"}),
        )
    ]


def test_load_retrieval_evaluation_cases_supports_object_dataset(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "retrieval.json"
    dataset.write_text(
        '{"cases": [{"query": "gradient descent", "relevant": ["gradient"]}]}',
        encoding="utf-8",
    )

    cases = load_retrieval_evaluation_cases(dataset)

    assert cases[0].query == "gradient descent"
    assert cases[0].relevant == frozenset({"gradient"})


def test_load_retrieval_evaluation_cases_rejects_invalid_dataset(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "retrieval.json"
    dataset.write_text('{"cases": []}', encoding="utf-8")

    with pytest.raises(ValueError, match="cannot be empty"):
        load_retrieval_evaluation_cases(dataset)

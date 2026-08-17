from knowledgeforge.application.evaluation import (
    RetrievalEvaluationCase,
    evaluate_retrieval,
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
        RetrievalEvaluationCase(
            "linear regression",
            frozenset({"linear"}),
        ),
        RetrievalEvaluationCase(
            "gradient descent",
            frozenset({"gradient"}),
        ),
    ]

    result = evaluate_retrieval(
        cases,
        [["linear", "noise"], ["noise", "gradient"]],
        k=2,
    )

    assert result.queries_evaluated == 2
    assert result.precision_at_k == 0.5

    # Both relevant documents are retrieved within top-2.
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

    # One of two retrieved results is relevant.
    assert result.precision_at_k == 0.5

    # One of two relevant results was retrieved.
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
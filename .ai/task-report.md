# Task Report — KF-006 Retrieval Explainability & Evaluation

## Status

Implemented retrieval explainability and the first offline evaluation slice on `agent/persistent-semantic-index`.

## Delivered

- Preserved the existing hybrid retrieval API.
- Added immutable `RetrievalEvidence` at the application boundary.
- Added `HybridRetriever.search_with_evidence()`.
- Exposed semantic, lexical, and metadata scores plus weighted contributions.
- Added deterministic human-readable retrieval reasons and deterministic tie-breaking.
- Preserved bounded context construction and stable `[S1]`, `[S2]`, ... source markers.
- Added offline retrieval metrics in `application/evaluation.py`:
  - `precision_at_k`
  - `recall_at_k`
  - `reciprocal_rank`
  - `mean_reciprocal_rank`
  - `evaluate_retrieval`
- Added `RetrievalEvaluationCase` and `RetrievalEvaluationResult` DTOs.
- Added unit tests covering the metrics and aggregate evaluation behavior.

## Architecture

Evaluation is intentionally provider-independent. It accepts ranked note identifiers and a small gold set, so retrieval quality can be measured without calling an LLM.

## Validation

The latest locally reported baseline before these changes was 121 passing tests with clean Ruff output. The new files have not been executed in the user's local checkout yet.

Run locally after pulling the branch:

```powershell
python -m uv run ruff check .
python -m uv run pytest -vv
```

## Next Step

Expose `RetrievalEvidence` through an application/CLI explain command, then add a small real KnowledgeForge evaluation dataset and regression thresholds for Precision@K, Recall@K, and MRR.

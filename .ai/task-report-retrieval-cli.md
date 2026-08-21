# Task Report — Retrieval Explain CLI

## Status

Implemented on `agent/retrieval-cli-evaluation`, based on the green persistent semantic-index branch.

## Delivered

- Exposed `KnowledgeAgent.search_with_evidence()` as an application-level retrieval use case.
- Added `knowledgeforge-ai search <query>`.
- Added `knowledgeforge-ai search <query> --explain` for score breakdown and retrieval reasons.
- Added CLI tests for ranking output, explain output, result limits, and empty results.
- Added an application test proving retrieval evidence is available without an LLM call.
- Documented the boundary decision in ADR-0006.
- Updated README usage and architecture notes.

## Design constraints preserved

- `HybridRetriever` remains the single owner of ranking weights and semantics.
- `RetrievalEvidence` remains an application-level DTO.
- The domain layer has no AI/retrieval dependency.
- Existing `ask` and `chat` behavior is unchanged.
- The CLI renders evidence and does not duplicate ranking calculations.

## Validation

The parent branch was locally validated at 135 passing tests with clean Ruff output before this slice.
This branch was edited through the repository API and therefore has not been executed in the user's local checkout yet.

Run locally:

```powershell
python -m uv run ruff check .
python -m uv run pytest -vv
```

## Next milestone

Build a small regression evaluation dataset from real KnowledgeForge note/query pairs, add configurable metric thresholds for Precision@K, Recall@K, and MRR, and expose the evaluation as a developer command before expanding agent capabilities.

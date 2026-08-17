# KF-006 — Retrieval Explainability

## Status
Implemented on `agent/persistent-semantic-index`.

## Changes
- Added `RetrievalEvidence` as an immutable application-level DTO.
- Added `HybridRetriever.search_with_evidence()` while preserving the existing `search()` API.
- Centralized score contributions using the existing semantic, lexical, and metadata weights.
- Added human-readable retrieval reasons for semantic, lexical, and metadata signals.
- Made ranking tie-breaking explicitly deterministic with case-insensitive title ordering.
- Added tests covering score breakdown, reason generation, deterministic ordering, and zero-signal filtering.
- Preserved `ContextBuilder` source markers and existing grounded AI behavior.

## Validation
The implementation was committed directly to the branch through GitHub's repository API. The local checkout is not available in this execution environment, so the full Ruff/Pytest suite could not be executed here after the changes.

Before merging, run:

```powershell
python -m uv run ruff check .
python -m uv run pytest -vv
```

Expected baseline before this slice: 121 tests passing.

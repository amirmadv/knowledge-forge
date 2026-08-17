# Task Report — KF-006 Grounded Retrieval

## Status

Implemented the first production-oriented slice of KF-006 on `agent/persistent-semantic-index`.

## Specification

Strengthen the retrieval-to-agent boundary so that retrieved knowledge can be explicitly referenced by stable source markers in the LLM context.

Requirements:

- Preserve the existing hybrid retrieval API.
- Keep context size bounded.
- Assign deterministic source markers to retrieved notes.
- Include source titles in the prompt separately from note content.
- Instruct the model to cite factual claims using only available source markers.
- Preserve existing graph context and source-title behavior.

## Implementation

### `src/knowledgeforge/application/retrieval.py`

- Added `SourceRef`.
- Added `ContextBuilder.build_with_sources()`.
- Preserved `ContextBuilder.build()` as a compatibility wrapper.
- Added `[S1]`, `[S2]`, ... markers to note context sections.
- Returned source metadata alongside rendered context.
- Preserved character and note-count limits.

### `src/knowledgeforge/application/ai.py`

- `KnowledgeAgent.ask()` now consumes grounded context with source references.
- Prompt explicitly requires source markers for factual claims.
- Prompt prevents fabricated source markers/titles.
- Prompt asks the model to identify conflicting sources.
- Existing `AIAnswer.sources` contract remains unchanged.

### Tests

- Added retrieval source-reference coverage.
- Extended agent tests to verify source markers and grounding instructions.

## Validation

Run locally after pulling the branch:

```powershell
python -m uv run ruff check .
python -m uv run pytest -vv
```

Expected result: all checks pass.

## Next Step

KF-006 next slice: make retrieval explainable at the application boundary by exposing ranking evidence (semantic, lexical, metadata, and graph expansion reasons) without coupling the domain layer to the LLM provider.

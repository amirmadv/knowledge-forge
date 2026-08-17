# ADR-0005: Retrieval Explainability at the Application Boundary

## Status
Accepted

## Context
KnowledgeForge now combines persistent semantic retrieval with lexical and metadata signals. The AI layer also uses graph-aware context and stable source markers. Ranking quality needs to be inspectable without coupling retrieval to an LLM provider.

## Decision
Expose retrieval evidence from the application layer through `RetrievalEvidence` and `HybridRetriever.search_with_evidence()`.

Evidence contains:

- final score
- raw semantic, lexical, and metadata scores
- weighted contributions for each signal
- deterministic human-readable reasons

The existing `HybridRetriever.search()` API remains compatible and converts evidence back to `RetrievalMatch`.

## Consequences
- Retrieval behavior can be inspected and evaluated independently of the LLM.
- CLI/UI layers can display why a note ranked highly.
- Score weights remain centralized in `HybridRetriever`.
- Domain models remain independent from provider concerns.
- Future retrieval evaluation can consume the same evidence without duplicating ranking logic.

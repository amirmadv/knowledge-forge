# ADR-0006: Expose Explainable Retrieval Through the AI CLI

## Status

Accepted

## Context

KnowledgeForge already computes structured retrieval evidence at the application boundary.
The next useful boundary for developers and users is a small CLI command that can inspect
ranking behavior without invoking an LLM answer.

## Decision

Add `knowledgeforge-ai search` with an optional `--explain` flag.

The command delegates to `KnowledgeAgent.search_with_evidence()` and renders:

- final combined score;
- semantic score;
- lexical score;
- metadata score;
- deterministic human-readable reasons.

The CLI does not duplicate ranking logic and does not introduce provider-specific objects
into the domain layer.

The existing `ask` and `chat` commands remain unchanged.

## Consequences

Positive:

- retrieval can be inspected independently from generated answers;
- ranking regressions are easier to diagnose;
- future UI layers can reuse the same application evidence model;
- no additional persistence or vector-database dependency is introduced.

Trade-offs:

- the CLI still requires the configured AI client because semantic retrieval may use its
  embedding endpoint;
- graph expansion used by the AI answer pipeline is intentionally not presented as a
  ranked retrieval result; it remains context expansion rather than a ranking signal.

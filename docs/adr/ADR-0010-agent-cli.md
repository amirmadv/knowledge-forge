# ADR-0010: Bounded Agent CLI Contract

## Status

Accepted

## Context

KnowledgeForge now has a bounded provider-neutral runtime and an OpenAI-compatible provider adapter. The runtime needs a stable user-facing entry point so the new tool-using behavior can be exercised locally, debugged, and later evaluated without exposing provider-specific payloads through the CLI.

The existing `ask` command remains a retrieval-grounded single-shot path. The new agent command should therefore be explicit rather than silently changing the behavior of the established command.

## Decision

Add `knowledgeforge-ai agent <prompt>` as the explicit CLI entry point for bounded tool-using execution.

The command supports:

- human-readable `text` output by default;
- machine-readable `json` output through `--output json`;
- the final answer;
- termination reason;
- total tool-call count;
- model-step count;
- tool names and success/failure status in text mode;
- complete deterministic step/observation details in JSON mode.

Unsupported output formats fail before the agent is invoked.

The command delegates to `KnowledgeAgent.run_agent()` and does not construct provider-specific messages or inspect provider JSON itself.

## Consequences

### Positive

- Developers can exercise the new runtime from the existing CLI.
- JSON traces provide a stable input for future automated evaluation.
- The existing `ask` command remains backward-compatible.
- Provider-specific concerns remain behind the application/infrastructure boundary.

### Trade-offs

- The CLI exposes execution traces that may contain tool results, so callers should treat JSON output as potentially sensitive local knowledge.
- Streaming output is not included yet.
- Runtime safety limits remain configured in application code rather than exposed as arbitrary CLI overrides.

## Rejected Alternatives

### Replace `ask` with the new runtime

Rejected because the established `ask` path has different retrieval and citation behavior and should not change implicitly.

### Print provider-native JSON

Rejected because the CLI contract should remain stable if the provider adapter changes.

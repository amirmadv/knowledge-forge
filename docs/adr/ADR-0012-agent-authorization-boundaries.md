# ADR-0012 — Agent Authorization Boundaries

## Status

Accepted

## Context

KnowledgeForge now has a bounded provider-neutral agent runtime, deterministic offline evaluation, and a provider adapter capable of tool calling. The next safety boundary is mutation: an agent that can write to the local vault must not receive write capabilities accidentally, and a model-generated write request must not bypass application policy.

The existing tool registry already owns tool contracts and execution. The runtime owns iteration and tool-call guardrails. Authorization belongs at both boundaries:

1. provider exposure must hide unauthorized capabilities from the model;
2. runtime execution must reject unauthorized capabilities even if a caller manually constructs a write tool call.

## Decision

Introduce an explicit `ToolAccess` policy with two levels:

- `READ_ONLY` — read tools are exposed and executable; write tools are hidden from provider declarations and rejected if invoked directly;
- `WRITE` — read tools and explicitly registered write tools may be exposed and executed.

Every `ToolSpec` declares its required access level. The `KnowledgeToolRegistry` filters provider declarations using the requested policy and enforces the same policy during execution.

The `AgentRuntimeConfig` defaults to `READ_ONLY`, making read-only execution the safe default. The runtime passes its policy to both provider-tool exposure and registry execution.

The first mutation capability is `create_note`. It is registered only when the application explicitly builds a write-capable registry. The tool creates a note through the existing `NoteService` and returns the resulting note metadata/content as structured tool data.

The application `KnowledgeAgent` exposes `tools_for_access()` and `runtime(access=...)`. Its existing `tools` behavior remains read-only for backward compatibility. `run_agent()` also defaults to read-only and accepts an explicit access policy for controlled write workflows.

## Consequences

### Positive

- accidental model access to mutating tools is prevented by default;
- defense in depth exists at both tool discovery and execution boundaries;
- authorization failures become structured tool observations, allowing the planner to recover without crashing the runtime;
- deterministic tests can prove both denied and authorized mutation behavior;
- future write tools can declare their required access without changing runtime semantics.

### Negative

- callers that intentionally need mutations must explicitly opt into `ToolAccess.WRITE`;
- adding a new mutating tool requires deciding its access level and adding authorization coverage;
- this is an application-level authorization boundary, not an operating-system sandbox. File-system and process isolation remain separate concerns.

## Rejected Alternatives

### Expose all tools and rely on the system prompt

Rejected because prompt instructions are not an authorization mechanism. A model can still request a write tool.

### Put authorization only in the CLI

Rejected because other application callers could bypass the CLI. Authorization must be enforced in the application runtime/registry.

### Put authorization only in the runtime

Rejected because the model should not be told about capabilities it is not permitted to use. Provider declarations should be filtered as well as execution.

## Validation

The milestone adds deterministic tests for:

- write tools hidden under read-only provider exposure;
- direct write execution rejected under read-only policy;
- authorized `create_note` execution succeeding;
- the runtime defaulting to read-only even when a write-capable registry is supplied.

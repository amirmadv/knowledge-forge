# Task Report — KF-012 Agent Authorization Boundaries

## Status

Advanced the bounded KnowledgeForge agent with explicit read-only versus write-capable tool authorization while preserving the provider-neutral runtime boundary.

## Delivered

### Tool authorization model

- Added `ToolAccess` with `READ_ONLY` and `WRITE` policies.
- Extended `ToolSpec` with an explicit access requirement.
- Added `ToolAuthorizationError` for deterministic authorization failures.
- Updated `KnowledgeToolRegistry.provider_tools()` to expose only tools allowed by the active policy.
- Updated `KnowledgeToolRegistry.execute()` to enforce the same policy at execution time.

### Runtime enforcement

- Extended `AgentRuntimeConfig` with `tool_access`.
- The runtime defaults to `READ_ONLY`.
- The runtime passes the policy into provider-tool exposure and tool execution.
- Unauthorized write requests are returned to the planner as structured failed tool observations, preserving the existing tool-error recovery behavior.

### First controlled write workflow

- Added `CreateNoteTool` as the first write-capable agent tool.
- Policy-aware runtime registries can contain the write capability, but `READ_ONLY` provider exposure hides it and `READ_ONLY` execution rejects it. This gives defense in depth instead of relying on discovery filtering alone.
- The tool creates a note through the existing `NoteService`, then applies the requested Markdown content through the existing update path.
- The tool returns structured note metadata and content.

### Application boundary

- `KnowledgeAgent.tools` remains read-only by default for backward compatibility.
- Added `KnowledgeAgent.tools_for_access()` for explicit policy-aware registry construction.
- Added `KnowledgeAgent.runtime(access=...)` for explicit runtime policy selection.
- Added `KnowledgeAgent.run_agent(..., access=...)`, defaulting to read-only.

### Documentation

- Added `docs/adr/ADR-0012-agent-authorization-boundaries.md`.

## Tests

Added deterministic coverage for:

- write tools being hidden from read-only provider declarations;
- direct write execution being rejected under read-only policy;
- authorized `create_note` execution;
- runtime read-only default behavior;
- write-tool metadata/access declarations.

## Current Architecture

```text
                         +----------------------+
                         | KnowledgeAgent       |
                         +----------+-----------+
                                    |
                    explicit ToolAccess policy
                                    |
                    +---------------+---------------+
                    |                               |
               READ_ONLY                         WRITE
                    |                               |
                    v                               v
          core read tools                 core read tools
          + hidden writes                 + create_note
                    |                               |
                    +---------------+---------------+
                                    v
                         KnowledgeToolRegistry
                         /                  \
                provider_tools()          execute()
                    |                       |
             exposure filter          auth enforcement
                    \                       /
                     +---------+-----------+
                               v
                         AgentRuntime
                               |
                    bounded execution guardrails
                               |
                         AgentEvaluation
```

## Guardrails

The runtime continues to enforce:

- 8 model iterations per run;
- 16 total tool calls per run;
- consecutive identical tool-call detection;
- unique tool-call IDs within a model response;
- isolated tool failures returned as structured observations;
- explicit tool access policy with read-only default.

## Validation

The previous milestone reached 181 passing tests after fixing the `create_note(content=...)` application integration regression.

The previous checkout still reported two Ruff cleanup findings: import ordering in `src/knowledgeforge/ai_cli.py` and a simplifiable conditional in `src/knowledgeforge/application/agent_evaluation.py`. The latter is now corrected in this milestone; the `ai_cli.py` import ordering remains a small cleanup item for the next verification pass.

## Next Step

Next milestone: **controlled write workflows beyond note creation and policy-aware agent evaluation**.

The next implementation should add authorization-aware evaluation cases and then introduce the next mutation only after its input validation, policy behavior, and deterministic evaluation are covered. Candidate mutations should remain behind explicit `ToolAccess.WRITE` until a stronger permission model is designed.

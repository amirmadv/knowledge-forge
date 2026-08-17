# Task Report — KF-008 Provider-Neutral Agent Runtime

## Status

Implemented the first bounded provider-neutral agent execution loop on `agent/retrieval-cli-evaluation`.

## Delivered

- Added `AgentToolCall` for model-requested tool invocations.
- Added `AgentModelResponse` for provider-neutral model decisions.
- Added `AgentMessage` for provider-neutral conversation state.
- Added `ToolObservation` for successful and failed tool executions.
- Added `AgentStep`, `AgentTrace`, and `AgentRunResult` for deterministic execution traces.
- Added `AgentPlanner` protocol as the future provider adapter boundary.
- Added `AgentRuntimeConfig` with explicit safety limits.
- Added `AgentRuntime` with:
  - bounded iteration loop
  - total tool-call limit
  - duplicate tool-call loop protection
  - response validation
  - tool-error isolation
  - final-answer termination
- Exposed `KnowledgeAgent.runtime` as the runtime over its existing provider-neutral tool registry.
- Added ADR-0008 documenting the runtime architecture and provider boundary.
- Added focused runtime tests covering successful tool execution, planner recovery after tool failure, prompt validation, iteration limits, tool-call limits, duplicate-call protection, and duplicate call IDs.

## Architecture

The agent execution boundary is now:

```text
KnowledgeAgent
    |
    +-- AgentRuntime
          |
          +-- AgentPlanner (provider adapter boundary)
          |
          +-- KnowledgeToolRegistry
                |
                +-- search_knowledge
                +-- inspect_note_graph
                +-- get_note
                +-- list_related_notes
```

The runtime knows only application-level contracts. A future OpenAI-compatible, Ollama, or other provider adapter can translate its native response into `AgentModelResponse` and translate `AgentMessage` values back to provider messages.

## Guardrails

The default runtime limits are:

- 8 model iterations per run
- 16 total tool calls per run
- 2 consecutive identical tool-call signatures

Tool failures are returned to the planner as structured observations instead of immediately terminating the run.

## Validation

The latest user-reported local baseline before this milestone was 157 passing tests with clean Ruff output. The new runtime commits were created directly in GitHub and still need to be pulled into the user's Windows checkout and validated locally.

Run locally after pulling:

```powershell
python -m uv run ruff check .
python -m uv run pytest -vv
```

## Next Step

Implement the first concrete provider adapter for the existing OpenAI-compatible client. It should translate provider tool-call responses into `AgentModelResponse`, translate runtime messages back into provider chat messages, and remain independently testable with mocked HTTP responses. Do not couple the runtime to the provider implementation.

# Task Report — KF-007 Agent Tool Architecture

## Status

Implemented the first provider-neutral agent tool boundary on `agent/retrieval-cli-evaluation`.

## Delivered

- Added `ToolSpec` for stable tool metadata and JSON-compatible input schemas.
- Added `ToolResult` for deterministic tool outputs.
- Added the `KnowledgeTool` protocol.
- Added `KnowledgeToolRegistry` with:
  - registration
  - duplicate protection
  - deterministic lookup errors
  - argument-object validation
  - execution
  - provider-neutral function declarations
- Exposed the registry through `KnowledgeAgent.tools`.
- Added four core read/search tools:
  - `search_knowledge`
  - `inspect_note_graph`
  - `get_note`
  - `list_related_notes`
- Preserved retrieval evidence, graph relationships, note metadata, and Markdown content in tool results.
- Added ADR-0007 describing the boundary and why LLM planning is kept separate from tool execution.
- Added focused unit tests for tool discovery, schemas, execution, validation, duplicate protection, retrieval evidence, graph data, and note content.

## Architecture

The agent now has an explicit capability boundary:

```text
KnowledgeAgent
    |
    +-- KnowledgeToolRegistry
          |
          +-- search_knowledge
          +-- inspect_note_graph
          +-- get_note
          +-- list_related_notes
```

The registry is intentionally not an LLM planner. A future provider adapter can consume `provider_tools()` and translate model tool calls into `registry.execute(...)` without coupling the model layer to domain services.

Mutation tools such as `create_note` and `improve_note` are intentionally deferred until tool authorization and mutation semantics are defined.

## Validation

The user's previous local baseline was 151 passing tests with clean Ruff output. The latest tool-architecture commits have not been executed in the user's local checkout because this environment does not contain the user's Windows working directory.

Run locally after pulling:

```powershell
python -m uv run ruff check .
python -m uv run pytest -vv
```

## Next Step

Implement the provider-neutral agent execution loop: represent a model response as text or tool calls, execute approved tools through `KnowledgeToolRegistry`, append tool results to the conversation, and continue until the model returns a final grounded answer. The first implementation should remain testable without a live LLM.

# ADR-0007: Agent Tool Architecture

## Status

Accepted

## Context

KnowledgeForge already has stable application boundaries for retrieval, graph inspection, note access, and semantic indexing. The next milestone is to make those capabilities usable by an agent without coupling the agent planner to internal domain services.

The system also needs a provider-neutral representation of tools so an eventual LLM tool-calling adapter can expose the same capabilities to OpenAI-compatible providers, local models, or deterministic tests.

## Decision

Introduce a small application-layer tool boundary:

- `ToolSpec` describes a stable public tool name, description, and JSON-compatible input schema.
- `ToolResult` contains a deterministic tool name and JSON-compatible result payload.
- `KnowledgeTool` defines the execution contract.
- `KnowledgeToolRegistry` owns registration, lookup, execution, and provider-neutral function declarations.
- Core tools are:
  - `search_knowledge`
  - `inspect_note_graph`
  - `get_note`
  - `list_related_notes`

`KnowledgeAgent.tools` exposes the registry. The registry does not perform LLM planning; it only defines and executes capabilities. A later planner/provider adapter can decide which tools to call and can consume `provider_tools()` without depending on domain implementation details.

Tool arguments are validated at the application boundary and tool outputs are JSON-like structures suitable for serialization into an LLM tool result message.

## Consequences

### Positive

- Agent planning is separated from business logic.
- Tool contracts are deterministic and independently testable.
- Provider-specific function-calling formats can be adapted later without changing domain services.
- Retrieval evidence remains available to the agent instead of being reduced to note titles.
- Graph and note operations become reusable by future CLI, API, and UI layers.

### Trade-offs

- The first milestone intentionally exposes only core read/search capabilities.
- Copilot write operations such as `create_note` and `improve` will be added after tool authorization and mutation semantics are defined.
- The current `provider_tools()` shape targets the common OpenAI-compatible function-tool convention; providers can be adapted at the infrastructure boundary later.

## Rejected Alternatives

### Directly expose domain services to the LLM adapter

Rejected because it leaks internal service APIs into the agent/provider boundary and makes future refactoring harder.

### Put tool definitions inside the CLI

Rejected because the tools are application capabilities, not presentation-layer commands. CLI, REST API, and future UI clients should be able to reuse the same registry.

# ADR-0009: OpenAI-Compatible Agent Provider Adapter

## Status

Accepted

## Context

The provider-neutral `AgentRuntime` now defines the execution contract for KnowledgeForge agents, while `KnowledgeToolRegistry` defines the approved application tools. A concrete provider adapter is required to connect that runtime to the existing OpenAI-compatible HTTP client without moving provider-specific message formats into the application layer.

KnowledgeForge intentionally uses a small standard-library HTTP client rather than coupling the core application to an LLM SDK. The adapter therefore needs to translate both directions:

- provider-neutral `AgentMessage` values into OpenAI Chat Completions messages;
- OpenAI-compatible assistant responses and function tool calls into `AgentModelResponse` and `AgentToolCall`.

The adapter must also preserve tool-call IDs and arguments so the runtime can send exact tool observations back to the provider.

## Decision

Introduce `knowledgeforge.infrastructure.ai.agent_planner.OpenAICompatibleAgentPlanner` as the first concrete `AgentPlanner` implementation.

The adapter:

- depends on `OpenAICompatibleClient` only;
- sends the runtime's provider-neutral messages and tool declarations to `/chat/completions`;
- serializes assistant tool calls using OpenAI-compatible `tool_calls` entries;
- serializes tool observations as `role=tool` messages with `tool_call_id` and tool name;
- parses function arguments from JSON strings into dictionaries;
- rejects malformed provider responses instead of guessing missing structure;
- optionally injects one system prompt when the runtime has not already supplied one;
- remains independently testable with a fake client and no network access.

Extend `OpenAICompatibleClient` with a generic `chat_completion` operation. The existing `chat` operation continues to provide the simple text-only API and delegates to the same HTTP transport.

Expose the adapter through `KnowledgeAgent.planner` and add `KnowledgeAgent.run_agent()` as the application-level entry point for bounded tool-using execution. The provider-neutral runtime remains independently usable and does not import infrastructure code.

## Consequences

### Positive

- Live tool calling can now use the existing OpenAI-compatible endpoint.
- Provider-specific JSON stays in the infrastructure layer.
- The runtime remains independent from the provider and SDK choice.
- The same runtime guardrails apply to every future provider adapter.
- The adapter can support OpenAI-compatible services such as hosted APIs or local compatible gateways without changing application contracts.

### Trade-offs

- OpenAI-compatible providers must support the Chat Completions tool-calling shape used by the adapter.
- Provider-specific features outside that contract are intentionally not exposed yet.
- The current adapter does not stream partial responses.
- Mutation tools still require a separate authorization design; this milestone only connects the existing read-oriented core tools.

## Rejected Alternatives

### Put OpenAI message conversion inside `AgentRuntime`

Rejected because it would make the application runtime provider-aware and weaken the adapter boundary established by ADR-0008.

### Add an LLM SDK dependency to the application layer

Rejected because KnowledgeForge already has a small provider-neutral HTTP client and does not need an SDK to define its core agent contracts.

### Keep live tool calling outside `KnowledgeAgent`

Rejected because the project now has a bounded runtime and tool registry that should be reachable through one application-level agent entry point.

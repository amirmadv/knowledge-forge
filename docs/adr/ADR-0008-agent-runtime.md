# ADR-0008: Provider-Neutral Agent Runtime

## Status

Accepted

## Context

KnowledgeForge now has a provider-neutral tool registry, but tool declarations alone do not provide an agent execution model. The system needs a bounded loop that can consume a model decision, execute approved tools, return observations to the model, and stop on a grounded final response.

The runtime must remain independent of any particular LLM SDK so deterministic tests can exercise agent behavior without network access or provider credentials.

## Decision

Introduce `knowledgeforge.application.agent_runtime` with these contracts:

- `AgentToolCall` represents one model-requested tool invocation.
- `AgentModelResponse` represents model text and/or tool calls.
- `AgentMessage` is the provider-neutral conversation message exchanged with a planner adapter.
- `ToolObservation` represents successful or failed tool execution and can serialize to a tool message.
- `AgentStep` records one model decision and its observations.
- `AgentTrace` records the complete execution history and termination reason.
- `AgentRunResult` contains the final answer and trace.
- `AgentPlanner` is the small adapter protocol implemented by a future provider integration.
- `AgentRuntimeConfig` controls bounded execution.
- `AgentRuntime` owns the execution loop and calls only `KnowledgeToolRegistry` for tools.

The runtime enforces:

- non-empty prompts;
- validated model responses;
- unique tool-call IDs within one response;
- maximum iterations;
- maximum total tool calls;
- protection against repeated identical tool calls;
- tool-error isolation so a failed tool becomes an observation the planner can recover from.

The runtime does not know about OpenAI, Ollama, LangChain, or any other provider. Provider adapters are responsible for translating provider-specific messages and tool-call payloads into these application contracts.

## Consequences

### Positive

- Agent orchestration is deterministic and testable without a live model.
- Tool execution remains behind the existing registry boundary.
- Runtime traces provide an explicit debugging and evaluation artifact.
- Provider integrations can evolve independently from domain and application tools.
- Safety limits prevent unbounded tool loops and runaway execution.

### Trade-offs

- A provider adapter is still required before live LLM tool calling can use the runtime.
- The first runtime milestone does not include mutation authorization.
- Repeated identical tool calls are treated as a loop signal rather than being silently executed forever.

## Rejected Alternatives

### Put the execution loop inside `KnowledgeAgent`

Rejected because it would couple retrieval, conversation behavior, and provider orchestration into one application service.

### Let each provider implement its own tool loop

Rejected because guardrails, tracing, and tool semantics would diverge across providers and become difficult to test consistently.

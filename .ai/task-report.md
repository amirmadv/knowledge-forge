# Task Report — KF-009 OpenAI-Compatible Agent Adapter

## Status

Implemented the first concrete provider adapter for the provider-neutral KnowledgeForge agent runtime on `agent/retrieval-cli-evaluation`.

## Delivered

- Added `OpenAICompatibleAgentPlanner` under `infrastructure.ai`.
- Added generic `OpenAICompatibleClient.chat_completion()` support while preserving the existing text-only `chat()` API.
- Added bidirectional message translation between runtime contracts and OpenAI-compatible Chat Completions payloads.
- Added parsing and validation for provider function tool calls.
- Preserved provider tool-call IDs and JSON arguments across runtime turns.
- Added `KnowledgeAgent.planner` as the concrete provider adapter boundary.
- Added `KnowledgeAgent.run_agent()` as the application-level entry point for bounded tool-using execution.
- Added ADR-0009 documenting the provider adapter architecture and boundaries.
- Added adapter tests for final text responses, tool-call normalization, assistant/tool message translation, malformed arguments, and malformed provider responses.
- Added HTTP client tests for generic chat completion payloads and tool declarations.
- Added an application integration test proving a real `get_note` core tool can execute through the runtime and provider adapter before the final answer is returned.
- Fixed the two Ruff `UP037` findings in `agent_runtime.py`.

## Architecture

The agent execution boundary is now:

```text
KnowledgeAgent
    |
    +-- AgentRuntime ------------------+
    |                                  |
    |                                  +-- AgentPlanner protocol
    |                                           |
    |                                           +-- OpenAICompatibleAgentPlanner
    |                                                   |
    |                                                   +-- OpenAICompatibleClient
    |
    +-- KnowledgeToolRegistry
            |
            +-- search_knowledge
            +-- inspect_note_graph
            +-- get_note
            +-- list_related_notes
```

The runtime still knows only application-level contracts. The provider adapter owns provider-specific message and tool-call JSON. The existing HTTP client owns transport and endpoint details.

## Guardrails

The runtime continues to enforce:

- 8 model iterations per run
- 16 total tool calls per run
- 2 consecutive identical tool-call signatures
- unique tool-call IDs within a model response
- isolated tool failures returned as structured observations

## Validation

The user-reported local baseline before this milestone was 164 passing tests, with Ruff reporting two fixable type-annotation findings in the new runtime. This milestone includes the lint fix and new adapter/application tests.

After pulling the new commits into the Windows checkout, run:

```powershell
python -m uv run ruff check .
python -m uv run pytest -vv
```

Expected direction: all tests should remain green and Ruff should be clean.

## Next Step

Build the agent-facing CLI command around `KnowledgeAgent.run_agent()` with structured output and trace visibility. The command should expose the bounded runtime without leaking provider-specific payloads, support a human-readable mode first, and provide JSON output suitable for later evaluation and automated workflows. Mutation tools should remain out of the agent registry until authorization is explicitly designed.

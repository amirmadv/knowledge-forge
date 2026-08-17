# Task Report — KF-010 Bounded Agent CLI

## Status

Implemented the first explicit CLI entry point for the bounded KnowledgeForge tool-using agent on `agent/retrieval-cli-evaluation`.

## Delivered

### Provider integration

- Added `OpenAICompatibleAgentPlanner` as the concrete `AgentPlanner` adapter.
- Extended `OpenAICompatibleClient` with generic `chat_completion()` support.
- Kept provider-specific message and tool-call JSON inside infrastructure.
- Added strict parsing for tool-call IDs, function names, and JSON argument objects.

### Application integration

- Added `KnowledgeAgent.planner` for the configured provider adapter.
- Added `KnowledgeAgent.run_agent()` as the application-level bounded agent entry point.
- Kept `AgentRuntime` provider-neutral and independently testable.

### CLI

- Added `knowledgeforge-ai agent <prompt>`.
- Default output is human-readable text with final answer and trace summary.
- Added `--output json` for deterministic machine-readable execution traces.
- Added validation for unsupported output formats.
- Preserved the existing `ask`, `search`, `evaluate`, `chat`, `index`, and Copilot commands.

### Tests and architecture

- Added provider adapter unit tests.
- Added generic chat-completion HTTP payload tests.
- Added end-to-end application coverage for a real `get_note` tool call through the runtime and adapter.
- Added CLI tests for human-readable and JSON agent output.
- Added ADR-0009 for the provider adapter.
- Added ADR-0010 for the bounded agent CLI contract.
- Fixed the two Ruff `UP037` findings in `agent_runtime.py`.

## Current Architecture

```text
knowledgeforge-ai agent
          |
          v
KnowledgeAgent.run_agent()
          |
          v
    AgentRuntime
          |
          +---- AgentPlanner protocol
          |          |
          |          +---- OpenAICompatibleAgentPlanner
          |                       |
          |                       +---- OpenAICompatibleClient
          |
          +---- KnowledgeToolRegistry
                     |
                     +---- search_knowledge
                     +---- inspect_note_graph
                     +---- get_note
                     +---- list_related_notes
```

The runtime remains the single place for execution guardrails. The provider adapter remains the single place for OpenAI-compatible wire translation. The CLI consumes only application-level results.

## Guardrails

The runtime continues to enforce:

- 8 model iterations per run
- 16 total tool calls per run
- 2 consecutive identical tool-call signatures
- unique tool-call IDs within a model response
- isolated tool failures returned as structured observations

## Validation

The latest user-reported local baseline before these milestones was 164 passing tests, with two Ruff type-annotation findings. The code changes above include additional tests and the lint fixes.

After pulling the branch into the Windows checkout, run:

```powershell
python -m uv run ruff check .
python -m uv run pytest -vv
```

Then verify the new CLI surface with:

```powershell
python -m uv run knowledgeforge-ai agent "What is in my knowledge vault?"
python -m uv run knowledgeforge-ai agent "What is in my knowledge vault?" --output json
```

The second command is intended to become the stable machine-readable interface for future agent evaluation.

## Next Step

The next milestone should focus on **agent evaluation and observability**, not adding more tools yet. Build an offline agent-evaluation harness that can replay deterministic planner decisions, measure tool-call efficiency and successful task completion, detect repeated-tool behavior, and produce JSON reports. Once that harness is stable, introduce explicit read/write authorization boundaries before any mutating tool is exposed to the agent.

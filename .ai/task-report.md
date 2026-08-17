# Task Report — KF-011 Agent Evaluation and Observability

## Status

Advanced the bounded KnowledgeForge agent with a deterministic, offline evaluation harness while preserving the provider-neutral runtime boundary.

## Delivered

### Compatibility fix

- Extended the application-level `create_note()` command with an optional `content` argument.
- The command still creates notes through the configured template and metadata pipeline; when explicit content is supplied, it replaces the generated body through the existing note-service update path.
- This resolves the application integration regression reported after pulling the agent CLI milestone.

### Deterministic agent evaluation

- Added `application.agent_evaluation` with a provider-independent evaluation model.
- Added `ScriptedAgentPlanner` so planner decisions can be replayed without network access or a live model.
- Added evaluation assertions for:
  - successful task completion;
  - expected answer text;
  - required tool usage;
  - per-case tool-call budgets;
  - tool-call success/failure counts;
  - repeated tool-call signatures.
- Added aggregate metrics for completion rate, average tool calls, tool success rate, and repeated-tool cases.
- Added stable JSON serialization for machine-readable evaluation reports.
- Added JSON dataset loading with validation.

### Tests

- Added `tests/test_agent_evaluation.py` covering runtime replay, evaluation failures, repeated calls, dataset loading, invalid datasets, and report shape.

## Current Architecture

```text
AgentEvaluationCase
        |
        v
ScriptedAgentPlanner ----> AgentRuntime ----> KnowledgeToolRegistry
        |                         |
        |                         +---- bounded execution guardrails
        |
        +---- deterministic model decisions

                |
                v
        AgentEvaluationReport
          |      |      |
          |      |      +---- repeated-tool cases
          |      +----------- tool success rate
          +------------------ completion / efficiency metrics
```

The evaluator deliberately reuses the real `AgentRuntime` instead of duplicating execution logic. This keeps evaluation aligned with production guardrails.

## Guardrails

The runtime continues to enforce:

- 8 model iterations per run;
- 16 total tool calls per run;
- consecutive identical tool-call detection;
- unique tool-call IDs within a model response;
- isolated tool failures returned as structured observations.

## Validation

The latest user-reported checkout had 174 passing tests and one application integration failure caused by `create_note(content=...)`. That compatibility issue is now addressed in the application command layer.

The same checkout also reported one Ruff import-order finding in `src/knowledgeforge/ai_cli.py`; this remains a small cleanup item to verify/fix in the user's working tree.

## Next Step

Next milestone: **authorization boundaries and controlled tool capabilities**.

Before exposing any mutating capability to the agent, introduce explicit read-only versus write-capable tool policies, make the runtime enforce the policy, and add evaluation cases proving that read-only runs cannot invoke mutating tools. After that, add the first controlled write workflow behind an explicit application-level authorization boundary.

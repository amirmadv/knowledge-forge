# ADR-0011: Deterministic Agent Evaluation

## Status

Accepted

## Context

KnowledgeForge now has a bounded provider-neutral agent runtime and an OpenAI-compatible planner adapter. The runtime produces deterministic execution traces containing model steps, tool calls, and tool observations.

Evaluation must be possible without a live provider so regressions can be reproduced locally and quality checks can run in CI. Reusing production execution behavior is important because an independent evaluator could silently diverge from runtime guardrails.

## Decision

Add a provider-neutral offline evaluation harness in `application.agent_evaluation`.

The harness uses a `ScriptedAgentPlanner` to replay deterministic `AgentModelResponse` sequences through the real `AgentRuntime`.

Each evaluation case may assert expected answer text, required tool usage, maximum tool calls, successful and failed tool executions, and repeated tool-call signatures.

The aggregate report exposes completion rate, average tool calls, tool success rate, number of cases containing repeated tool calls, and per-case failures and trace-derived observations.

Reports have a stable JSON representation suitable for CI and future CLI integration.

## Consequences

### Positive

- Evaluation is deterministic and does not require network access.
- Tests exercise the same runtime and guardrails used by the production agent.
- Tool efficiency and repeated-call behavior become measurable.
- JSON reports provide a stable contract for future automation.

### Trade-offs

- Scripted planner cases do not measure the quality of a live model's reasoning.
- Evaluation datasets must be maintained as model/provider behavior changes.
- Full end-to-end provider evaluation remains a separate future layer.

## Rejected Alternatives

### Mock the runtime

Rejected because it would not verify the real execution guardrails.

### Call a live LLM in unit tests

Rejected because network-dependent tests are nondeterministic, slower, and unsuitable as the first regression layer.

### Evaluate only final answers

Rejected because tool selection and execution efficiency are central to the bounded-agent contract.

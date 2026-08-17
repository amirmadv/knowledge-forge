"""Deterministic evaluation primitives for the bounded KnowledgeForge agent."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from knowledgeforge.application.agent_runtime import (
    AgentModelResponse,
    AgentRuntime,
    AgentRuntimeError,
    AgentToolCall,
)


@dataclass(frozen=True, slots=True)
class AgentEvaluationCase:
    """One deterministic agent scenario used by the offline evaluator."""

    case_id: str
    prompt: str
    responses: tuple[AgentModelResponse, ...]
    expected_answer_contains: tuple[str, ...] = ()
    required_tools: tuple[str, ...] = ()
    max_tool_calls: int | None = None

    def __post_init__(self) -> None:
        if not self.case_id.strip():
            raise ValueError("case_id cannot be empty.")
        if not self.prompt.strip():
            raise ValueError("prompt cannot be empty.")
        if not self.responses:
            raise ValueError("responses cannot be empty.")
        if self.max_tool_calls is not None and self.max_tool_calls < 0:
            raise ValueError("max_tool_calls cannot be negative.")


@dataclass(frozen=True, slots=True)
class AgentEvaluationItem:
    """Result of evaluating one deterministic agent scenario."""

    case_id: str
    passed: bool
    answer: str
    termination_reason: str
    tool_call_count: int
    observed_tools: tuple[str, ...]
    successful_tool_calls: int
    failed_tool_calls: int
    repeated_tool_call_count: int
    failures: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        """Return a stable JSON-compatible representation."""
        return {
            "case_id": self.case_id,
            "passed": self.passed,
            "answer": self.answer,
            "termination_reason": self.termination_reason,
            "tool_call_count": self.tool_call_count,
            "observed_tools": list(self.observed_tools),
            "successful_tool_calls": self.successful_tool_calls,
            "failed_tool_calls": self.failed_tool_calls,
            "repeated_tool_call_count": self.repeated_tool_call_count,
            "failures": list(self.failures),
        }


@dataclass(frozen=True, slots=True)
class AgentEvaluationReport:
    """Aggregate report for deterministic agent evaluation cases."""

    items: tuple[AgentEvaluationItem, ...]

    @property
    def cases_evaluated(self) -> int:
        """Return the number of evaluated scenarios."""
        return len(self.items)

    @property
    def passed_cases(self) -> int:
        """Return the number of scenarios that passed all assertions."""
        return sum(item.passed for item in self.items)

    @property
    def completion_rate(self) -> float:
        """Return the fraction of scenarios that completed successfully."""
        if not self.items:
            return 0.0
        return self.passed_cases / len(self.items)

    @property
    def average_tool_calls(self) -> float:
        """Return the average number of tool calls per evaluated scenario."""
        if not self.items:
            return 0.0
        return sum(item.tool_call_count for item in self.items) / len(self.items)

    @property
    def tool_success_rate(self) -> float:
        """Return the fraction of tool invocations that succeeded."""
        successful = sum(item.successful_tool_calls for item in self.items)
        failed = sum(item.failed_tool_calls for item in self.items)
        total = successful + failed
        if total == 0:
            return 1.0
        return successful / total

    @property
    def repeated_tool_call_cases(self) -> int:
        """Return the number of cases containing repeated tool signatures."""
        return sum(item.repeated_tool_call_count > 0 for item in self.items)

    def as_dict(self) -> dict[str, Any]:
        """Return a stable JSON-compatible report."""
        return {
            "cases_evaluated": self.cases_evaluated,
            "passed_cases": self.passed_cases,
            "completion_rate": self.completion_rate,
            "average_tool_calls": self.average_tool_calls,
            "tool_success_rate": self.tool_success_rate,
            "repeated_tool_call_cases": self.repeated_tool_call_cases,
            "cases": [item.as_dict() for item in self.items],
        }

    def to_json(self) -> str:
        """Serialize the report as deterministic JSON."""
        return json.dumps(
            self.as_dict(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )


class ScriptedAgentPlanner:
    """Replay a fixed sequence of model responses without network access."""

    def __init__(self, responses: tuple[AgentModelResponse, ...]) -> None:
        if not responses:
            raise ValueError("responses cannot be empty.")
        self._responses = responses
        self._index = 0

    def respond(
        self,
        messages: tuple[Any, ...],
        tools: list[dict[str, Any]],
    ) -> AgentModelResponse:
        """Return the next scripted response and reject unexpected extra turns."""
        del messages, tools
        if self._index >= len(self._responses):
            raise AgentRuntimeError("Scripted planner exhausted before final answer.")
        response = self._responses[self._index]
        self._index += 1
        return response


def evaluate_agent_cases(
    cases: list[AgentEvaluationCase] | tuple[AgentEvaluationCase, ...],
    runtime: AgentRuntime,
) -> AgentEvaluationReport:
    """Replay deterministic cases through the real bounded runtime."""
    items: list[AgentEvaluationItem] = []

    for case in cases:
        planner = ScriptedAgentPlanner(case.responses)
        failures: list[str] = []
        answer = ""
        termination_reason = "runtime_error"
        tool_call_count = 0
        observed_tools: list[str] = []
        successful_tool_calls = 0
        failed_tool_calls = 0
        repeated_tool_call_count = 0

        try:
            result = runtime.run(case.prompt, planner)
        except Exception as exc:
            failures.append(str(exc))
            result = None

        if result is not None:
            answer = result.answer
            termination_reason = result.trace.termination_reason
            tool_call_count = result.trace.tool_call_count
            signatures: list[tuple[str, str]] = []
            for step in result.trace.steps:
                for call in step.response.tool_calls:
                    signature = (
                        call.name,
                        json.dumps(
                            call.arguments,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                    )
                    if signature in signatures:
                        repeated_tool_call_count += 1
                    signatures.append(signature)
                for observation in step.observations:
                    observed_tools.append(observation.tool_name)
                    if observation.success:
                        successful_tool_calls += 1
                    else:
                        failed_tool_calls += 1

            normalized_answer = answer.casefold()
            for expected in case.expected_answer_contains:
                if expected.casefold() not in normalized_answer:
                    failures.append(
                        "Answer does not contain expected text: "
                        f"{expected!r}."
                    )

            missing_tools = [
                tool
                for tool in case.required_tools
                if tool not in observed_tools
            ]
            for tool in missing_tools:
                failures.append(f"Required tool was not called: {tool}.")

            if (
                case.max_tool_calls is not None
                and tool_call_count > case.max_tool_calls
            ):
                failures.append(
                    "Tool-call budget exceeded: "
                    f"{tool_call_count} > {case.max_tool_calls}."
                )

        items.append(
            AgentEvaluationItem(
                case_id=case.case_id,
                passed=not failures,
                answer=answer,
                termination_reason=termination_reason,
                tool_call_count=tool_call_count,
                observed_tools=tuple(observed_tools),
                successful_tool_calls=successful_tool_calls,
                failed_tool_calls=failed_tool_calls,
                repeated_tool_call_count=repeated_tool_call_count,
                failures=tuple(failures),
            )
        )

    return AgentEvaluationReport(items=tuple(items))


def load_agent_evaluation_cases(path: Path) -> list[AgentEvaluationCase]:
    """Load deterministic agent evaluation cases from a JSON file."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_cases = payload.get("cases") if isinstance(payload, dict) else payload

    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError(
            "Agent evaluation dataset must contain a non-empty cases list."
        )

    return [_parse_case(item) for item in raw_cases]


def _parse_case(item: Any) -> AgentEvaluationCase:
    if not isinstance(item, dict):
        raise ValueError("Each agent evaluation case must be an object.")

    responses_payload = item.get("responses")
    if not isinstance(responses_payload, list) or not responses_payload:
        raise ValueError("Each case must contain a non-empty responses list.")

    responses = tuple(_parse_response(response) for response in responses_payload)
    return AgentEvaluationCase(
        case_id=_required_string(item, "id"),
        prompt=_required_string(item, "prompt"),
        responses=responses,
        expected_answer_contains=_string_tuple(
            item.get("expected_answer_contains", [])
        ),
        required_tools=_string_tuple(item.get("required_tools", [])),
        max_tool_calls=_optional_non_negative_int(item.get("max_tool_calls")),
    )


def _parse_response(item: Any) -> AgentModelResponse:
    if not isinstance(item, dict):
        raise ValueError("Each scripted response must be an object.")

    raw_tool_calls = item.get("tool_calls", [])
    if not isinstance(raw_tool_calls, list):
        raise ValueError("tool_calls must be a list.")

    tool_calls = tuple(_parse_tool_call(call) for call in raw_tool_calls)
    content = item.get("content", "")
    if not isinstance(content, str):
        raise ValueError("Response content must be a string.")

    return AgentModelResponse(content=content, tool_calls=tool_calls)


def _parse_tool_call(item: Any) -> AgentToolCall:
    if not isinstance(item, dict):
        raise ValueError("Each scripted tool call must be an object.")

    arguments = item.get("arguments", {})
    if not isinstance(arguments, dict):
        raise ValueError("Tool call arguments must be an object.")

    return AgentToolCall(
        call_id=_required_string(item, "call_id"),
        name=_required_string(item, "name"),
        arguments=arguments,
    )


def _required_string(item: dict[str, Any], key: str) -> str:
    value = item.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string.")
    return value


def _string_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError("Expected a list of strings.")
    return tuple(value)


def _optional_non_negative_int(value: Any) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError("max_tool_calls must be a non-negative integer.")
    return value

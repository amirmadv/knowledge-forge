"""Provider-neutral execution runtime for KnowledgeForge agents."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol

from knowledgeforge.application.tools import KnowledgeToolRegistry, ToolResult


@dataclass(frozen=True, slots=True)
class AgentToolCall:
    """A tool call requested by an agent model."""

    call_id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True, slots=True)
class AgentModelResponse:
    """Provider-neutral model response containing text and/or tool calls."""

    content: str = ""
    tool_calls: tuple[AgentToolCall, ...] = ()


@dataclass(frozen=True, slots=True)
class AgentMessage:
    """Message exchanged between the runtime and a model adapter."""

    role: str
    content: str = ""
    tool_calls: tuple[AgentToolCall, ...] = ()
    tool_call_id: str | None = None
    tool_name: str | None = None


@dataclass(frozen=True, slots=True)
class ToolObservation:
    """Outcome of one tool invocation, including isolated failures."""

    call_id: str
    tool_name: str
    success: bool
    data: dict[str, Any]
    error: str | None = None

    @classmethod
    def from_result(cls, call: AgentToolCall, result: ToolResult) -> ToolObservation:
        return cls(
            call_id=call.call_id,
            tool_name=result.tool_name,
            success=True,
            data=result.data,
        )

    @classmethod
    def from_error(cls, call: AgentToolCall, error: Exception) -> ToolObservation:
        return cls(
            call_id=call.call_id,
            tool_name=call.name,
            success=False,
            data={},
            error=str(error),
        )

    def as_message(self) -> AgentMessage:
        """Serialize the observation as a provider-neutral tool message."""
        payload = {
            "tool_name": self.tool_name,
            "success": self.success,
            "data": self.data,
        }
        if self.error is not None:
            payload["error"] = self.error
        return AgentMessage(
            role="tool",
            content=json.dumps(payload, ensure_ascii=False, sort_keys=True),
            tool_call_id=self.call_id,
            tool_name=self.tool_name,
        )


@dataclass(frozen=True, slots=True)
class AgentStep:
    """One model decision and the observations produced by its tool calls."""

    response: AgentModelResponse
    observations: tuple[ToolObservation, ...] = ()


@dataclass(frozen=True, slots=True)
class AgentTrace:
    """Deterministic execution trace suitable for debugging and evaluation."""

    steps: tuple[AgentStep, ...]
    tool_call_count: int
    termination_reason: str


@dataclass(frozen=True, slots=True)
class AgentRunResult:
    """Final answer plus the complete runtime trace."""

    answer: str
    trace: AgentTrace


class AgentPlanner(Protocol):
    """Provider adapter capable of producing the next model response."""

    def respond(
        self,
        messages: tuple[AgentMessage, ...],
        tools: list[dict[str, Any]],
    ) -> AgentModelResponse: ...


class AgentRuntimeError(RuntimeError):
    """Raised when the agent cannot produce a valid final answer."""


@dataclass(frozen=True, slots=True)
class AgentRuntimeConfig:
    """Safety limits for one bounded agent execution."""

    max_iterations: int = 8
    max_tool_calls: int = 16
    max_consecutive_duplicate_calls: int = 2

    def __post_init__(self) -> None:
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be at least 1.")
        if self.max_tool_calls < 1:
            raise ValueError("max_tool_calls must be at least 1.")
        if self.max_consecutive_duplicate_calls < 1:
            raise ValueError("max_consecutive_duplicate_calls must be at least 1.")


class AgentRuntime:
    """Execute bounded model/tool turns against a provider-neutral registry."""

    def __init__(
        self,
        tool_registry: KnowledgeToolRegistry,
        config: AgentRuntimeConfig | None = None,
    ) -> None:
        self._tool_registry = tool_registry
        self._config = config or AgentRuntimeConfig()

    @property
    def tool_registry(self) -> KnowledgeToolRegistry:
        """Return the registry used by this runtime."""
        return self._tool_registry

    def run(self, prompt: str, planner: AgentPlanner) -> AgentRunResult:
        """Run until the planner returns final text or a guardrail is reached."""
        normalized_prompt = prompt.strip()
        if not normalized_prompt:
            raise ValueError("Prompt cannot be empty.")

        messages: list[AgentMessage] = [
            AgentMessage(role="user", content=normalized_prompt)
        ]
        steps: list[AgentStep] = []
        total_tool_calls = 0
        previous_signature: tuple[str, str] | None = None
        consecutive_duplicates = 0

        for _ in range(self._config.max_iterations):
            response = planner.respond(
                tuple(messages),
                self._tool_registry.provider_tools(),
            )
            self._validate_response(response)

            if not response.tool_calls:
                answer = response.content.strip()
                if not answer:
                    raise AgentRuntimeError("Model returned neither final text nor tool calls.")
                steps.append(AgentStep(response=response))
                trace = AgentTrace(
                    steps=tuple(steps),
                    tool_call_count=total_tool_calls,
                    termination_reason="final_answer",
                )
                return AgentRunResult(answer=answer, trace=trace)

            total_tool_calls += len(response.tool_calls)
            if total_tool_calls > self._config.max_tool_calls:
                raise AgentRuntimeError(
                    "Agent tool-call limit exceeded: "
                    f"maximum is {self._config.max_tool_calls}."
                )

            messages.append(
                AgentMessage(
                    role="assistant",
                    content=response.content.strip(),
                    tool_calls=response.tool_calls,
                )
            )

            observations: list[ToolObservation] = []
            for call in response.tool_calls:
                signature = (call.name, _canonical_arguments(call.arguments))
                if signature == previous_signature:
                    consecutive_duplicates += 1
                else:
                    consecutive_duplicates = 1
                    previous_signature = signature

                if (
                    consecutive_duplicates
                    > self._config.max_consecutive_duplicate_calls
                ):
                    raise AgentRuntimeError(
                        "Agent repeated the same tool call too many times: "
                        f"{call.name}."
                    )

                try:
                    result = self._tool_registry.execute(call.name, call.arguments)
                except Exception as exc:
                    observation = ToolObservation.from_error(call, exc)
                else:
                    observation = ToolObservation.from_result(call, result)

                observations.append(observation)
                messages.append(observation.as_message())

            steps.append(
                AgentStep(
                    response=response,
                    observations=tuple(observations),
                )
            )

        raise AgentRuntimeError(
            "Agent iteration limit exceeded: "
            f"maximum is {self._config.max_iterations}."
        )

    @staticmethod
    def _validate_response(response: AgentModelResponse) -> None:
        if not isinstance(response.content, str):
            raise AgentRuntimeError("Model response content must be text.")
        seen_ids: set[str] = set()
        for call in response.tool_calls:
            if not call.call_id.strip():
                raise AgentRuntimeError("Every tool call must have a non-empty call_id.")
            if call.call_id in seen_ids:
                raise AgentRuntimeError(
                    f"Duplicate tool call id in model response: {call.call_id}."
                )
            seen_ids.add(call.call_id)
            if not call.name.strip():
                raise AgentRuntimeError("Every tool call must have a non-empty name.")
            if not isinstance(call.arguments, dict):
                raise AgentRuntimeError(
                    f"Tool call arguments must be an object: {call.name}."
                )


def _canonical_arguments(arguments: dict[str, Any]) -> str:
    return json.dumps(arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

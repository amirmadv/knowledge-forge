"""Tests for the provider-neutral KnowledgeForge agent runtime."""

from pathlib import Path

import pytest

from knowledgeforge.application.agent_runtime import (
    AgentModelResponse,
    AgentPlanner,
    AgentRuntime,
    AgentRuntimeConfig,
    AgentRuntimeError,
    AgentToolCall,
)
from knowledgeforge.application.commands import create_note, update_note
from knowledgeforge.application.tools import KnowledgeToolRegistry, SearchKnowledgeTool
from knowledgeforge.infrastructure.config.settings import Settings


class FakePlanner:
    """Return scripted model responses while recording runtime messages."""

    def __init__(self, responses: list[AgentModelResponse]) -> None:
        self.responses = responses
        self.calls = 0
        self.messages = []
        self.tools = []

    def respond(self, messages, tools):
        self.messages.append(messages)
        self.tools.append(tools)
        response = self.responses[self.calls]
        self.calls += 1
        return response


def _registry(vault_path: Path) -> KnowledgeToolRegistry:
    settings = Settings(
        vault_path=vault_path,
        ai_enabled=True,
        ai_api_key="test-key",
    )
    from knowledgeforge.application.ai import KnowledgeAgent

    agent = KnowledgeAgent(
        settings=settings,
        client=FakeAIClient(),
    )
    return KnowledgeToolRegistry((SearchKnowledgeTool(agent.search_with_evidence),))


class FakeAIClient:
    """Provide deterministic embedding and chat behavior for runtime setup."""

    def chat(self, prompt: str, system: str | None = None) -> str:
        return "unused"

    def embed(self, text: str) -> tuple[float, ...]:
        return (1.0, 0.0, 0.0)


def test_runtime_executes_tool_then_returns_final_answer(tmp_path: Path) -> None:
    vault_path = tmp_path / "vault"
    create_note("Linear Regression", vault_path=vault_path)
    update_note(
        "Linear Regression",
        "Linear regression predicts a target from input features.",
        vault_path=vault_path,
    )

    planner = FakePlanner(
        [
            AgentModelResponse(
                tool_calls=(
                    AgentToolCall(
                        call_id="call-1",
                        name="search_knowledge",
                        arguments={"query": "linear regression", "limit": 1},
                    ),
                )
            ),
            AgentModelResponse(content="Linear regression predicts a target [S1]."),
        ]
    )

    result = AgentRuntime(_registry(vault_path)).run(
        "What is linear regression?",
        planner,
    )

    assert result.answer == "Linear regression predicts a target [S1]."
    assert result.trace.tool_call_count == 1
    assert result.trace.termination_reason == "final_answer"
    assert len(result.trace.steps) == 2
    observation = result.trace.steps[0].observations[0]
    assert observation.success is True
    assert observation.data["results"][0]["title"] == "Linear Regression"
    assert planner.tools[0][0]["function"]["name"] == "search_knowledge"
    assert planner.messages[1][-1].role == "tool"
    assert planner.messages[1][-1].tool_call_id == "call-1"


def test_runtime_isolates_tool_errors_for_planner_recovery(tmp_path: Path) -> None:
    registry = _registry(tmp_path / "vault")
    planner = FakePlanner(
        [
            AgentModelResponse(
                tool_calls=(
                    AgentToolCall(
                        call_id="call-1",
                        name="search_knowledge",
                        arguments={"query": "   "},
                    ),
                )
            ),
            AgentModelResponse(content="I could not search because the query was empty."),
        ]
    )

    result = AgentRuntime(registry).run("Find something", planner)

    observation = result.trace.steps[0].observations[0]
    assert observation.success is False
    assert "non-empty string" in (observation.error or "")
    assert result.answer.startswith("I could not search")


def test_runtime_rejects_empty_prompt(tmp_path: Path) -> None:
    planner = FakePlanner([AgentModelResponse(content="unused")])

    with pytest.raises(ValueError, match="Prompt cannot be empty"):
        AgentRuntime(_registry(tmp_path / "vault")).run("   ", planner)


def test_runtime_enforces_tool_call_limit(tmp_path: Path) -> None:
    planner = FakePlanner(
        [
            AgentModelResponse(
                tool_calls=(
                    AgentToolCall("a", "search_knowledge", {"query": "one"}),
                    AgentToolCall("b", "search_knowledge", {"query": "two"}),
                )
            )
        ]
    )
    runtime = AgentRuntime(
        _registry(tmp_path / "vault"),
        AgentRuntimeConfig(max_tool_calls=1),
    )

    with pytest.raises(AgentRuntimeError, match="tool-call limit"):
        runtime.run("search", planner)


def test_runtime_enforces_iteration_limit(tmp_path: Path) -> None:
    response = AgentModelResponse(
        tool_calls=(AgentToolCall("call", "search_knowledge", {"query": "x"}),)
    )
    planner = FakePlanner([response, response])
    runtime = AgentRuntime(
        _registry(tmp_path / "vault"),
        AgentRuntimeConfig(max_iterations=1),
    )

    with pytest.raises(AgentRuntimeError, match="iteration limit"):
        runtime.run("search", planner)


def test_runtime_detects_repeated_identical_tool_calls(tmp_path: Path) -> None:
    response_one = AgentModelResponse(
        tool_calls=(AgentToolCall("a", "search_knowledge", {"query": "x"}),)
    )
    response_two = AgentModelResponse(
        tool_calls=(AgentToolCall("b", "search_knowledge", {"query": "x"}),)
    )
    response_three = AgentModelResponse(
        tool_calls=(AgentToolCall("c", "search_knowledge", {"query": "x"}),)
    )
    planner = FakePlanner([response_one, response_two, response_three])
    runtime = AgentRuntime(
        _registry(tmp_path / "vault"),
        AgentRuntimeConfig(max_iterations=4, max_consecutive_duplicate_calls=2),
    )

    with pytest.raises(AgentRuntimeError, match="repeated the same tool call"):
        runtime.run("search", planner)


def test_runtime_rejects_duplicate_call_ids(tmp_path: Path) -> None:
    planner = FakePlanner(
        [
            AgentModelResponse(
                tool_calls=(
                    AgentToolCall("same", "search_knowledge", {"query": "one"}),
                    AgentToolCall("same", "search_knowledge", {"query": "two"}),
                )
            )
        ]
    )

    with pytest.raises(AgentRuntimeError, match="Duplicate tool call id"):
        AgentRuntime(_registry(tmp_path / "vault")).run("search", planner)

"""Tests for explicit KnowledgeForge agent tool authorization policies."""

from pathlib import Path

from knowledgeforge.application.agent_runtime import (
    AgentModelResponse,
    AgentRuntime,
    AgentRuntimeConfig,
    AgentToolCall,
)
from knowledgeforge.application.ai import KnowledgeAgent
from knowledgeforge.application.commands import create_note
from knowledgeforge.application.tools import ToolAccess
from knowledgeforge.infrastructure.config.settings import Settings


class FakeAIClient:
    """Provide deterministic client behavior without network access."""

    def chat(self, prompt: str, system: str | None = None) -> str:
        return "unused"

    def embed(self, text: str) -> tuple[float, ...]:
        return (1.0, 0.0, 0.0)


class FakePlanner:
    """Return one scripted tool call followed by a final answer."""

    def __init__(self, responses: list[AgentModelResponse]) -> None:
        self.responses = responses
        self.index = 0
        self.tools_seen: list[list[dict[str, object]]] = []

    def respond(self, messages, tools):
        del messages
        self.tools_seen.append(tools)
        response = self.responses[self.index]
        self.index += 1
        return response


def _agent(vault_path: Path) -> KnowledgeAgent:
    return KnowledgeAgent(
        settings=Settings(
            vault_path=vault_path,
            ai_enabled=True,
            ai_api_key="test-key",
        ),
        client=FakeAIClient(),
    )


def test_read_only_runtime_hides_and_blocks_write_tool(tmp_path: Path) -> None:
    """Read-only execution must not expose or mutate through write tools."""
    vault_path = tmp_path / "vault"
    agent = _agent(vault_path)
    planner = FakePlanner(
        [
            AgentModelResponse(
                tool_calls=(
                    AgentToolCall(
                        call_id="create-1",
                        name="create_note",
                        arguments={
                            "title": "Should Not Exist",
                            "content": "blocked",
                        },
                    ),
                )
            ),
            AgentModelResponse(content="The write was blocked by policy."),
        ]
    )

    result = agent.runtime(ToolAccess.READ_ONLY).run("Create a note", planner)

    assert result.answer == "The write was blocked by policy."
    assert result.trace.steps[0].observations[0].success is False
    assert "requires write authorization" in (
        result.trace.steps[0].observations[0].error or ""
    )
    assert planner.tools_seen[0]
    assert all(
        item["function"]["name"] != "create_note" for item in planner.tools_seen[0]
    )
    assert not (vault_path / "Should Not Exist.md").exists()


def test_write_runtime_exposes_and_executes_create_note(tmp_path: Path) -> None:
    """Write authorization must be explicit before the create-note capability is exposed."""
    vault_path = tmp_path / "vault"
    agent = _agent(vault_path)
    planner = FakePlanner(
        [
            AgentModelResponse(
                tool_calls=(
                    AgentToolCall(
                        call_id="create-1",
                        name="create_note",
                        arguments={
                            "title": "Authorized Note",
                            "content": "Created through the authorized workflow.",
                        },
                    ),
                )
            ),
            AgentModelResponse(content="Created the note."),
        ]
    )

    result = agent.runtime(ToolAccess.WRITE).run("Create a note", planner)

    assert result.answer == "Created the note."
    assert result.trace.steps[0].observations[0].success is True
    assert any(
        item["function"]["name"] == "create_note" for item in planner.tools_seen[0]
    )
    note = agent.note_service.get("Authorized Note")
    assert "Created through the authorized workflow." in note.path.read_text(
        encoding="utf-8"
    )


def test_read_only_is_the_default_runtime_policy(tmp_path: Path) -> None:
    """The runtime default remains safe even when a registry contains write tools."""
    vault_path = tmp_path / "vault"
    create_note("Existing", vault_path=vault_path)
    agent = _agent(vault_path)
    runtime = AgentRuntime(
        agent.tools_for_access(ToolAccess.WRITE),
        AgentRuntimeConfig(),
    )

    assert runtime.config.tool_access is ToolAccess.READ_ONLY

"""Tests for conversational AI sessions."""

from pathlib import Path

from knowledgeforge.application.ai import KnowledgeAgent
from knowledgeforge.application.chat import KnowledgeChatSession
from knowledgeforge.application.commands import create_note
from knowledgeforge.infrastructure.config.settings import Settings


class FakeAIClient:
    """Capture prompts without calling a real provider."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def chat(self, prompt: str, system: str | None = None) -> str:
        self.prompts.append(prompt)
        return f"answer-{len(self.prompts)}"


def _agent(vault_path: Path, client: FakeAIClient) -> KnowledgeAgent:
    return KnowledgeAgent(
        settings=Settings(
            vault_path=vault_path,
            ai_enabled=True,
            ai_api_key="test-key",
        ),
        client=client,
    )


def test_chat_session_keeps_bounded_history(tmp_path: Path) -> None:
    """Conversation history should be passed to later agent calls."""
    vault_path = tmp_path / "vault"
    create_note(title="Linear Regression", vault_path=vault_path)
    client = FakeAIClient()
    session = KnowledgeChatSession(_agent(vault_path, client))

    first = session.ask("What is linear regression?")
    second = session.ask("How is it related to loss?")

    assert first.answer == "answer-1"
    assert second.answer == "answer-2"
    assert "User: What is linear regression?" in client.prompts[1]
    assert "Assistant: answer-1" in client.prompts[1]
    assert len(session.turns) == 2


def test_chat_session_limits_history_and_can_clear(tmp_path: Path) -> None:
    """Sessions should retain only recent turns and support clearing."""
    vault_path = tmp_path / "vault"
    create_note(title="AI", vault_path=vault_path)
    client = FakeAIClient()
    session = KnowledgeChatSession(_agent(vault_path, client))

    for index in range(10):
        session.ask(f"Question {index}")

    assert len(session.turns) == session.MAX_TURNS
    assert session.turns[0].question == "Question 2"

    session.clear()

    assert session.turns == ()

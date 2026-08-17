"""Interactive conversational AI support for KnowledgeForge."""

from __future__ import annotations

from dataclasses import dataclass

from knowledgeforge.application.ai import AIAnswer, KnowledgeAgent


@dataclass(frozen=True, slots=True)
class ChatTurn:
    """One user/assistant exchange in a KnowledgeForge session."""

    question: str
    answer: str
    sources: tuple[str, ...] = ()


class KnowledgeChatSession:
    """Maintain short conversational memory while keeping answers vault-grounded."""

    MAX_TURNS = 8

    def __init__(self, agent: KnowledgeAgent) -> None:
        self._agent = agent
        self._turns: list[ChatTurn] = []

    @property
    def turns(self) -> tuple[ChatTurn, ...]:
        """Return the current conversation history."""
        return tuple(self._turns)

    def ask(self, question: str) -> AIAnswer:
        """Ask a follow-up question with bounded conversation memory."""
        result = self._agent.ask(
            question,
            history=tuple(
                (turn.question, turn.answer)
                for turn in self._turns
            ),
        )
        self._turns.append(
            ChatTurn(
                question=question.strip(),
                answer=result.answer,
                sources=result.sources,
            )
        )
        if len(self._turns) > self.MAX_TURNS:
            del self._turns[:-self.MAX_TURNS]
        return result

    def clear(self) -> None:
        """Clear the conversation history."""
        self._turns.clear()

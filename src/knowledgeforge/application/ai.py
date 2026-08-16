"""Application-level AI use cases for KnowledgeForge."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from knowledgeforge.domain.note import NoteService
from knowledgeforge.infrastructure.ai.client import (
    AIClientError,
    OpenAICompatibleClient,
)
from knowledgeforge.infrastructure.config.settings import Settings


@dataclass(frozen=True, slots=True)
class AIAnswer:
    """Result returned by an AI application use case."""

    answer: str


class KnowledgeAgent:
    """Minimal agent facade that can answer using local vault context."""

    def __init__(
        self,
        settings: Settings,
        vault_path: Path | None = None,
    ) -> None:
        if not settings.ai_enabled:
            raise AIClientError("AI is disabled. Set KNOWLEDGEFORGE_AI_ENABLED=true.")
        if not settings.ai_api_key:
            raise AIClientError("AI API key is not configured.")

        self._client = OpenAICompatibleClient(
            base_url=settings.ai_base_url,
            api_key=settings.ai_api_key,
            model=settings.ai_model,
            timeout=settings.ai_timeout,
        )
        self._note_service = NoteService(vault_path or settings.vault_path)

    def ask(self, question: str) -> AIAnswer:
        """Answer a question with relevant local note context."""
        notes = self._note_service.search(question)
        context_parts: list[str] = []

        for note in notes[:8]:
            try:
                _, content = self._note_service.get(note.title), self._note_service.read(note.title)
            except (AttributeError, TypeError):
                content = ""
            context_parts.append(f"# {note.title}\n{content}")

        context = "\n\n".join(context_parts)
        prompt = (
            "Answer the user's question using the supplied KnowledgeForge notes when relevant. "
            "If the notes do not contain enough information, say so clearly.\n\n"
            f"Question:\n{question}\n\n"
            f"Knowledge context:\n{context or '(no matching notes)'}"
        )

        return AIAnswer(
            answer=self._client.chat(
                prompt,
                system="You are the KnowledgeForge personal knowledge assistant.",
            )
        )

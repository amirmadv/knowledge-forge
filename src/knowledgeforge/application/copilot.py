"""AI Copilot use cases for enriching and operating on KnowledgeForge notes."""

from __future__ import annotations

from dataclasses import dataclass

from knowledgeforge.application.ai import KnowledgeAgent
from knowledgeforge.domain.note import Note, NoteService
from knowledgeforge.infrastructure.ai.client import OpenAICompatibleClient


@dataclass(frozen=True, slots=True)
class CopilotResult:
    """Result of an AI Copilot operation."""

    answer: str
    sources: tuple[str, ...] = ()
    note: Note | None = None


class KnowledgeCopilot:
    """AI-assisted note operations grounded in the local KnowledgeForge vault."""

    def __init__(
        self,
        agent: KnowledgeAgent,
        note_service: NoteService | None = None,
        client: OpenAICompatibleClient | None = None,
    ) -> None:
        self._agent = agent
        self._note_service = note_service or agent.note_service
        self._client = client or agent.client

    def summarize(self, title: str) -> CopilotResult:
        """Create a concise, structured summary of a note."""
        note, content = self._load_note(title)
        prompt = self._operation_prompt(
            "Summarize this note in Persian. Return exactly three sections: "
            "خلاصه، نکات کلیدی، و مفاهیم مرتبط. Do not invent information.",
            note.title,
            content,
        )
        answer = self._client.chat(prompt, system=self._system())
        return CopilotResult(answer=answer, sources=(note.title,))

    def suggest_tags(self, title: str) -> CopilotResult:
        """Suggest normalized tags without modifying the note."""
        note, content = self._load_note(title)
        prompt = self._operation_prompt(
            "Suggest 3 to 8 concise lowercase tags. Return only a comma-separated "
            "list. Use existing vocabulary where possible and do not invent topics.",
            note.title,
            content,
        )
        answer = self._client.chat(prompt, system=self._system())
        return CopilotResult(answer=answer, sources=(note.title,))

    def improve(self, title: str) -> CopilotResult:
        """Rewrite a note for clarity while preserving its factual content."""
        note, content = self._load_note(title)
        prompt = self._operation_prompt(
            "Rewrite the Markdown body for clarity, structure, and readability. "
            "Preserve all factual information. Return only the Markdown body.",
            note.title,
            content,
        )
        answer = self._client.chat(prompt, system=self._system())
        updated = self._note_service.update_content(title, answer)
        return CopilotResult(
            answer="Note improved successfully.",
            sources=(note.title,),
            note=updated,
        )

    def related(self, title: str) -> CopilotResult:
        """Find related notes using the existing graph-aware agent."""
        note = self._note_service.get(title)
        question = (
            "Which existing KnowledgeForge notes are most closely related to "
            f"'{note.title}'? Explain the relationship using the supplied vault "
            "context and list the strongest candidates."
        )
        result = self._agent.ask(question)
        return CopilotResult(
            answer=result.answer,
            sources=result.sources or (note.title,),
        )

    def knowledge_gaps(self, title: str) -> CopilotResult:
        """Identify useful knowledge gaps around a note."""
        note, content = self._load_note(title)
        graph = self._agent.inspect_graph(title, depth=1)
        prompt = self._operation_prompt(
            "Identify the most important missing concepts, unanswered questions, "
            "or missing supporting notes needed to understand this topic well. "
            "Prioritize actionable gaps and do not invent claims about the vault.",
            note.title,
            content,
            extra=f"Graph nodes: {', '.join(node.slug for node in graph.nodes)}",
        )
        answer = self._client.chat(prompt, system=self._system())
        return CopilotResult(answer=answer, sources=(note.title,))

    def create_note(
        self,
        title: str,
        instruction: str,
    ) -> CopilotResult:
        """Create a new note body from a user instruction."""
        normalized_instruction = instruction.strip()
        normalized_title = title.strip()
        if not normalized_title:
            raise ValueError("Note title cannot be empty.")
        if not normalized_instruction:
            raise ValueError("Instruction cannot be empty.")

        prompt = (
            "Create a KnowledgeForge Markdown note body from the instruction below. "
            "Use only information explicitly supported by the instruction. "
            "If information is missing, keep the note focused instead of inventing it. "
            "Return only the Markdown body.\n\n"
            f"Title: {normalized_title}\n"
            f"Instruction: {normalized_instruction}"
        )
        body = self._client.chat(prompt, system=self._system())
        note = self._note_service.create(normalized_title)
        updated = self._note_service.update_content(note.title, body)
        return CopilotResult(
            answer=f"Created note: {updated.title}",
            sources=(),
            note=updated,
        )

    @staticmethod
    def _system() -> str:
        return (
            "You are the KnowledgeForge AI Copilot. The user's Markdown vault is "
            "the source of truth. Be precise, concise, and grounded. Never fabricate "
            "facts. Preserve Markdown when an operation requests Markdown."
        )

    @staticmethod
    def _operation_prompt(
        instruction: str,
        title: str,
        content: str,
        extra: str = "",
    ) -> str:
        """Build a bounded Copilot prompt."""
        suffix = f"\n\n{extra}" if extra else ""
        return (
            f"{instruction}\n\n"
            f"Note title: {title}\n"
            f"Note content:\n{content}{suffix}"
        )

    def _load_note(self, title: str) -> tuple[Note, str]:
        """Load a note and its complete Markdown content."""
        note = self._note_service.get(title)
        return note, self._note_service.read_content(note.title)

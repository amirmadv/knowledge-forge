"""Domain service for KnowledgeForge note management."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path

from knowledgeforge.domain.note.model import Note
from knowledgeforge.domain.template import TemplateService


class NoteAlreadyExistsError(FileExistsError):
    """Raised when attempting to create an existing note."""


class NoteNotFoundError(FileNotFoundError):
    """Raised when a requested note does not exist."""


class InvalidNoteTitleError(ValueError):
    """Raised when a note title is invalid."""


class NoteService:
    """Create and manage KnowledgeForge notes."""

    NOTES_DIRECTORY = "notes"

    def __init__(
        self,
        vault_path: Path,
        template_service: TemplateService | None = None,
    ) -> None:
        """Initialize the note service.

        Args:
            vault_path: Root directory of the KnowledgeForge vault.
            template_service: Optional service used to render note templates.
        """
        self._vault_path = vault_path
        self._template_service = template_service

    @property
    def vault_path(self) -> Path:
        """Return the configured vault path."""
        return self._vault_path

    def create(
        self,
        title: str,
        template: str = "default",
    ) -> Note:
        """Create a new Markdown note.

        Args:
            title: Human-readable title of the note.
            template: Name of the template used to render the note.

        Returns:
            The newly created Note.

        Raises:
            InvalidNoteTitleError: If the title is empty or invalid.
            NoteAlreadyExistsError: If the note already exists.
        """
        normalized_title = title.strip()

        if not normalized_title:
            raise InvalidNoteTitleError("Note title cannot be empty.")

        slug = self._create_slug(normalized_title)

        if not slug:
            raise InvalidNoteTitleError(
                "Note title must contain usable characters."
            )

        notes_path = self._vault_path / self.NOTES_DIRECTORY
        notes_path.mkdir(parents=True, exist_ok=True)

        note_path = notes_path / f"{slug}.md"

        if note_path.exists():
            raise NoteAlreadyExistsError(
                f"Note already exists: {note_path}"
            )

        now = datetime.now(UTC)

        if self._template_service is None:
            content = self._build_initial_content(
                title=normalized_title,
                created_at=now,
            )
        else:
            template_model = self._template_service.get(template)
            content = self._template_service.render(
                template_model,
                {
                    "title": normalized_title,
                    "content": "",
                },
            )

        note_path.write_text(content, encoding="utf-8")

        return Note(
            title=normalized_title,
            path=note_path,
            created_at=now,
            updated_at=now,
        )

    def list_notes(self) -> list[Note]:
        """Return all Markdown notes in the vault.

        Returns:
            Notes sorted alphabetically by title.
        """
        notes_path = self._vault_path / self.NOTES_DIRECTORY

        if not notes_path.is_dir():
            return []

        notes: list[Note] = []

        for note_path in sorted(notes_path.glob("*.md")):
            notes.append(self._read_note_metadata(note_path))

        return sorted(notes, key=lambda note: note.title.lower())

    def search(self, query: str) -> list[Note]:
        """Search notes by title and Markdown content.

        The search is case-insensitive and returns each matching note
        at most once.

        Args:
            query: Text to search for.

        Returns:
            Notes containing the query in their title or content,
            sorted alphabetically by title.

        Raises:
            ValueError: If the search query is empty.
        """
        normalized_query = query.strip()

        if not normalized_query:
            raise ValueError("Search query cannot be empty.")

        query_lower = normalized_query.casefold()
        matches: list[Note] = []

        for note in self.list_notes():
            if query_lower in note.title.casefold():
                matches.append(note)
                continue

            content = note.path.read_text(encoding="utf-8")

            if query_lower in content.casefold():
                matches.append(note)

        return matches

    def get(self, title: str) -> Note:
        """Return a note by its title.

        Args:
            title: Human-readable note title.

        Returns:
            The matching Note.

        Raises:
            NoteNotFoundError: If the note does not exist.
            InvalidNoteTitleError: If the title cannot produce a valid slug.
        """
        normalized_title = title.strip()

        if not normalized_title:
            raise InvalidNoteTitleError("Note title cannot be empty.")

        slug = self._create_slug(normalized_title)

        if not slug:
            raise InvalidNoteTitleError(
                "Note title must contain usable characters."
            )

        note_path = self._vault_path / self.NOTES_DIRECTORY / f"{slug}.md"

        if not note_path.exists():
            raise NoteNotFoundError(f"Note not found: {normalized_title}")

        return self._read_note_metadata(note_path)

    def read_content(self, title: str) -> str:
        """Read the complete Markdown content of a note.

        Args:
            title: Human-readable note title.

        Returns:
            Complete Markdown content.

        Raises:
            NoteNotFoundError: If the note does not exist.
        """
        note = self.get(title)
        return note.path.read_text(encoding="utf-8")

    def update_content(self, title: str, content: str) -> Note:
        """Replace the Markdown content of an existing note.

        The title and creation timestamp in the front matter are preserved.

        Args:
            title: Human-readable note title.
            content: New Markdown body content.

        Returns:
            The updated Note.

        Raises:
            NoteNotFoundError: If the note does not exist.
        """
        note = self.get(title)

        normalized_content = content.rstrip() + "\n"

        if not normalized_content.strip():
            normalized_content = "\n"

        existing_content = note.path.read_text(encoding="utf-8")

        front_matter, _ = self._split_front_matter(existing_content)

        now = datetime.now(UTC)

        updated_document = (
            front_matter
            + "\n"
            + normalized_content.lstrip("\n")
        )

        note.path.write_text(updated_document, encoding="utf-8")

        return Note(
            title=note.title,
            path=note.path,
            created_at=note.created_at,
            updated_at=now,
        )

    def update(self, title: str, content: str) -> Note:
        """Update an existing note.

        This is the public application-facing alias for update_content().
        """
        return self.update_content(title, content)

    @staticmethod
    def _create_slug(title: str) -> str:
        """Convert a note title into a filesystem-friendly slug."""
        slug = title.lower()
        slug = re.sub(r"[^\w\s-]", "", slug, flags=re.UNICODE)
        slug = re.sub(r"[-\s]+", "-", slug)
        return slug.strip("-_")

    @staticmethod
    def _build_initial_content(
        title: str,
        created_at: datetime,
    ) -> str:
        """Build the initial Markdown content for a note."""
        return (
            "---\n"
            f"title: {title}\n"
            f"created_at: {created_at.isoformat()}\n"
            "---\n\n"
            f"# {title}\n\n"
        )

    @staticmethod
    def _split_front_matter(content: str) -> tuple[str, str]:
        """Split a Markdown document into front matter and body."""
        if not content.startswith("---\n"):
            return "", content

        end_marker = content.find("\n---", 4)

        if end_marker == -1:
            return "", content

        front_matter_end = end_marker + len("\n---")

        return (
            content[:front_matter_end],
            content[front_matter_end:],
        )

    @staticmethod
    def _read_note_metadata(note_path: Path) -> Note:
        """Read basic note metadata from a Markdown file."""
        content = note_path.read_text(encoding="utf-8")

        title = note_path.stem.replace("-", " ").title()

        created_at = datetime.fromtimestamp(
            note_path.stat().st_ctime,
            tz=UTC,
        )

        updated_at = datetime.fromtimestamp(
            note_path.stat().st_mtime,
            tz=UTC,
        )

        if content.startswith("---\n"):
            front_matter, _ = NoteService._split_front_matter(content)

            title_match = re.search(
                r"^title:\s*(.+)$",
                front_matter,
                flags=re.MULTILINE,
            )

            if title_match:
                title = title_match.group(1).strip()

        return Note(
            title=title,
            path=note_path,
            created_at=created_at,
            updated_at=updated_at,
        )
"""Domain service for KnowledgeForge note management."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path

from knowledgeforge.domain.note.model import Note, NoteMetadata
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

    VALID_NOTE_TYPES = frozenset(
        {
            "concept",
            "research",
            "project",
            "reference",
            "question",
            "idea",
            "tutorial",
            "meeting",
        }
    )

    VALID_STATUSES = frozenset(
        {
            "draft",
            "active",
            "review",
            "archived",
            "completed",
        }
    )

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

    @staticmethod
    def _build_tags_yaml(tags: tuple[str, ...]) -> str:
        """Build YAML list entries for note tags."""
        if not tags:
            return "  -"

        return "\n".join(
            f"  - {tag}"
            for tag in tags
        )

    @property
    def vault_path(self) -> Path:
        """Return the configured vault path."""
        return self._vault_path

    def create(
        self,
        title: str,
        template: str = "default",
        note_type: str = "concept",
        status: str = "draft",
        tags: tuple[str, ...] = (),
    ) -> Note:
        """Create a new Markdown note.

        Args:
            title: Human-readable title of the note.
            template: Name of the template used to render the note.
            note_type: Type of the note.
            status: Lifecycle status of the note.
            tags: Tags associated with the note.

        Returns:
            The newly created Note.

        Raises:
            InvalidNoteTitleError: If the title is empty or invalid.
            NoteAlreadyExistsError: If the note already exists.
            ValueError: If metadata values are invalid.
        """
        normalized_title = title.strip()

        if not normalized_title:
            raise InvalidNoteTitleError("Note title cannot be empty.")

        slug = self._create_slug(normalized_title)

        if not slug:
            raise InvalidNoteTitleError(
                "Note title must contain usable characters."
            )

        metadata = self._normalize_metadata(
            note_type=note_type,
            status=status,
            tags=tags,
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
                metadata=metadata,
                created_at=now,
                updated_at=now,
            )
        else:
            template_model = self._template_service.get(template)
            content = self._template_service.render(
                template_model,
                {
                    "title": normalized_title,
                    "content": "",
                    "note_type": metadata.note_type,
                    "status": metadata.status,
                    "tags": ", ".join(metadata.tags),
                    "tags_yaml": self._build_tags_yaml(metadata.tags),
                    "created_at": now.isoformat(),
                    "updated_at": now.isoformat(),
                },
            )

        note_path.write_text(content, encoding="utf-8")

        return Note(
            title=normalized_title,
            path=note_path,
            created_at=now,
            updated_at=now,
            metadata=metadata,
        )

    def list_notes(
        self,
        note_type: str | None = None,
        status: str | None = None,
        tag: str | None = None,
    ) -> list[Note]:
        """List notes with optional metadata filters.

        Args:
            note_type: Optional note type filter.
            status: Optional lifecycle status filter.
            tag: Optional tag filter.

        Returns:
            Notes matching all provided filters.
        """
        notes_path = self._vault_path / self.NOTES_DIRECTORY

        if not notes_path.is_dir():
            return []

        notes = [
            self._read_note_metadata(note_path)
            for note_path in sorted(notes_path.glob("*.md"))
        ]

        normalized_note_type = (
            note_type.strip().casefold()
            if note_type is not None
            else None
        )

        normalized_status = (
            status.strip().casefold()
            if status is not None
            else None
        )

        normalized_tag = (
            re.sub(
                r"\s+",
                "-",
                tag.strip().casefold(),
            )
            if tag is not None
            else None
        )

        filtered_notes = notes

        if normalized_note_type:
            filtered_notes = [
                note
                for note in filtered_notes
                if note.metadata.note_type == normalized_note_type
            ]

        if normalized_status:
            filtered_notes = [
                note
                for note in filtered_notes
                if note.metadata.status == normalized_status
            ]

        if normalized_tag:
            filtered_notes = [
                note
                for note in filtered_notes
                if normalized_tag in note.metadata.tags
            ]

        return sorted(
            filtered_notes,
            key=lambda note: note.title.casefold(),
        )

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

        note_path = (
            self._vault_path
            / self.NOTES_DIRECTORY
            / f"{slug}.md"
        )

        if not note_path.exists():
            raise NoteNotFoundError(
                f"Note not found: {normalized_title}"
            )

        return self._read_note_metadata(note_path)

    def delete(self, title: str) -> Note:
        """Delete an existing note from the vault.

        Args:
            title: Human-readable note title.

        Returns:
            The deleted Note.

        Raises:
            NoteNotFoundError: If the note does not exist.
            InvalidNoteTitleError: If the title cannot produce a valid slug.
        """
        note = self.get(title)

        note.path.unlink()

        return note

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

        The title, metadata, and creation timestamp in the front matter
        are preserved. The updated timestamp is refreshed and persisted.

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

        updated_front_matter = self._replace_front_matter_value(
            front_matter,
            "updated_at",
            now.isoformat(),
        )

        updated_document = (
            updated_front_matter
            + "\n"
            + normalized_content.lstrip("\n")
        )

        note.path.write_text(
            updated_document,
            encoding="utf-8",
        )

        return Note(
            title=note.title,
            path=note.path,
            created_at=note.created_at,
            updated_at=now,
            metadata=note.metadata,
        )

    def update(self, title: str, content: str) -> Note:
        """Update an existing note.

        This is the public application-facing alias for update_content().
        """
        return self.update_content(title, content)

    @classmethod
    def _normalize_metadata(
        cls,
        note_type: str,
        status: str,
        tags: tuple[str, ...],
    ) -> NoteMetadata:
        """Normalize and validate note metadata."""
        normalized_note_type = note_type.strip().casefold()
        normalized_status = status.strip().casefold()

        if normalized_note_type not in cls.VALID_NOTE_TYPES:
            raise ValueError(f"Invalid note type: {note_type}")

        if normalized_status not in cls.VALID_STATUSES:
            raise ValueError(f"Invalid note status: {status}")

        normalized_tags: list[str] = []

        for tag in tags:
            normalized_tag = re.sub(
                r"\s+",
                "-",
                tag.strip().casefold(),
            )

            if (
                normalized_tag
                and normalized_tag not in normalized_tags
            ):
                normalized_tags.append(normalized_tag)

        return NoteMetadata(
            note_type=normalized_note_type,
            status=normalized_status,
            tags=tuple(normalized_tags),
        )

    @staticmethod
    def _create_slug(title: str) -> str:
        """Convert a note title into a filesystem-friendly slug."""
        slug = title.lower()

        slug = re.sub(
            r"[^\w\s-]",
            "",
            slug,
            flags=re.UNICODE,
        )

        slug = re.sub(
            r"[-\s]+",
            "-",
            slug,
        )

        return slug.strip("-_")

    @staticmethod
    def _build_initial_content(
        title: str,
        metadata: NoteMetadata,
        created_at: datetime,
        updated_at: datetime,
    ) -> str:
        """Build the initial Markdown content for a note."""
        lines = [
            "---",
            f"title: {title}",
            f"note_type: {metadata.note_type}",
            f"status: {metadata.status}",
            "tags:",
        ]

        if metadata.tags:
            lines.extend(
                f"  - {tag}"
                for tag in metadata.tags
            )
        else:
            lines.append("  -")

        lines.extend(
            [
                f"created_at: {created_at.isoformat()}",
                f"updated_at: {updated_at.isoformat()}",
                "---",
                "",
                f"# {title}",
                "",
            ]
        )

        return "\n".join(lines)

    @staticmethod
    def _split_front_matter(
        content: str,
    ) -> tuple[str, str]:
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
    def _replace_front_matter_value(
        front_matter: str,
        key: str,
        value: str,
    ) -> str:
        """Replace or append a scalar front-matter value."""
        pattern = re.compile(
            rf"^{re.escape(key)}:[ \t]*.*$",
            flags=re.MULTILINE,
        )

        replacement = f"{key}: {value}"

        if pattern.search(front_matter):
            return pattern.sub(
                replacement,
                front_matter,
                count=1,
            )

        return (
            front_matter.rstrip("\n")
            + "\n"
            + replacement
        )

    @staticmethod
    def _parse_front_matter_metadata(
        front_matter: str,
    ) -> tuple[
        str | None,
        str | None,
        tuple[str, ...],
        datetime | None,
        datetime | None,
    ]:
        """Parse note metadata from YAML-like front matter.

        This parser intentionally handles only the small YAML subset
        generated by KnowledgeForge. It avoids a full YAML dependency
        while keeping parsing deterministic and safe.
        """
        note_type: str | None = None
        status: str | None = None
        tags: list[str] = []
        created_at: datetime | None = None
        updated_at: datetime | None = None

        lines = front_matter.splitlines()
        index = 0

        while index < len(lines):
            line = lines[index].strip()

            if not line or line == "---":
                index += 1
                continue

            if line.startswith("note_type:"):
                note_type = line.split(
                    ":",
                    1,
                )[1].strip()

                index += 1
                continue

            if line.startswith("status:"):
                status = line.split(
                    ":",
                    1,
                )[1].strip()

                index += 1
                continue

            if line.startswith("created_at:"):
                raw_created_at = line.split(
                    ":",
                    1,
                )[1].strip()

                try:
                    created_at = datetime.fromisoformat(
                        raw_created_at
                    )
                except ValueError:
                    created_at = None

                index += 1
                continue

            if line.startswith("updated_at:"):
                raw_updated_at = line.split(
                    ":",
                    1,
                )[1].strip()

                try:
                    updated_at = datetime.fromisoformat(
                        raw_updated_at
                    )
                except ValueError:
                    updated_at = None

                index += 1
                continue

            if line == "tags:":
                index += 1

                while index < len(lines):
                    tag_line = lines[index]
                    stripped_tag_line = tag_line.strip()

                    if not stripped_tag_line:
                        index += 1
                        continue

                    if not tag_line.startswith((" ", "\t")):
                        break

                    if not stripped_tag_line.startswith("-"):
                        break

                    tag = stripped_tag_line[1:].strip()

                    if tag:
                        tags.append(tag)

                    index += 1

                continue

            index += 1

        normalized_tags: list[str] = []

        for tag in tags:
            normalized_tag = re.sub(
                r"\s+",
                "-",
                tag.casefold().strip(),
            )

            if (
                normalized_tag
                and normalized_tag not in normalized_tags
            ):
                normalized_tags.append(normalized_tag)

        return (
            note_type,
            status,
            tuple(normalized_tags),
            created_at,
            updated_at,
        )

    @staticmethod
    def _read_note_metadata(note_path: Path) -> Note:
        """Read note metadata from a Markdown file."""
        content = note_path.read_text(
            encoding="utf-8",
        )

        title = (
            note_path.stem
            .replace("-", " ")
            .title()
        )

        stat = note_path.stat()

        created_at = datetime.fromtimestamp(
            stat.st_ctime,
            tz=UTC,
        )

        updated_at = datetime.fromtimestamp(
            stat.st_mtime,
            tz=UTC,
        )

        metadata = NoteMetadata()

        if content.startswith("---\n"):
            front_matter, _ = NoteService._split_front_matter(
                content
            )

            title_match = re.search(
                r"^title:[ \t]*(.+)$",
                front_matter,
                flags=re.MULTILINE,
            )

            if title_match:
                title = title_match.group(1).strip()

            (
                note_type,
                status,
                tags,
                parsed_created_at,
                parsed_updated_at,
            ) = NoteService._parse_front_matter_metadata(
                front_matter
            )

            normalized_note_type = (
                note_type.strip().casefold()
                if note_type
                else "concept"
            )

            normalized_status = (
                status.strip().casefold()
                if status
                else "draft"
            )

            metadata = NoteMetadata(
                note_type=normalized_note_type,
                status=normalized_status,
                tags=tags,
            )

            if parsed_created_at is not None:
                created_at = parsed_created_at

            if parsed_updated_at is not None:
                updated_at = parsed_updated_at

        return Note(
            title=title,
            path=note_path,
            created_at=created_at,
            updated_at=updated_at,
            metadata=metadata,
        )
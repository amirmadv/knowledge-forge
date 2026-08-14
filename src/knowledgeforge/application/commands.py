"""Application commands for KnowledgeForge."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from knowledgeforge.domain.note import Note, NoteService
from knowledgeforge.domain.template import (
    InvalidTemplateNameError,
    Template,
    TemplateService,
)
from knowledgeforge.domain.vault import VaultService
from knowledgeforge.infrastructure.config.settings import Settings
from knowledgeforge.infrastructure.template.filesystem import (
    FilesystemTemplateRepository,
)

INITIALIZATION_SUCCESS_MESSAGE = "KnowledgeForge initialized successfully."


@dataclass(frozen=True, slots=True)
class CommandResult:
    """Result returned by an application command."""

    message: str
    note: Note | None = None
    template: Template | None = None
    vault_path: Path | None = None


def _resolve_vault_path(vault_path: Path | None) -> Path:
    """Resolve the configured or explicitly provided vault path."""
    return vault_path or Path(Settings().vault_path)


def _resolve_templates_path(
    templates_path: Path | None,
) -> Path:
    """Resolve the configured or explicitly provided templates path."""
    return templates_path or Path(Settings().templates_path)


def _create_template_service(
    templates_path: Path | None = None,
) -> TemplateService:
    """Create the template service backed by the filesystem."""
    repository = FilesystemTemplateRepository(
        _resolve_templates_path(templates_path)
    )

    return TemplateService(repository)


def _create_note_service(
    vault_path: Path | None = None,
    templates_path: Path | None = None,
) -> NoteService:
    """Create a note service with its template service."""
    return NoteService(
        vault_path=_resolve_vault_path(vault_path),
        template_service=_create_template_service(templates_path),
    )


def initialize_knowledgeforge(
    vault_path: Path | None = None,
) -> CommandResult:
    """Initialize a KnowledgeForge vault."""
    resolved_vault_path = _resolve_vault_path(vault_path)

    vault_service = VaultService()
    created_path = vault_service.create(resolved_vault_path)

    _create_template_service()

    return CommandResult(
        message=INITIALIZATION_SUCCESS_MESSAGE,
        vault_path=created_path,
    )


def create_note(
    title: str,
    template: str = "default",
    vault_path: Path | None = None,
    templates_path: Path | None = None,
    note_type: str = "concept",
    status: str = "draft",
    tags: tuple[str, ...] = (),
) -> CommandResult:
    """Create a new note using the requested template."""
    note_service = _create_note_service(
        vault_path=vault_path,
        templates_path=templates_path,
    )

    note = note_service.create(
        title=title,
        template=template,
        note_type=note_type,
        status=status,
        tags=tags,
    )

    return CommandResult(
        message="Note created successfully.",
        note=note,
    )


def list_notes(
    vault_path: Path | None = None,
) -> list[Note]:
    """List all notes in the configured vault."""
    service = NoteService(
        vault_path=_resolve_vault_path(vault_path),
    )

    return service.list_notes()


def show_note(
    title: str,
    vault_path: Path | None = None,
) -> tuple[Note, str]:
    """Return a note and its complete Markdown content."""
    service = NoteService(
        vault_path=_resolve_vault_path(vault_path),
    )

    note = service.get(title)
    content = service.read_content(title)

    return note, content


def update_note(
    title: str,
    content: str,
    vault_path: Path | None = None,
) -> CommandResult:
    """Replace the content of an existing note."""
    service = NoteService(
        vault_path=_resolve_vault_path(vault_path),
    )

    note = service.update(
        title=title,
        content=content,
    )

    return CommandResult(
        message="Note updated successfully.",
        note=note,
    )


def search_notes(
    query: str,
    vault_path: Path | None = None,
) -> list[Note]:
    """Search notes by title, content, and metadata."""
    service = NoteService(
        vault_path=_resolve_vault_path(vault_path),
    )

    return service.search(query)


def list_templates(
    templates_path: Path | None = None,
) -> list[str]:
    """List all available template names."""
    template_service = _create_template_service(templates_path)

    templates = template_service.list()

    return [template.name for template in templates]


def show_template(
    name: str,
    templates_path: Path | None = None,
) -> Template:
    """Return a template by name."""
    template_service = _create_template_service(templates_path)

    return template_service.get(name)


def create_template(
    name: str,
    content: str,
    templates_path: Path | None = None,
) -> CommandResult:
    """Create a new Markdown template."""
    template_service = _create_template_service(templates_path)

    normalized_name = name.strip()

    if not normalized_name:
        raise InvalidTemplateNameError(
            "Template name cannot be empty."
        )

    template_service._validate_name(normalized_name)

    repository = FilesystemTemplateRepository(
        _resolve_templates_path(templates_path)
    )

    template = repository.create(
        name=normalized_name,
        content=content,
    )

    return CommandResult(
        message="Template created successfully.",
        template=template,
    )
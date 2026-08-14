"""Filesystem-backed template repository."""

from pathlib import Path

from knowledgeforge.domain.template import (
    Template,
    TemplateNotFoundError,
    TemplateRepository,
)

DEFAULT_TEMPLATE_CONTENT = """---
title: {{ title }}
note_type: {{ note_type }}
status: {{ status }}
tags:
{{ tags_yaml }}
created_at: {{ created_at }}
updated_at: {{ updated_at }}
---

# {{ title }}

{{ content }}
"""


class FilesystemTemplateRepository(TemplateRepository):
    """Store and retrieve templates from the filesystem."""

    def __init__(self, templates_path: Path) -> None:
        self.templates_path = Path(templates_path)
        self.templates_path.mkdir(parents=True, exist_ok=True)
        self._ensure_default_template()

    def list(self) -> list[Template]:
        """Return all Markdown templates sorted by name."""
        templates: list[Template] = []

        for path in sorted(self.templates_path.glob("*.md")):
            templates.append(self._load(path))

        return templates

    def get(self, name: str) -> Template:
        """Return a template by name."""
        template_path = self.templates_path / f"{name}.md"

        if not template_path.exists():
            raise TemplateNotFoundError(
                f"Template not found: {name}"
            )

        return self._load(template_path)

    def create(self, name: str, content: str) -> Template:
        """Create a new template."""
        template_path = self.templates_path / f"{name}.md"

        if template_path.exists():
            raise FileExistsError(
                f"Template already exists: {name}"
            )

        template = Template(
            name=name,
            path=template_path,
            content=content,
        )

        return self.save(template)

    def save(self, template: Template) -> Template:
        """Save a template to the filesystem."""
        template_path = self.templates_path / f"{template.name}.md"

        template_path.write_text(
            template.content,
            encoding="utf-8",
        )

        return template

    def _load(self, path: Path) -> Template:
        """Load a template from the filesystem."""
        return Template(
            name=path.stem,
            path=path,
            content=path.read_text(encoding="utf-8"),
        )

    def _ensure_default_template(self) -> None:
        """Create the default template when it does not exist."""
        default_path = self.templates_path / "default.md"

        if not default_path.exists():
            default_path.write_text(
                DEFAULT_TEMPLATE_CONTENT,
                encoding="utf-8",
            )


# Backward-compatible alias for callers using the other capitalization.
FileSystemTemplateRepository = FilesystemTemplateRepository
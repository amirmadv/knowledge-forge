"""Domain service for KnowledgeForge templates."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Protocol

from knowledgeforge.domain.template.model import Template


class TemplateNotFoundError(FileNotFoundError):
    """Raised when a requested template does not exist."""


class InvalidTemplateNameError(ValueError):
    """Raised when a template name is invalid."""


class UnknownTemplatePlaceholderError(ValueError):
    """Raised when template content contains an unknown placeholder."""


class TemplateRepository(Protocol):
    """Repository interface required by the template service."""

    def list(self) -> list[Template]:
        """Return all available templates."""

    def get(self, name: str) -> Template:
        """Return a template by name."""


class TemplateService:
    """Provide reusable template discovery and rendering behavior."""

    _VALID_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$")

    def __init__(self, repository: TemplateRepository) -> None:
        """Initialize the template service.

        Args:
            repository: Repository used to access templates.
        """
        self._repository = repository

    def list(self) -> list[Template]:
        """Return all available templates."""
        return self._repository.list()

    def get(self, name: str) -> Template:
        """Return a template by name.

        Args:
            name: Template name.

        Raises:
            InvalidTemplateNameError: If the name is invalid.
            TemplateNotFoundError: If the template does not exist.
        """
        normalized_name = name.strip()

        self._validate_name(normalized_name)

        try:
            return self._repository.get(normalized_name)
        except TemplateNotFoundError:
            raise
        except KeyError as exc:
            raise TemplateNotFoundError(
                f"Template not found: {normalized_name}"
            ) from exc

    def render(
        self,
        template: Template,
        values: Mapping[str, object],
    ) -> str:
        """Render a template using named placeholders.

        Supported placeholders use the following syntax:

            {{ title }}

        Args:
            template: Template to render.
            values: Placeholder values.

        Returns:
            Rendered template content.

        Raises:
            UnknownTemplatePlaceholderError:
                If a placeholder has no corresponding value.
        """
        content = template.content

        placeholders = self._extract_placeholders(content)

        unknown_placeholders = [
            placeholder
            for placeholder in placeholders
            if placeholder not in values
        ]

        if unknown_placeholders:
            names = ", ".join(sorted(unknown_placeholders))
            raise UnknownTemplatePlaceholderError(
                f"Unknown template placeholder(s): {names}"
            )

        for placeholder in placeholders:
            value = str(values[placeholder])
            content = content.replace(
                "{{" + placeholder + "}}",
                value,
            )
            content = content.replace(
                "{{ " + placeholder + " }}",
                value,
            )

        return content

    @classmethod
    def _validate_name(cls, name: str) -> None:
        """Validate a template name."""
        if not name or not cls._VALID_NAME_PATTERN.fullmatch(name):
            raise InvalidTemplateNameError(
                "Template name must contain only letters, "
                "numbers, hyphens, and underscores."
            )

    @staticmethod
    def _extract_placeholders(content: str) -> set[str]:
        """Extract placeholders from template content."""
        placeholders: set[str] = set()

        for match in re.finditer(
            r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}",
            content,
        ):
            placeholders.add(match.group(1))

        return placeholders
"""Tests for the KnowledgeForge template domain service."""

from pathlib import Path

import pytest

from knowledgeforge.domain.template import (
    InvalidTemplateNameError,
    Template,
    TemplateNotFoundError,
    TemplateService,
    UnknownTemplatePlaceholderError,
)


class InMemoryTemplateRepository:
    """Simple in-memory repository for testing."""

    def __init__(self, templates: list[Template]) -> None:
        self._templates = {template.name: template for template in templates}

    def list(self) -> list[Template]:
        return list(self._templates.values())

    def get(self, name: str) -> Template:
        if name not in self._templates:
            raise TemplateNotFoundError(f"Template not found: {name}")

        return self._templates[name]


@pytest.fixture
def repository() -> InMemoryTemplateRepository:
    """Return a repository containing test templates."""
    return InMemoryTemplateRepository(
        [
            Template(
                name="default",
                path=Path("default.md"),
                content=(
                    "# {{ title }}\n\n"
                    "Created: {{ created_at }}\n\n"
                    "{{ content }}\n"
                ),
            ),
            Template(
                name="technical",
                path=Path("technical.md"),
                content=(
                    "# {{ title }}\n\n"
                    "## Technical Notes\n\n"
                    "{{ content }}\n"
                ),
            ),
        ]
    )


@pytest.fixture
def service(
    repository: InMemoryTemplateRepository,
) -> TemplateService:
    """Return a template service backed by the test repository."""
    return TemplateService(repository)


def test_list_returns_available_templates(
    service: TemplateService,
) -> None:
    """Service should return templates from the repository."""
    templates = service.list()

    assert [template.name for template in templates] == [
        "default",
        "technical",
    ]


def test_get_returns_requested_template(
    service: TemplateService,
) -> None:
    """Service should return the requested template."""
    template = service.get("technical")

    assert template.name == "technical"
    assert "Technical Notes" in template.content


def test_get_raises_for_missing_template(
    service: TemplateService,
) -> None:
    """Service should raise when a template does not exist."""
    with pytest.raises(TemplateNotFoundError):
        service.get("does-not-exist")


@pytest.mark.parametrize(
    "name",
    [
        "",
        " ",
        "../default",
        "default.md",
        "template/name",
        "template name",
    ],
)
def test_get_rejects_invalid_template_name(
    service: TemplateService,
    name: str,
) -> None:
    """Service should reject unsafe or invalid template names."""
    with pytest.raises(InvalidTemplateNameError):
        service.get(name)


def test_render_replaces_all_placeholders(
    service: TemplateService,
) -> None:
    """Service should replace every known placeholder."""
    template = service.get("default")

    rendered = service.render(
        template,
        {
            "title": "Machine Learning",
            "created_at": "2026-08-10T10:00:00+00:00",
            "content": "Linear regression is supervised learning.",
        },
    )

    assert "# Machine Learning" in rendered
    assert "Created: 2026-08-10T10:00:00+00:00" in rendered
    assert "Linear regression is supervised learning." in rendered
    assert "{{ title }}" not in rendered
    assert "{{ created_at }}" not in rendered
    assert "{{ content }}" not in rendered


def test_render_supports_repeated_placeholders(
    service: TemplateService,
) -> None:
    """Service should replace repeated placeholders."""
    template = Template(
        name="repeated",
        path=Path("repeated.md"),
        content="{{ title }}\n{{ title }}\n",
    )

    rendered = service.render(
        template,
        {"title": "Deep Learning"},
    )

    assert rendered == "Deep Learning\nDeep Learning\n"


def test_render_raises_for_unknown_placeholder(
    service: TemplateService,
) -> None:
    """Service should reject placeholders without supplied values."""
    template = Template(
        name="broken",
        path=Path("broken.md"),
        content="# {{ title }}\n{{ unknown }}\n",
    )

    with pytest.raises(UnknownTemplatePlaceholderError):
        service.render(
            template,
            {"title": "Machine Learning"},
        )
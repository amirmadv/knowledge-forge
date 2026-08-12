"""Tests for the filesystem-backed template repository."""

from pathlib import Path

from knowledgeforge.domain.template import TemplateNotFoundError
from knowledgeforge.infrastructure.template.filesystem import (
    FilesystemTemplateRepository,
)


def test_repository_creates_default_template(tmp_path: Path) -> None:
    """Repository should create the default template automatically."""
    templates_path = tmp_path / "templates"

    repository = FilesystemTemplateRepository(templates_path)

    default_path = templates_path / "default.md"

    assert default_path.is_file()
    assert repository.get("default").name == "default"


def test_repository_lists_markdown_templates(tmp_path: Path) -> None:
    """Repository should discover Markdown templates."""
    templates_path = tmp_path / "templates"

    repository = FilesystemTemplateRepository(templates_path)

    (templates_path / "technical.md").write_text(
        "# Technical\n",
        encoding="utf-8",
    )
    (templates_path / "research.md").write_text(
        "# Research\n",
        encoding="utf-8",
    )

    templates = repository.list()

    names = [template.name for template in templates]

    assert names == ["default", "research", "technical"]


def test_repository_loads_template_content(tmp_path: Path) -> None:
    """Repository should load template content from disk."""
    templates_path = tmp_path / "templates"
    templates_path.mkdir()

    template_path = templates_path / "technical.md"
    template_path.write_text(
        "# {{ title }}\n\n{{ content }}\n",
        encoding="utf-8",
    )

    repository = FilesystemTemplateRepository(templates_path)

    template = repository.get("technical")

    assert template.name == "technical"
    assert template.path == template_path
    assert template.content == "# {{ title }}\n\n{{ content }}\n"


def test_repository_raises_for_missing_template(tmp_path: Path) -> None:
    """Repository should raise a clear error for missing templates."""
    repository = FilesystemTemplateRepository(tmp_path / "templates")

    try:
        repository.get("does-not-exist")
    except TemplateNotFoundError as exc:
        assert "does-not-exist" in str(exc)
    else:
        raise AssertionError(
            "Expected TemplateNotFoundError for a missing template."
        )
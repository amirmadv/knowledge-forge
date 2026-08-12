"""Tests for KnowledgeForge template application commands."""

from pathlib import Path

from knowledgeforge.application.commands import (
    list_templates,
    show_template,
)


def test_list_templates_returns_available_templates(
    tmp_path: Path,
) -> None:
    """Listing templates should return available template names."""
    templates_path = tmp_path / "templates"
    templates_path.mkdir()

    (templates_path / "default.md").write_text(
        "# {{ title }}\n",
        encoding="utf-8",
    )
    (templates_path / "technical.md").write_text(
        "# Technical\n",
        encoding="utf-8",
    )

    result = list_templates(templates_path)

    assert result == ["default", "technical"]


def test_show_template_returns_requested_template(
    tmp_path: Path,
) -> None:
    """Showing a template should return its domain model."""
    templates_path = tmp_path / "templates"
    templates_path.mkdir()

    template_path = templates_path / "technical.md"
    template_path.write_text(
        "# Technical\n",
        encoding="utf-8",
    )

    result = show_template(
        name="technical",
        templates_path=templates_path,
    )

    assert result.name == "technical"
    assert result.path == template_path
    assert result.content == "# Technical\n"
"""Tests for the KnowledgeForge template domain model."""

from pathlib import Path

from knowledgeforge.domain.template import Template


def test_template_exposes_expected_fields() -> None:
    """A template should expose its name, path, and content."""
    template = Template(
        name="default",
        path=Path("templates/default.md"),
        content="# {{ title }}",
    )

    assert template.name == "default"
    assert template.path == Path("templates/default.md")
    assert template.content == "# {{ title }}"


def test_template_filename_returns_path_name() -> None:
    """A template should expose its filename."""
    template = Template(
        name="technical",
        path=Path("templates/technical.md"),
        content="# {{ title }}",
    )

    assert template.filename == "technical.md"
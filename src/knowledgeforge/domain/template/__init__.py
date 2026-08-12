"""Template domain package."""

from knowledgeforge.domain.template.model import Template
from knowledgeforge.domain.template.service import (
    InvalidTemplateNameError,
    TemplateNotFoundError,
    TemplateRepository,
    TemplateService,
    UnknownTemplatePlaceholderError,
)

__all__ = [
    "InvalidTemplateNameError",
    "Template",
    "TemplateNotFoundError",
    "TemplateRepository",
    "TemplateService",
    "UnknownTemplatePlaceholderError",
]
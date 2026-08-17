# KnowledgeForge

KnowledgeForge is an AI-powered personal knowledge management system designed
to organize, structure, search, connect, and enrich a personal knowledge base.

The project is intentionally local-first: notes live as Markdown files, relationships
form a knowledge graph, and the AI agent retrieves local note content plus graph context
before asking an OpenAI-compatible model for an answer.

## Current MVP

- Markdown vault and note lifecycle management
- Note metadata, tags, templates, and search
- Directed knowledge graph with ancestors, descendants, backlinks, and neighborhoods
- Graph-aware AI agent with local-first RAG retrieval
- OpenAI-compatible provider client
- CLI for normal KnowledgeForge operations and AI questions
- 99 automated tests currently passing on the development environment

## Requirements

- Python 3.12+
- uv
- Git
- An OpenAI-compatible AI provider for `knowledgeforge-ai`

## Installation

Synchronize the project dependencies:

```powershell
python -m uv sync
```

Initialize a vault:

```powershell
python -m knowledgeforge init
```

Or use the installed script after `uv sync`:

```powershell
knowledgeforge init
```

## Enable AI

Copy `.env.example` to `.env` and configure the provider.

For OpenAI:

```text
KNOWLEDGEFORGE_AI_ENABLED=true
KNOWLEDGEFORGE_AI_BASE_URL=https://api.openai.com/v1
KNOWLEDGEFORGE_AI_API_KEY=your-api-key
KNOWLEDGEFORGE_AI_MODEL=gpt-4o-mini
```

For a local Ollama server using its OpenAI-compatible endpoint:

```text
KNOWLEDGEFORGE_AI_ENABLED=true
KNOWLEDGEFORGE_AI_BASE_URL=http://localhost:11434/v1
KNOWLEDGEFORGE_AI_API_KEY=ollama
KNOWLEDGEFORGE_AI_MODEL=qwen3:4b
```

Then ask the agent:

```powershell
knowledgeforge-ai ask "How does linear regression relate to gradient descent?"
```

The agent first searches the local vault. If a full-question match is not found,
it falls back to meaningful query terms and then expands matching notes through
the knowledge graph. This keeps the AI grounded in the user's own knowledge.

## Development checks

Run linting and the full test suite:

```powershell
python -m uv run ruff check .
python -m uv run pytest -vv
```

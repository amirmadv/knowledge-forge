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
- Semantic embedding retrieval with lexical fallback
- Bounded conversational memory for follow-up AI questions
- Grounded source-note reporting for every AI answer
- OpenAI-compatible chat and embedding client
- CLI for normal KnowledgeForge operations and AI questions
- Interactive `knowledgeforge-ai chat` mode
- 105 automated tests currently passing on the development environment

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
KNOWLEDGEFORGE_AI_EMBEDDING_MODEL=text-embedding-3-small
```

For a local Ollama server using its OpenAI-compatible endpoint, use an embedding-capable
model such as `nomic-embed-text` alongside your chat model:

```text
KNOWLEDGEFORGE_AI_ENABLED=true
KNOWLEDGEFORGE_AI_BASE_URL=http://localhost:11434/v1
KNOWLEDGEFORGE_AI_API_KEY=ollama
KNOWLEDGEFORGE_AI_MODEL=qwen3:4b
KNOWLEDGEFORGE_AI_EMBEDDING_MODEL=nomic-embed-text
```

Then ask the agent:

```powershell
knowledgeforge-ai ask "How does linear regression relate to gradient descent?"
```

For a multi-turn session with short-term conversational memory:

```powershell
knowledgeforge-ai chat
```

Inside chat, use `/clear` to reset conversation memory and `/exit` to leave.

The agent retrieves knowledge in this order: exact local matches first, then semantic
embedding similarity when the provider supports embeddings, then keyword matches. The
selected notes are expanded through the knowledge graph before the final prompt is sent
to the chat model. If embeddings are unavailable, KnowledgeForge automatically falls back
to lexical retrieval so the core agent remains usable.

The CLI prints the note titles used as grounded sources for each answer.

## Development checks

Run linting and the full test suite:

```powershell
python -m uv run ruff check .
python -m uv run pytest -vv
```

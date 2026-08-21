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
- Persistent, incremental semantic embedding index
- Explainable hybrid retrieval with semantic, lexical, and metadata score breakdowns
- Retrieval evaluation with Precision@K, Recall@K, MRR, and configurable quality gates
- Bounded conversational memory for follow-up AI questions
- Grounded source-note reporting for every AI answer
- OpenAI-compatible chat and embedding client
- CLI for normal KnowledgeForge operations and AI questions
- Interactive `knowledgeforge-ai chat` mode
- Knowledge Copilot for summaries, tags, note improvement, related notes, knowledge gaps, and note creation

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

Search the vault directly:

```powershell
knowledgeforge-ai search "gradient descent"
```

Inspect why notes were ranked:

```powershell
knowledgeforge-ai search "gradient descent" --explain
```

The explain mode reports the final score, semantic/lexical/metadata signals, and
human-readable reasons for each ranked note. The ranking logic remains in the application
retrieval layer; the CLI only renders the evidence.

For a multi-turn session with short-term conversational memory:

```powershell
knowledgeforge-ai chat
```

Inside chat, use `/clear` to reset conversation memory and `/exit` to leave.

## Semantic index

Semantic embeddings are stored locally at:

```text
vault/.knowledgeforge/semantic-index.json
```

The index is updated incrementally. Unchanged notes reuse their existing vectors;
changed notes are re-embedded automatically. Changing the configured embedding model
invalidates the previous vectors.

To explicitly rebuild the complete index:

```powershell
knowledgeforge-ai index
```

The agent retrieves knowledge using hybrid semantic, lexical, and metadata signals, then
expands selected notes through the knowledge graph before the final prompt is sent to the
chat model. If embeddings are unavailable, KnowledgeForge automatically falls back to
lexical retrieval so the core agent remains usable.

The CLI prints the note titles used as grounded sources for each AI answer.

## Retrieval evaluation

Create a small gold dataset containing query-to-note-slug relationships, for example:

```json
{
  "cases": [
    {
      "query": "linear regression",
      "relevant": ["linear-regression"]
    },
    {
      "query": "gradient descent",
      "relevant": ["gradient-descent", "optimization"]
    }
  ]
}
```

Run the evaluation:

```powershell
knowledgeforge-ai evaluate .\evals\retrieval.json `
  --k 5 `
  --min-precision 0.60 `
  --min-recall 0.80 `
  --min-mrr 0.75
```

The command reports Precision@K, Recall@K, and MRR. A failed quality gate exits with code `2`
by default, making the command suitable for CI. Use `--details` for per-query diagnostics or
`--no-fail-on-gate` when you want a report without failing the process.

See `docs/evaluation/retrieval-evaluation.md` for the dataset contract and evaluation guidance.

## Development checks

Run linting and the full test suite:

```powershell
python -m uv run ruff check .
python -m uv run pytest -vv
```
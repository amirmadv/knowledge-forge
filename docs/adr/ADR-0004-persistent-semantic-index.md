# ADR-0004: Persist the Semantic Embedding Index

## Status

Accepted

## Context

KnowledgeForge already supports semantic retrieval, but the embedding cache was
held only in memory. Every new `knowledgeforge-ai` process therefore had to
re-embed every note before semantic search could rank the vault.

The project is local-first and stores notes as Markdown files, so the semantic
index should follow the same principle without introducing a database dependency.

## Decision

Persist semantic embeddings under the vault at:

`.knowledgeforge/semantic-index.json`

Each record contains a content fingerprint and embedding vector. The fingerprint
includes the configured embedding model, note title, and complete Markdown
content. This gives us:

- reuse across CLI processes;
- automatic re-embedding when a note changes;
- automatic invalidation when the embedding model changes;
- removal of stale records when notes are deleted;
- no additional runtime dependency or database.

The index is an optimization, not the source of truth. If it cannot be written,
semantic retrieval continues in memory. If an index is missing or incompatible,
it is rebuilt incrementally on demand.

A `knowledgeforge-ai index` command is provided for explicit full rebuilds.

## Consequences

Positive:

- repeated AI sessions become significantly cheaper for unchanged vaults;
- semantic retrieval remains local-first and file-based;
- embedding model changes do not silently reuse incompatible vectors;
- users can explicitly rebuild the index when needed.

Trade-offs:

- the vault contains an additional generated JSON file;
- embedding vectors can increase local disk usage;
- the current index is a simple linear scan and is not yet a vector database.

A vector database or ANN index can be introduced later if vault size requires it.

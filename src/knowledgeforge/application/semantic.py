"""Semantic retrieval primitives for KnowledgeForge."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

from knowledgeforge.domain.note import Note, NoteService


@dataclass(frozen=True, slots=True)
class SemanticMatch:
    """A note ranked by embedding similarity."""

    note: Note
    score: float


class SemanticRetriever:
    """Retrieve notes with an OpenAI-compatible embedding client.

    Embeddings are persisted inside the local vault so repeated AI questions do
    not re-embed unchanged notes. A content fingerprint invalidates an entry
    whenever its note changes, and changing the embedding model invalidates the
    complete index.
    """

    INDEX_VERSION = 1
    INDEX_DIRECTORY = ".knowledgeforge"
    INDEX_FILENAME = "semantic-index.json"

    def __init__(
        self,
        note_service: NoteService,
        client: object,
        index_path: Path | None = None,
    ) -> None:
        self._note_service = note_service
        self._client = client
        self._index_path = index_path or (
            note_service.vault_path
            / self.INDEX_DIRECTORY
            / self.INDEX_FILENAME
        )
        self._embedding_model = str(
            getattr(client, "embedding_model", "")
        )
        self._cache: dict[str, tuple[float, ...]] = {}
        self._fingerprints: dict[str, str] = {}
        self._load_index()

    def search(self, query: str, limit: int = 5) -> list[SemanticMatch]:
        """Return the most semantically similar notes."""
        if not query.strip() or limit <= 0:
            return []

        embed = getattr(self._client, "embed", None)
        if not callable(embed):
            return []

        query_vector = tuple(embed(query))
        if not query_vector:
            return []

        matches: list[SemanticMatch] = []
        current_keys: set[str] = set()
        changed = False

        for note in self._note_service.list_notes():
            content = self._note_service.read_content(note.title)
            key = note.path.stem
            fingerprint = self._fingerprint(note.title, content)
            current_keys.add(key)

            vector = self._cache.get(key)
            if vector is None or self._fingerprints.get(key) != fingerprint:
                vector = tuple(embed(f"{note.title}\n{content}"))
                if vector:
                    self._cache[key] = vector
                    self._fingerprints[key] = fingerprint
                    changed = True

            score = cosine_similarity(query_vector, vector)
            if score > 0:
                matches.append(SemanticMatch(note=note, score=score))

        stale_keys = set(self._cache) - current_keys
        if stale_keys:
            changed = True
            for key in stale_keys:
                self._cache.pop(key, None)
                self._fingerprints.pop(key, None)

        if changed:
            self._persist_index()

        matches.sort(key=lambda item: item.score, reverse=True)
        return matches[:limit]

    def rebuild(self) -> int:
        """Rebuild and persist embeddings for every note in the vault."""
        embed = getattr(self._client, "embed", None)
        if not callable(embed):
            return 0

        self._cache.clear()
        self._fingerprints.clear()

        notes = self._note_service.list_notes()
        for note in notes:
            content = self._note_service.read_content(note.title)
            vector = tuple(embed(f"{note.title}\n{content}"))
            if vector:
                key = note.path.stem
                self._cache[key] = vector
                self._fingerprints[key] = self._fingerprint(
                    note.title,
                    content,
                )

        self._persist_index()
        return len(self._cache)

    def _fingerprint(self, title: str, content: str) -> str:
        """Create a stable fingerprint for one note and embedding model."""
        payload = f"{self._embedding_model}\n{title}\n{content}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _load_index(self) -> None:
        """Load a compatible persisted embedding index if one exists."""
        try:
            payload = json.loads(self._index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return

        if not isinstance(payload, dict):
            return
        if payload.get("version") != self.INDEX_VERSION:
            return
        if payload.get("embedding_model", "") != self._embedding_model:
            return

        records = payload.get("notes", {})
        if not isinstance(records, dict):
            return

        for key, record in records.items():
            if not isinstance(key, str) or not isinstance(record, dict):
                continue

            fingerprint = record.get("fingerprint")
            vector = record.get("vector")
            if not isinstance(fingerprint, str) or not isinstance(vector, list):
                continue
            if not all(isinstance(value, (int, float)) for value in vector):
                continue

            self._fingerprints[key] = fingerprint
            self._cache[key] = tuple(float(value) for value in vector)

    def _persist_index(self) -> None:
        """Persist the current embedding index without affecting retrieval."""
        payload = {
            "version": self.INDEX_VERSION,
            "embedding_model": self._embedding_model,
            "notes": {
                key: {
                    "fingerprint": self._fingerprints[key],
                    "vector": list(self._cache[key]),
                }
                for key in sorted(self._cache)
                if key in self._fingerprints
            },
        }

        try:
            self._index_path.parent.mkdir(parents=True, exist_ok=True)
            self._index_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError:
            # Retrieval must remain usable even when the local cache cannot be written.
            return


def cosine_similarity(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    """Calculate cosine similarity for two equal-length vectors."""
    if not left or not right or len(left) != len(right):
        return 0.0

    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0

    return sum(a * b for a, b in zip(left, right, strict=True)) / (
        left_norm * right_norm
    )

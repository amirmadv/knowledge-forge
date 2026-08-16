"""Domain service for KnowledgeForge note relationships."""

from __future__ import annotations

import json
from pathlib import Path

from knowledgeforge.domain.note import Note, NoteService
from knowledgeforge.domain.relationship.model import (
    NoteRelation,
    RelationshipType,
)


class RelationshipAlreadyExistsError(ValueError):
    """Raised when a relationship already exists."""


class RelationshipNotFoundError(ValueError):
    """Raised when a relationship does not exist."""


class InvalidRelationshipError(ValueError):
    """Raised when a relationship is invalid."""


class RelationshipService:
    """Create and manage relationships between KnowledgeForge notes."""

    KNOWLEDGE_DIRECTORY = ".knowledge"
    RELATIONS_FILENAME = "relations.json"

    def __init__(
        self,
        vault_path: Path,
        note_service: NoteService | None = None,
    ) -> None:
        """Initialize the relationship service.

        Args:
            vault_path: Root directory of the KnowledgeForge vault.
            note_service: Optional note service used to validate notes.
        """
        self._vault_path = vault_path
        self._note_service = note_service or NoteService(vault_path)

    @property
    def relations_path(self) -> Path:
        """Return the path used to persist relationships."""
        return (
            self._vault_path
            / self.KNOWLEDGE_DIRECTORY
            / self.RELATIONS_FILENAME
        )

    def get_note(
        self,
        title: str,
    ) -> Note:
        """Return a note using the configured note service.

        Args:
            title: Title of the requested note.

        Returns:
            The requested Note.

        Raises:
            NoteNotFoundError: If the note does not exist.
            InvalidNoteTitleError: If the title is invalid.
        """
        return self._note_service.get(title)

    def add(
        self,
        source_title: str,
        target_title: str,
        relation_type: RelationshipType | str = RelationshipType.RELATED,
    ) -> NoteRelation:
        """Create a relationship between two existing notes.

        Args:
            source_title: Title of the source note.
            target_title: Title of the target note.
            relation_type: Type of relationship.

        Returns:
            The newly created relationship.

        Raises:
            NoteNotFoundError: If either note does not exist.
            InvalidRelationshipError: If the relationship is invalid.
            RelationshipAlreadyExistsError: If it already exists.
        """
        source = self.get_note(source_title)
        target = self.get_note(target_title)

        if source.path == target.path:
            raise InvalidRelationshipError(
                "A note cannot have a relationship with itself."
            )

        normalized_type = self._normalize_relation_type(relation_type)

        relation = NoteRelation(
            source=source.slug,
            target=target.slug,
            relation_type=normalized_type,
        )

        relations = self._load()

        if relation in relations:
            raise RelationshipAlreadyExistsError(
                "Relationship already exists."
            )

        relations.append(relation)
        self._save(relations)

        return relation

    def remove(
        self,
        source_title: str,
        target_title: str,
        relation_type: RelationshipType | str = RelationshipType.RELATED,
    ) -> NoteRelation:
        """Remove an existing relationship.

        Args:
            source_title: Title of the source note.
            target_title: Title of the target note.
            relation_type: Type of relationship.

        Returns:
            The removed relationship.

        Raises:
            NoteNotFoundError: If either note does not exist.
            RelationshipNotFoundError: If the relationship does not exist.
        """
        source = self.get_note(source_title)
        target = self.get_note(target_title)

        normalized_type = self._normalize_relation_type(relation_type)

        relation = NoteRelation(
            source=source.slug,
            target=target.slug,
            relation_type=normalized_type,
        )

        relations = self._load()

        if relation not in relations:
            raise RelationshipNotFoundError(
                "Relationship not found."
            )

        remaining = [
            item
            for item in relations
            if item != relation
        ]

        self._save(remaining)

        return relation

    def list_for(
        self,
        title: str,
    ) -> list[NoteRelation]:
        """Return all relationships involving a note.

        Args:
            title: Title of the note.

        Returns:
            Relationships where the note is either source or target.
        """
        note = self.get_note(title)

        return sorted(
            [
                relation
                for relation in self._load()
                if (
                    relation.source == note.slug
                    or relation.target == note.slug
                )
            ],
            key=lambda relation: (
                relation.relation_type.value,
                relation.source,
                relation.target,
            ),
        )

    def list_all(self) -> list[NoteRelation]:
        """Return all relationships in the vault."""
        return sorted(
            self._load(),
            key=lambda relation: (
                relation.source,
                relation.target,
                relation.relation_type.value,
            ),
        )

    @staticmethod
    def _normalize_relation_type(
        relation_type: RelationshipType | str,
    ) -> RelationshipType:
        """Normalize a relationship type."""
        if isinstance(relation_type, RelationshipType):
            return relation_type

        normalized = relation_type.strip().casefold()

        try:
            return RelationshipType(normalized)
        except ValueError as exc:
            valid_types = ", ".join(
                relation.value
                for relation in RelationshipType
            )

            raise InvalidRelationshipError(
                f"Invalid relationship type: {relation_type}. "
                f"Valid types: {valid_types}"
            ) from exc

    def _load(self) -> list[NoteRelation]:
        """Load relationships from persistent storage."""
        path = self.relations_path

        if not path.exists():
            return []

        raw_content = path.read_text(
            encoding="utf-8",
        ).strip()

        if not raw_content:
            return []

        try:
            raw_relations = json.loads(raw_content)
        except json.JSONDecodeError as exc:
            raise InvalidRelationshipError(
                f"Invalid relationship store: {path}"
            ) from exc

        if not isinstance(raw_relations, list):
            raise InvalidRelationshipError(
                "Relationship store must contain a JSON array."
            )

        relations: list[NoteRelation] = []

        for item in raw_relations:
            if not isinstance(item, dict):
                raise InvalidRelationshipError(
                    "Each relationship must be a JSON object."
                )

            try:
                source = item["source"]
                target = item["target"]
                relation_type = item["type"]
            except KeyError as exc:
                raise InvalidRelationshipError(
                    "Relationship is missing required fields."
                ) from exc

            if not all(
                isinstance(value, str)
                for value in (
                    source,
                    target,
                    relation_type,
                )
            ):
                raise InvalidRelationshipError(
                    "Relationship fields must be strings."
                )

            relations.append(
                NoteRelation(
                    source=source,
                    target=target,
                    relation_type=self._normalize_relation_type(
                        relation_type
                    ),
                )
            )

        return relations

    def _save(
        self,
        relations: list[NoteRelation],
    ) -> None:
        """Persist relationships to disk."""
        path = self.relations_path

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        payload = [
            {
                "source": relation.source,
                "target": relation.target,
                "type": relation.relation_type.value,
            }
            for relation in relations
        ]

        path.write_text(
            json.dumps(
                payload,
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
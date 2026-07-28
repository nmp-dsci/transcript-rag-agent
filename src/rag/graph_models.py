"""Record types for the GraphRAG knowledge graph (P4).

Three record kinds, per the s05 design: entities (canonical things the corpus
talks about), relations (typed edges between entities), and claims — the
load-bearing record. A claim carries its source chunk id, video id, timestamps
and ``upload_date``, which is what makes graph answers citable with the same
deep links vector answers produce, and what makes the temporal trend layer a
sort instead of a feature.

``ChunkExtraction`` is the validated JSON contract one LLM extraction call must
satisfy for one chunk; the chunk-level provenance fields are stamped by the
extractor from the chunk itself, never trusted from the model.
"""

from __future__ import annotations

import hashlib
import re

from pydantic import BaseModel, Field, field_validator


def entity_id_for(name: str) -> str:
    """Canonical entity id: lowercased, non-alphanumerics collapsed to ``-``.

    Ids are derived from names so the same entity extracted from two chunks
    merges without a resolution pass ("Negative Gearing" and "negative
    gearing" collide on purpose).
    """
    slug = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
    return slug or "unknown"


def claim_id_for(chunk_id: str, text: str) -> str:
    """Stable claim identity: the same sentence from the same chunk upserts once."""
    digest = hashlib.sha256(f"{chunk_id}\n{text.strip()}".encode("utf-8")).hexdigest()
    return f"claim:{digest[:16]}"


class ExtractedEntity(BaseModel):
    name: str
    type: str = "concept"
    aliases: list[str] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("entity name must not be empty")
        return value.strip()

    @property
    def entity_id(self) -> str:
        return entity_id_for(self.name)


class ExtractedRelation(BaseModel):
    source: str
    target: str
    type: str = "related_to"
    weight: float = 0.5

    @field_validator("weight")
    @classmethod
    def _unit_interval(cls, value: float) -> float:
        return min(1.0, max(0.0, value))


class ExtractedClaim(BaseModel):
    text: str
    entities: list[str] = Field(default_factory=list)
    polarity: str = "asserts"

    @field_validator("text")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("claim text must not be empty")
        return value.strip()


class ChunkExtraction(BaseModel):
    """One chunk's extracted graph records plus the provenance it inherits.

    Provenance fields default to empty/None so the model contract can be
    validated straight from LLM output; the extractor overwrites them from the
    ``TranscriptChunk`` before anything reaches the store.
    """

    entities: list[ExtractedEntity] = Field(default_factory=list)
    relations: list[ExtractedRelation] = Field(default_factory=list)
    claims: list[ExtractedClaim] = Field(default_factory=list)
    # Stamped from the chunk, never from the LLM.
    chunk_id: str = ""
    video_id: str = ""
    source_url: str = ""
    video_title: str | None = None
    channel_id: str | None = None
    upload_date: str | None = None
    start_seconds: float | None = None
    end_seconds: float | None = None
    error: str | None = None


class GraphClaim(BaseModel):
    """A claim as read back from the store, ready to cite."""

    id: str
    text: str
    entities: list[str] = Field(default_factory=list)
    chunk_id: str = ""
    video_id: str = ""
    source_url: str = ""
    video_title: str | None = None
    upload_date: str | None = None
    start_seconds: float | None = None
    end_seconds: float | None = None
    polarity: str = "asserts"

    @property
    def chunk_index(self) -> int:
        match = re.match(r"^chunk:[^:]+:(\d+)$", self.chunk_id)
        return int(match.group(1)) if match else 0


class GraphEntity(BaseModel):
    id: str
    name: str
    type: str = "concept"
    aliases: list[str] = Field(default_factory=list)
    mentions: int = 0
    community_id: int | None = None


class GraphCommunity(BaseModel):
    id: int
    entity_ids: list[str] = Field(default_factory=list)
    entity_names: list[str] = Field(default_factory=list)
    summary: str | None = None
    claim_count: int = 0

"""Extraction-quality check for the GraphRAG graph (s05 P4.1).

GraphRAG has a failure mode vector RAG doesn't: bad entity/claim extraction
silently corrupts every graph answer downstream. This module scores extraction
against a small hand-labelled sample (``graph_extraction_sample.json``) the
same way ``context_recall`` makes a retrieval regression visible.

The labels are deliberately *recall-shaped*: each labelled chunk lists the
entities a human reading the chunk says extraction **must** find, and keyword
sets that some extracted claim must contain. Extraction finding *more* than
the labels is expected (the labels are not exhaustive), so no precision number
is reported — a false precision score from non-exhaustive labels would be
worse than none.

Matching is normalization-tolerant: "budget" matches "federal budget" via slug
containment in either direction, and aliases count.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from src.rag.graph_models import ChunkExtraction, entity_id_for

DEFAULT_SAMPLE_PATH = Path(__file__).with_name("graph_extraction_sample.json")


class LabelledChunk(BaseModel):
    """What extraction must find in one chunk, per a human reading of it."""

    chunk_id: str
    entities: list[str] = Field(default_factory=list)
    #: Each inner list is one expected claim, expressed as keywords that must
    #: all appear (case-insensitive) in a single extracted claim's text.
    claim_keywords: list[list[str]] = Field(default_factory=list)
    notes: str = ""


def load_labelled(path: str | Path | None = None) -> list[LabelledChunk]:
    sample_path = Path(path) if path is not None else DEFAULT_SAMPLE_PATH
    raw = json.loads(sample_path.read_text(encoding="utf-8"))
    return [LabelledChunk.model_validate(record) for record in raw["labelled"]]


def _slug_matches(label: str, candidates: set[str]) -> bool:
    slug = entity_id_for(label)
    return any(
        slug == candidate or slug in candidate or candidate in slug for candidate in candidates
    )


def score_extraction(extraction: ChunkExtraction, label: LabelledChunk) -> dict[str, Any]:
    """Recall of labelled entities and claims for one chunk."""
    extracted_slugs = {entity.entity_id for entity in extraction.entities}
    for entity in extraction.entities:
        extracted_slugs.update(entity_id_for(alias) for alias in entity.aliases)

    entity_hits = [name for name in label.entities if _slug_matches(name, extracted_slugs)]
    claim_texts = [claim.text.lower() for claim in extraction.claims]
    claim_hits = [
        keywords
        for keywords in label.claim_keywords
        if any(all(word.lower() in text for word in keywords) for text in claim_texts)
    ]
    return {
        "chunk_id": label.chunk_id,
        "error": extraction.error,
        "entity_recall": (
            round(len(entity_hits) / len(label.entities), 4) if label.entities else None
        ),
        "claim_recall": (
            round(len(claim_hits) / len(label.claim_keywords), 4) if label.claim_keywords else None
        ),
        "missed_entities": [name for name in label.entities if name not in entity_hits],
        "missed_claims": [
            keywords for keywords in label.claim_keywords if keywords not in claim_hits
        ],
    }


def score_all(
    extractions: dict[str, ChunkExtraction], labelled: list[LabelledChunk]
) -> dict[str, Any]:
    """Score every labelled chunk; missing extractions score as failures."""
    results = []
    for label in labelled:
        extraction = extractions.get(label.chunk_id)
        if extraction is None:
            extraction = ChunkExtraction(error="chunk not extracted")
        results.append(score_extraction(extraction, label))

    def mean_of(key: str) -> float | None:
        values = [r[key] for r in results if isinstance(r[key], float | int)]
        return round(sum(values) / len(values), 4) if values else None

    return {
        "kind": "graph-extraction-eval",
        "labelled_chunks": len(labelled),
        "entity_recall": mean_of("entity_recall"),
        "claim_recall": mean_of("claim_recall"),
        "results": results,
    }

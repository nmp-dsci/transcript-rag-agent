"""Extraction-quality scoring: recall matching, misses, and failure handling."""

from __future__ import annotations

from src.evals.graph_extraction import (
    LabelledChunk,
    load_labelled,
    score_all,
    score_extraction,
)
from src.rag.graph_models import ChunkExtraction, ExtractedClaim, ExtractedEntity


def extraction_with(entities: list[str], claims: list[str]) -> ChunkExtraction:
    return ChunkExtraction(
        entities=[ExtractedEntity(name=name) for name in entities],
        claims=[ExtractedClaim(text=text) for text in claims],
        chunk_id="chunk:v:1",
    )


def test_entity_matching_tolerates_normalization_and_containment() -> None:
    label = LabelledChunk(
        chunk_id="chunk:v:1",
        entities=["budget", "Pauline Hanson", "rate cuts"],
    )
    extraction = extraction_with(["Federal Budget", "pauline hanson", "housing supply"], [])
    result = score_extraction(extraction, label)
    # "budget" ⊂ "federal-budget"; "rate cuts" missing.
    assert result["entity_recall"] == round(2 / 3, 4)
    assert result["missed_entities"] == ["rate cuts"]


def test_aliases_count_as_matches() -> None:
    label = LabelledChunk(chunk_id="chunk:v:1", entities=["gearing"])
    extraction = ChunkExtraction(
        entities=[ExtractedEntity(name="negative gearing policy", aliases=["gearing"])]
    )
    assert score_extraction(extraction, label)["entity_recall"] == 1.0


def test_claim_keywords_must_all_land_in_one_claim() -> None:
    label = LabelledChunk(
        chunk_id="chunk:v:1",
        claim_keywords=[["senate", "inquiry"], ["perth", "100%"]],
    )
    extraction = extraction_with(
        [], ["The bill sits with a Senate inquiry committee.", "Perth grew strongly."]
    )
    result = score_extraction(extraction, label)
    assert result["claim_recall"] == 0.5
    assert result["missed_claims"] == [["perth", "100%"]]


def test_score_all_treats_missing_extraction_as_failure() -> None:
    labelled = [
        LabelledChunk(chunk_id="chunk:v:1", entities=["budget"]),
        LabelledChunk(chunk_id="chunk:v:2", entities=["perth"]),
    ]
    report = score_all(
        {"chunk:v:1": extraction_with(["budget"], [])},
        labelled,
    )
    assert report["entity_recall"] == 0.5
    assert report["results"][1]["error"] == "chunk not extracted"


def test_shipped_sample_is_well_formed() -> None:
    labelled = load_labelled()
    assert len(labelled) >= 3
    for label in labelled:
        assert label.chunk_id.startswith("chunk:")
        assert label.entities, label.chunk_id

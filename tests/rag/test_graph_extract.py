"""Extraction contract, provenance stamping, retry, and the chunk-hash cache."""

from __future__ import annotations

import json

import pytest

from src.rag.graph_extract import (
    GraphExtractor,
    extraction_cache_key,
    parse_extraction,
    stamp_provenance,
)
from src.rag.graph_models import claim_id_for, entity_id_for
from src.rag.models import TranscriptChunk


VALID_RESPONSE = json.dumps(
    {
        "entities": [{"name": "Negative Gearing", "type": "policy", "aliases": ["gearing"]}],
        "relations": [
            {
                "source": "negative gearing",
                "target": "budget",
                "type": "changed_by",
                "weight": 0.8,
            }
        ],
        "claims": [
            {
                "text": "Negative gearing is capped from July 2027.",
                "entities": ["negative gearing"],
                "polarity": "asserts",
            }
        ],
    }
)


def make_chunk(text: str = "some spoken text", index: int = 3) -> TranscriptChunk:
    return TranscriptChunk(
        transcript_id="transcript:vid1",
        video_id="vid1",
        source_url="https://www.youtube.com/watch?v=vid1",
        chunk_index=index,
        text=text,
        start_seconds=10.0,
        end_seconds=60.0,
        title="Test video",
        channel_id="chan1",
        upload_date="2026-06-10",
    )


class FakeLLM:
    """Returns queued responses; records how many calls were made."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls = 0

    def invoke(self, messages: list) -> object:
        self.calls += 1
        return type("R", (), {"content": self.responses.pop(0)})()


def test_entity_and_claim_ids_are_canonical() -> None:
    assert entity_id_for("Negative Gearing") == "negative-gearing"
    assert entity_id_for("  50% CGT discount! ") == "50-cgt-discount"
    assert entity_id_for("") == "unknown"
    first = claim_id_for("chunk:v:1", "Rates will fall.")
    assert first == claim_id_for("chunk:v:1", "Rates will fall.")
    assert first != claim_id_for("chunk:v:2", "Rates will fall.")


def test_parse_extraction_accepts_plain_and_fenced_json() -> None:
    plain = parse_extraction(VALID_RESPONSE)
    fenced = parse_extraction(f"```json\n{VALID_RESPONSE}\n```")
    assert plain.entities[0].name == "Negative Gearing"
    assert fenced.claims[0].polarity == "asserts"


@pytest.mark.parametrize("bad", ["not json", "[1, 2]", '{"claims": [{"text": ""}]}'])
def test_parse_extraction_rejects_contract_violations(bad: str) -> None:
    with pytest.raises(ValueError):
        parse_extraction(bad)


def test_stamp_provenance_overwrites_llm_values() -> None:
    extraction = parse_extraction(VALID_RESPONSE).model_copy(
        update={"video_id": "spoofed", "upload_date": "1999-01-01"}
    )
    stamped = stamp_provenance(extraction, make_chunk())
    assert stamped.chunk_id == "chunk:vid1:3"
    assert stamped.video_id == "vid1"
    assert stamped.upload_date == "2026-06-10"
    assert stamped.start_seconds == 10.0


def test_extractor_caches_by_chunk_content(tmp_path) -> None:
    llm = FakeLLM([VALID_RESPONSE])
    extractor = GraphExtractor(llm, cache_dir=tmp_path)
    chunk = make_chunk()

    first = extractor.extract(chunk)
    second = extractor.extract(chunk)
    assert llm.calls == 1  # second call served from cache
    assert first.claims[0].text == second.claims[0].text

    # Changing the text changes the cache key: the chunk is re-extracted.
    changed = make_chunk(text="different words entirely")
    assert extraction_cache_key(chunk) != extraction_cache_key(changed)


def test_extractor_retries_once_then_records_failure(tmp_path) -> None:
    llm = FakeLLM(["not json", "still not json"])
    extractor = GraphExtractor(llm, cache_dir=tmp_path)

    result = extractor.extract(make_chunk())
    assert llm.calls == 2
    assert result.error is not None
    assert result.entities == [] and result.claims == []
    # Failures are not pinned: the next run tries the LLM again.
    llm_retry = FakeLLM([VALID_RESPONSE])
    retried = GraphExtractor(llm_retry, cache_dir=tmp_path).extract(make_chunk())
    assert retried.error is None
    assert llm_retry.calls == 1


def test_extractor_recovers_on_retry(tmp_path) -> None:
    llm = FakeLLM(["not json", VALID_RESPONSE])
    extractor = GraphExtractor(llm, cache_dir=tmp_path)
    result = extractor.extract(make_chunk())
    assert result.error is None
    assert result.entities[0].entity_id == "negative-gearing"

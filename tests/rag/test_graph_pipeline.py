"""build_graph: the extraction + community-rebuild pipeline shared by the
`index-graph` CLI command and the ingestion queue's automatic post-ingest
hook. No real Neo4j or LLM involved — a fake store and a fake chat model
stand in for both."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.config import Settings
from src.rag.graph_models import GraphCommunity
from src.rag.graph_pipeline import build_graph
from src.rag.models import TranscriptChunk

VALID_RESPONSE = json.dumps(
    {
        "entities": [{"name": "Negative Gearing", "type": "policy", "aliases": []}],
        "relations": [],
        "claims": [
            {
                "text": "Negative gearing is capped from July 2027.",
                "entities": ["negative gearing"],
                "polarity": "asserts",
            }
        ],
    }
)


def make_chunk(
    video_id: str = "vid1", index: int = 0, text: str = "some spoken text"
) -> TranscriptChunk:
    return TranscriptChunk(
        transcript_id=f"transcript:{video_id}",
        video_id=video_id,
        source_url=f"https://www.youtube.com/watch?v={video_id}",
        chunk_index=index,
        text=text,
        start_seconds=10.0,
        end_seconds=60.0,
        title="Test video",
        channel_id="chan1",
        upload_date="2026-06-10",
    )


def make_settings(tmp_path: Path) -> Settings:
    return Settings(
        superdata_api_key="",
        deepseek_api_key="test-key",
        deepseek_model="test-model",
        deepseek_base_url=None,
        chroma_path=tmp_path / "chroma",
        mlflow_tracking_uri="",
        mlflow_experiment_name="test",
        log_transcript_artifacts=False,
        graph_cache_dir=tmp_path / "graph_cache",
    )


class FakeLLM:
    """Returns queued responses; records how many calls were made."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls = 0

    def invoke(self, messages: list) -> object:
        self.calls += 1
        return type("R", (), {"content": self.responses.pop(0)})()


class FakeGraphStore:
    """Enough of GraphStore's surface for build_graph, no Neo4j required."""

    def __init__(self, entity_edges: tuple = ((), ())) -> None:
        self.schema_ensured = False
        self.wiped = False
        self.upserted: list = []
        self.closed = False
        self._entity_edges = entity_edges
        self.stored_assignments: dict | None = None
        self.summaries: dict[int, str] = {}

    def ensure_schema(self) -> None:
        self.schema_ensured = True

    def wipe(self) -> None:
        self.wiped = True

    def upsert_extraction(self, extraction) -> None:
        self.upserted.append(extraction)

    def counts(self) -> dict[str, int]:
        return {
            "entities": len(self._entity_edges[0]),
            "relations": len(self._entity_edges[1]),
            "claims": sum(len(x.claims) for x in self.upserted),
            "communities": 0,
        }

    def entity_edges(self):
        return self._entity_edges

    def store_communities(self, assignments: dict) -> None:
        self.stored_assignments = assignments

    def communities(self) -> list[GraphCommunity]:
        return []

    def top_claims_for_community(self, community_id: int, limit: int = 12) -> list:
        return []

    def set_community_summary(self, community_id: int, summary: str) -> None:
        self.summaries[community_id] = summary

    def close(self) -> None:
        self.closed = True


@pytest.fixture(autouse=True)
def fake_llm(monkeypatch):
    """No test in this file should construct a real ChatOpenAI client."""
    instance = FakeLLM([VALID_RESPONSE])
    monkeypatch.setattr("src.rag.graph_pipeline.build_llm", lambda settings: instance)
    return instance


def test_build_graph_returns_zeroed_stats_for_no_chunks(tmp_path) -> None:
    stats = build_graph(make_settings(tmp_path), [], store=FakeGraphStore())
    assert stats == {
        "extracted": 0,
        "failed": 0,
        "failed_chunk_ids": [],
        "failed_details": [],
        "communities_summarized": 0,
        "counts": {},
    }


def test_build_graph_extracts_and_upserts_chunks(tmp_path) -> None:
    store = FakeGraphStore()
    stats = build_graph(make_settings(tmp_path), [make_chunk()], store=store, skip_communities=True)
    assert stats["extracted"] == 1
    assert stats["failed"] == 0
    assert len(store.upserted) == 1
    assert store.schema_ensured
    # skip_communities means the (expensive, whole-graph) rebuild never runs.
    assert store.stored_assignments is None
    assert stats["communities_summarized"] == 0


def test_build_graph_reuses_the_chunk_hash_cache(tmp_path) -> None:
    settings = make_settings(tmp_path)
    chunk = make_chunk()
    store_a = FakeGraphStore()
    build_graph(settings, [chunk], store=store_a, skip_communities=True)

    store_b = FakeGraphStore()
    build_graph(settings, [chunk], store=store_b, skip_communities=True)

    # Same fake LLM instance across both calls (patched once by the fixture);
    # a second extraction of the identical chunk must be served from the
    # on-disk cache, not a second LLM call.
    assert len(store_b.upserted) == 1


def test_build_graph_rebuilds_communities_unless_skipped(tmp_path) -> None:
    store = FakeGraphStore(entity_edges=(["negative-gearing"], []))
    stats = build_graph(make_settings(tmp_path), [make_chunk()], store=store)
    assert store.stored_assignments == {"negative-gearing": 0}
    assert stats["counts"]["entities"] == 1


def test_build_graph_wipes_the_store_on_refresh(tmp_path) -> None:
    store = FakeGraphStore()
    build_graph(
        make_settings(tmp_path), [make_chunk()], store=store, refresh=True, skip_communities=True
    )
    assert store.wiped


def test_build_graph_records_failed_extractions_without_upserting(tmp_path, monkeypatch) -> None:
    failing_llm = FakeLLM(["not json", "still not json"])
    monkeypatch.setattr("src.rag.graph_pipeline.build_llm", lambda settings: failing_llm)
    store = FakeGraphStore()

    stats = build_graph(make_settings(tmp_path), [make_chunk()], store=store, skip_communities=True)
    assert stats["extracted"] == 0
    assert stats["failed"] == 1
    assert stats["failed_chunk_ids"] == ["chunk:vid1:0"]
    assert stats["failed_details"][0][0] == "chunk:vid1:0"
    assert store.upserted == []


def test_build_graph_closes_a_store_it_opened_itself(tmp_path, monkeypatch) -> None:
    opened = FakeGraphStore()
    monkeypatch.setattr("src.rag.graph_pipeline.GraphStore.from_settings", lambda settings: opened)
    build_graph(make_settings(tmp_path), [make_chunk()], skip_communities=True)
    assert opened.closed


def test_build_graph_does_not_close_a_store_passed_in(tmp_path) -> None:
    store = FakeGraphStore()
    build_graph(make_settings(tmp_path), [make_chunk()], store=store, skip_communities=True)
    assert not store.closed

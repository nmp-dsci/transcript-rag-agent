"""The Cypher GraphStore builds, against a recording stand-in for the driver.

Neo4j itself is not exercised here — what matters is that the scoping a caller
asks for reaches the *query*, since a filter applied to the returned rows is
silently defeated by the query's own LIMIT.
"""

from __future__ import annotations

from typing import Any

from src.rag.graph_models import GraphClaim
from src.rag.graph_store import GraphStore


class RecordingStore(GraphStore):
    """A store whose ``_run`` records the query instead of connecting."""

    def __init__(self, rows: list[dict[str, Any]] | None = None) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.rows = rows or []
        self.uri = "bolt://recording"

    def _run(self, query: str, **params: Any) -> list[dict[str, Any]]:
        self.calls.append((query, params))
        return self.rows


def claim_row(claim_id: str, video_id: str) -> dict[str, Any]:
    return {
        "id": claim_id,
        "text": f"claim {claim_id}",
        "entities": ["negative gearing"],
        "chunk_id": f"chunk:{video_id}:0",
        "video_id": video_id,
        "source_url": f"https://youtu.be/{video_id}",
        "video_title": None,
        "upload_date": "2026-06-10",
        "start_seconds": None,
        "end_seconds": None,
        "polarity": "neutral",
    }


def test_claims_about_scopes_to_a_video_inside_the_query_at_zero_hops() -> None:
    store = RecordingStore()
    store.claims_about(["a"], limit=200, hops=0, video_id="v1")

    query, params = store.calls[0]
    assert "c.video_id = $video_id" in query
    assert params["video_id"] == "v1"
    # Before the LIMIT, not after it: otherwise other videos' claims spend the
    # quota and the scoped call comes back empty.
    assert query.index("c.video_id = $video_id") < query.index("LIMIT $limit")


def test_claims_about_scopes_to_a_video_on_the_neighbour_hop_too() -> None:
    store = RecordingStore()
    store.claims_about(["a"], limit=40, hops=1, video_id="v1")

    query, params = store.calls[0]
    assert "c.video_id = $video_id" in query
    assert params["video_id"] == "v1"
    # The hop anchor is a disjunction; the video scope must apply to the whole
    # of it, not just to the RELATES branch.
    assert "WHERE (e.id IN $ids OR (e)-[:RELATES]-(seed)) AND c.video_id = $video_id" in query


def test_claims_about_without_a_video_stays_corpus_wide() -> None:
    store = RecordingStore(rows=[claim_row("c1", "v1"), claim_row("c2", "v2")])

    claims = store.claims_about(["a"], limit=40, hops=1)

    query, params = store.calls[0]
    assert "video_id" not in params
    assert "c.video_id = $video_id" not in query
    assert [claim.video_id for claim in claims] == ["v1", "v2"]
    assert all(isinstance(claim, GraphClaim) for claim in claims)


def test_claims_about_returns_empty_without_entity_ids() -> None:
    store = RecordingStore()
    assert store.claims_about([], video_id="v1") == []
    assert store.calls == []

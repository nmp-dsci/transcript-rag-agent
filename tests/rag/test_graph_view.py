"""Knowledge-graph layout and the entity-detail assembly, against a fake store."""

from __future__ import annotations

from src.rag.graph_models import GraphClaim, GraphCommunity, GraphEntity
from src.rag.graph_view import (
    _fr_layout_coordinates,
    _normalise,
    build_knowledge_graph,
    chunk_enrichment_for_video,
    entity_claims,
    rank_chunks_by_graph,
)


class FakeStore:
    def __init__(self) -> None:
        self.entities = [
            GraphEntity(id="a", name="Negative gearing", mentions=10, community_id=0),
            GraphEntity(id="b", name="Budget 2026", mentions=5, community_id=0),
            GraphEntity(id="c", name="Agentic coding", mentions=3, community_id=1),
        ]
        self.edges = [("a", "b", 0.8)]
        self.claims = {
            "a": [
                GraphClaim(id="c1", text="Gearing is grandfathered.", upload_date="2026-06-10"),
                GraphClaim(id="c2", text="New rules apply from 2027.", upload_date="2026-06-15"),
            ]
        }

    def all_entities(self, limit: int = 2000):
        return self.entities

    def entity_edges(self):
        return [e.id for e in self.entities], self.edges

    def communities(self):
        return [
            GraphCommunity(
                id=0,
                entity_ids=["a", "b"],
                entity_names=["Negative gearing", "Budget 2026"],
                summary="Property tax changes.",
                claim_count=2,
            ),
            GraphCommunity(id=1, entity_ids=["c"], entity_names=["Agentic coding"], claim_count=0),
        ]

    def get_entity(self, entity_id: str):
        return next((e for e in self.entities if e.id == entity_id), None)

    def claims_about(self, entity_ids, limit=40, hops=1):
        return self.claims.get(entity_ids[0], []) if entity_ids else []


def test_normalise_centres_and_scales_into_unit_square() -> None:
    result = _normalise([(0.0, 0.0), (10.0, 4.0), (-2.0, -6.0)])
    xs = [x for x, _ in result]
    ys = [y for _, y in result]
    assert max(xs) <= 1.0 and min(xs) >= -1.0
    assert max(ys) <= 1.0 and min(ys) >= -1.0
    # The wider axis (x: span 12) reaches the edges; y (span 10) does not.
    assert max(xs) == 1.0


def test_normalise_handles_identical_points() -> None:
    assert _normalise([(3.0, 3.0), (3.0, 3.0)]) == [(0.0, 0.0), (0.0, 0.0)]


def test_fr_layout_handles_empty_and_singleton() -> None:
    assert _fr_layout_coordinates([], []) == {}
    assert _fr_layout_coordinates(["only"], []) == {"only": (0.0, 0.0)}


def test_fr_layout_places_every_node() -> None:
    positions = _fr_layout_coordinates(["a", "b", "c"], [("a", "b", 1.0)])
    assert set(positions) == {"a", "b", "c"}
    for x, y in positions.values():
        assert -1.0 <= x <= 1.0
        assert -1.0 <= y <= 1.0


def test_build_knowledge_graph_assembles_nodes_edges_and_communities() -> None:
    result = build_knowledge_graph(FakeStore())
    node_ids = {node["id"] for node in result["nodes"]}
    assert node_ids == {"a", "b", "c"}
    assert all(-1.0 <= node["x"] <= 1.0 and -1.0 <= node["y"] <= 1.0 for node in result["nodes"])
    assert result["edges"] == [{"source": "a", "target": "b", "weight": 0.8}]
    assert len(result["communities"]) == 2
    assert result["communities"][0]["summary"] == "Property tax changes."


def test_entity_claims_returns_metadata_and_dated_timeline() -> None:
    result = entity_claims(FakeStore(), "a")
    assert result["entity"]["name"] == "Negative gearing"
    assert [c["id"] for c in result["claims"]] == ["c1", "c2"]


def test_entity_claims_reports_none_for_unknown_entity() -> None:
    result = entity_claims(FakeStore(), "nonexistent")
    assert result["entity"] is None
    assert result["claims"] == []


class VideoFakeStore:
    """A fake with real chunk_id/video_id/entities on its claims, for the
    chunk-enrichment and graph-ranking functions (which read those fields —
    the minimal ``FakeStore`` above leaves them at their pydantic defaults)."""

    def __init__(self) -> None:
        self.claims = [
            GraphClaim(
                id="c1",
                text="Negative gearing is grandfathered for pre-budget properties.",
                entities=["negative gearing", "budget"],
                chunk_id="chunk:v1:2",
                video_id="v1",
                upload_date="2026-06-10",
            ),
            GraphClaim(
                id="c2",
                text="The budget takes effect from July 2027.",
                entities=["budget"],
                chunk_id="chunk:v1:2",
                video_id="v1",
                upload_date="2026-06-10",
            ),
            GraphClaim(
                id="c3",
                text="Agentic coding uses memory files.",
                entities=["agentic coding"],
                chunk_id="chunk:v1:5",
                video_id="v1",
                upload_date="2026-06-20",
            ),
            GraphClaim(
                id="c4",
                text="A different video's claim.",
                entities=["negative gearing"],
                chunk_id="chunk:v2:0",
                video_id="v2",
                upload_date="2026-06-11",
            ),
        ]

    def claims_for_video(self, video_id: str, limit: int = 1000):
        return [claim for claim in self.claims if claim.video_id == video_id]

    def resolve_entities(self, terms, limit=6):
        wanted = {term.lower() for term in terms}
        catalogue = [
            GraphEntity(id="negative-gearing", name="negative gearing", mentions=10),
            GraphEntity(id="budget", name="budget", mentions=8),
            GraphEntity(id="agentic-coding", name="agentic coding", mentions=4),
        ]
        return [entity for entity in catalogue if any(term in entity.name for term in wanted)]

    def claims_about(self, entity_ids, limit=200, hops=0):
        names = {
            "negative-gearing": "negative gearing",
            "budget": "budget",
            "agentic-coding": "agentic coding",
        }
        wanted = {names[eid] for eid in entity_ids if eid in names}
        return [claim for claim in self.claims if wanted & set(claim.entities)]


def test_chunk_enrichment_groups_claims_by_chunk_index() -> None:
    result = chunk_enrichment_for_video(VideoFakeStore(), "v1")
    chunks = result["chunks"]
    assert set(chunks) == {"2", "5"}
    assert len(chunks["2"]["claims"]) == 2
    assert chunks["2"]["entities"] == ["negative gearing", "budget"]
    assert chunks["5"]["entities"] == ["agentic coding"]


def test_chunk_enrichment_excludes_other_videos() -> None:
    result = chunk_enrichment_for_video(VideoFakeStore(), "v2")
    assert set(result["chunks"]) == {"0"}


def test_chunk_enrichment_empty_for_unindexed_video() -> None:
    result = chunk_enrichment_for_video(VideoFakeStore(), "no-such-video")
    assert result["chunks"] == {}


def test_rank_chunks_by_graph_ranks_multi_entity_chunk_first() -> None:
    records = [
        {"video_id": "v1", "chunk_index": 2, "text": "raw chunk text", "start_seconds": 30.0},
        {"video_id": "v1", "chunk_index": 5, "text": "other raw text", "start_seconds": 90.0},
    ]
    rows = rank_chunks_by_graph(VideoFakeStore(), "negative gearing budget", records, top_k=10)
    assert rows[0]["chunk_index"] == 2  # covers both matched entities
    assert rows[0]["text"] == "raw chunk text"  # real chunk text, not the claim
    assert set(rows[0]["matched_entities"]) == {"negative gearing", "budget"}
    assert rows[0]["score"] == 1.0  # both resolved entities covered


def test_rank_chunks_by_graph_scopes_to_a_video_before_truncating() -> None:
    """Scoping must narrow the ranking, not slice it after the fact.

    claims_about is corpus-wide, so filtering after the top_k cut would rank
    globally, keep the best k, and then discard everything outside the chosen
    video — leaving the column empty whenever the winners came from elsewhere.
    """
    records = [
        {"video_id": "v2", "chunk_index": 0, "text": "other video text"},
    ]
    rows = rank_chunks_by_graph(
        VideoFakeStore(), "negative gearing", records, top_k=1, video_id="v2"
    )
    # v1 chunk 2 covers "negative gearing" too and would win a corpus-wide
    # top-1; scoping to v2 must still return v2's matching chunk.
    assert [row["video_id"] for row in rows] == ["v2"]
    assert rows[0]["chunk_index"] == 0


def test_rank_chunks_by_graph_without_a_video_ranks_the_whole_corpus() -> None:
    records = [
        {"video_id": "v1", "chunk_index": 2, "text": "v1 text"},
        {"video_id": "v2", "chunk_index": 0, "text": "v2 text"},
    ]
    rows = rank_chunks_by_graph(VideoFakeStore(), "negative gearing", records, top_k=10)
    assert {row["video_id"] for row in rows} == {"v1", "v2"}


def test_rank_chunks_by_graph_returns_empty_when_no_entities_resolve() -> None:
    records = [{"video_id": "v1", "chunk_index": 2, "text": "x"}]
    assert rank_chunks_by_graph(VideoFakeStore(), "zzz nonexistent", records, top_k=10) == []


def test_rank_chunks_by_graph_falls_back_to_claim_text_without_a_record() -> None:
    rows = rank_chunks_by_graph(VideoFakeStore(), "agentic coding", [], top_k=10)
    assert rows[0]["chunk_index"] == 5
    assert "Agentic coding uses memory files" in rows[0]["text"]

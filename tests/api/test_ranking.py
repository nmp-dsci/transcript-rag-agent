from __future__ import annotations

from src.api.ranking import build_rankings
from src.rag import bm25


def record(index: int, text: str, video_id: str = "vid1", score: float | None = None):
    item = {"video_id": video_id, "chunk_index": index, "text": text}
    if score is not None:
        item["score"] = score
    return item


CORPUS = [
    record(0, "capital gains tax discount for property investors"),
    record(1, "negative gearing rules and rental deductions"),
    record(2, "capital gains tax is being grandfathered"),
]


def rankings(semantic_hits, modes=("semantic", "bm25"), query="capital gains tax"):
    bm25.clear_cache()
    return build_rankings(
        query,
        modes=list(modes),
        top_k=5,
        semantic_fn=lambda q, k: semantic_hits[:k],
        records_fn=lambda: CORPUS,
    )


def test_aligns_ranks_across_modes() -> None:
    # Semantic prefers chunk 2; BM25 (term frequency) will order differently.
    result = rankings([record(2, CORPUS[2]["text"], score=0.9), record(0, CORPUS[0]["text"], score=0.5)])

    semantic = result["modes"]["semantic"]
    keyword = result["modes"]["bm25"]
    assert semantic[0]["chunk_id"] == "vid1:2"
    # Each row knows where the other mode placed the same chunk.
    by_id = {row["chunk_id"]: row for row in keyword}
    assert semantic[0]["other_rank"] == by_id["vid1:2"]["rank"]


def test_marks_chunks_only_one_mode_found() -> None:
    # A semantic-only hit that shares no query term with the BM25 query.
    result = rankings([record(1, CORPUS[1]["text"], score=0.4)])
    semantic = result["modes"]["semantic"]
    assert semantic[0]["chunk_id"] == "vid1:1"
    assert semantic[0]["other_rank"] is None


def test_reports_overlap_between_modes() -> None:
    result = rankings([record(0, CORPUS[0]["text"], score=0.9)])
    assert result["overlap"]["count"] == 1
    assert "vid1:0" in result["overlap"]["chunk_ids"]


def test_single_mode_has_no_alignment_or_overlap() -> None:
    result = rankings([], modes=("bm25",))
    assert set(result["modes"]) == {"bm25"}
    assert all(row["other_rank"] is None for row in result["modes"]["bm25"])
    assert result["overlap"]["count"] == 0


def test_truncates_long_previews() -> None:
    long_text = "capital " * 200
    bm25.clear_cache()
    result = build_rankings(
        "capital",
        modes=["semantic"],
        top_k=1,
        semantic_fn=lambda q, k: [record(0, long_text, score=1.0)],
        records_fn=lambda: [],
    )
    preview = result["modes"]["semantic"][0]["preview"]
    assert len(preview) <= 221 and preview.endswith("…")


def test_carries_scope_and_query_through() -> None:
    bm25.clear_cache()
    result = build_rankings(
        "anything",
        modes=["bm25"],
        top_k=3,
        semantic_fn=lambda q, k: [],
        records_fn=lambda: CORPUS,
        video_id="vid1",
    )
    assert result["query"] == "anything"
    assert result["video_id"] == "vid1"
    assert result["top_k"] == 3

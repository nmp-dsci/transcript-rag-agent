from __future__ import annotations

import pytest

from src.rag.fusion import (
    chunk_key,
    fuse_chunks,
    fuse_chunks_with_scores,
    key_of,
    reciprocal_rank_fusion,
)
from src.rag.models import RetrievedChunk


def chunk(index: int, video_id: str = "vid1", score: float | None = None) -> RetrievedChunk:
    return RetrievedChunk(
        transcript_id=f"t-{video_id}",
        video_id=video_id,
        source_url="https://www.youtube.com/watch?v=vid1",
        chunk_index=index,
        text=f"chunk {index}",
        score=score,
    )


def record(index: int, video_id: str = "vid1") -> dict:
    return {"video_id": video_id, "chunk_index": index, "text": f"chunk {index}"}


# --- keys -------------------------------------------------------------------


def test_chunk_key_joins_video_and_index():
    assert chunk_key("vid1", 3) == "vid1:3"


def test_key_of_matches_across_chunk_objects_and_bm25_dicts():
    """The whole fusion rests on these two shapes agreeing on identity."""
    assert key_of(chunk(3)) == key_of(record(3)) == "vid1:3"


# --- RRF math ---------------------------------------------------------------


def test_rrf_single_list_is_a_passthrough_ordering():
    fused = reciprocal_rank_fusion([["a", "b", "c"]], k=0)
    assert [key for key, _ in fused] == ["a", "b", "c"]
    assert [score for _, score in fused] == [1.0, 0.5, pytest.approx(1 / 3)]


def test_rrf_scores_match_hand_computation():
    # k=1, so ranks 1,2,3 contribute 1/2, 1/3, 1/4.
    #   a: 1/2 + 1/3 = 0.833333
    #   c: 1/4 + 1/2 = 0.75
    #   b: 1/3       = 0.333333
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["c", "a"]], k=1)
    assert [key for key, _ in fused] == ["a", "c", "b"]
    scores = dict(fused)
    assert scores["a"] == pytest.approx(1 / 2 + 1 / 3)
    assert scores["c"] == pytest.approx(1 / 4 + 1 / 2)
    assert scores["b"] == pytest.approx(1 / 3)


def test_rrf_default_k_damps_the_top_position():
    """With k=60 a rank-1 hit only narrowly beats a rank-2 hit."""
    fused = dict(reciprocal_rank_fusion([["a", "b"]]))
    assert fused["a"] == pytest.approx(1 / 61)
    assert fused["b"] == pytest.approx(1 / 62)


def test_rrf_promotes_a_document_both_lists_agree_on():
    """The point of fusion: rank 2 in both beats rank 1 in only one."""
    fused = reciprocal_rank_fusion([["a", "shared"], ["b", "shared"]], k=1)
    assert fused[0][0] == "shared"
    assert fused[0][1] == pytest.approx(1 / 3 + 1 / 3)


def test_rrf_missing_documents_contribute_nothing():
    fused = dict(reciprocal_rank_fusion([["a"], ["b"]], k=0))
    assert fused == {"a": pytest.approx(1.0), "b": pytest.approx(1.0)}


def test_rrf_weights_scale_each_list():
    # Same lists as the hand-computed case, with list 0 counted double.
    #   a: 2*(1/2) + 1*(1/3) = 1.333333
    #   c: 2*(1/4) + 1*(1/2) = 1.0
    #   b: 2*(1/3)           = 0.666667
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["c", "a"]], k=1, weights=[2.0, 1.0])
    assert [key for key, _ in fused] == ["a", "c", "b"]
    scores = dict(fused)
    assert scores["a"] == pytest.approx(2 / 2 + 1 / 3)
    assert scores["c"] == pytest.approx(2 / 4 + 1 / 2)
    assert scores["b"] == pytest.approx(2 / 3)


def test_rrf_zero_weight_disables_a_list_entirely():
    fused = dict(reciprocal_rank_fusion([["a"], ["b"]], k=0, weights=[1.0, 0.0]))
    assert fused["a"] == pytest.approx(1.0)
    assert fused["b"] == pytest.approx(0.0)


def test_rrf_weights_can_reorder_the_result():
    by_first = reciprocal_rank_fusion([["a", "b"], ["b", "a"]], k=1, weights=[3.0, 1.0])
    assert [key for key, _ in by_first] == ["a", "b"]
    by_second = reciprocal_rank_fusion([["a", "b"], ["b", "a"]], k=1, weights=[1.0, 3.0])
    assert [key for key, _ in by_second] == ["b", "a"]


# --- edges ------------------------------------------------------------------


def test_rrf_ties_break_by_first_appearance():
    """Symmetric lists score identically; order must still be deterministic."""
    fused = reciprocal_rank_fusion([["a", "b"], ["b", "a"]], k=1)
    assert [key for key, _ in fused] == ["a", "b"]
    assert fused[0][1] == pytest.approx(fused[1][1])


def test_rrf_handles_empty_input_and_empty_lists():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []
    assert [key for key, _ in reciprocal_rank_fusion([["a"], []], k=0)] == ["a"]


def test_rrf_counts_a_repeated_key_once_at_its_best_rank():
    fused = reciprocal_rank_fusion([["a", "a", "b"]], k=0)
    assert dict(fused)["a"] == pytest.approx(1.0)
    assert [key for key, _ in fused] == ["a", "b"]


def test_rrf_rejects_mismatched_weights():
    with pytest.raises(ValueError, match="weights"):
        reciprocal_rank_fusion([["a"], ["b"]], weights=[1.0])


def test_rrf_rejects_negative_k():
    with pytest.raises(ValueError, match="non-negative"):
        reciprocal_rank_fusion([["a"]], k=-1)


# --- fuse_chunks ------------------------------------------------------------


def test_fuse_chunks_returns_retrieved_chunks_in_fused_order():
    semantic = [chunk(0), chunk(1), chunk(2)]
    bm25 = [record(2), record(0)]
    fused = fuse_chunks(semantic, bm25, top_k=3, k=1)
    assert all(isinstance(item, RetrievedChunk) for item in fused)
    # chunk 0: 1/2 + 1/3 = 0.833 ; chunk 2: 1/4 + 1/2 = 0.75 ; chunk 1: 1/3
    assert [item.chunk_index for item in fused] == [0, 2, 1]


def test_fuse_chunks_lets_bm25_agreement_promote_a_weak_semantic_hit():
    semantic = [chunk(0), chunk(1), chunk(2)]
    bm25 = [record(2)]
    fused = fuse_chunks(semantic, bm25, top_k=3, k=1)
    assert fused[0].chunk_index == 2


def test_fuse_chunks_respects_top_k():
    fused = fuse_chunks([chunk(0), chunk(1), chunk(2)], [record(1)], top_k=2)
    assert len(fused) == 2


def test_fuse_chunks_preserves_the_original_retrieval_score():
    fused = fuse_chunks([chunk(0, score=0.42)], [record(0)], top_k=1)
    assert fused[0].score == 0.42


def test_fuse_chunks_drops_bm25_only_keys_by_default():
    """BM25 records lack transcript_id, so they cannot become RetrievedChunks."""
    fused = fuse_chunks([chunk(0)], [record(9), record(0)], top_k=5)
    assert [item.chunk_index for item in fused] == [0]


def test_dropping_bm25_only_keys_does_not_reorder_the_survivors():
    semantic = [chunk(0), chunk(1)]
    with_extra = fuse_chunks(semantic, [record(7), record(1), record(8)], top_k=5)
    without_extra = fuse_chunks(semantic, [record(1)], top_k=5)
    assert [c.chunk_index for c in with_extra] == [c.chunk_index for c in without_extra]


def test_fuse_chunks_resolver_recovers_bm25_only_chunks():
    recovered = chunk(9)
    # Weight BM25 above semantic so the recovered chunk must outrank chunk 0,
    # proving it was truly fused in rather than just appended.
    fused = fuse_chunks(
        [chunk(0)],
        [record(9)],
        top_k=5,
        k=1,
        weights=[1.0, 5.0],
        resolver=lambda key: recovered if key == "vid1:9" else None,
    )
    assert [item.chunk_index for item in fused] == [9, 0]
    assert fused[0] is recovered


def test_fuse_chunks_ignores_a_repeated_bm25_record():
    """One document listed twice must not score twice."""
    once = fuse_chunks_with_scores([chunk(0)], [record(0)], top_k=1)
    twice = fuse_chunks_with_scores([chunk(0)], [record(0), record(0)], top_k=1)
    assert once[0][1] == pytest.approx(twice[0][1])


def test_fuse_chunks_pool_recovers_bm25_only_chunks():
    fused = fuse_chunks([chunk(0)], [record(9)], top_k=5, pool=[chunk(9)])
    assert sorted(item.chunk_index for item in fused) == [0, 9]


def test_semantic_chunk_object_wins_over_the_pool_copy():
    """The semantic object carries the retrieval score; the pool copy may not."""
    fused = fuse_chunks([chunk(0, score=0.9)], [], top_k=1, pool=[chunk(0, score=None)])
    assert fused[0].score == 0.9


def test_fuse_chunks_separates_videos_with_the_same_chunk_index():
    semantic = [chunk(0, video_id="vidA"), chunk(0, video_id="vidB")]
    fused = fuse_chunks(semantic, [record(0, video_id="vidB")], top_k=5, k=1)
    assert [item.video_id for item in fused] == ["vidB", "vidA"]


def test_fuse_chunks_handles_empty_inputs():
    assert fuse_chunks([], [], top_k=5) == []
    assert fuse_chunks([], [record(0)], top_k=5) == []
    assert [c.chunk_index for c in fuse_chunks([chunk(0)], [], top_k=5)] == [0]


def test_fuse_chunks_returns_nothing_for_non_positive_top_k():
    assert fuse_chunks([chunk(0)], [record(0)], top_k=0) == []
    assert fuse_chunks([chunk(0)], [record(0)], top_k=-1) == []


def test_fuse_chunks_accepts_weights():
    semantic = [chunk(0), chunk(1)]
    bm25 = [record(1), record(0)]
    assert [c.chunk_index for c in fuse_chunks(semantic, bm25, 2, 1, [3.0, 1.0])] == [0, 1]
    assert [c.chunk_index for c in fuse_chunks(semantic, bm25, 2, 1, [1.0, 3.0])] == [1, 0]


def test_fuse_chunks_with_scores_exposes_the_fused_score():
    scored = fuse_chunks_with_scores([chunk(0), chunk(1)], [record(0)], top_k=2, k=1)
    chunks = [item[0] for item in scored]
    scores = [item[1] for item in scored]
    assert [c.chunk_index for c in chunks] == [0, 1]
    assert scores[0] == pytest.approx(1 / 2 + 1 / 2)
    assert scores[1] == pytest.approx(1 / 3)


def test_fuse_chunks_matches_the_chunks_from_fuse_chunks_with_scores():
    semantic, bm25 = [chunk(0), chunk(1)], [record(1)]
    scored = fuse_chunks_with_scores(semantic, bm25, top_k=2)
    assert fuse_chunks(semantic, bm25, top_k=2) == [item[0] for item in scored]

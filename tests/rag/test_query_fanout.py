"""The retrieval provider's behaviour when a query transform is configured."""

from __future__ import annotations

import pytest

from src.rag.context import MultiTranscriptRagContextProvider
from src.rag.models import RetrievedChunk
from src.rag.query_transform import QueryPlan


def _chunk(index: int, score: float | None = 0.5) -> RetrievedChunk:
    return RetrievedChunk(
        transcript_id="raw_transcript:video",
        video_id="video",
        source_url="https://www.youtube.com/watch?v=video",
        chunk_index=index,
        text=f"chunk {index}",
        segment_count=1,
        score=score,
    )


class RecordingChunkStore:
    """Answers ``query_all`` from a per-query script, recording what it saw."""

    def __init__(self, by_query: dict[str, list[RetrievedChunk]]) -> None:
        self.by_query = by_query
        self.queries: list[str] = []

    def has_any_chunks(self) -> bool:
        return True

    def query_all(self, query: str, top_k: int):
        self.queries.append(query)
        return self.by_query.get(query, [])[:top_k]


class FakeTransform:
    kind = "multi_query"
    model = "deepseek-v4"

    def __init__(self, plan: QueryPlan) -> None:
        self.plan = plan
        self.calls: list[str] = []

    def expand(self, question: str) -> QueryPlan:
        self.calls.append(question)
        return self.plan


def _provider(store, transform=None) -> MultiTranscriptRagContextProvider:
    return MultiTranscriptRagContextProvider(
        raw_store=None,
        chunk_store=store,
        query_transform=transform,
    )


def test_without_a_transform_retrieval_embeds_the_question_as_asked() -> None:
    store = RecordingChunkStore({"what changed?": [_chunk(0)]})

    context = _provider(store).get_context("what changed?", top_k=5)

    assert store.queries == ["what changed?"]
    assert [chunk.chunk_index for chunk in context.retrieved_chunks] == [0]
    # No transform step is invented for a retrieval that never ran one.
    assert [step.label for step in context.trace] == ["Retrieve candidates"]


def test_hyde_retrieves_on_the_passage_and_never_on_the_question() -> None:
    store = RecordingChunkStore({"a passage about the cap": [_chunk(3)]})
    transform = FakeTransform(QueryPlan(kind="hyde", queries=["a passage about the cap"]))

    context = _provider(store, transform).get_context("what changed?", top_k=5)

    assert store.queries == ["a passage about the cap"]
    assert [chunk.chunk_index for chunk in context.retrieved_chunks] == [3]


def test_a_single_query_expansion_does_not_claim_a_fusion_that_never_ran() -> None:
    store = RecordingChunkStore({"passage": [_chunk(1)]})
    transform = FakeTransform(QueryPlan(kind="hyde", queries=["passage"]))

    context = _provider(store, transform).get_context("q", top_k=5)

    labels = [step.label for step in context.trace]
    assert "Fuse query variants" not in labels
    assert labels == ["HyDE hypothetical passage", "Retrieve candidates"]


def test_multi_query_searches_every_variant_and_fuses_the_rankings() -> None:
    store = RecordingChunkStore(
        {
            "q": [_chunk(0), _chunk(1)],
            "rephrased": [_chunk(2), _chunk(0)],
        }
    )
    transform = FakeTransform(QueryPlan(kind="multi_query", queries=["q", "rephrased"]))

    context = _provider(store, transform).get_context("q", top_k=5)

    assert store.queries == ["q", "rephrased"]
    # chunk 0 is the only one both phrasings found, so RRF puts it first.
    assert [chunk.chunk_index for chunk in context.retrieved_chunks][0] == 0
    assert {chunk.chunk_index for chunk in context.retrieved_chunks} == {0, 1, 2}


def test_a_fused_chunk_keeps_the_object_the_first_query_retrieved() -> None:
    """Per-query scores come from different embeddings and are not comparable,
    so a chunk several phrasings found keeps the original query's own score."""
    store = RecordingChunkStore({"q": [_chunk(0, score=0.9)], "rephrased": [_chunk(0, score=0.1)]})
    transform = FakeTransform(QueryPlan(kind="multi_query", queries=["q", "rephrased"]))

    context = _provider(store, transform).get_context("q", top_k=5)

    assert context.retrieved_chunks[0].score == pytest.approx(0.9)


def test_the_trace_counts_the_candidate_pool_not_the_sum_of_the_rankings() -> None:
    store = RecordingChunkStore({"q": [_chunk(0), _chunk(1)], "rephrased": [_chunk(0)]})
    transform = FakeTransform(QueryPlan(kind="multi_query", queries=["q", "rephrased"]))

    context = _provider(store, transform).get_context("q", top_k=5)

    retrieve = next(step for step in context.trace if step.label == "Retrieve candidates")
    # Three results across two searches, but only two distinct chunks.
    assert "2 candidates" in retrieve.detail
    assert "× 2 queries" in retrieve.detail
    assert retrieve.chunk_ids == ["chunk:video:0", "chunk:video:1"]


def test_the_trace_records_what_the_expansion_actually_did() -> None:
    store = RecordingChunkStore({"q": [_chunk(0)], "rephrased": [_chunk(1)]})
    transform = FakeTransform(
        QueryPlan(kind="multi_query", queries=["q", "rephrased"], elapsed_ms=42)
    )

    context = _provider(store, transform).get_context("q", top_k=5)

    step = context.trace[0]
    assert step.phase == "llm"
    assert step.label == "Multi-query expansion"
    assert step.elapsed_ms == 42
    assert step.model == "deepseek-v4"
    assert "rephrased" in step.detail


def test_a_degraded_expansion_still_retrieves_and_says_it_degraded() -> None:
    store = RecordingChunkStore({"what changed?": [_chunk(0)]})
    transform = FakeTransform(QueryPlan(kind="hyde", queries=["what changed?"], degraded=True))

    context = _provider(store, transform).get_context("what changed?", top_k=5)

    assert store.queries == ["what changed?"]
    assert [chunk.chunk_index for chunk in context.retrieved_chunks] == [0]
    assert "failed" in context.trace[0].detail

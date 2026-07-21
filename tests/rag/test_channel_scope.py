"""Channel-scoped retrieval and neighbour expansion in the context provider."""

from __future__ import annotations

import pytest

from src.rag.context import MultiTranscriptRagContextProvider
from src.rag.models import RetrievedChunk, TranscriptChunk


def chunk(video_id: str, index: int, text: str = "text", score: float = 0.9):
    return RetrievedChunk(
        transcript_id=f"raw_transcript:{video_id}",
        video_id=video_id,
        source_url="https://youtu.be/" + video_id,
        chunk_index=index,
        text=text,
        segment_count=1,
        score=score,
    )


class FakeChunkStore:
    def __init__(self, by_channel=None, by_video=None, neighbors=None):
        self.by_channel = by_channel or {}
        self.by_video = by_video or {}
        self._neighbors = neighbors or {}
        self.channel_calls: list[tuple[str, str, int]] = []
        self.all_calls: list[tuple[str, int]] = []
        self.collection = self

    def has_any_chunks(self) -> bool:
        return True

    def query_by_channel(self, channel_id, query, top_k):
        self.channel_calls.append((channel_id, query, top_k))
        return self.by_channel.get(channel_id, [])[:top_k]

    def query_all(self, query, top_k):
        self.all_calls.append((query, top_k))
        return [c for chunks in self.by_channel.values() for c in chunks][:top_k]

    def neighbors(self, video_id, chunk_index, span):
        return self._neighbors.get((video_id, chunk_index), [])

    def get(self, **kwargs):
        return {"documents": [], "metadatas": []}


def provider(store, **kwargs):
    return MultiTranscriptRagContextProvider(
        raw_store=None, chunk_store=store, **kwargs
    )


def test_channel_scope_uses_the_native_channel_filter():
    store = FakeChunkStore(by_channel={"UC1": [chunk("v1", 0), chunk("v1", 1)]})
    context = provider(store).get_context("q", channel_id="UC1", top_k=2)
    assert store.channel_calls == [("UC1", "q", 2)]
    assert store.all_calls == []
    assert len(context.retrieved_chunks) == 2


def test_no_channel_falls_back_to_whole_corpus():
    store = FakeChunkStore(by_channel={"UC1": [chunk("v1", 0)]})
    provider(store).get_context("q", top_k=5)
    assert store.channel_calls == []
    assert store.all_calls == [("q", 5)]


def test_unknown_channel_reports_it_rather_than_answering_from_everything():
    store = FakeChunkStore(by_channel={"UC1": [chunk("v1", 0)]})
    with pytest.raises(ValueError, match="No indexed chunks found for channel"):
        provider(store).get_context("q", channel_id="UC_missing")


def test_neighbour_expansion_widens_hits_without_duplicating_them():
    neighbor = TranscriptChunk(
        transcript_id="raw_transcript:v1",
        video_id="v1",
        source_url="https://youtu.be/v1",
        chunk_index=1,
        text="neighbour",
        segment_count=1,
    )
    store = FakeChunkStore(
        by_channel={"UC1": [chunk("v1", 2, "hit")]},
        neighbors={("v1", 2): [neighbor]},
    )
    context = provider(store, neighbor_span=1).get_context(
        "q", channel_id="UC1", top_k=1
    )
    texts = [c.text for c in context.retrieved_chunks]
    assert texts == ["neighbour", "hit"]
    # Neighbours are context, not retrieval results, so they carry no score.
    assert context.retrieved_chunks[0].score is None


def test_neighbour_already_retrieved_is_not_added_twice():
    store = FakeChunkStore(
        by_channel={"UC1": [chunk("v1", 0, "a"), chunk("v1", 1, "b")]},
        neighbors={("v1", 0): [], ("v1", 1): []},
    )
    context = provider(store, neighbor_span=1).get_context(
        "q", channel_id="UC1", top_k=2
    )
    keys = [(c.video_id, c.chunk_index) for c in context.retrieved_chunks]
    assert len(keys) == len(set(keys))


def test_hybrid_mode_retrieves_wider_than_top_k_before_narrowing():
    store = FakeChunkStore(
        by_channel={"UC1": [chunk("v1", i) for i in range(30)]}
    )
    context = provider(store, retrieval_candidates=25).get_context(
        "q", channel_id="UC1", top_k=5, retrieval_mode="hybrid"
    )
    # Wide candidate pull, then narrowed to top_k for the answer.
    assert store.channel_calls[0][2] == 25
    assert len(context.retrieved_chunks) == 5

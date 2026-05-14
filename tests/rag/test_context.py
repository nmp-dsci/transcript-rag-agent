from __future__ import annotations

import pytest

from src.rag.context import MultiTranscriptRagContextProvider, RagTranscriptContextProvider
from src.rag.models import RawTranscriptDocument, RawTranscriptSegment, RetrievedChunk


class FakeRawStore:
    def ensure_raw_document(self, source_url: str, refresh: bool = False):
        return (
            RawTranscriptDocument(
                transcript_id="raw_transcript:video",
                video_id="video",
                source_url=source_url,
                fetched_at="2026-05-14T00:00:00+00:00",
                segments=[RawTranscriptSegment(text="full transcript")],
            ),
            "hit",
        )


class FakeChunkStore:
    def __init__(self) -> None:
        self.indexed = False
        self.query_text = None

    def has_chunks(self, video_id: str) -> bool:
        return self.indexed

    def query(self, video_id: str, query: str, top_k: int):
        self.query_text = query
        return [
            RetrievedChunk(
                transcript_id="raw_transcript:video",
                video_id=video_id,
                source_url="https://www.youtube.com/watch?v=video",
                chunk_index=0,
                text="capital gains tax answer",
                start_seconds=754,
                end_seconds=782,
                segment_count=1,
            )
        ]


class FakeIndexer:
    def __init__(self, chunk_store: FakeChunkStore) -> None:
        self.chunk_store = chunk_store
        self.calls = 0

    def index(self, source_url: str, refresh: bool = False):
        self.calls += 1
        self.chunk_store.indexed = True

        class Result:
            cache_status = "miss"

        return Result()


def test_rag_context_auto_indexes_and_formats_timestamped_chunks() -> None:
    chunk_store = FakeChunkStore()
    indexer = FakeIndexer(chunk_store)
    provider = RagTranscriptContextProvider(
        raw_store=FakeRawStore(),
        chunk_store=chunk_store,
        indexer=indexer,
        top_k=10,
    )

    context = provider.get_transcript(
        "video", "https://www.youtube.com/watch?v=video", query="capital gains"
    )

    assert indexer.calls == 1
    assert chunk_store.query_text == "capital gains"
    assert context.context_mode == "rag"
    assert "[1] 12:34-13:02" in (context.context_text or "")
    assert context.retrieved_chunks


class FakeMultiChunkStore:
    def __init__(self, has_any: bool = True, has_url: bool = True) -> None:
        self.has_any = has_any
        self.has_url = has_url
        self.calls = []

    def has_any_chunks(self) -> bool:
        return self.has_any

    def has_chunks(self, video_id: str) -> bool:
        return self.has_url

    def query_all(self, query: str, top_k: int):
        self.calls.append(("all", query, top_k))
        return [_multi_chunk("aaaaaaaaaaa")]

    def query_by_url(self, source_url: str, query: str, top_k: int):
        self.calls.append(("url", source_url, query, top_k))
        return [_multi_chunk("aaaaaaaaaaa")]


class FakeMultiRawStore:
    def ensure_raw_document(self, source_url: str, refresh: bool = False):
        return (
            RawTranscriptDocument(
                transcript_id="raw_transcript:aaaaaaaaaaa",
                video_id="aaaaaaaaaaa",
                source_url=source_url,
                fetched_at="2026-05-14T00:00:00+00:00",
                segments=[RawTranscriptSegment(text="full transcript")],
            ),
            "hit",
        )


class FakeMultiIndexer:
    def __init__(self, chunk_store: FakeMultiChunkStore) -> None:
        self.chunk_store = chunk_store
        self.calls = []

    def index(self, source_url: str, refresh: bool = False):
        self.calls.append((source_url, refresh))
        self.chunk_store.has_url = True

        class Result:
            cache_status = "miss"

        return Result()


def test_multi_transcript_context_queries_all_when_url_is_missing() -> None:
    chunk_store = FakeMultiChunkStore()
    provider = MultiTranscriptRagContextProvider(
        raw_store=FakeMultiRawStore(),
        chunk_store=chunk_store,
    )

    context = provider.get_context("capital gains", top_k=10)

    assert chunk_store.calls == [("all", "capital gains", 10)]
    assert "url=https://www.youtube.com/watch?v=aaaaaaaaaaa&t=10s" in (
        context.context_text or ""
    )


def test_multi_transcript_context_filters_by_url() -> None:
    chunk_store = FakeMultiChunkStore()
    provider = MultiTranscriptRagContextProvider(
        raw_store=FakeMultiRawStore(),
        chunk_store=chunk_store,
    )

    provider.get_context(
        "capital gains",
        source_url="https://www.youtube.com/watch?v=aaaaaaaaaaa",
        top_k=5,
    )

    assert chunk_store.calls == [
        (
            "url",
            "https://www.youtube.com/watch?v=aaaaaaaaaaa",
            "capital gains",
            5,
        )
    ]


def test_multi_transcript_context_auto_indexes_filtered_url() -> None:
    chunk_store = FakeMultiChunkStore(has_url=False)
    indexer = FakeMultiIndexer(chunk_store)
    provider = MultiTranscriptRagContextProvider(
        raw_store=FakeMultiRawStore(),
        chunk_store=chunk_store,
        indexer=indexer,
    )

    context = provider.get_context(
        "capital gains",
        source_url="https://www.youtube.com/watch?v=aaaaaaaaaaa",
    )

    assert indexer.calls == [("https://www.youtube.com/watch?v=aaaaaaaaaaa", False)]
    assert context.cache_status == "miss"


def test_multi_transcript_context_errors_when_all_mode_has_no_chunks() -> None:
    provider = MultiTranscriptRagContextProvider(
        raw_store=FakeMultiRawStore(),
        chunk_store=FakeMultiChunkStore(has_any=False),
    )

    with pytest.raises(ValueError, match="No indexed transcript chunks"):
        provider.get_context("capital gains")


def _multi_chunk(video_id: str) -> RetrievedChunk:
    return RetrievedChunk(
        transcript_id=f"raw_transcript:{video_id}",
        video_id=video_id,
        source_url=f"https://www.youtube.com/watch?v={video_id}",
        chunk_index=0,
        text="capital gains tax answer",
        start_seconds=10,
        end_seconds=20,
        segment_count=1,
    )

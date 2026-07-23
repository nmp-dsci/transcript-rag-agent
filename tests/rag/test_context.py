from __future__ import annotations

import pytest

from src.rag.context import MultiTranscriptRagContextProvider, RagTranscriptContextProvider
from src.rag.models import (
    RawTranscriptDocument,
    RawTranscriptSegment,
    RetrievedChunk,
    RetrievedTranscriptSummary,
)


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
    def __init__(
        self,
        has_any: bool = True,
        has_url: bool = True,
        channel_videos: dict[str, list[str]] | None = None,
    ) -> None:
        self.has_any = has_any
        self.has_url = has_url
        self.calls = []
        self.channel_videos = channel_videos or {}

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

    def query_by_video_ids(self, video_ids: list[str], query: str, top_k: int):
        self.calls.append(("video_ids", video_ids, query, top_k))
        return [_multi_chunk(video_ids[0])]

    def channel_video_ids(self, channel_id: str) -> list[str]:
        self.calls.append(("channel_video_ids", channel_id))
        return self.channel_videos.get(channel_id, [])


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


class FakeSummaryStore:
    def query_relevant_transcripts(
        self,
        question: str,
        top_k: int,
        min_score: float,
    ):
        return [
            RetrievedTranscriptSummary(
                transcript_id="raw_transcript:bbbbbbbbbbb",
                video_id="bbbbbbbbbbb",
                source_url="https://www.youtube.com/watch?v=bbbbbbbbbbb",
                summary="capital gains tax summary",
                summary_model="deepseek-test",
                summary_generated_at="2026-05-16T00:00:00+00:00",
                summary_embedding=[1.0, 0.0, 1.0],
                summary_embedding_model="fake",
                summary_embedded_at="2026-05-16T00:01:00+00:00",
                score=0.8,
            )
        ]


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


def test_multi_transcript_context_filters_by_summary_before_chunks() -> None:
    chunk_store = FakeMultiChunkStore()
    provider = MultiTranscriptRagContextProvider(
        raw_store=FakeMultiRawStore(),
        chunk_store=chunk_store,
        summary_store=FakeSummaryStore(),
    )

    context = provider.get_context(
        "capital gains",
        top_k=5,
        filter_transcripts=True,
        transcript_filter_top_k=3,
        transcript_filter_min_score=0.25,
    )

    assert chunk_store.calls == [
        ("video_ids", ["bbbbbbbbbbb"], "capital gains", 5)
    ]
    assert context.selected_transcripts
    assert context.selected_transcripts[0].video_id == "bbbbbbbbbbb"


class FakeChannelFilterSummaryStore:
    """Matches two transcripts on relevance, only one of which is in-channel."""

    def query_relevant_transcripts(
        self,
        question: str,
        top_k: int,
        min_score: float,
    ):
        return [
            RetrievedTranscriptSummary(
                transcript_id="raw_transcript:bbbbbbbbbbb",
                video_id="bbbbbbbbbbb",
                source_url="https://www.youtube.com/watch?v=bbbbbbbbbbb",
                summary="capital gains tax summary",
                summary_model="deepseek-test",
                summary_generated_at="2026-05-16T00:00:00+00:00",
                summary_embedding=[1.0, 0.0, 1.0],
                summary_embedding_model="fake",
                summary_embedded_at="2026-05-16T00:01:00+00:00",
                score=0.8,
            ),
            RetrievedTranscriptSummary(
                transcript_id="raw_transcript:ccccccccccc",
                video_id="ccccccccccc",
                source_url="https://www.youtube.com/watch?v=ccccccccccc",
                summary="a different channel's summary",
                summary_model="deepseek-test",
                summary_generated_at="2026-05-16T00:00:00+00:00",
                summary_embedding=[0.0, 1.0, 1.0],
                summary_embedding_model="fake",
                summary_embedded_at="2026-05-16T00:01:00+00:00",
                score=0.6,
            ),
        ]


def test_multi_transcript_context_filter_transcripts_and_channel_id_compose() -> None:
    """Both toggles set at once must narrow WITHIN the channel, not drop it."""
    chunk_store = FakeMultiChunkStore(channel_videos={"UC1": ["bbbbbbbbbbb"]})
    provider = MultiTranscriptRagContextProvider(
        raw_store=FakeMultiRawStore(),
        chunk_store=chunk_store,
        summary_store=FakeChannelFilterSummaryStore(),
    )

    context = provider.get_context(
        "capital gains",
        top_k=5,
        filter_transcripts=True,
        channel_id="UC1",
    )

    assert ("video_ids", ["bbbbbbbbbbb"], "capital gains", 5) in chunk_store.calls
    assert not any(
        call[0] == "video_ids" and "ccccccccccc" in call[1]
        for call in chunk_store.calls
    )
    assert context.retrieved_chunks


def test_multi_transcript_context_filter_transcripts_and_channel_id_empty_intersection_raises() -> None:
    """No summary-matched transcript belongs to the channel: fail loudly."""
    chunk_store = FakeMultiChunkStore(channel_videos={"UC_other": ["zzzzzzzzzzz"]})
    provider = MultiTranscriptRagContextProvider(
        raw_store=FakeMultiRawStore(),
        chunk_store=chunk_store,
        summary_store=FakeChannelFilterSummaryStore(),
    )

    with pytest.raises(ValueError, match="within channel 'UC_other'"):
        provider.get_context(
            "capital gains",
            filter_transcripts=True,
            channel_id="UC_other",
        )
    assert not any(call[0] == "video_ids" for call in chunk_store.calls)


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


class FakeHybridChunkStore:
    """A store whose semantic pass misses a chunk that only BM25 will find.

    ``query_all`` returns one chunk that lacks the rare query term; the whole
    tiny corpus (that chunk plus a keyword-only one) is exposed to BM25 through
    ``collection.get``. Hybrid fusion should surface the keyword-only chunk
    rather than drop it — the recall-widening behaviour the resolver enables.
    """

    def __init__(self, records_metadata: list[dict] | None = None) -> None:
        self._semantic = RetrievedChunk(
            transcript_id="raw_transcript:vidsem",
            video_id="vidsem",
            source_url="https://www.youtube.com/watch?v=vidsem",
            chunk_index=0,
            text="a general discussion of property investment returns",
            start_seconds=10,
            end_seconds=20,
            segment_count=1,
        )
        self._metadatas = records_metadata or [
            {
                "transcript_id": "raw_transcript:vidsem",
                "video_id": "vidsem",
                "chunk_index": 0,
                "source_url": "https://www.youtube.com/watch?v=vidsem",
            },
            {
                "transcript_id": "raw_transcript:vidkw",
                "video_id": "vidkw",
                "chunk_index": 5,
                "source_url": "https://www.youtube.com/watch?v=vidkw",
            },
        ]
        self.collection = self  # _bm25_records calls chunk_store.collection.get(...)

    def has_any_chunks(self) -> bool:
        return True

    def query_all(self, query: str, top_k: int):
        return [self._semantic]

    def get(self, where=None, include=None):
        return {
            "documents": [
                "a general discussion of property investment returns",
                "existing properties are grandfathered under the old negative gearing rules",
            ],
            "metadatas": self._metadatas,
        }


def test_hybrid_widens_recall_with_a_bm25_only_chunk() -> None:
    from src.rag import bm25

    bm25.clear_cache()
    chunk_store = FakeHybridChunkStore()
    provider = MultiTranscriptRagContextProvider(
        raw_store=FakeMultiRawStore(),
        chunk_store=chunk_store,
        retrieval_mode="hybrid",
    )

    context = provider.get_context("grandfathered", top_k=5)

    surfaced = {(c.video_id, c.chunk_index) for c in context.retrieved_chunks}
    # Semantic never returned vidkw:5; only BM25 found the rare term, and fusion
    # now surfaces it instead of dropping it for want of a resolvable identity.
    assert ("vidkw", 5) in surfaced
    keyword_only = next(c for c in context.retrieved_chunks if c.video_id == "vidkw")
    # Found by keyword alone, so it carries no invented semantic score.
    assert keyword_only.score is None


def test_hybrid_does_not_fabricate_identity_for_unresolvable_hits() -> None:
    from src.rag import bm25

    bm25.clear_cache()
    # The keyword-only record is missing source_url, so it cannot be cited.
    chunk_store = FakeHybridChunkStore(
        records_metadata=[
            {
                "transcript_id": "raw_transcript:vidsem",
                "video_id": "vidsem",
                "chunk_index": 0,
                "source_url": "https://www.youtube.com/watch?v=vidsem",
            },
            {"transcript_id": "raw_transcript:vidkw", "video_id": "vidkw", "chunk_index": 5},
        ]
    )
    provider = MultiTranscriptRagContextProvider(
        raw_store=FakeMultiRawStore(),
        chunk_store=chunk_store,
        retrieval_mode="hybrid",
    )

    context = provider.get_context("grandfathered", top_k=5)

    surfaced = {(c.video_id, c.chunk_index) for c in context.retrieved_chunks}
    assert ("vidkw", 5) not in surfaced


def test_chunk_from_record_rebuilds_a_citable_chunk() -> None:
    from src.rag.context import _chunk_from_record

    chunk = _chunk_from_record(
        {
            "transcript_id": "raw_transcript:v",
            "video_id": "v",
            "chunk_index": 3,
            "text": "hello",
            "source_url": "https://www.youtube.com/watch?v=v",
        }
    )
    assert chunk is not None
    assert chunk.chunk_id == "chunk:v:3"
    assert chunk.score is None


def test_chunk_from_record_refuses_to_invent_missing_identity() -> None:
    from src.rag.context import _chunk_from_record

    # No source_url and no transcript_id, respectively → dropped, never fabricated.
    assert (
        _chunk_from_record(
            {"transcript_id": "t", "video_id": "v", "chunk_index": 1, "text": "x"}
        )
        is None
    )
    assert (
        _chunk_from_record(
            {"video_id": "v", "chunk_index": 1, "text": "x", "source_url": "https://y.com"}
        )
        is None
    )


def test_chunk_from_record_drops_malformed_metadata_instead_of_raising() -> None:
    from src.rag.context import _chunk_from_record

    # A malformed source_url must drop the record, not crash hybrid retrieval.
    assert (
        _chunk_from_record(
            {
                "transcript_id": "t",
                "video_id": "v",
                "chunk_index": 1,
                "text": "x",
                "source_url": "not a url",
            }
        )
        is None
    )
    # Same for an unparsable start_seconds.
    assert (
        _chunk_from_record(
            {
                "transcript_id": "t",
                "video_id": "v",
                "chunk_index": 1,
                "text": "x",
                "source_url": "https://www.youtube.com/watch?v=v",
                "start_seconds": "not-a-number",
            }
        )
        is None
    )

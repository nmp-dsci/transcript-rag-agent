from __future__ import annotations

import pytest

from src.rag.context import (
    MultiTranscriptRagContextProvider,
    RagTranscriptContextProvider,
    select_diverse_sources,
    source_breadth,
)
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
    assert "url=https://www.youtube.com/watch?v=aaaaaaaaaaa&t=10s" in (context.context_text or "")


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

    assert chunk_store.calls == [("video_ids", ["bbbbbbbbbbb"], "capital gains", 5)]
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
        call[0] == "video_ids" and "ccccccccccc" in call[1] for call in chunk_store.calls
    )
    assert context.retrieved_chunks


def test_multi_transcript_context_filter_transcripts_and_channel_id_empty_intersection_raises() -> (
    None
):
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

    exclude_video_ids: list[str] = []

    @property
    def exclusion_key(self) -> str:
        return ""

    def scoped_where(self, where=None):
        return where

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
        _chunk_from_record({"transcript_id": "t", "video_id": "v", "chunk_index": 1, "text": "x"})
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


# ── retrieval trace ──────────────────────────────────────────────────────────


class FakeTraceReranker:
    def rerank(self, query: str, chunks: list, top_k: int) -> list:
        return list(reversed(chunks))[:top_k]


class FakeNeighborChunkStore(FakeMultiChunkStore):
    def neighbors(self, video_id: str, chunk_index: int, span: int) -> list:
        return []


def test_get_context_records_a_retrieve_trace_step() -> None:
    provider = MultiTranscriptRagContextProvider(
        raw_store=FakeMultiRawStore(),
        chunk_store=FakeMultiChunkStore(),
    )

    context = provider.get_context("capital gains", top_k=10)

    assert [step.phase for step in context.trace] == ["retrieve"]
    step = context.trace[0]
    assert "whole corpus" in step.detail
    assert step.chunk_ids == ["chunk:aaaaaaaaaaa:0"]
    assert isinstance(step.elapsed_ms, int)


def test_get_context_trace_includes_filter_and_rerank_stages() -> None:
    provider = MultiTranscriptRagContextProvider(
        raw_store=FakeMultiRawStore(),
        chunk_store=FakeMultiChunkStore(),
        summary_store=FakeSummaryStore(),
        reranker=FakeTraceReranker(),
    )

    context = provider.get_context(
        "capital gains",
        top_k=5,
        filter_transcripts=True,
        transcript_filter_top_k=3,
        transcript_filter_min_score=0.25,
    )

    assert [step.phase for step in context.trace] == ["filter", "retrieve", "rerank"]
    assert "1 videos matched" in context.trace[0].detail
    # The rerank step records what it kept, which is what the answer call sees.
    assert context.trace[2].chunk_ids == ["chunk:bbbbbbbbbbb:0"]


class FakeTitledSummaryStore(FakeSummaryStore):
    """Two matches with titles, so the note has a list to render."""

    def query_relevant_transcripts(self, question: str, top_k: int, min_score: float):
        return [
            _summary("bbbbbbbbbbb", "Job Interview Simulation", 0.325),
            _summary("ccccccccccc", "How to Get a Job in 2026", 0.2749),
        ]


def _summary(video_id: str, title: str | None, score: float) -> RetrievedTranscriptSummary:
    return RetrievedTranscriptSummary(
        transcript_id=f"raw_transcript:{video_id}",
        video_id=video_id,
        source_url=f"https://www.youtube.com/watch?v={video_id}",
        summary="summary text",
        summary_model="deepseek-test",
        summary_generated_at="2026-05-16T00:00:00+00:00",
        summary_embedding=[1.0, 0.0, 1.0],
        summary_embedding_model="fake",
        summary_embedded_at="2026-05-16T00:01:00+00:00",
        title=title,
        score=score,
    )


def test_the_filter_step_names_every_video_it_routed_to() -> None:
    """The count is the measurement; which videos is the check on it."""
    provider = MultiTranscriptRagContextProvider(
        raw_store=FakeMultiRawStore(),
        chunk_store=FakeMultiChunkStore(),
        summary_store=FakeTitledSummaryStore(),
    )

    context = provider.get_context("behavioural interviews", filter_transcripts=True)

    note = context.trace[0].note
    assert note is not None
    assert "Job Interview Simulation (bbbbbbbbbbb) 0.33" in note
    assert "How to Get a Job in 2026 (ccccccccccc) 0.27" in note


class FakeUntitledSummaryStore(FakeSummaryStore):
    def query_relevant_transcripts(self, question: str, top_k: int, min_score: float):
        return [_summary("bbbbbbbbbbb", None, 0.4)]


def test_the_filter_step_falls_back_to_the_video_id_when_a_title_is_missing() -> None:
    provider = MultiTranscriptRagContextProvider(
        raw_store=FakeMultiRawStore(),
        chunk_store=FakeMultiChunkStore(),
        summary_store=FakeUntitledSummaryStore(),
    )

    context = provider.get_context("behavioural interviews", filter_transcripts=True)

    assert context.trace[0].note == "bbbbbbbbbbb 0.40"


class FakeWideChunkStore(FakeMultiChunkStore):
    """Returns more candidates than top_k, so the final trim actually cuts."""

    def query_all(self, query: str, top_k: int):
        self.calls.append(("all", query, top_k))
        return [_multi_chunk("aaaaaaaaaaa"), _multi_chunk("bbbbbbbbbbb")]


def test_get_context_trace_records_the_trim_to_top_k() -> None:
    """Without this the retrieve step would overstate what the LLM saw."""
    provider = MultiTranscriptRagContextProvider(
        raw_store=FakeMultiRawStore(),
        chunk_store=FakeWideChunkStore(),
    )

    context = provider.get_context("capital gains", top_k=1)

    assert [step.phase for step in context.trace] == ["retrieve", "merge"]
    trim = context.trace[1]
    assert trim.label == "Trim to top_k"
    assert "kept the first 1 of 2" in trim.detail
    assert trim.chunk_ids == ["chunk:aaaaaaaaaaa:0"]
    assert len(context.retrieved_chunks) == 1


def test_get_context_trace_omits_the_trim_step_when_nothing_was_cut() -> None:
    provider = MultiTranscriptRagContextProvider(
        raw_store=FakeMultiRawStore(),
        chunk_store=FakeWideChunkStore(),
    )

    context = provider.get_context("capital gains", top_k=10)

    assert [step.phase for step in context.trace] == ["retrieve"]


def test_failed_summary_filter_carries_the_stage_it_measured() -> None:
    """The filter ran and found nothing; that step is the whole diagnostic."""
    from src.rag.context import RetrievalError

    class EmptySummaryStore:
        def query_relevant_transcripts(self, question, top_k, min_score):
            return []

    provider = MultiTranscriptRagContextProvider(
        raw_store=FakeMultiRawStore(),
        chunk_store=FakeMultiChunkStore(),
        summary_store=EmptySummaryStore(),
    )

    with pytest.raises(RetrievalError) as excinfo:
        provider.get_context("capital gains", filter_transcripts=True)

    assert [step.label for step in excinfo.value.trace] == ["Summary filter"]
    assert "0 videos matched" in excinfo.value.trace[0].detail
    assert isinstance(excinfo.value.trace[0].elapsed_ms, int)


def test_a_retrieval_failure_before_any_stage_carries_an_empty_trace() -> None:
    """Nothing was measured, so an empty trace is the honest record."""
    from src.rag.context import RetrievalError

    provider = MultiTranscriptRagContextProvider(
        raw_store=FakeMultiRawStore(),
        chunk_store=FakeMultiChunkStore(has_any=False),
    )

    with pytest.raises(RetrievalError) as excinfo:
        provider.get_context("capital gains")

    assert excinfo.value.trace == []


def test_get_context_trace_records_neighbor_expansion() -> None:
    provider = MultiTranscriptRagContextProvider(
        raw_store=FakeMultiRawStore(),
        chunk_store=FakeNeighborChunkStore(),
        neighbor_span=1,
    )

    context = provider.get_context("capital gains", top_k=10)

    assert [step.phase for step in context.trace] == ["retrieve", "merge"]
    assert "±1 adjacent" in context.trace[1].detail


def test_the_retrieve_step_records_what_was_actually_searched_for() -> None:
    """A trace that hides the query cannot show you the corpus was searched for
    the wrong thing — the failure most worth catching here."""
    provider = MultiTranscriptRagContextProvider(
        raw_store=FakeMultiRawStore(),
        chunk_store=FakeMultiChunkStore(),
    )

    context = provider.get_context("what recruiters look for in a portfolio", top_k=10)

    assert context.trace[0].query == "what recruiters look for in a portfolio"


def test_a_very_long_query_is_shortened_rather_than_dropped() -> None:
    from src.rag.context import MAX_TRACE_QUERY_CHARS

    provider = MultiTranscriptRagContextProvider(
        raw_store=FakeMultiRawStore(),
        chunk_store=FakeMultiChunkStore(),
    )

    context = provider.get_context("capital gains " * 100, top_k=10)

    query = context.trace[0].query
    assert query is not None
    assert len(query) <= MAX_TRACE_QUERY_CHARS
    assert query.endswith("…")


# --- The summary filter's cap on a corpus-wide question -----------------------


class RecordingSummaryStore:
    """A summary store that reports the ``top_k`` it was asked for.

    ``video_count`` is what :meth:`count` returns — the ceiling the cap rises
    to — and every video is returned so the filter's own top_k does the cutting.
    """

    def __init__(self, video_count: int = 40) -> None:
        self.video_count = video_count
        self.requested_top_k: list[int] = []

    def count(self) -> int:
        return self.video_count

    def query_relevant_transcripts(self, question: str, top_k: int, min_score: float):
        self.requested_top_k.append(top_k)
        return [
            RetrievedTranscriptSummary(
                transcript_id=f"raw_transcript:vid{index:09d}",
                video_id=f"vid{index:08d}",
                source_url=f"https://www.youtube.com/watch?v=vid{index:08d}",
                summary=f"summary {index}",
                summary_model="deepseek-test",
                summary_generated_at="2026-05-16T00:00:00+00:00",
                summary_embedding=[1.0, 0.0, 1.0],
                summary_embedding_model="fake",
                summary_embedded_at="2026-05-16T00:01:00+00:00",
                score=0.8,
            )
            for index in range(min(top_k, self.video_count))
        ]


def _filtered_context(question: str, summary_store, *, chunk_store=None, top_k=10, **kwargs):
    provider = MultiTranscriptRagContextProvider(
        raw_store=FakeMultiRawStore(),
        chunk_store=chunk_store or FakeMultiChunkStore(),
        summary_store=summary_store,
        **kwargs,
    )
    return provider.get_context(
        question,
        top_k=top_k,
        filter_transcripts=True,
        transcript_filter_top_k=5,
    )


def test_a_corpus_wide_question_lifts_the_summary_filter_cap() -> None:
    """ "Main themes across this corpus" has no top five — the answer is the spread."""
    summary_store = RecordingSummaryStore(video_count=40)

    context = _filtered_context(
        "What are the main themes across this corpus of videos?", summary_store
    )

    assert summary_store.requested_top_k == [40]
    assert len(context.selected_transcripts or []) == 40


def test_a_specific_question_keeps_the_cap() -> None:
    summary_store = RecordingSummaryStore(video_count=40)

    context = _filtered_context("How do I make my resume ATS-friendly?", summary_store)

    assert summary_store.requested_top_k == [5]
    assert len(context.selected_transcripts or []) == 5


def test_lifting_the_cap_is_recorded_in_the_trace_with_the_signal_that_fired() -> None:
    """A silent behaviour change is a behaviour change nobody can audit."""
    summary_store = RecordingSummaryStore(video_count=40)

    context = _filtered_context(
        "What are the main themes across this corpus of videos?", summary_store
    )

    steps = [step for step in (context.trace or []) if step.label == "Corpus-wide question"]
    assert len(steps) == 1
    assert "'corpus'" in steps[0].detail
    assert "5 to 40" in steps[0].detail
    filter_step = next(step for step in context.trace if step.label == "Summary filter")
    assert "cap lifted" in filter_step.detail


def test_no_corpus_wide_step_is_recorded_for_a_specific_question() -> None:
    summary_store = RecordingSummaryStore(video_count=40)

    context = _filtered_context("How do I make my resume ATS-friendly?", summary_store)

    assert not [step for step in (context.trace or []) if step.label == "Corpus-wide question"]
    filter_step = next(step for step in context.trace if step.label == "Summary filter")
    assert "cap lifted" not in filter_step.detail


def test_the_cap_lift_can_be_switched_off_for_a_controlled_comparison() -> None:
    """The eval harness has to be able to produce the pre-change arm."""
    summary_store = RecordingSummaryStore(video_count=40)

    _filtered_context(
        "What are the main themes across this corpus of videos?",
        summary_store,
        corpus_wide_filter=False,
    )

    assert summary_store.requested_top_k == [5]


def test_a_corpus_smaller_than_the_cap_never_shrinks_the_filter() -> None:
    """Lifting a cap must not lower it: max(), not replace."""
    summary_store = RecordingSummaryStore(video_count=2)

    _filtered_context("What are the main themes across this corpus?", summary_store)

    assert summary_store.requested_top_k == [5]


# --- Diversity-constrained selection on a corpus-wide question ----------------


def _ranked(*specs) -> list[RetrievedChunk]:
    """Chunks in ranking order, given ``(video_id, chunk_index)`` pairs."""
    return [
        RetrievedChunk(
            transcript_id=f"raw_transcript:{video_id}",
            video_id=video_id,
            source_url=f"https://www.youtube.com/watch?v={video_id}",
            chunk_index=index,
            text=f"{video_id} chunk {index}",
            start_seconds=0,
            end_seconds=10,
            score=1.0 - position / 100,
        )
        for position, (video_id, index) in enumerate(specs)
    ]


def test_a_per_video_cap_promotes_chunks_the_budget_had_crowded_out() -> None:
    """The defect: ten chunks from three videos is a three-video answer.

    ``vidaaa`` monopolises the ranking, so lifting the *video* cap admitted
    seventy-one videos to a budget that four of them were already spending.
    """
    pool = _ranked(
        ("vidaaa", 0),
        ("vidaaa", 1),
        ("vidaaa", 2),
        ("vidbbb", 0),
        ("vidccc", 0),
        ("vidddd", 0),
    )

    kept = select_diverse_sources(pool, top_k=4, max_per_video=1)

    assert [chunk.video_id for chunk in kept] == ["vidaaa", "vidbbb", "vidccc", "vidddd"]
    assert source_breadth(kept) == 4
    assert source_breadth(pool[:4]) == 2


def test_the_cap_never_changes_the_order_the_retriever_ranked_in() -> None:
    """It changes *which* chunks reach the answer, not what order they arrive."""
    pool = _ranked(("vidaaa", 0), ("vidaaa", 1), ("vidbbb", 0), ("vidaaa", 2), ("vidccc", 0))

    kept = select_diverse_sources(pool, top_k=3, max_per_video=1)

    assert [(c.video_id, c.chunk_index) for c in kept] == [
        ("vidaaa", 0),
        ("vidbbb", 0),
        ("vidccc", 0),
    ]


def test_the_selection_is_never_smaller_than_it_would_have_been() -> None:
    """Backfill, not truncation: a narrow corpus must not lose context.

    Three videos and a cap of one would leave a four-chunk budget three-quarters
    spent if capped chunks were dropped rather than parked.
    """
    pool = _ranked(("vidaaa", 0), ("vidaaa", 1), ("vidaaa", 2), ("vidbbb", 0), ("vidccc", 0))

    kept = select_diverse_sources(pool, top_k=4, max_per_video=1)

    assert len(kept) == 4
    # The backfilled chunk is the best-ranked parked one, restored to its place.
    assert [(c.video_id, c.chunk_index) for c in kept] == [
        ("vidaaa", 0),
        ("vidaaa", 1),
        ("vidbbb", 0),
        ("vidccc", 0),
    ]


def test_every_selected_chunk_came_from_the_ranking() -> None:
    """The honest half: the cap can only promote what relevance already ranked.

    A round-robin over the *admitted videos* would have added a chunk from a
    video the ranking never surfaced — breadth measuring the filter's
    admissions rather than the corpus's evidence.
    """
    pool = _ranked(("vidaaa", 0), ("vidaaa", 1), ("vidbbb", 7))

    kept = select_diverse_sources(pool, top_k=3, max_per_video=1)

    assert all(chunk in pool for chunk in kept)


def test_a_cap_at_or_above_top_k_is_a_no_op() -> None:
    """The switch degrades to nothing, not to a surprise."""
    pool = _ranked(("vidaaa", 0), ("vidaaa", 1), ("vidaaa", 2), ("vidbbb", 0))

    assert select_diverse_sources(pool, top_k=3, max_per_video=3) == pool[:3]


def test_diversity_is_off_when_the_cap_is_zero() -> None:
    pool = _ranked(("vidaaa", 0), ("vidaaa", 1), ("vidbbb", 0))

    assert select_diverse_sources(pool, top_k=2, max_per_video=0) == pool[:2]


def test_source_breadth_reads_chunk_ids_as_well_as_chunks() -> None:
    """The instrument has to work on a committed run, which stores only ids."""
    pool = _ranked(("vidaaa", 0), ("vidaaa", 1), ("vidbbb", 0))

    assert source_breadth(pool) == 2
    assert source_breadth(["chunk:vidaaa:0", "chunk:vidaaa:1", "chunk:vidbbb:0"]) == 2
    assert source_breadth([]) == 0
    assert source_breadth(["not-a-chunk-id"]) == 0


class BroadChunkStore(FakeMultiChunkStore):
    """A store whose ranking is dominated by one video, like the real one."""

    def query_by_video_ids(self, video_ids: list[str], query: str, top_k: int):
        self.calls.append(("video_ids", video_ids, query, top_k))
        return _ranked(
            ("vid00000000", 0),
            ("vid00000000", 1),
            ("vid00000000", 2),
            ("vid00000001", 0),
            ("vid00000002", 0),
        )


def test_the_diversity_cap_is_off_by_default() -> None:
    """No shipped behaviour moves before the matrix has measured it."""
    provider = MultiTranscriptRagContextProvider(
        raw_store=FakeMultiRawStore(),
        chunk_store=BroadChunkStore(),
        summary_store=RecordingSummaryStore(video_count=40),
    )

    context = provider.get_context(
        "What are the main themes across this corpus?",
        top_k=3,
        filter_transcripts=True,
        transcript_filter_top_k=5,
    )

    assert [c.video_id for c in context.retrieved_chunks] == ["vid00000000"] * 3
    assert not [s for s in (context.trace or []) if s.label == "Source diversity"]


def test_the_diversity_cap_widens_a_corpus_wide_answer_when_switched_on() -> None:
    summary_store = RecordingSummaryStore(video_count=40)

    context = _filtered_context(
        "What are the main themes across this corpus?",
        summary_store,
        chunk_store=BroadChunkStore(),
        corpus_wide_max_per_video=1,
        top_k=3,
    )

    assert source_breadth(context.retrieved_chunks) == 3
    step = next(s for s in context.trace if s.label == "Source diversity")
    assert "at most 1 chunk(s) per video" in step.detail
    assert "from 3 videos" in step.detail


def test_a_specific_question_is_untouched_by_the_diversity_cap() -> None:
    """It is gated on the corpus-wide signal, where the spread *is* the answer."""
    summary_store = RecordingSummaryStore(video_count=40)

    context = _filtered_context(
        "How do I make my resume ATS-friendly?",
        summary_store,
        chunk_store=BroadChunkStore(),
        corpus_wide_max_per_video=1,
        top_k=3,
    )

    assert [c.video_id for c in context.retrieved_chunks] == ["vid00000000"] * 3
    assert not [s for s in (context.trace or []) if s.label == "Source diversity"]


def test_switching_the_corpus_wide_path_off_disables_the_diversity_cap_too() -> None:
    """``corpus_wide_filter=False`` reproduces the pre-change arm end to end."""
    summary_store = RecordingSummaryStore(video_count=40)

    context = _filtered_context(
        "What are the main themes across this corpus?",
        summary_store,
        chunk_store=BroadChunkStore(),
        corpus_wide_filter=False,
        corpus_wide_max_per_video=1,
        top_k=3,
    )

    assert [c.video_id for c in context.retrieved_chunks] == ["vid00000000"] * 3


def test_the_provider_reports_the_behaviour_its_switches_implement() -> None:
    """The fingerprint reads this rather than a literal, so they cannot drift."""

    def provider(**kwargs):
        return MultiTranscriptRagContextProvider(
            raw_store=FakeMultiRawStore(), chunk_store=FakeMultiChunkStore(), **kwargs
        )

    assert provider().retrieval_behavior == "budget"
    assert provider(corpus_wide_max_per_video=2).retrieval_behavior == "budget+diversity"
    assert provider(corpus_wide_filter=False).retrieval_behavior == "capped"

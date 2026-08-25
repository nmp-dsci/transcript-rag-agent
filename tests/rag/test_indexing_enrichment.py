"""A provider failure must degrade an index, never fail one.

The incident these tests exist for: DeepSeek returned HTTP 402 ``Insufficient
Balance`` during summary generation, the exception propagated out of
``RagIndexer.index``, and the CLI exited 1 — for videos whose transcript and
chunks were already committed. The queue reported "Indexing failed (exit 1)"
while the database held a perfectly retrievable, summary-less video, and
nothing recorded that it needed repair.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.rag.indexing import RagIndexer
from src.rag.storage import RawTranscriptStore, TranscriptChunkStore
from src.rag.summaries import TranscriptSummaryStore
from src.transcripts.models import Transcript, TranscriptSegment

VIDEO_ID = "3hk7nO_q0a8"
SOURCE_URL = f"https://www.youtube.com/watch?v={VIDEO_ID}"


class FakeEmbeddingModel:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text)), 1.0, 0.0]


class FakeFetcher:
    def __init__(self, description: str = "A fixture description.") -> None:
        self.description = description

    def fetch(self, source_url: str) -> Transcript:
        return Transcript(
            video_id=VIDEO_ID,
            url=SOURCE_URL,
            title="Fixture video",
            channel_id="channel-1",
            channel_name="Fixture channel",
            raw_text="segment one segment two",
            segments=[
                TranscriptSegment(text="segment one", start_seconds=0.0, end_seconds=10.0),
                TranscriptSegment(text="segment two", start_seconds=10.0, end_seconds=20.0),
            ],
            fetched_at=datetime(2026, 8, 25, tzinfo=timezone.utc),
        )

    def fetch_metadata(self, source_url: str) -> dict:
        return {"description": self.description}


class ExplodingGenerator:
    """Stands in for DeepSeek with a zero balance."""

    model_name = "deepseek-v4-flash"

    def summarize(self, raw_document) -> str:
        raise RuntimeError(
            "Error code: 402 - {'error': {'message': 'Insufficient Balance'}}"
        )


def _indexer(tmp_path, generator) -> tuple[RagIndexer, TranscriptChunkStore]:
    embedding_model = FakeEmbeddingModel()
    raw_store = RawTranscriptStore(tmp_path / "chroma", fetcher=FakeFetcher())
    chunk_store = TranscriptChunkStore(tmp_path / "chroma", embedding_model)
    summary_store = TranscriptSummaryStore(
        tmp_path / "chroma",
        embedding_model=embedding_model,
        embedding_model_name="fake",
        raw_store=raw_store,
    )
    indexer = RagIndexer(
        raw_store=raw_store,
        chunk_store=chunk_store,
        target_chars=1,
        overlap_chars=0,
        summary_store=summary_store,
        summary_generator=generator,
    )
    return indexer, chunk_store


def test_summary_provider_failure_does_not_fail_the_index(tmp_path) -> None:
    indexer, chunk_store = _indexer(tmp_path, ExplodingGenerator())

    result = indexer.index(SOURCE_URL)

    # The whole point: no exception, and the vector half is real.
    assert result.summary_status == "failed"
    assert "402" in (result.summary_error or "")
    assert len(result.chunks) == 2
    assert chunk_store.count_chunks(VIDEO_ID) == 2


def test_the_video_stays_retrievable_without_a_summary(tmp_path) -> None:
    indexer, chunk_store = _indexer(tmp_path, ExplodingGenerator())

    indexer.index(SOURCE_URL)

    retrieved = chunk_store.query_by_video_id(VIDEO_ID, "segment", top_k=10)
    assert [chunk.chunk_index for chunk in retrieved] == [0, 1]


def test_a_working_generator_still_reports_a_real_status(tmp_path) -> None:
    class WorkingGenerator:
        model_name = "fixture"

        def summarize(self, raw_document) -> str:
            return "A real summary of the fixture video."

    indexer, _ = _indexer(tmp_path, WorkingGenerator())

    result = indexer.index(SOURCE_URL)

    assert result.summary_status in {"created", "refresh", "indexed", "hit"}
    assert result.summary_error is None


def test_no_generator_configured_is_not_a_failure(tmp_path) -> None:
    raw_store = RawTranscriptStore(tmp_path / "chroma", fetcher=FakeFetcher())
    chunk_store = TranscriptChunkStore(tmp_path / "chroma", FakeEmbeddingModel())
    indexer = RagIndexer(raw_store=raw_store, chunk_store=chunk_store)

    result = indexer.index(SOURCE_URL)

    assert result.summary_status is None
    assert result.summary_error is None


def test_a_failed_summary_is_persisted_not_just_returned(tmp_path) -> None:
    """The repair pass has to be able to find these later.

    An unrecorded failure is indistinguishable from "nobody has tried yet",
    which is the exact ambiguity that let 14 videos accumulate unnoticed.
    """
    indexer, _ = _indexer(tmp_path, ExplodingGenerator())

    indexer.index(SOURCE_URL)

    stored = indexer.raw_store.get_raw_document(VIDEO_ID)
    assert stored is not None
    assert stored.summary_status == "failed"


def test_a_successful_summary_records_which_generator_wrote_it(tmp_path) -> None:
    class Descriptionish:
        model_name = "youtube-description"
        source = "description"

        def summarize(self, raw_document) -> str:
            return "Straight from the creator's own blurb."

    indexer, _ = _indexer(tmp_path, Descriptionish())

    indexer.index(SOURCE_URL)

    stored = indexer.raw_store.get_raw_document(VIDEO_ID)
    assert stored is not None
    assert stored.summary_status == "done"
    assert stored.summary_source == "description"


def test_legacy_documents_without_the_field_still_read_correctly(tmp_path) -> None:
    """Everything indexed before these fields existed carries None."""
    from src.rag.models import RawTranscriptDocument, graph_state, summary_state

    legacy_with = RawTranscriptDocument(
        transcript_id="t",
        video_id=VIDEO_ID,
        source_url=SOURCE_URL,
        fetched_at="2026-01-01T00:00:00Z",
        summary="An older LLM summary.",
    )
    legacy_without = legacy_with.model_copy(update={"summary": None})

    assert summary_state(legacy_with) == "done"
    assert summary_state(legacy_without) == "pending"
    # Nothing has ever extracted a graph for either.
    assert graph_state(legacy_with) == "pending"


def test_clearing_a_summary_actually_clears_it(tmp_path) -> None:
    """Chroma's upsert merges metadata — it does not replace it.

    A key omitted from the new metadata keeps its previous value, so the
    "write only if not None" pattern can never unset a field. That makes
    clearing a stale summary — when a rewrite to a new source fails and the
    previous generator's text must not be left behind — silently impossible.
    """
    raw_store = RawTranscriptStore(tmp_path / "chroma", fetcher=FakeFetcher())
    document, _ = raw_store.ensure_raw_document(SOURCE_URL)
    raw_store.upsert_raw_document(
        document.model_copy(
            update={"summary": "An older LLM summary.", "summary_model": "deepseek-v4-flash"}
        )
    )
    assert raw_store.get_raw_document(VIDEO_ID).summary == "An older LLM summary."

    stored = raw_store.get_raw_document(VIDEO_ID)
    raw_store.upsert_raw_document(
        stored.model_copy(update={"summary": None, "summary_model": None})
    )

    cleared = raw_store.get_raw_document(VIDEO_ID)
    assert cleared.summary is None
    assert cleared.summary_model is None

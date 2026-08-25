"""Stages are reported when they begin, not announced before any work starts.

The bug this replaces: the queue set stage to "discover", then "fetch", then
"processing" in three consecutive statements with no work between them, and
then blocked on one opaque call for the whole run. Any progress bar built on
that jumped to 3/6 instantly and froze there — worse than none, because it
looked like information.
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone

from src.rag.indexing import RagIndexer
from src.rag.progress import (
    CORE_STAGES,
    report_stage,
    stage_index,
    stage_reporter,
)
from src.rag.storage import RawTranscriptStore, TranscriptChunkStore
from src.transcripts.models import Transcript, TranscriptSegment

VIDEO_ID = "3hk7nO_q0a8"
SOURCE_URL = f"https://www.youtube.com/watch?v={VIDEO_ID}"


class FakeEmbeddingModel:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text)), 1.0, 0.0]


class FakeFetcher:
    def fetch(self, source_url: str) -> Transcript:
        return Transcript(
            video_id=VIDEO_ID,
            url=SOURCE_URL,
            title="Fixture",
            channel_id="c",
            channel_name="Fixture channel",
            raw_text="a b",
            segments=[
                TranscriptSegment(text="a", start_seconds=0.0, end_seconds=1.0),
                TranscriptSegment(text="b", start_seconds=1.0, end_seconds=2.0),
            ],
            fetched_at=datetime(2026, 8, 25, tzinfo=timezone.utc),
        )

    def fetch_metadata(self, source_url: str) -> dict:
        return {}


class TestStageReporter:
    def test_nothing_listening_is_a_silent_no_op(self) -> None:
        report_stage("fetch")  # must not raise

    def test_a_reporter_that_raises_cannot_fail_the_index(self) -> None:
        """Progress reporting is never allowed to break the thing it reports on."""

        def broken(_stage: str) -> None:
            raise RuntimeError("boom")

        with stage_reporter(broken):
            report_stage("fetch")  # must not raise

    def test_the_previous_reporter_is_restored(self) -> None:
        outer: list[str] = []
        inner: list[str] = []
        with stage_reporter(outer.append):
            with stage_reporter(inner.append):
                report_stage("chunk")
            report_stage("embed")
        assert inner == ["chunk"]
        assert outer == ["embed"]

    def test_concurrent_jobs_do_not_report_into_each_other(self) -> None:
        """Three workers run at once — a global would cross the streams."""
        seen: dict[str, list[str]] = {}
        barrier = threading.Barrier(3)

        def worker(name: str) -> None:
            collected: list[str] = []
            with stage_reporter(collected.append):
                barrier.wait(timeout=5)
                report_stage(name)
            seen[name] = collected

        threads = [threading.Thread(target=worker, args=(n,)) for n in ("a", "b", "c")]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)

        assert seen == {"a": ["a"], "b": ["b"], "c": ["c"]}

    def test_stage_index_is_one_based_and_rejects_unknown_stages(self) -> None:
        assert stage_index("discover") == 1
        assert stage_index("embed") == len(CORE_STAGES)
        assert stage_index("done") is None


class TestIndexerReportsRealStages:
    def test_every_core_stage_fires_once_in_order(self, tmp_path) -> None:
        raw_store = RawTranscriptStore(tmp_path / "chroma", fetcher=FakeFetcher())
        chunk_store = TranscriptChunkStore(tmp_path / "chroma", FakeEmbeddingModel())
        indexer = RagIndexer(raw_store=raw_store, chunk_store=chunk_store, target_chars=1)

        stages: list[str] = []
        with stage_reporter(stages.append):
            indexer.index(SOURCE_URL)

        assert stages == list(CORE_STAGES)

    def test_chunk_is_not_reported_before_the_transcript_exists(self, tmp_path) -> None:
        """The ordering that makes the reading truthful.

        `chunk` must fire *after* the fetch returns, otherwise the progress
        bar is once again describing work that has not begun.
        """
        raw_store = RawTranscriptStore(tmp_path / "chroma", fetcher=FakeFetcher())
        chunk_store = TranscriptChunkStore(tmp_path / "chroma", FakeEmbeddingModel())
        indexer = RagIndexer(raw_store=raw_store, chunk_store=chunk_store, target_chars=1)

        observed: list[tuple[str, bool]] = []

        def record(stage: str) -> None:
            observed.append((stage, raw_store.get_raw_document(VIDEO_ID) is not None))

        with stage_reporter(record):
            indexer.index(SOURCE_URL)

        by_stage = dict(observed)
        assert by_stage["fetch"] is False
        assert by_stage["chunk"] is True

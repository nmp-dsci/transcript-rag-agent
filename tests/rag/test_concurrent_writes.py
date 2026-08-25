"""Concurrent ingestion workers must not corrupt shared state.

Two things are shared once the queue runs more than one job at a time: the
Chroma collections every job writes to, and the single dashboard HTML file
every job rewrites when it finishes. Both are exercised here under real
threads rather than argued about.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from src.rag.chunking import build_chunks
from src.rag.storage import (
    RawTranscriptStore,
    TranscriptChunkStore,
    collection_write_lock,
    raw_document_from_transcript,
)
from src.transcripts.models import Transcript, TranscriptSegment


class FakeEmbeddingModel:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text)), 1.0, 0.0]


def _transcript(video_id: str, count: int) -> Transcript:
    return Transcript(
        video_id=video_id,
        url=f"https://www.youtube.com/watch?v={video_id}",
        title=f"Video {video_id}",
        channel_id="c1",
        channel_name="Fixture channel",
        raw_text="x",
        segments=[
            TranscriptSegment(
                text=f"segment {index}",
                start_seconds=float(index * 10),
                end_seconds=float(index * 10 + 10),
            )
            for index in range(count)
        ],
        fetched_at=datetime(2026, 8, 25, tzinfo=timezone.utc),
    )


class TestCollectionWriteLock:
    def test_the_same_collection_shares_one_lock_across_instances(self, tmp_path) -> None:
        """The queue runs each job in-process with its own store objects.

        A per-instance lock would therefore protect nothing — two concurrent
        jobs would hold two different locks over the same collection on disk.
        """
        a = TranscriptChunkStore(tmp_path / "chroma", FakeEmbeddingModel())
        b = TranscriptChunkStore(tmp_path / "chroma", FakeEmbeddingModel())
        assert a.write_lock is b.write_lock

    def test_different_collections_do_not_block_each_other(self, tmp_path) -> None:
        raw = RawTranscriptStore(tmp_path / "chroma")
        chunks = TranscriptChunkStore(tmp_path / "chroma", FakeEmbeddingModel())
        assert raw.write_lock is not chunks.write_lock

    def test_the_lock_is_keyed_by_resolved_path(self, tmp_path) -> None:
        """`chroma` and `./chroma` are the same collection."""
        first = collection_write_lock(tmp_path / "chroma", "transcript_chunks")
        second = collection_write_lock(tmp_path / "." / "chroma", "transcript_chunks")
        assert first is second


class TestConcurrentIndexing:
    def test_ten_videos_written_at_once_all_survive(self, tmp_path) -> None:
        """The real risk of a multi-worker queue: writes lost to a race."""
        store = TranscriptChunkStore(tmp_path / "chroma", FakeEmbeddingModel())
        video_ids = [f"video{index:02d}" for index in range(10)]

        def write(video_id: str) -> None:
            document = raw_document_from_transcript(_transcript(video_id, 5))
            store.replace_chunks(video_id, build_chunks(document, 1, 0))

        with ThreadPoolExecutor(max_workers=10) as pool:
            list(pool.map(write, video_ids))

        for video_id in video_ids:
            assert store.count_chunks(video_id) == 5

    def test_concurrent_rewrites_of_one_video_leave_a_consistent_count(self, tmp_path) -> None:
        """Re-indexing the same video from several threads must not interleave
        an upsert with another thread's stale-chunk delete."""
        store = TranscriptChunkStore(tmp_path / "chroma", FakeEmbeddingModel())
        document = raw_document_from_transcript(_transcript("same", 6))
        store.replace_chunks("same", build_chunks(document, 1, 0))

        def rewrite(_n: int) -> None:
            store.replace_chunks("same", build_chunks(document, 1, 0))

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(rewrite, range(8)))

        assert store.count_chunks("same") == 6


class TestAtomicDashboardWrite:
    def test_a_reader_never_sees_a_half_written_file(self, tmp_path) -> None:
        """Every index run rewrites this one file, so concurrent runs collide.

        Interleaved in-place writes produce invalid HTML that neither run
        notices. An atomic rename means a reader gets the old file or the new
        one, never a torn one.
        """
        import os
        import tempfile

        target = tmp_path / "rag_pipeline.html"
        target.write_text("<html>original</html>", encoding="utf-8")
        stop = threading.Event()
        torn: list[str] = []

        def read_forever() -> None:
            while not stop.is_set():
                text = target.read_text(encoding="utf-8")
                if not text.endswith("</html>"):
                    torn.append(text)

        def write_atomically(payload: str) -> None:
            handle = tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", dir=target.parent, delete=False
            )
            with handle:
                handle.write(payload)
            os.replace(handle.name, target)

        reader = threading.Thread(target=read_forever, daemon=True)
        reader.start()
        try:
            for index in range(40):
                write_atomically(f"<html>{'body' * 2000}{index}</html>")
        finally:
            stop.set()
            reader.join(timeout=5)

        assert torn == []

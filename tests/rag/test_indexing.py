"""Re-indexing must leave a video with exactly the chunks it now has.

The failure these tests exist for is real and was found in the live store: a
video re-chunked from 67 chunks to 63 kept ``chunk:<video>:63`` through
``:66`` from the previous run, because chunk ids are positional and an upsert
only ever overwrites the ids it is given. The orphans stayed indexed, stayed
retrievable, and kept answering with text that was no longer in the transcript.
"""

from __future__ import annotations

from datetime import datetime, timezone

from src.rag.indexing import RagIndexer
from src.rag.storage import RawTranscriptStore, TranscriptChunkStore
from src.transcripts.models import Transcript, TranscriptSegment

VIDEO_ID = "3hk7nO_q0a8"
SOURCE_URL = f"https://www.youtube.com/watch?v={VIDEO_ID}"


class FakeEmbeddingModel:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float("stale" in text.lower()), float("fresh" in text.lower()), 1.0]


class FakeFetcher:
    """Returns a different transcript on each ``fetch``, oldest first."""

    def __init__(self, transcripts: list[Transcript]) -> None:
        self.transcripts = list(transcripts)
        self.calls = 0

    def fetch(self, source_url: str) -> Transcript:
        transcript = self.transcripts[min(self.calls, len(self.transcripts) - 1)]
        self.calls += 1
        return transcript

    def fetch_metadata(self, source_url: str) -> dict:
        return {}


def _transcript(texts: list[str]) -> Transcript:
    return Transcript(
        video_id=VIDEO_ID,
        url=SOURCE_URL,
        title="Fixture video",
        channel_id="channel-1",
        channel_name="Fixture channel",
        raw_text=" ".join(texts),
        segments=[
            TranscriptSegment(
                text=text,
                start_seconds=float(index * 10),
                end_seconds=float(index * 10 + 10),
            )
            for index, text in enumerate(texts)
        ],
        fetched_at=datetime(2026, 8, 11, tzinfo=timezone.utc),
    )


def _indexer(tmp_path, fetcher: FakeFetcher) -> tuple[RagIndexer, TranscriptChunkStore]:
    raw_store = RawTranscriptStore(tmp_path / "chroma", fetcher=fetcher)
    chunk_store = TranscriptChunkStore(tmp_path / "chroma", FakeEmbeddingModel())
    # One segment per chunk, so chunk count is segment count and the shrink is
    # exact rather than an artefact of where the packer happened to split.
    indexer = RagIndexer(
        raw_store=raw_store,
        chunk_store=chunk_store,
        target_chars=1,
        overlap_chars=0,
    )
    return indexer, chunk_store


def test_refresh_removes_chunks_the_shorter_rechunk_no_longer_produces(tmp_path) -> None:
    fetcher = FakeFetcher(
        [
            _transcript([f"stale segment {index}" for index in range(6)]),
            _transcript([f"fresh segment {index}" for index in range(3)]),
        ]
    )
    indexer, chunk_store = _indexer(tmp_path, fetcher)

    first = indexer.index(SOURCE_URL)
    assert len(first.chunks) == 6
    assert first.removed_chunk_ids == []

    second = indexer.index(SOURCE_URL, refresh=True)

    assert len(second.chunks) == 3
    assert second.removed_chunk_ids == [
        f"chunk:{VIDEO_ID}:3",
        f"chunk:{VIDEO_ID}:4",
        f"chunk:{VIDEO_ID}:5",
    ]
    # Nothing from the previous run survives: not as an id, not as a count, and
    # — the part that actually corrupted retrieval — not as a returnable chunk.
    assert chunk_store.chunk_ids_for_video(VIDEO_ID) == [
        f"chunk:{VIDEO_ID}:0",
        f"chunk:{VIDEO_ID}:1",
        f"chunk:{VIDEO_ID}:2",
    ]
    assert chunk_store.count_chunks(VIDEO_ID) == 3
    orphans = chunk_store.collection.get(ids=second.removed_chunk_ids, include=["documents"])
    assert orphans["ids"] == []
    retrieved = chunk_store.query_by_video_id(VIDEO_ID, "stale segment", top_k=50)
    assert [chunk.chunk_index for chunk in retrieved] == [0, 1, 2]
    assert all("stale" not in chunk.text for chunk in retrieved)


def test_refresh_leaves_other_videos_alone(tmp_path) -> None:
    fetcher = FakeFetcher(
        [
            _transcript([f"stale segment {index}" for index in range(6)]),
            _transcript([f"fresh segment {index}" for index in range(2)]),
        ]
    )
    indexer, chunk_store = _indexer(tmp_path, fetcher)
    other = _transcript(["stale segment 0"]).model_copy(update={"video_id": "otherVideoId"})
    from src.rag.chunking import build_chunks
    from src.rag.storage import raw_document_from_transcript

    chunk_store.upsert_chunks(build_chunks(raw_document_from_transcript(other), 1, 0))

    indexer.index(SOURCE_URL)
    indexer.index(SOURCE_URL, refresh=True)

    assert chunk_store.count_chunks("otherVideoId") == 1


def test_growing_rechunk_removes_nothing(tmp_path) -> None:
    fetcher = FakeFetcher(
        [
            _transcript([f"fresh segment {index}" for index in range(2)]),
            _transcript([f"fresh segment {index}" for index in range(5)]),
        ]
    )
    indexer, chunk_store = _indexer(tmp_path, fetcher)

    indexer.index(SOURCE_URL)
    second = indexer.index(SOURCE_URL, refresh=True)

    assert second.removed_chunk_ids == []
    assert chunk_store.count_chunks(VIDEO_ID) == 5


def test_rebuild_to_zero_chunks_does_not_wipe_the_video(tmp_path) -> None:
    """An empty rebuild is a suspect fetch, not a video that lost its transcript."""
    fetcher = FakeFetcher(
        [
            _transcript([f"fresh segment {index}" for index in range(4)]),
            _transcript([]),
        ]
    )
    indexer, chunk_store = _indexer(tmp_path, fetcher)

    indexer.index(SOURCE_URL)
    second = indexer.index(SOURCE_URL, refresh=True)

    assert second.chunks == []
    assert second.removed_chunk_ids == []
    assert chunk_store.count_chunks(VIDEO_ID) == 4

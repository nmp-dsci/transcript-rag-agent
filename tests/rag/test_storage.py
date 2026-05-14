from __future__ import annotations

from datetime import datetime, timezone

from src.rag.models import TranscriptChunk
from src.rag.storage import (
    RawTranscriptStore,
    TranscriptChunkStore,
    raw_document_from_transcript,
)
from src.transcripts.models import Transcript, TranscriptSegment


class FakeEmbeddingModel:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float("capital" in text.lower()), float("agent" in text.lower()), 1.0]


def test_raw_transcript_store_serializes_segments_in_document_body(tmp_path) -> None:
    transcript = Transcript(
        video_id="3hk7nO_q0a8",
        url="https://www.youtube.com/watch?v=3hk7nO_q0a8",
        raw_text="hello world",
        segments=[
            TranscriptSegment(
                text="hello",
                offset_ms=8150,
                duration_ms=1200,
                start_seconds=8.15,
                end_seconds=9.35,
                language="en",
            )
        ],
        fetched_at=datetime.now(timezone.utc),
    )
    store = RawTranscriptStore(tmp_path / "chroma")
    document = raw_document_from_transcript(transcript)

    store.upsert_raw_document(document)
    loaded = store.get_raw_document(transcript.video_id)
    stored = store.collection.get(ids=[document.transcript_id], include=["documents", "metadatas"])

    assert loaded is not None
    assert loaded.segments[0].offset_ms == 8150
    assert loaded.segments[0].duration_ms == 1200
    assert "segments" in stored["documents"][0]
    assert "segments" not in stored["metadatas"][0]


def test_chunk_store_queries_top_k_with_metadata(tmp_path) -> None:
    store = TranscriptChunkStore(tmp_path / "chroma", FakeEmbeddingModel())
    chunks = [
        TranscriptChunk(
            transcript_id="raw_transcript:video",
            video_id="video",
            source_url="https://www.youtube.com/watch?v=video",
            chunk_index=0,
            text="agent systems",
            start_seconds=1,
            end_seconds=2,
            start_segment_index=0,
            end_segment_index=0,
            segment_count=1,
        ),
        TranscriptChunk(
            transcript_id="raw_transcript:video",
            video_id="video",
            source_url="https://www.youtube.com/watch?v=video",
            chunk_index=1,
            text="capital gains tax",
            start_seconds=10,
            end_seconds=12,
            start_segment_index=1,
            end_segment_index=1,
            segment_count=1,
        ),
    ]

    store.upsert_chunks(chunks)
    retrieved = store.query("video", "capital gains", top_k=1)

    assert store.has_chunks("video")
    assert len(retrieved) == 1
    assert retrieved[0].chunk_index == 1
    assert retrieved[0].start_seconds == 10


def test_chunk_store_query_all_searches_across_videos(tmp_path) -> None:
    store = TranscriptChunkStore(tmp_path / "chroma", FakeEmbeddingModel())
    store.upsert_chunks(
        [
            TranscriptChunk(
                transcript_id="raw_transcript:aaaaaaaaaaa",
                video_id="aaaaaaaaaaa",
                source_url="https://www.youtube.com/watch?v=aaaaaaaaaaa",
                chunk_index=0,
                text="agent systems",
                segment_count=1,
            ),
            TranscriptChunk(
                transcript_id="raw_transcript:bbbbbbbbbbb",
                video_id="bbbbbbbbbbb",
                source_url="https://www.youtube.com/watch?v=bbbbbbbbbbb",
                chunk_index=0,
                text="capital gains tax",
                segment_count=1,
            ),
        ]
    )

    retrieved = store.query_all("capital gains", top_k=2)

    assert {chunk.video_id for chunk in retrieved} == {"aaaaaaaaaaa", "bbbbbbbbbbb"}


def test_chunk_store_query_by_url_filters_to_one_video(tmp_path) -> None:
    store = TranscriptChunkStore(tmp_path / "chroma", FakeEmbeddingModel())
    store.upsert_chunks(
        [
            TranscriptChunk(
                transcript_id="raw_transcript:aaaaaaaaaaa",
                video_id="aaaaaaaaaaa",
                source_url="https://www.youtube.com/watch?v=aaaaaaaaaaa",
                chunk_index=0,
                text="capital gains tax",
                segment_count=1,
            ),
            TranscriptChunk(
                transcript_id="raw_transcript:bbbbbbbbbbb",
                video_id="bbbbbbbbbbb",
                source_url="https://www.youtube.com/watch?v=bbbbbbbbbbb",
                chunk_index=0,
                text="capital gains tax",
                segment_count=1,
            ),
        ]
    )

    retrieved = store.query_by_url("https://www.youtube.com/watch?v=bbbbbbbbbbb", "capital", 10)

    assert {chunk.video_id for chunk in retrieved} == {"bbbbbbbbbbb"}

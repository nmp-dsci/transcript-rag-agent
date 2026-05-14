from __future__ import annotations

from src.rag.chunking import build_chunks, format_timestamp
from src.rag.models import RawTranscriptDocument, RawTranscriptSegment


def test_chunking_preserves_order_timestamps_and_segment_indexes() -> None:
    document = RawTranscriptDocument(
        transcript_id="raw_transcript:video",
        video_id="video",
        source_url="https://www.youtube.com/watch?v=video",
        fetched_at="2026-05-14T00:00:00+00:00",
        segments=[
            RawTranscriptSegment(text="alpha", start_seconds=0, end_seconds=1),
            RawTranscriptSegment(text="beta", start_seconds=1, end_seconds=2),
            RawTranscriptSegment(text="gamma", start_seconds=2, end_seconds=3),
        ],
    )

    chunks = build_chunks(document, target_chars=11, overlap_chars=0)

    assert [chunk.text for chunk in chunks] == ["alpha beta", "gamma"]
    assert chunks[0].chunk_id == "chunk:video:0"
    assert chunks[0].start_seconds == 0
    assert chunks[0].end_seconds == 2
    assert chunks[0].start_segment_index == 0
    assert chunks[0].end_segment_index == 1


def test_format_timestamp() -> None:
    assert format_timestamp(None) == "unknown"
    assert format_timestamp(754) == "12:34"
    assert format_timestamp(3723) == "01:02:03"

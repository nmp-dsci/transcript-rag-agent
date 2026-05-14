from __future__ import annotations

from src.rag.models import RetrievedChunk
from src.rag.references import format_chunk_reference, youtube_timestamp_url


def test_youtube_timestamp_url_adds_seconds() -> None:
    assert (
        youtube_timestamp_url("https://www.youtube.com/watch?v=abc", 593.6)
        == "https://www.youtube.com/watch?v=abc&t=593s"
    )


def test_youtube_timestamp_url_preserves_base_when_unknown() -> None:
    assert youtube_timestamp_url("https://www.youtube.com/watch?v=abc", None) == (
        "https://www.youtube.com/watch?v=abc"
    )


def test_format_chunk_reference_includes_url_and_time() -> None:
    chunk = RetrievedChunk(
        transcript_id="raw_transcript:abc",
        video_id="abc",
        source_url="https://www.youtube.com/watch?v=abc",
        chunk_index=1,
        text="text",
        start_seconds=593,
        end_seconds=665,
        segment_count=1,
    )

    reference = format_chunk_reference(1, chunk)

    assert "[1]" in reference
    assert "09:53-11:05" in reference
    assert "https://www.youtube.com/watch?v=abc&t=593s" in reference

from __future__ import annotations

from src.transcripts.fetcher import SuperdataTranscriptFetcher


def test_normalizes_supadata_segment_response() -> None:
    fetcher = SuperdataTranscriptFetcher("key")

    transcript = fetcher._normalize_response(
        url="https://www.youtube.com/watch?v=3hk7nO_q0a8",
        video_id="3hk7nO_q0a8",
        data={
            "content": [
                {"text": "hello", "offset": 0, "duration": 1000, "lang": "en"},
                {"text": "world", "offset": 1000, "duration": 1000, "lang": "en"},
            ],
            "lang": "en",
        },
    )

    assert transcript.raw_text == "hello world"
    assert transcript.segments[0].offset_ms == 0
    assert transcript.segments[0].duration_ms == 1000
    assert transcript.segments[0].start_seconds == 0
    assert transcript.segments[0].end_seconds == 1
    assert transcript.segments[0].language == "en"
    assert transcript.language == "en"


def test_normalizes_supadata_text_response() -> None:
    fetcher = SuperdataTranscriptFetcher("key")

    transcript = fetcher._normalize_response(
        url="https://www.youtube.com/watch?v=3hk7nO_q0a8",
        video_id="3hk7nO_q0a8",
        data={"content": "plain transcript", "lang": "en"},
    )

    assert transcript.raw_text == "plain transcript"
    assert transcript.segments == []


def test_normalizes_supadata_metadata_response() -> None:
    fetcher = SuperdataTranscriptFetcher("key")

    transcript = fetcher._normalize_response(
        url="https://www.youtube.com/watch?v=3hk7nO_q0a8",
        video_id="3hk7nO_q0a8",
        data={"content": "plain transcript", "lang": "en"},
        metadata={
            "title": "Video title",
            "description": "Video description",
            "author": {"displayName": "Channel name"},
            "stats": {"views": 123, "likes": 45},
            "media": {
                "duration": 90,
                "thumbnailUrl": "https://i.ytimg.com/vi/3hk7nO_q0a8/hqdefault.jpg",
            },
            "tags": ["tax", "property"],
            "createdAt": "2026-05-16T00:00:00Z",
            "additionalData": {
                "channelId": "channel-1",
                "transcriptLanguages": ["en", "es"],
            },
        },
    )

    assert transcript.title == "Video title"
    assert transcript.channel_name == "Channel name"
    assert transcript.channel_id == "channel-1"
    assert transcript.duration_seconds == 90
    assert transcript.view_count == 123
    assert transcript.like_count == 45
    assert transcript.tags == ["tax", "property"]
    assert transcript.transcript_languages == ["en", "es"]

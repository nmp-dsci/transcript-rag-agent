from __future__ import annotations

import pytest

from src.transcripts.youtube import YouTubeUrlError, extract_video_id, video_id_from_url


def test_extracts_video_id_from_watch_url() -> None:
    assert (
        extract_video_id("https://www.youtube.com/watch?v=3hk7nO_q0a8")
        == "3hk7nO_q0a8"
    )


def test_extracts_video_id_from_short_url() -> None:
    assert extract_video_id("https://youtu.be/3hk7nO_q0a8") == "3hk7nO_q0a8"


def test_rejects_invalid_url() -> None:
    with pytest.raises(YouTubeUrlError):
        extract_video_id("https://example.com/nope")


@pytest.mark.parametrize(
    "value",
    [
        "https://www.youtube.com/watch?v=ruiLq0OzjkI&pp=ygUNZ3VhcmRyYWlscyBhaQ%3D%3D",
        "youtube.com/watch?v=ruiLq0OzjkI",
        "https://youtu.be/ruiLq0OzjkI",
        "https://www.youtube.com/shorts/ruiLq0OzjkI",
        "  https://m.youtube.com/watch?v=ruiLq0OzjkI  ",
    ],
)
def test_video_id_from_url_accepts_video_urls(value: str) -> None:
    assert video_id_from_url(value) == "ruiLq0OzjkI"


@pytest.mark.parametrize(
    "value",
    [
        "@raycfu",
        "https://www.youtube.com/@raycfu",
        "https://www.youtube.com/channel/UCxxxxxxxxxxxxxxxxxxxxxx",
        "youtube.com/@raycfu/videos",
        # A bare handle can be 11 characters too; only a URL counts as a video.
        "abcdefghijk",
        "",
    ],
)
def test_video_id_from_url_rejects_channels_and_bare_ids(value: str) -> None:
    assert video_id_from_url(value) is None

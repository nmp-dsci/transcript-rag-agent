from __future__ import annotations

import httpx

from src.transcripts.discovery import SupadataDiscoveryClient


def test_discover_latest_channel_videos_normalizes_ids(monkeypatch, tmp_path) -> None:
    def fake_get(url, params, headers, timeout):
        assert url.endswith("/youtube/channel/videos")
        assert params["id"] == "@channel"
        return httpx.Response(200, json={"videoIds": ["aaaaaaaaaaa", "bbbbbbbbbbb"]})

    monkeypatch.setattr(httpx, "get", fake_get)
    client = SupadataDiscoveryClient("key", cache_dir=tmp_path)

    videos = client.discover_latest_channel_videos("@channel", limit=2)

    assert [video.video_id for video in videos] == ["aaaaaaaaaaa", "bbbbbbbbbbb"]
    assert str(videos[0].source_url) == "https://www.youtube.com/watch?v=aaaaaaaaaaa"


def test_discover_search_results_discards_non_video_items(monkeypatch, tmp_path) -> None:
    def fake_get(url, params, headers, timeout):
        assert url.endswith("/youtube/search")
        return httpx.Response(
            200,
            json={
                "items": [
                    {"type": "channel", "id": "not-a-video", "title": "Channel"},
                    {
                        "type": "video",
                        "videoId": "ccccccccccc",
                        "title": "Video",
                        "channelName": "Channel name",
                    },
                ]
            },
        )

    monkeypatch.setattr(httpx, "get", fake_get)
    client = SupadataDiscoveryClient("key", cache_dir=tmp_path)

    videos = client.discover_search_results("query", top_n=1)

    assert len(videos) == 1
    assert videos[0].video_id == "ccccccccccc"
    assert videos[0].title == "Video"

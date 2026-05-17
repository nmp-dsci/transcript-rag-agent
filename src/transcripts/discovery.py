from __future__ import annotations

import hashlib
import json
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, HttpUrl

from src.transcripts.youtube import extract_video_id


class DiscoveryError(RuntimeError):
    pass


class DiscoveredVideo(BaseModel):
    video_id: str
    source_url: HttpUrl
    title: str | None = None
    channel_id: str | None = None
    channel_name: str | None = None
    published_at: str | None = None
    duration_seconds: float | None = None


class SupadataDiscoveryClient:
    base_url = "https://api.supadata.ai/v1"

    def __init__(
        self,
        api_key: str,
        timeout_seconds: float = 120.0,
        cache_dir: Path | str | None = None,
        cache_ttl_hours: float = 24.0,
        use_cache: bool = True,
    ) -> None:
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.cache_dir = Path(cache_dir or ".yt-agent/discovery_cache")
        self.cache_ttl = timedelta(hours=cache_ttl_hours)
        self.use_cache = use_cache

    def discover_latest_channel_videos(
        self,
        channel: str,
        limit: int = 5,
    ) -> list[DiscoveredVideo]:
        return self.discover_channel_videos(channel, max_results=limit)[:limit]

    def discover_channel_videos(
        self,
        channel: str,
        published_after: date | None = None,
        published_before: date | None = None,
        max_results: int = 50,
    ) -> list[DiscoveredVideo]:
        params = {"id": channel, "limit": max_results, "type": "video"}
        data = self._request("youtube/channel/videos", params=params)
        ids = data.get("videoIds") if isinstance(data, dict) else []
        videos = [
            _video_from_id(str(video_id))
            for video_id in ids or []
            if _looks_like_discoverable_video_id(str(video_id))
        ]
        if published_after is None and published_before is None:
            return videos[:max_results]

        filtered: list[DiscoveredVideo] = []
        for video in videos:
            metadata = self.fetch_metadata(video.video_id)
            enriched = _video_from_metadata(video.video_id, metadata) or video
            published = _parse_date(enriched.published_at)
            if published is None:
                continue
            if published_after is not None and published < published_after:
                break
            if published_before is not None and published > published_before:
                continue
            filtered.append(enriched)
            if len(filtered) >= max_results:
                break
        return filtered

    def discover_search_results(self, query: str, top_n: int = 10) -> list[DiscoveredVideo]:
        videos: list[DiscoveredVideo] = []
        next_page_token: str | None = None
        while len(videos) < top_n:
            params: dict[str, Any] = {"query": query, "limit": top_n - len(videos)}
            if next_page_token:
                params["nextPageToken"] = next_page_token
            data = self._request("youtube/search", params=params)
            batch, next_page_token = _videos_from_search_response(data)
            videos.extend(batch)
            if not next_page_token or not batch:
                break
        return videos[:top_n]

    def fetch_metadata(self, video_id: str) -> dict[str, Any]:
        return self._request(
            "metadata",
            params={"url": _youtube_url(video_id)},
            cache_namespace="metadata",
        )

    def _request(
        self,
        path: str,
        params: dict[str, Any],
        cache_namespace: str | None = None,
    ) -> dict[str, Any]:
        cache_path = self._cache_path(cache_namespace or path, params)
        if self.use_cache:
            cached = self._read_cache(cache_path)
            if cached is not None:
                return cached
        headers = {"x-api-key": self.api_key}
        url = f"{self.base_url}/{path.lstrip('/')}"
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                response = httpx.get(
                    url,
                    params=params,
                    headers=headers,
                    timeout=self.timeout_seconds,
                )
                if response.status_code == 429 and attempt < 2:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                if response.status_code >= 400:
                    raise DiscoveryError(
                        f"Supadata discovery failed with HTTP {response.status_code}: "
                        f"{response.text}"
                    )
                data = response.json()
                if not isinstance(data, dict):
                    raise DiscoveryError("Supadata discovery returned a non-object response")
                self._write_cache(cache_path, data)
                return data
            except httpx.HTTPError as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep(0.5 * (attempt + 1))
                    continue
        raise DiscoveryError(f"Supadata discovery request failed: {last_error}")

    def _cache_path(self, namespace: str, params: dict[str, Any]) -> Path:
        payload = json.dumps(params, sort_keys=True, default=str)
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
        safe_namespace = namespace.replace("/", "_")
        return self.cache_dir / safe_namespace / f"{digest}.json"

    def _read_cache(self, path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        cached_at = _parse_datetime(data.get("cached_at"))
        if cached_at is None or datetime.now(timezone.utc) - cached_at > self.cache_ttl:
            return None
        payload = data.get("payload")
        return payload if isinstance(payload, dict) else None

    def _write_cache(self, path: Path, payload: dict[str, Any]) -> None:
        if not self.use_cache:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "cached_at": datetime.now(timezone.utc).isoformat(),
                    "payload": payload,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )


def discover_channel_videos(
    channel: str,
    published_after: date | None = None,
    published_before: date | None = None,
    max_results: int = 50,
    *,
    client: SupadataDiscoveryClient,
) -> list[DiscoveredVideo]:
    return client.discover_channel_videos(
        channel,
        published_after=published_after,
        published_before=published_before,
        max_results=max_results,
    )


def discover_latest_channel_videos(
    channel: str,
    limit: int = 5,
    *,
    client: SupadataDiscoveryClient,
) -> list[DiscoveredVideo]:
    return client.discover_latest_channel_videos(channel, limit=limit)


def discover_search_results(
    query: str,
    top_n: int = 10,
    *,
    client: SupadataDiscoveryClient,
) -> list[DiscoveredVideo]:
    return client.discover_search_results(query, top_n=top_n)


def _videos_from_search_response(data: dict[str, Any]) -> tuple[list[DiscoveredVideo], str | None]:
    items = (
        data.get("items")
        or data.get("results")
        or data.get("videos")
        or data.get("data")
        or []
    )
    if isinstance(items, dict):
        items = items.get("items") or items.get("results") or []
    videos: list[DiscoveredVideo] = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("type") or item.get("kind") or item.get("resultType") or "").lower()
        if kind and "video" not in kind:
            continue
        video_id = _extract_search_video_id(item)
        if not video_id:
            continue
        videos.append(_video_from_metadata(video_id, item) or _video_from_id(video_id))
    next_page_token = data.get("nextPageToken") or data.get("next_page_token")
    return videos, str(next_page_token) if next_page_token else None


def _extract_search_video_id(item: dict[str, Any]) -> str | None:
    candidates = [
        item.get("videoId"),
        item.get("video_id"),
        item.get("id"),
        item.get("url"),
        item.get("link"),
    ]
    id_value = item.get("id")
    if isinstance(id_value, dict):
        candidates.extend([id_value.get("videoId"), id_value.get("id")])
    for candidate in candidates:
        if candidate is None:
            continue
        text = str(candidate)
        try:
            return extract_video_id(text)
        except ValueError:
            if _looks_like_discoverable_video_id(text):
                return text
    return None


def _video_from_id(video_id: str) -> DiscoveredVideo:
    return DiscoveredVideo(video_id=video_id, source_url=HttpUrl(_youtube_url(video_id)))


def _video_from_metadata(video_id: str, metadata: dict[str, Any]) -> DiscoveredVideo | None:
    if not metadata:
        return None
    author = metadata.get("author") if isinstance(metadata.get("author"), dict) else {}
    channel = metadata.get("channel") if isinstance(metadata.get("channel"), dict) else {}
    additional = (
        metadata.get("additionalData")
        if isinstance(metadata.get("additionalData"), dict)
        else {}
    )
    media = metadata.get("media") if isinstance(metadata.get("media"), dict) else {}
    return DiscoveredVideo(
        video_id=video_id,
        source_url=HttpUrl(str(metadata.get("url") or metadata.get("link") or _youtube_url(video_id))),
        title=_str_or_none(metadata.get("title")),
        channel_id=_str_or_none(
            channel.get("id") or additional.get("channelId") or additional.get("channel_id") or author.get("id")
        ),
        channel_name=_str_or_none(
            channel.get("name") or author.get("displayName") or author.get("username") or metadata.get("channelName")
        ),
        published_at=_str_or_none(
            metadata.get("createdAt") or metadata.get("publishedAt") or metadata.get("published_at")
        ),
        duration_seconds=_float_or_none(media.get("duration") or metadata.get("duration")),
    )


def _youtube_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"


def _parse_date(value: str | None) -> date | None:
    parsed = _parse_datetime(value)
    return parsed.date() if parsed is not None else None


def _parse_datetime(value: object) -> datetime | None:
    if not value:
        return None
    try:
        text = str(value).replace("Z", "+00:00")
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _looks_like_discoverable_video_id(value: str) -> bool:
    return bool(value) and all(char.isalnum() or char in {"_", "-"} for char in value)


def _str_or_none(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text or None


def _float_or_none(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

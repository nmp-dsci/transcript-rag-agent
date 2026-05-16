from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

import httpx
from pydantic import HttpUrl, ValidationError

from src.transcripts.models import Transcript, TranscriptSegment
from src.transcripts.youtube import extract_video_id


class TranscriptFetchError(RuntimeError):
    pass


class SuperdataTranscriptFetcher:
    """Fetch transcripts from Supadata while preserving the spec's env naming."""

    endpoint = "https://api.supadata.ai/v1/transcript"
    metadata_endpoint = "https://api.supadata.ai/v1/metadata"

    def __init__(
        self,
        api_key: str,
        timeout_seconds: float = 120.0,
        poll_interval_seconds: float = 2.0,
        max_poll_seconds: float = 600.0,
    ) -> None:
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.poll_interval_seconds = poll_interval_seconds
        self.max_poll_seconds = max_poll_seconds

    def fetch(self, url: str) -> Transcript:
        video_id = extract_video_id(url)
        data = self._request_transcript(url)
        metadata = self._request_metadata(url)
        return self._normalize_response(
            url=url,
            video_id=video_id,
            data=data,
            metadata=metadata,
        )

    def fetch_metadata(self, url: str) -> dict[str, Any]:
        return self._request_metadata(url)

    def _request_transcript(self, url: str) -> dict[str, Any]:
        headers = {"x-api-key": self.api_key}
        params: dict[str, Any] = {"url": url, "text": "false", "mode": "auto"}
        try:
            response = httpx.get(
                self.endpoint,
                params=params,
                headers=headers,
                timeout=self.timeout_seconds,
            )
        except httpx.HTTPError as exc:
            raise TranscriptFetchError(f"Supadata transcript request failed: {exc}") from exc

        if response.status_code == 202:
            job_id = response.json().get("jobId")
            if not job_id:
                raise TranscriptFetchError("Supadata returned 202 without jobId")
            return self._poll_job(job_id)

        if response.status_code >= 400:
            raise TranscriptFetchError(
                f"Supadata transcript request failed with HTTP {response.status_code}: "
                f"{response.text}"
            )
        return response.json()

    def _request_metadata(self, url: str) -> dict[str, Any]:
        headers = {"x-api-key": self.api_key}
        try:
            response = httpx.get(
                self.metadata_endpoint,
                params={"url": url},
                headers=headers,
                timeout=self.timeout_seconds,
            )
        except httpx.HTTPError:
            return {}
        if response.status_code >= 400:
            return {}
        data = response.json()
        return data if isinstance(data, dict) else {}

    def _poll_job(self, job_id: str) -> dict[str, Any]:
        deadline = time.monotonic() + self.max_poll_seconds
        url = f"{self.endpoint}/{job_id}"
        headers = {"x-api-key": self.api_key}
        while time.monotonic() < deadline:
            response = httpx.get(url, headers=headers, timeout=self.timeout_seconds)
            if response.status_code >= 400:
                raise TranscriptFetchError(
                    f"Supadata job status failed with HTTP {response.status_code}: "
                    f"{response.text}"
                )
            data = response.json()
            status = data.get("status")
            if status == "completed":
                return data
            if status == "failed":
                raise TranscriptFetchError(f"Supadata transcript job failed: {data}")
            time.sleep(self.poll_interval_seconds)
        raise TranscriptFetchError(f"Supadata transcript job timed out: {job_id}")

    def _normalize_response(
        self,
        url: str,
        video_id: str,
        data: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> Transcript:
        metadata = metadata or {}
        content = data.get("content") or data.get("result")
        language = data.get("lang") or data.get("language")
        segments: list[TranscriptSegment] = []

        if isinstance(content, str):
            raw_text = content.strip()
        elif isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    continue
                text = str(item.get("text", "")).strip()
                if not text:
                    continue
                offset_ms = item.get("offset")
                duration_ms = item.get("duration")
                start_seconds = (
                    float(offset_ms) / 1000 if isinstance(offset_ms, int | float) else None
                )
                end_seconds = None
                if start_seconds is not None and isinstance(duration_ms, int | float):
                    end_seconds = start_seconds + (float(duration_ms) / 1000)
                segments.append(
                    TranscriptSegment(
                        text=text,
                        offset_ms=int(offset_ms) if isinstance(offset_ms, int | float) else None,
                        duration_ms=(
                            int(duration_ms) if isinstance(duration_ms, int | float) else None
                        ),
                        start_seconds=start_seconds,
                        end_seconds=end_seconds,
                        language=item.get("lang"),
                    )
                )
            raw_text = " ".join(segment.text for segment in segments).strip()
            if not language and content:
                first = next((item for item in content if isinstance(item, dict)), {})
                language = first.get("lang")
        else:
            raise TranscriptFetchError("Supadata response did not include transcript content")

        if not raw_text:
            raise TranscriptFetchError("Supadata returned an empty transcript")

        author = metadata.get("author") if isinstance(metadata.get("author"), dict) else {}
        media = metadata.get("media") if isinstance(metadata.get("media"), dict) else {}
        additional = (
            metadata.get("additionalData")
            if isinstance(metadata.get("additionalData"), dict)
            else {}
        )
        channel = metadata.get("channel") if isinstance(metadata.get("channel"), dict) else {}
        tags = metadata.get("tags") if isinstance(metadata.get("tags"), list) else []
        transcript_languages = (
            metadata.get("transcriptLanguages")
            if isinstance(metadata.get("transcriptLanguages"), list)
            else additional.get("transcriptLanguages")
            if isinstance(additional.get("transcriptLanguages"), list)
            else []
        )
        return Transcript(
            video_id=video_id,
            url=HttpUrl(url),
            title=metadata.get("title") or data.get("title"),
            description=_str_or_none(metadata.get("description")),
            channel_id=_str_or_none(
                channel.get("id")
                or additional.get("channelId")
                or author.get("id")
            ),
            channel_name=_str_or_none(
                channel.get("name")
                or author.get("displayName")
                or author.get("username")
            ),
            duration_seconds=_float_or_none(media.get("duration") or metadata.get("duration")),
            thumbnail_url=_http_url_or_none(
                media.get("thumbnailUrl")
                or metadata.get("thumbnail")
                or metadata.get("thumbnailUrl")
            ),
            upload_date=_str_or_none(metadata.get("uploadDate") or metadata.get("createdAt")),
            view_count=_int_or_none(_nested(metadata, "stats", "views") or metadata.get("viewCount")),
            like_count=_int_or_none(_nested(metadata, "stats", "likes") or metadata.get("likeCount")),
            tags=[str(tag) for tag in tags],
            transcript_languages=[str(lang) for lang in transcript_languages],
            language=language,
            provider="supadata",
            raw_text=raw_text,
            segments=segments,
            fetched_at=datetime.now(timezone.utc),
        )


def _nested(data: dict[str, Any], parent: str, child: str) -> Any:
    value = data.get(parent)
    if not isinstance(value, dict):
        return None
    return value.get(child)


def _str_or_none(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text or None


def _int_or_none(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_or_none(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _http_url_or_none(value: object) -> HttpUrl | None:
    text = _str_or_none(value)
    if text is None:
        return None
    try:
        return HttpUrl(text)
    except ValidationError:
        return None

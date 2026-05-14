from __future__ import annotations

from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from src.rag.chunking import format_timestamp
from src.rag.models import RetrievedChunk


def youtube_timestamp_url(source_url: str, seconds: float | None) -> str:
    if seconds is None:
        return source_url
    parsed = urlparse(source_url)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query["t"] = f"{max(0, int(seconds))}s"
    return urlunparse(parsed._replace(query=urlencode(query)))


def format_chunk_reference(index: int, chunk: RetrievedChunk) -> str:
    start = format_timestamp(chunk.start_seconds)
    end = format_timestamp(chunk.end_seconds)
    timestamp_url = youtube_timestamp_url(str(chunk.source_url), chunk.start_seconds)
    return (
        f"[{index}] video={chunk.video_id} time={start}-{end} "
        f"url={timestamp_url}"
    )

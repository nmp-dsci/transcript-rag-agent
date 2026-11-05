from __future__ import annotations

from urllib.parse import parse_qs, urlparse


class YouTubeUrlError(ValueError):
    pass


def extract_video_id(url_or_id: str) -> str:
    value = url_or_id.strip()
    if not value:
        raise YouTubeUrlError("YouTube URL or video ID is required")

    if _looks_like_video_id(value):
        return value

    parsed = urlparse(value)
    host = parsed.netloc.lower()
    path_parts = [part for part in parsed.path.split("/") if part]

    if host in {"youtube.com", "www.youtube.com", "m.youtube.com"}:
        query_id = parse_qs(parsed.query).get("v", [""])[0]
        if _looks_like_video_id(query_id):
            return query_id
        if path_parts and path_parts[0] in {"shorts", "embed", "live"}:
            candidate = path_parts[1] if len(path_parts) > 1 else ""
            if _looks_like_video_id(candidate):
                return candidate

    if host in {"youtu.be", "www.youtu.be"} and path_parts:
        candidate = path_parts[0]
        if _looks_like_video_id(candidate):
            return candidate

    raise YouTubeUrlError(f"Could not extract YouTube video ID from: {url_or_id}")


def video_id_from_url(value: str) -> str | None:
    """The video ID when ``value`` is a YouTube *video URL*, else ``None``.

    Stricter than :func:`extract_video_id`: a bare 11-character string is not
    accepted, because a channel handle typed without its ``@`` can look exactly
    like a video ID. Used to catch a video URL pasted into a channel field.
    """
    text = value.strip()
    if not text or _looks_like_video_id(text):
        return None
    if "://" not in text:
        text = f"https://{text}"
    try:
        return extract_video_id(text)
    except YouTubeUrlError:
        return None


def _looks_like_video_id(value: str) -> bool:
    return len(value) == 11 and all(
        char.isalnum() or char in {"_", "-"} for char in value
    )

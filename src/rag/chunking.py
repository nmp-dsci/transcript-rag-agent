from __future__ import annotations

from src.rag.models import RawTranscriptDocument, RawTranscriptSegment, TranscriptChunk


def build_chunks(
    raw_document: RawTranscriptDocument,
    target_chars: int = 1200,
    overlap_chars: int = 150,
) -> list[TranscriptChunk]:
    segments = raw_document.segments
    if not segments:
        return []

    chunks: list[TranscriptChunk] = []
    start = 0
    while start < len(segments):
        end = start
        char_count = 0
        while end < len(segments):
            next_text = segments[end].text
            separator_chars = 1 if char_count else 0
            if end > start and char_count + separator_chars + len(next_text) > target_chars:
                break
            char_count += separator_chars + len(next_text)
            end += 1
        if end == start:
            end += 1

        chunk_segments = segments[start:end]
        chunks.append(_chunk(raw_document, len(chunks), start, end - 1, chunk_segments))

        if end >= len(segments):
            break
        start = _next_start_with_overlap(segments, start, end, overlap_chars)

    return chunks


def _next_start_with_overlap(
    segments: list[RawTranscriptSegment],
    current_start: int,
    current_end_exclusive: int,
    overlap_chars: int,
) -> int:
    if overlap_chars <= 0:
        return current_end_exclusive
    overlap_start = current_end_exclusive
    chars = 0
    while overlap_start > current_start:
        candidate = segments[overlap_start - 1].text
        if chars and chars + 1 + len(candidate) > overlap_chars:
            break
        if not chars and len(candidate) > overlap_chars:
            break
        chars += (1 if chars else 0) + len(candidate)
        overlap_start -= 1
    return max(overlap_start, current_start + 1)


def _chunk(
    raw_document: RawTranscriptDocument,
    chunk_index: int,
    start_index: int,
    end_index: int,
    segments: list[RawTranscriptSegment],
) -> TranscriptChunk:
    first = segments[0]
    last = segments[-1]
    return TranscriptChunk(
        transcript_id=raw_document.transcript_id,
        video_id=raw_document.video_id,
        source_url=raw_document.source_url,
        chunk_index=chunk_index,
        text=" ".join(segment.text for segment in segments).strip(),
        start_seconds=first.start_seconds,
        end_seconds=last.end_seconds,
        start_segment_index=start_index,
        end_segment_index=end_index,
        segment_count=len(segments),
        channel_id=raw_document.channel_id,
        channel_name=raw_document.channel_name,
        title=raw_document.title,
        upload_date=raw_document.upload_date,
        context_header=build_context_header(
            channel_name=raw_document.channel_name,
            title=raw_document.title,
            start_seconds=first.start_seconds,
            end_seconds=last.end_seconds,
        ),
    )


def build_context_header(
    channel_name: str | None,
    title: str | None,
    start_seconds: float | None,
    end_seconds: float | None,
) -> str | None:
    """A one-line preamble naming the source of a chunk, for embedding.

    Returns ``None`` when nothing identifying is known, so chunks without video
    metadata embed exactly as they did before contextual headers existed.
    """
    parts = [part for part in (channel_name, title) if part]
    if not parts:
        return None
    window = f"{format_timestamp(start_seconds)}-{format_timestamp(end_seconds)}"
    return f"[{' — '.join(parts)} @ {window}]"


def format_timestamp(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    total_seconds = max(0, int(seconds))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"

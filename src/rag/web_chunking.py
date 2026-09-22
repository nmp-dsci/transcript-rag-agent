"""Cutting an extracted document into section-anchored chunks.

One rule decides the design: **a chunk never spans two headings.** That costs
some packing efficiency — a 40-word section becomes a 40-word chunk rather than
being padded out with the next section's text — and buys the thing articles need
to cite as well as videos do. A transcript chunk knows the second it started at;
an article chunk has to know the section it sits under, and a chunk straddling
two headings knows neither.

Sections longer than the target are split on paragraph boundaries, with the same
overlap the transcript chunker uses, and every part keeps its section's heading.
So a 12,000-character section — measured, one registered source has one —
becomes ten chunks that all still cite the same heading.
"""

from __future__ import annotations

from src.documents.models import Document
from src.rag.web_models import WebChunk, anchor_for, content_hash, source_key


def _split_paragraphs(text: str, target_chars: int, overlap_chars: int) -> list[str]:
    """Split one section's text into runs of about ``target_chars``.

    Paragraph boundaries rather than character offsets, so a chunk does not
    begin mid-sentence. A single paragraph longer than the target is left
    whole: breaking prose at an arbitrary character is worse than one oversized
    chunk, and the embedding model truncates gracefully.
    """
    paragraphs = [block.strip() for block in text.split("\n") if block.strip()]
    if not paragraphs:
        return []
    parts: list[str] = []
    current: list[str] = []
    length = 0
    for paragraph in paragraphs:
        if current and length + len(paragraph) + 1 > target_chars:
            parts.append("\n".join(current))
            # Carry the tail of the previous part forward, so a sentence that
            # answers a question started in the paragraph before it is not
            # split away from it.
            carried: list[str] = []
            carried_length = 0
            for previous in reversed(current):
                if carried and carried_length + len(previous) + 1 > overlap_chars:
                    break
                carried.insert(0, previous)
                carried_length += len(previous) + 1
            current = carried if overlap_chars > 0 else []
            length = carried_length
        current.append(paragraph)
        length += len(paragraph) + 1
    if current:
        parts.append("\n".join(current))
    return parts


def build_web_chunks(
    document: Document,
    external_id: str,
    channel_id: str = "",
    target_chars: int = 1200,
    overlap_chars: int = 150,
    revision: int = 1,
    published_at: str | None = None,
    url: str | None = None,
) -> list[WebChunk]:
    """Every chunk of one extracted document, in reading order.

    ``url`` overrides the document's own URL, for the sources fetched somewhere
    other than where a reader should be sent.
    """
    key = source_key(external_id)
    chunks: list[WebChunk] = []
    for section in document.sections:
        body = section.text.strip()
        if not body:
            continue
        for part_index, part in enumerate(_split_paragraphs(body, target_chars, overlap_chars)):
            chunks.append(
                WebChunk(
                    source_key=key,
                    external_id=external_id,
                    chunk_index=len(chunks),
                    text=part,
                    channel_id=channel_id,
                    url=url or document.url,
                    title=document.title,
                    heading=section.heading,
                    section_index=section.index,
                    part_index=part_index,
                    anchor=anchor_for(section.heading),
                    published_at=published_at,
                    content_hash=content_hash(part),
                    revision=revision,
                )
            )
    return chunks

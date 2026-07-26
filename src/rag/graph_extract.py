"""Per-chunk entity/claim extraction for the GraphRAG knowledge graph (P4).

One LLM call per chunk against a validated JSON contract
(:class:`src.rag.graph_models.ChunkExtraction`). Extraction is the whole
indexing bill and its quality silently shapes every graph answer, so two
properties matter here:

* **Cache by chunk hash** — results are cached under
  ``.yt-agent/graph_cache/<sha>.json`` keyed on the chunk id *and* its text, so
  re-running ``index-graph`` only re-extracts chunks whose content changed.
  The graph stays cheap to rebuild, which is what lets it stay derived state.
* **Fail per chunk, not per run** — a chunk whose extraction cannot be parsed
  after one retry is recorded as an empty extraction with ``error`` set. One
  malformed response must not abort a 281-chunk backfill.

Provenance fields (chunk id, video id, ``upload_date``, timestamps) are stamped
from the :class:`~src.rag.models.TranscriptChunk`, never taken from the model.
``upload_date`` is what the temporal trend layer keys on.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Callable, Protocol

from pydantic import ValidationError

from src.agents.prompts import (
    GRAPH_EXTRACTION_SYSTEM_PROMPT,
    build_graph_extraction_prompt,
)
from src.rag.graph_models import ChunkExtraction
from src.rag.models import TranscriptChunk

logger = logging.getLogger(__name__)


class ChatModel(Protocol):
    def invoke(self, messages: list) -> object: ...


def extraction_cache_key(chunk: TranscriptChunk) -> str:
    """Cache identity: the chunk id plus a hash of its text.

    Chunk ids renumber when chunking config changes, and text changes when a
    transcript is re-fetched; either invalidates the cached extraction.
    """
    digest = hashlib.sha256(f"{chunk.chunk_id}\n{chunk.text}".encode("utf-8")).hexdigest()
    return digest[:32]


def parse_extraction(content: str) -> ChunkExtraction:
    """Parse and validate one extraction response.

    Accepts the model's JSON with or without a markdown fence. Raises
    ``ValueError`` on anything that does not satisfy the contract — the caller
    decides whether to retry or record a failed chunk.
    """
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"extraction response was not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("extraction response must be a JSON object")
    try:
        return ChunkExtraction.model_validate(data)
    except ValidationError as exc:
        raise ValueError(f"extraction response failed the contract: {exc}") from exc


def stamp_provenance(extraction: ChunkExtraction, chunk: TranscriptChunk) -> ChunkExtraction:
    """Overwrite provenance from the chunk. LLM-supplied values never survive."""
    return extraction.model_copy(
        update={
            "chunk_id": chunk.chunk_id,
            "video_id": chunk.video_id,
            "source_url": str(chunk.source_url),
            "video_title": chunk.title,
            "channel_id": chunk.channel_id,
            "upload_date": chunk.upload_date,
            "start_seconds": chunk.start_seconds,
            "end_seconds": chunk.end_seconds,
        }
    )


class GraphExtractor:
    """Extract entities/relations/claims for chunks, with a disk cache."""

    def __init__(self, llm: ChatModel, cache_dir: Path | None = None) -> None:
        self.llm = llm
        self.cache_dir = cache_dir
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)

    def extract(self, chunk: TranscriptChunk) -> ChunkExtraction:
        cached = self._read_cache(chunk)
        if cached is not None:
            return cached
        extraction = self._extract_uncached(chunk)
        self._write_cache(chunk, extraction)
        return extraction

    def extract_all(
        self,
        chunks: list[TranscriptChunk],
        on_progress: Callable[[str], None] | None = None,
        max_workers: int = 8,
    ) -> list[ChunkExtraction]:
        """Extract every chunk, concurrently, preserving input order.

        Extraction calls are independent (one chunk, one prompt, one cache
        file), so they parallelize trivially — sequential extraction is the
        wall-clock bottleneck of a backfill, not tokens. Results come back in
        input order regardless of completion order; progress reports in
        completion order.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from threading import Lock

        results: list[ChunkExtraction | None] = [None] * len(chunks)
        progress_lock = Lock()
        completed = 0

        def work(index: int, chunk: TranscriptChunk) -> tuple[int, ChunkExtraction, bool]:
            cached = self._read_cache(chunk) is not None
            return index, self.extract(chunk), cached

        workers = max(1, min(max_workers, len(chunks)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(work, index, chunk) for index, chunk in enumerate(chunks)]
            for future in as_completed(futures):
                index, extraction, cached = future.result()
                results[index] = extraction
                if on_progress is not None:
                    detail = (
                        f"{len(extraction.entities)}e/{len(extraction.claims)}c"
                        if extraction.error is None
                        else f"FAILED: {extraction.error[:60]}"
                    )
                    with progress_lock:
                        completed += 1
                        on_progress(
                            f"[{completed}/{len(chunks)}] {chunks[index].chunk_id} "
                            f"({'cache' if cached else 'llm'}) {detail}"
                        )
        return [extraction for extraction in results if extraction is not None]

    def _extract_uncached(self, chunk: TranscriptChunk) -> ChunkExtraction:
        from langchain_core.messages import HumanMessage, SystemMessage

        user_prompt = build_graph_extraction_prompt(chunk.text, chunk.title, chunk.upload_date)
        messages = [
            SystemMessage(content=GRAPH_EXTRACTION_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]
        last_error = ""
        for attempt in range(2):
            try:
                response = self.llm.invoke(messages)
            except Exception as exc:  # noqa: BLE001 - one bad chunk must not abort the backfill
                last_error = str(exc)
                logger.warning(
                    "extraction call failed for %s (attempt %s): %s",
                    chunk.chunk_id,
                    attempt + 1,
                    exc,
                )
                continue
            try:
                content = str(getattr(response, "content", response) or "")
                return stamp_provenance(parse_extraction(content), chunk)
            except ValueError as exc:
                last_error = str(exc)
                logger.warning(
                    "extraction parse failed for %s (attempt %s): %s",
                    chunk.chunk_id,
                    attempt + 1,
                    exc,
                )
                # Retry once with the error appended so the model can self-repair.
                messages = [
                    SystemMessage(content=GRAPH_EXTRACTION_SYSTEM_PROMPT),
                    HumanMessage(
                        content=(
                            f"{user_prompt}\n\nYour previous response was invalid: "
                            f"{last_error}\nReturn only the JSON object."
                        )
                    ),
                ]
        failed = ChunkExtraction(error=f"extraction failed after retry: {last_error}")
        return stamp_provenance(failed, chunk)

    # ── cache ─────────────────────────────────────────────────────────────

    def _cache_path(self, chunk: TranscriptChunk) -> Path | None:
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"{extraction_cache_key(chunk)}.json"

    def _read_cache(self, chunk: TranscriptChunk) -> ChunkExtraction | None:
        path = self._cache_path(chunk)
        if path is None or not path.exists():
            return None
        try:
            cached = ChunkExtraction.model_validate_json(path.read_text(encoding="utf-8"))
        except (ValidationError, ValueError):
            return None
        # A cached failure is retried on the next run rather than pinned forever.
        if cached.error is not None:
            return None
        return cached

    def _write_cache(self, chunk: TranscriptChunk, extraction: ChunkExtraction) -> None:
        path = self._cache_path(chunk)
        if path is None:
            return
        path.write_text(extraction.model_dump_json(indent=2) + "\n", encoding="utf-8")

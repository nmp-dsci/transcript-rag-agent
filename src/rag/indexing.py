"""Turn a source URL into retrievable chunks — and nothing that can fail on a
provider.

The core index is deliberately narrow: fetch, chunk, embed. Summarising and
knowledge-graph extraction are *enrichment*, and enrichment must never be able
to fail an index whose vector half already succeeded. That is not a
hypothetical: an ``Insufficient Balance`` 402 from the summary LLM used to
propagate out of :meth:`RagIndexer.index`, exit the CLI non-zero, and report
"Indexing failed" for a video whose transcript and chunks were already
committed to Chroma — leaving a half-indexed video with nothing recording that
it needed repair.

So a summary failure here is captured, not raised: the result carries
``summary_status="failed"`` plus the error text, and the caller decides what to
do about it. Graph extraction is already handled this way one level up, in
``src.api.ingestion_queue``.
"""

from __future__ import annotations

from typing import Any

import logging
from dataclasses import dataclass, field

from src.rag.chunking import build_chunks
from src.rag.models import FAILED, RawTranscriptDocument, TranscriptChunk
from src.rag.progress import report_stage
from src.rag.storage import RawTranscriptStore, TranscriptChunkStore
from src.rag.summaries import TranscriptSummaryGenerator, TranscriptSummaryStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RagIndexResult:
    raw_document: RawTranscriptDocument
    chunks: list[TranscriptChunk]
    cache_status: str
    #: ``hit`` | ``indexed`` | ``created`` | ``refresh`` when a summary exists,
    #: ``failed`` when the generator raised, ``None`` when no summary store or
    #: generator was configured. Never raises — see the module docstring.
    summary_status: str | None = None
    #: Why the summary step failed, when it did. ``None`` on every other path.
    summary_error: str | None = None
    #: Chunk ids this index run deleted because the rebuild no longer produces
    #: them — the tail a shrinking re-chunk leaves behind. Empty on a first
    #: index and on any re-index that did not shrink, which is the normal case.
    removed_chunk_ids: list[str] = field(default_factory=list)


class RagIndexer:
    def __init__(
        self,
        raw_store: RawTranscriptStore,
        chunk_store: TranscriptChunkStore,
        target_chars: int = 1200,
        overlap_chars: int = 150,
        summary_store: TranscriptSummaryStore | None = None,
        summary_generator: TranscriptSummaryGenerator | None = None,
    ) -> None:
        self.raw_store = raw_store
        self.chunk_store = chunk_store
        self.target_chars = target_chars
        self.overlap_chars = overlap_chars
        self.summary_store = summary_store
        self.summary_generator = summary_generator

    def index(
        self,
        source_url: str,
        refresh: bool = False,
        refresh_summary: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> RagIndexResult:
        # Each stage is announced as it *begins*, so a progress reading is a
        # fact about where the run is rather than an optimistic guess made
        # before any work started.
        report_stage("discover")
        report_stage("fetch")
        raw_document, cache_status = (
            self.raw_store.ensure_raw_document(source_url, refresh=refresh, metadata=metadata)
            if metadata is not None
            else self.raw_store.ensure_raw_document(source_url, refresh=refresh)
        )
        report_stage("chunk")
        chunks = build_chunks(
            raw_document,
            target_chars=self.target_chars,
            overlap_chars=self.overlap_chars,
        )
        report_stage("embed")
        # Replace rather than upsert: the rebuilt chunks are the whole truth for
        # this video, so anything the previous run left at a higher index has to
        # go with it. See ``TranscriptChunkStore.replace_chunks``.
        removed = self.chunk_store.replace_chunks(raw_document.video_id, chunks)
        # ── the core index is complete and durable from here ──────────────
        # Everything below is enrichment. It reports failure; it never raises.
        summary_status = None
        summary_error = None
        if self.summary_store is not None and self.summary_generator is not None:
            try:
                _summary, summary_status = self.summary_store.ensure_summary(
                    raw_document,
                    self.summary_generator,
                    refresh=refresh_summary,
                    chunk_count=len(chunks),
                )
            except Exception as exc:
                summary_status = FAILED
                summary_error = str(exc)
                logger.warning(
                    "summary generation failed for %s — indexed without one: %s",
                    raw_document.video_id,
                    exc,
                )
                # Persist the failure. An unrecorded one is indistinguishable
                # from "nobody has tried yet", and the repair pass has to be
                # able to find these.
                raw_document = raw_document.model_copy(update={"summary_status": FAILED})
                self.raw_store.upsert_raw_document(raw_document)
            else:
                refreshed = self.raw_store.get_raw_document(raw_document.video_id)
                if refreshed is not None:
                    raw_document = refreshed
        return RagIndexResult(
            raw_document=raw_document,
            chunks=chunks,
            cache_status=cache_status,
            summary_status=summary_status,
            summary_error=summary_error,
            removed_chunk_ids=removed,
        )

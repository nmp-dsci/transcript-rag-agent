from __future__ import annotations

from dataclasses import dataclass

from src.rag.chunking import build_chunks
from src.rag.models import RawTranscriptDocument, TranscriptChunk
from src.rag.storage import RawTranscriptStore, TranscriptChunkStore
from src.rag.summaries import TranscriptSummaryGenerator, TranscriptSummaryStore


@dataclass(frozen=True)
class RagIndexResult:
    raw_document: RawTranscriptDocument
    chunks: list[TranscriptChunk]
    cache_status: str
    summary_status: str | None = None


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
    ) -> RagIndexResult:
        raw_document, cache_status = self.raw_store.ensure_raw_document(
            source_url, refresh=refresh
        )
        chunks = build_chunks(
            raw_document,
            target_chars=self.target_chars,
            overlap_chars=self.overlap_chars,
        )
        self.chunk_store.upsert_chunks(chunks)
        summary_status = None
        if self.summary_store is not None and self.summary_generator is not None:
            _summary, summary_status = self.summary_store.ensure_summary(
                raw_document,
                self.summary_generator,
                refresh=refresh_summary,
                chunk_count=len(chunks),
            )
            refreshed = self.raw_store.get_raw_document(raw_document.video_id)
            if refreshed is not None:
                raw_document = refreshed
        return RagIndexResult(
            raw_document=raw_document,
            chunks=chunks,
            cache_status=cache_status,
            summary_status=summary_status,
        )

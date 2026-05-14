from __future__ import annotations

from dataclasses import dataclass

from src.rag.chunking import build_chunks
from src.rag.models import RawTranscriptDocument, TranscriptChunk
from src.rag.storage import RawTranscriptStore, TranscriptChunkStore


@dataclass(frozen=True)
class RagIndexResult:
    raw_document: RawTranscriptDocument
    chunks: list[TranscriptChunk]
    cache_status: str


class RagIndexer:
    def __init__(
        self,
        raw_store: RawTranscriptStore,
        chunk_store: TranscriptChunkStore,
        target_chars: int = 1200,
        overlap_chars: int = 150,
    ) -> None:
        self.raw_store = raw_store
        self.chunk_store = chunk_store
        self.target_chars = target_chars
        self.overlap_chars = overlap_chars

    def index(self, source_url: str, refresh: bool = False) -> RagIndexResult:
        raw_document, cache_status = self.raw_store.ensure_raw_document(
            source_url, refresh=refresh
        )
        chunks = build_chunks(
            raw_document,
            target_chars=self.target_chars,
            overlap_chars=self.overlap_chars,
        )
        self.chunk_store.upsert_chunks(chunks)
        return RagIndexResult(
            raw_document=raw_document,
            chunks=chunks,
            cache_status=cache_status,
        )

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from src.rag.models import RetrievedChunk, RetrievedTranscriptSummary
from src.rag.storage import RawTranscriptStore, transcript_from_raw_document
from src.transcripts.fetcher import SuperdataTranscriptFetcher
from src.transcripts.models import Transcript
from src.transcripts.storage import ChromaTranscriptStore


@dataclass(frozen=True)
class TranscriptContext:
    transcript: Transcript
    cache_status: str
    context_text: str | None = None
    context_mode: str = "raw"
    retrieved_chunks: list[RetrievedChunk] | None = None
    selected_transcripts: list[RetrievedTranscriptSummary] | None = None
    top_k: int | None = None

    def __post_init__(self) -> None:
        if self.context_text is None:
            object.__setattr__(self, "context_text", self.transcript.raw_text)
        if self.retrieved_chunks is None:
            object.__setattr__(self, "retrieved_chunks", [])
        if self.selected_transcripts is None:
            object.__setattr__(self, "selected_transcripts", [])


class TranscriptContextProvider(Protocol):
    def get_transcript(
        self, video_id: str, source_url: str, query: str | None = None
    ) -> TranscriptContext:
        ...


class RawTranscriptContextProvider:
    """Provides full raw transcript context today; replace with RAG later."""

    def __init__(
        self,
        store: RawTranscriptStore | ChromaTranscriptStore,
        fetcher: SuperdataTranscriptFetcher,
    ) -> None:
        self.store = store
        self.fetcher = fetcher

    def get_transcript(
        self, video_id: str, source_url: str, query: str | None = None
    ) -> TranscriptContext:
        if isinstance(self.store, RawTranscriptStore):
            raw_document, cache_status = self.store.ensure_raw_document(source_url)
            transcript = transcript_from_raw_document(raw_document)
            return TranscriptContext(
                transcript=transcript,
                cache_status=cache_status,
                context_text=transcript.raw_text,
                context_mode="raw",
            )

        cached = self.store.get(video_id)
        if cached is not None:
            return TranscriptContext(
                transcript=cached,
                cache_status="hit",
                context_text=cached.raw_text,
                context_mode="raw",
            )

        transcript = self.fetcher.fetch(source_url)
        self.store.upsert(transcript)
        return TranscriptContext(
            transcript=transcript,
            cache_status="miss",
            context_text=transcript.raw_text,
            context_mode="raw",
        )

    def refresh_transcript(self, source_url: str) -> TranscriptContext:
        if isinstance(self.store, RawTranscriptStore):
            raw_document, cache_status = self.store.ensure_raw_document(
                source_url, refresh=True
            )
            transcript = transcript_from_raw_document(raw_document)
            return TranscriptContext(
                transcript=transcript,
                cache_status=cache_status,
                context_text=transcript.raw_text,
                context_mode="raw",
            )

        transcript = self.fetcher.fetch(source_url)
        self.store.upsert(transcript)
        return TranscriptContext(
            transcript=transcript,
            cache_status="refresh",
            context_text=transcript.raw_text,
            context_mode="raw",
        )

    def get_or_refresh_transcript(
        self, video_id: str, source_url: str, no_refresh: bool
    ) -> TranscriptContext:
        if no_refresh:
            return self.get_transcript(video_id, source_url)
        return self.refresh_transcript(source_url)

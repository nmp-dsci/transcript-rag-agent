from __future__ import annotations

from src.agents.context import TranscriptContext
from src.rag.chunking import format_timestamp
from src.rag.indexing import RagIndexer
from src.rag.references import format_chunk_reference
from src.rag.storage import RawTranscriptStore, TranscriptChunkStore, transcript_from_raw_document


class RagTranscriptContextProvider:
    def __init__(
        self,
        raw_store: RawTranscriptStore,
        chunk_store: TranscriptChunkStore,
        indexer: RagIndexer | None = None,
        top_k: int = 10,
    ) -> None:
        self.raw_store = raw_store
        self.chunk_store = chunk_store
        self.indexer = indexer
        self.top_k = top_k

    def get_transcript(
        self, video_id: str, source_url: str, query: str | None = None
    ) -> TranscriptContext:
        cache_status = "hit"
        if not self.chunk_store.has_chunks(video_id):
            if self.indexer is None:
                raise ValueError(
                    "No RAG chunks found. Run index-rag first or configure auto-indexing."
                )
            result = self.indexer.index(source_url, refresh=False)
            cache_status = result.cache_status

        raw_document, raw_cache_status = self.raw_store.ensure_raw_document(
            source_url, refresh=False
        )
        if cache_status == "hit":
            cache_status = raw_cache_status
        retrieved = self.chunk_store.query(video_id, query or "", self.top_k)
        transcript = transcript_from_raw_document(raw_document)
        return TranscriptContext(
            transcript=transcript,
            cache_status=cache_status,
            context_text=format_retrieved_chunks(retrieved),
            context_mode="rag",
            retrieved_chunks=retrieved,
            top_k=self.top_k,
        )


def format_retrieved_chunks(chunks) -> str:
    parts: list[str] = []
    for index, chunk in enumerate(chunks, 1):
        start = format_timestamp(chunk.start_seconds)
        end = format_timestamp(chunk.end_seconds)
        parts.append(f"[{index}] {start}-{end}\n{chunk.text}")
    return "\n\n".join(parts)


class MultiTranscriptRagContextProvider:
    def __init__(
        self,
        raw_store: RawTranscriptStore,
        chunk_store: TranscriptChunkStore,
        indexer: RagIndexer | None = None,
    ) -> None:
        self.raw_store = raw_store
        self.chunk_store = chunk_store
        self.indexer = indexer

    def get_context(
        self,
        question: str,
        source_url: str | None = None,
        top_k: int = 10,
    ) -> TranscriptContext:
        cache_status = "hit"
        if source_url is None:
            if not self.chunk_store.has_any_chunks():
                raise ValueError(
                    "No indexed transcript chunks found. Run index-rag for one or more "
                    "YouTube URLs first."
                )
            retrieved = self.chunk_store.query_all(question, top_k)
            transcript = _context_transcript_from_chunks(retrieved)
        else:
            video_id = _extract_video_id(source_url)
            if not self.chunk_store.has_chunks(video_id):
                if self.indexer is None:
                    raise ValueError(
                        f"No RAG chunks found for {source_url}. Run index-rag first."
                    )
                result = self.indexer.index(source_url, refresh=False)
                cache_status = result.cache_status
            raw_document, raw_cache_status = self.raw_store.ensure_raw_document(
                source_url, refresh=False
            )
            if cache_status == "hit":
                cache_status = raw_cache_status
            retrieved = self.chunk_store.query_by_url(source_url, question, top_k)
            transcript = transcript_from_raw_document(raw_document)

        return TranscriptContext(
            transcript=transcript,
            cache_status=cache_status,
            context_text=format_retrieved_chunks_with_references(retrieved),
            context_mode="rag",
            retrieved_chunks=retrieved,
            top_k=top_k,
        )


def format_retrieved_chunks_with_references(chunks) -> str:
    parts: list[str] = []
    for index, chunk in enumerate(chunks, 1):
        parts.append(f"{format_chunk_reference(index, chunk)}\n{chunk.text}")
    return "\n\n".join(parts)


def _context_transcript_from_chunks(chunks):
    from datetime import datetime, timezone

    from src.transcripts.models import Transcript

    if chunks:
        first = chunks[0]
        return Transcript(
            video_id="all",
            url=first.source_url,
            provider="rag",
            raw_text=" ".join(chunk.text for chunk in chunks),
            fetched_at=datetime.now(timezone.utc),
        )
    return Transcript(
        video_id="all",
        url="https://www.youtube.com/watch?v=unknown",
        provider="rag",
        raw_text="",
        fetched_at=datetime.now(timezone.utc),
    )


def _extract_video_id(source_url: str) -> str:
    from src.transcripts.youtube import extract_video_id

    return extract_video_id(source_url)

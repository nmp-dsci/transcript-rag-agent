from __future__ import annotations

from src.agents.context import TranscriptContext
from src.rag.chunking import format_timestamp
from src.rag.indexing import RagIndexer
from src.rag.references import format_chunk_reference
from src.rag.storage import RawTranscriptStore, TranscriptChunkStore, transcript_from_raw_document
from src.rag.summaries import TranscriptSummaryStore


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
    """Retrieval for the multi-transcript agents.

    Scope narrows in three steps, most specific first: a single video
    (``source_url``), a whole channel (``channel_id``), or the entire corpus.
    Within the chosen scope, retrieval can run semantically or as a hybrid of
    semantic and BM25 rankings, optionally reranked and widened to neighbouring
    chunks before the answer call sees it.
    """

    def __init__(
        self,
        raw_store: RawTranscriptStore,
        chunk_store: TranscriptChunkStore,
        indexer: RagIndexer | None = None,
        summary_store: TranscriptSummaryStore | None = None,
        retrieval_mode: str = "semantic",
        retrieval_candidates: int = 30,
        reranker: object | None = None,
        neighbor_span: int = 0,
    ) -> None:
        self.raw_store = raw_store
        self.chunk_store = chunk_store
        self.indexer = indexer
        self.summary_store = summary_store
        self.retrieval_mode = retrieval_mode
        self.retrieval_candidates = retrieval_candidates
        self.reranker = reranker
        self.neighbor_span = neighbor_span

    def get_context(
        self,
        question: str,
        source_url: str | None = None,
        top_k: int = 10,
        filter_transcripts: bool = False,
        transcript_filter_top_k: int = 5,
        transcript_filter_min_score: float = 0.25,
        channel_id: str | None = None,
        retrieval_mode: str | None = None,
    ) -> TranscriptContext:
        cache_status = "hit"
        selected_transcripts = []
        mode = retrieval_mode or self.retrieval_mode
        # Retrieve wide, then let fusion/reranking narrow to top_k. With neither
        # enabled this collapses to the original single top_k query.
        candidates = (
            max(top_k, self.retrieval_candidates)
            if (mode == "hybrid" or self.reranker is not None)
            else top_k
        )
        if source_url is None:
            if not self.chunk_store.has_any_chunks():
                raise ValueError(
                    "No indexed transcript chunks found. Run index-rag for one or more "
                    "YouTube URLs first."
                )
            if filter_transcripts:
                if self.summary_store is None:
                    raise ValueError("Transcript filtering requires a summary store")
                selected_transcripts = self.summary_store.query_relevant_transcripts(
                    question,
                    top_k=transcript_filter_top_k,
                    min_score=transcript_filter_min_score,
                )
                if not selected_transcripts:
                    raise ValueError(
                        "No transcript summaries matched the question. Try lowering "
                        "--transcript-filter-min-score or run without "
                        "--filter-transcripts."
                    )
                retrieved = self.chunk_store.query_by_video_ids(
                    [summary.video_id for summary in selected_transcripts],
                    question,
                    candidates,
                )
            elif channel_id:
                retrieved = self.chunk_store.query_by_channel(
                    channel_id, question, candidates
                )
                if not retrieved:
                    raise ValueError(
                        f"No indexed chunks found for channel {channel_id!r}. The "
                        "channel may not be indexed, or its chunks predate the "
                        "channel metadata backfill."
                    )
            else:
                retrieved = self.chunk_store.query_all(question, candidates)
            retrieved = self._refine(question, retrieved, top_k, mode, channel_id)
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
            retrieved = self.chunk_store.query_by_url(source_url, question, candidates)
            retrieved = self._refine(question, retrieved, top_k, mode, None, video_id)
            transcript = transcript_from_raw_document(raw_document)

        return TranscriptContext(
            transcript=transcript,
            cache_status=cache_status,
            context_text=format_retrieved_chunks_with_references(retrieved),
            context_mode="rag",
            retrieved_chunks=retrieved,
            selected_transcripts=selected_transcripts,
            top_k=top_k,
        )

    def _refine(
        self,
        question: str,
        retrieved: list,
        top_k: int,
        mode: str,
        channel_id: str | None,
        video_id: str | None = None,
    ) -> list:
        """Fuse, rerank, and widen a candidate set down to the final top_k."""
        if mode == "hybrid":
            retrieved = self._fuse_with_bm25(
                question, retrieved, top_k, channel_id, video_id
            )
        if self.reranker is not None and retrieved:
            try:
                retrieved = self.reranker.rerank(question, retrieved, top_k)
            except Exception:
                # A reranker failure must degrade to the underlying ranking
                # rather than lose the answer entirely.
                retrieved = retrieved[:top_k]
        else:
            retrieved = retrieved[:top_k]
        if self.neighbor_span > 0:
            retrieved = self._expand_neighbors(retrieved)
        return retrieved

    def _fuse_with_bm25(
        self,
        question: str,
        semantic: list,
        top_k: int,
        channel_id: str | None,
        video_id: str | None,
    ) -> list:
        from src.rag import bm25
        from src.rag.fusion import fuse_chunks

        records = self._bm25_records(channel_id, video_id)
        if not records:
            return semantic
        keyword = bm25.search(
            records,
            question,
            max(top_k, self.retrieval_candidates),
            cache_key=f"hybrid:{video_id or channel_id or 'all'}",
        )
        return fuse_chunks(semantic, keyword, top_k=top_k)

    def _bm25_records(
        self, channel_id: str | None, video_id: str | None
    ) -> list[dict]:
        where: dict[str, str] | None = None
        if video_id:
            where = {"video_id": video_id}
        elif channel_id:
            where = {"channel_id": channel_id}
        result = self.chunk_store.collection.get(
            where=where, include=["documents", "metadatas"]
        )
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []
        records: list[dict] = []
        for index, meta in enumerate(metadatas):
            meta = meta or {}
            text = documents[index] if index < len(documents) else ""
            if not text:
                continue
            records.append(
                {
                    "video_id": str(meta.get("video_id", "")),
                    "chunk_index": int(meta.get("chunk_index", index) or 0),
                    "text": text,
                    "start_seconds": meta.get("start_seconds"),
                    "end_seconds": meta.get("end_seconds"),
                    "source_url": meta.get("source_url") or None,
                }
            )
        return records

    def _expand_neighbors(self, retrieved: list) -> list:
        """Paste adjacent chunks around each hit, keeping retrieval order.

        Neighbours inherit no score — they are context, not retrieval results,
        and are dropped if already present so a chunk never appears twice.
        """
        seen = {(chunk.video_id, chunk.chunk_index) for chunk in retrieved}
        widened: list = []
        for chunk in retrieved:
            neighbors = self.chunk_store.neighbors(
                chunk.video_id, chunk.chunk_index, self.neighbor_span
            )
            for neighbor in neighbors:
                key = (neighbor.video_id, neighbor.chunk_index)
                if key in seen:
                    continue
                seen.add(key)
                widened.append(_as_retrieved(neighbor))
            widened.append(chunk)
        return widened


def _as_retrieved(chunk):
    from src.rag.models import RetrievedChunk

    if isinstance(chunk, RetrievedChunk):
        return chunk
    return RetrievedChunk(**chunk.model_dump(), score=None)


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

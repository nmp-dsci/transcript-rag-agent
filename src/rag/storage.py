from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import chromadb
from pydantic import HttpUrl

from src.rag.embeddings import EmbeddingModel
from src.rag.models import RawTranscriptDocument, RawTranscriptSegment, RetrievedChunk, TranscriptChunk
from src.transcripts.fetcher import SuperdataTranscriptFetcher
from src.transcripts.models import Transcript
from src.transcripts.youtube import extract_video_id


class RawTranscriptStore:
    collection_name = "raw_transcripts"

    def __init__(
        self,
        path: Path | str,
        fetcher: SuperdataTranscriptFetcher | None = None,
        collection_name: str | None = None,
    ) -> None:
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.fetcher = fetcher
        self.collection_name = collection_name or self.collection_name
        self.client = chromadb.PersistentClient(path=str(self.path))
        self.collection = self.client.get_or_create_collection(self.collection_name)

    def upsert_raw_document(self, document: RawTranscriptDocument) -> None:
        self.collection.upsert(
            ids=[document.transcript_id],
            documents=[_raw_document_body(document)],
            metadatas=[_raw_document_metadata(document)],
        )

    def get_raw_document(self, video_id: str) -> RawTranscriptDocument | None:
        result = self.collection.get(
            ids=[_raw_transcript_id(video_id)],
            include=["documents", "metadatas"],
        )
        if not result.get("ids"):
            return None
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []
        if not documents or not metadatas:
            return None
        body = json.loads(documents[0])
        metadata = metadatas[0] or {}
        return RawTranscriptDocument(
            transcript_id=str(metadata.get("transcript_id", _raw_transcript_id(video_id))),
            video_id=str(metadata.get("video_id", video_id)),
            source_url=HttpUrl(str(metadata["source_url"])),
            provider=str(metadata.get("provider", "supadata")),
            title=_none_if_empty(metadata.get("title")),
            language=_none_if_empty(metadata.get("language")),
            segments=[
                RawTranscriptSegment.model_validate(segment)
                for segment in body.get("segments", [])
            ],
            fetched_at=str(metadata.get("fetched_at", _now_iso())),
            source_collection=str(metadata.get("source_collection", self.collection_name)),
        )

    def ensure_raw_document(
        self, source_url: str, refresh: bool = False
    ) -> tuple[RawTranscriptDocument, str]:
        video_id = extract_video_id(source_url)
        if not refresh:
            cached = self.get_raw_document(video_id)
            if cached is not None:
                return cached, "hit"
        if self.fetcher is None:
            raise ValueError("RawTranscriptStore requires a fetcher to refresh transcripts")
        transcript = self.fetcher.fetch(source_url)
        document = raw_document_from_transcript(transcript, self.collection_name)
        self.upsert_raw_document(document)
        return document, "refresh" if refresh else "miss"

    def join_raw_text(self, video_id: str) -> str:
        document = self.get_raw_document(video_id)
        if document is None:
            raise KeyError(f"Raw transcript not found: {video_id}")
        return " ".join(segment.text for segment in document.segments).strip()


class TranscriptChunkStore:
    collection_name = "transcript_chunks"

    def __init__(
        self,
        path: Path | str,
        embedding_model: EmbeddingModel,
        collection_name: str | None = None,
    ) -> None:
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.embedding_model = embedding_model
        self.collection_name = collection_name or self.collection_name
        self.client = chromadb.PersistentClient(path=str(self.path))
        self.collection = self.client.get_or_create_collection(self.collection_name)

    def upsert_chunks(self, chunks: list[TranscriptChunk]) -> None:
        if not chunks:
            return
        embeddings = self.embedding_model.embed_documents([chunk.text for chunk in chunks])
        self.collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            embeddings=embeddings,
            metadatas=[_chunk_metadata(chunk) for chunk in chunks],
        )

    def has_chunks(self, video_id: str) -> bool:
        result = self.collection.get(
            where={"video_id": video_id},
            limit=1,
            include=["metadatas"],
        )
        return bool(result.get("ids"))

    def has_any_chunks(self) -> bool:
        result = self.collection.get(limit=1, include=["metadatas"])
        return bool(result.get("ids"))

    def query(self, video_id: str, query: str, top_k: int) -> list[RetrievedChunk]:
        return self.query_by_video_id(video_id, query, top_k)

    def query_all(self, query: str, top_k: int) -> list[RetrievedChunk]:
        return self._query(query=query, top_k=top_k, where=None)

    def query_by_url(
        self, source_url: str, query: str, top_k: int
    ) -> list[RetrievedChunk]:
        return self.query_by_video_id(extract_video_id(source_url), query, top_k)

    def query_by_video_id(
        self, video_id: str, query: str, top_k: int
    ) -> list[RetrievedChunk]:
        return self._query(query=query, top_k=top_k, where={"video_id": video_id})

    def _query(
        self,
        query: str,
        top_k: int,
        where: dict[str, str] | None,
    ) -> list[RetrievedChunk]:
        embedding = self.embedding_model.embed_query(query)
        kwargs: dict[str, object] = {
            "query_embeddings": [embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where is not None:
            kwargs["where"] = where
        result = self.collection.query(**kwargs)
        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        chunks: list[RetrievedChunk] = []
        for index, _chunk_id in enumerate(ids):
            metadata = metadatas[index] or {}
            distance = distances[index] if index < len(distances) else None
            chunks.append(
                RetrievedChunk(
                    transcript_id=str(metadata["transcript_id"]),
                    video_id=str(metadata["video_id"]),
                    source_url=HttpUrl(str(metadata["source_url"])),
                    chunk_index=int(metadata["chunk_index"]),
                    text=documents[index],
                    start_seconds=_float_or_none(metadata.get("start_seconds")),
                    end_seconds=_float_or_none(metadata.get("end_seconds")),
                    start_segment_index=_int_or_none(metadata.get("start_segment_index")),
                    end_segment_index=_int_or_none(metadata.get("end_segment_index")),
                    segment_count=int(metadata.get("segment_count", 0)),
                    score=(1.0 - float(distance)) if distance is not None else None,
                )
            )
        return chunks


def raw_document_from_transcript(
    transcript: Transcript,
    source_collection: str = RawTranscriptStore.collection_name,
) -> RawTranscriptDocument:
    segments = [
        RawTranscriptSegment(
            text=segment.text,
            offset_ms=segment.offset_ms,
            duration_ms=segment.duration_ms,
            start_seconds=segment.start_seconds,
            end_seconds=segment.end_seconds,
            language=segment.language or transcript.language,
        )
        for segment in transcript.segments
    ]
    if not segments and transcript.raw_text:
        segments.append(RawTranscriptSegment(text=transcript.raw_text, language=transcript.language))
    return RawTranscriptDocument(
        transcript_id=_raw_transcript_id(transcript.video_id),
        video_id=transcript.video_id,
        source_url=transcript.url,
        provider=transcript.provider,
        title=transcript.title,
        language=transcript.language,
        segments=segments,
        fetched_at=transcript.fetched_at.isoformat(),
        source_collection=source_collection,
    )


def transcript_from_raw_document(document: RawTranscriptDocument) -> Transcript:
    fetched_at = _parse_datetime(document.fetched_at)
    return Transcript(
        video_id=document.video_id,
        url=document.source_url,
        title=document.title,
        language=document.language,
        provider=document.provider,
        raw_text=" ".join(segment.text for segment in document.segments).strip(),
        segments=[
            _transcript_segment_from_raw(segment) for segment in document.segments
        ],
        fetched_at=fetched_at,
    )


def _transcript_segment_from_raw(segment: RawTranscriptSegment):
    from src.transcripts.models import TranscriptSegment

    return TranscriptSegment(
        text=segment.text,
        offset_ms=segment.offset_ms,
        duration_ms=segment.duration_ms,
        start_seconds=segment.start_seconds,
        end_seconds=segment.end_seconds,
        language=segment.language,
    )


def _raw_transcript_id(video_id: str) -> str:
    return f"raw_transcript:{video_id}"


def _raw_document_body(document: RawTranscriptDocument) -> str:
    return json.dumps(
        {"segments": [segment.model_dump(mode="json") for segment in document.segments]},
        separators=(",", ":"),
    )


def _raw_document_metadata(document: RawTranscriptDocument) -> dict[str, str | int]:
    return {
        "transcript_id": document.transcript_id,
        "video_id": document.video_id,
        "source_url": str(document.source_url),
        "source_collection": document.source_collection,
        "provider": document.provider,
        "title": document.title or "",
        "language": document.language or "",
        "fetched_at": document.fetched_at,
        "segment_count": len(document.segments),
    }


def _chunk_metadata(chunk: TranscriptChunk) -> dict[str, str | int | float]:
    metadata: dict[str, str | int | float] = {
        "transcript_id": chunk.transcript_id,
        "video_id": chunk.video_id,
        "source_url": str(chunk.source_url),
        "source_collection": "raw_transcripts",
        "chunk_index": chunk.chunk_index,
        "segment_count": chunk.segment_count,
    }
    if chunk.start_seconds is not None:
        metadata["start_seconds"] = chunk.start_seconds
    if chunk.end_seconds is not None:
        metadata["end_seconds"] = chunk.end_seconds
    if chunk.start_segment_index is not None:
        metadata["start_segment_index"] = chunk.start_segment_index
    if chunk.end_segment_index is not None:
        metadata["end_segment_index"] = chunk.end_segment_index
    return metadata


def _parse_datetime(value: str) -> datetime:
    if not value:
        return datetime.now(timezone.utc)
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _none_if_empty(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text or None


def _float_or_none(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def _int_or_none(value: object) -> int | None:
    if value is None:
        return None
    return int(value)

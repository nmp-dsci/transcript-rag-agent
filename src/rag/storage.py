from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import chromadb
from pydantic import HttpUrl, ValidationError

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
            description=body.get("description") or _none_if_empty(metadata.get("description")),
            channel_id=_none_if_empty(metadata.get("channel_id")),
            channel_name=_none_if_empty(metadata.get("channel_name")),
            duration_seconds=_float_or_none(metadata.get("duration_seconds")),
            thumbnail_url=_http_url_or_none(metadata.get("thumbnail_url")),
            upload_date=_none_if_empty(metadata.get("upload_date")),
            view_count=_int_or_none(metadata.get("view_count")),
            like_count=_int_or_none(metadata.get("like_count")),
            tags=[str(tag) for tag in body.get("tags", [])],
            transcript_languages=[
                str(lang) for lang in body.get("transcript_languages", [])
            ],
            language=_none_if_empty(metadata.get("language")),
            segments=[
                RawTranscriptSegment.model_validate(segment)
                for segment in body.get("segments", [])
            ],
            fetched_at=str(metadata.get("fetched_at", _now_iso())),
            source_collection=str(metadata.get("source_collection", self.collection_name)),
            summary=_none_if_empty(metadata.get("summary")),
            summary_model=_none_if_empty(metadata.get("summary_model")),
            summary_generated_at=_none_if_empty(metadata.get("summary_generated_at")),
            summary_embedding=body.get("summary_embedding"),
            summary_embedding_model=_none_if_empty(
                metadata.get("summary_embedding_model")
            ),
            summary_embedded_at=_none_if_empty(metadata.get("summary_embedded_at")),
        )

    def ensure_raw_document(
        self, source_url: str, refresh: bool = False
    ) -> tuple[RawTranscriptDocument, str]:
        video_id = extract_video_id(source_url)
        if not refresh:
            cached = self.get_raw_document(video_id)
            if cached is not None:
                if self.fetcher is not None and _missing_video_metadata(cached):
                    updated = _raw_document_with_metadata(
                        cached,
                        self.fetcher.fetch_metadata(source_url),
                    )
                    if updated != cached:
                        self.upsert_raw_document(updated)
                        return updated, "metadata"
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
        # Embed the contextual header with the text, but store the spoken text
        # alone as the document — the header is retrieval scaffolding, not
        # something the answering LLM should quote back.
        embeddings = self.embedding_model.embed_documents(
            [chunk.embedding_text for chunk in chunks]
        )
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

    def count_chunks(self, video_id: str) -> int:
        result = self.collection.get(
            where={"video_id": video_id},
            include=["metadatas"],
        )
        return len(result.get("ids") or [])

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

    def query_by_video_ids(
        self, video_ids: list[str], query: str, top_k: int
    ) -> list[RetrievedChunk]:
        if not video_ids:
            return []
        chunks: list[RetrievedChunk] = []
        for video_id in dict.fromkeys(video_ids):
            chunks.extend(self.query_by_video_id(video_id, query, top_k))
        return sorted(
            chunks,
            key=lambda chunk: float("-inf") if chunk.score is None else chunk.score,
            reverse=True,
        )[:top_k]

    def query_by_channel(
        self, channel_id: str, query: str, top_k: int
    ) -> list[RetrievedChunk]:
        """Retrieve across every chunk belonging to one channel.

        A native Chroma metadata filter, which is why chunks carry channel
        identity — the alternative is querying each of the channel's videos
        separately and merging, which scales with video count.
        """
        return self._query(query=query, top_k=top_k, where={"channel_id": channel_id})

    def channel_video_ids(self, channel_id: str) -> list[str]:
        """The video ids whose chunks are tagged with this channel."""
        result = self.collection.get(
            where={"channel_id": channel_id}, include=["metadatas"]
        )
        seen: dict[str, None] = {}
        for meta in result.get("metadatas") or []:
            video_id = str((meta or {}).get("video_id", ""))
            if video_id:
                seen.setdefault(video_id, None)
        return list(seen)

    def neighbors(self, video_id: str, chunk_index: int, span: int = 1) -> list[TranscriptChunk]:
        """Chunks immediately before and after one chunk, in index order.

        Used to widen precise retrieval hits back out to readable context so
        answers stop getting cut off mid-thought.
        """
        if span <= 0:
            return []
        wanted = [
            index
            for index in range(chunk_index - span, chunk_index + span + 1)
            if index >= 0 and index != chunk_index
        ]
        if not wanted:
            return []
        result = self.collection.get(
            ids=[f"chunk:{video_id}:{index}" for index in wanted],
            include=["documents", "metadatas"],
        )
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []
        chunks: list[TranscriptChunk] = []
        for index, meta in enumerate(metadatas):
            meta = meta or {}
            chunks.append(
                _chunk_from_metadata(
                    meta, documents[index] if index < len(documents) else ""
                )
            )
        return sorted(chunks, key=lambda chunk: chunk.chunk_index)

    def all_embeddings(self) -> list[dict[str, object]]:
        """Every chunk with its embedding, for similarity-graph construction."""
        result = self.collection.get(include=["embeddings", "documents", "metadatas"])
        embeddings = result.get("embeddings")
        embeddings = [] if embeddings is None else list(embeddings)
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []
        records: list[dict[str, object]] = []
        for index, meta in enumerate(metadatas):
            meta = meta or {}
            if index >= len(embeddings):
                continue
            records.append(
                {
                    "chunk_id": f"chunk:{meta.get('video_id', '')}:{meta.get('chunk_index', index)}",
                    "video_id": str(meta.get("video_id", "")),
                    "chunk_index": int(meta.get("chunk_index", index) or 0),
                    "channel_id": meta.get("channel_id") or None,
                    "channel_name": meta.get("channel_name") or None,
                    "title": meta.get("title") or None,
                    "text": documents[index] if index < len(documents) else "",
                    "start_seconds": _float_or_none(meta.get("start_seconds")),
                    "end_seconds": _float_or_none(meta.get("end_seconds")),
                    "source_url": meta.get("source_url") or None,
                    "embedding": [float(value) for value in embeddings[index]],
                }
            )
        return records

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
                    channel_id=_none_if_empty(metadata.get("channel_id")),
                    channel_name=_none_if_empty(metadata.get("channel_name")),
                    title=_none_if_empty(metadata.get("title")),
                    upload_date=_none_if_empty(metadata.get("upload_date")),
                    score=(1.0 - float(distance)) if distance is not None else None,
                )
            )
        return chunks


def _chunk_from_metadata(metadata: dict, text: str) -> TranscriptChunk:
    return TranscriptChunk(
        transcript_id=str(metadata.get("transcript_id", "")),
        video_id=str(metadata.get("video_id", "")),
        source_url=HttpUrl(str(metadata["source_url"])),
        chunk_index=int(metadata.get("chunk_index", 0) or 0),
        text=text,
        start_seconds=_float_or_none(metadata.get("start_seconds")),
        end_seconds=_float_or_none(metadata.get("end_seconds")),
        start_segment_index=_int_or_none(metadata.get("start_segment_index")),
        end_segment_index=_int_or_none(metadata.get("end_segment_index")),
        segment_count=int(metadata.get("segment_count", 0) or 0),
        channel_id=_none_if_empty(metadata.get("channel_id")),
        channel_name=_none_if_empty(metadata.get("channel_name")),
        title=_none_if_empty(metadata.get("title")),
        upload_date=_none_if_empty(metadata.get("upload_date")),
    )


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
        description=transcript.description,
        channel_id=transcript.channel_id,
        channel_name=transcript.channel_name,
        duration_seconds=transcript.duration_seconds,
        thumbnail_url=transcript.thumbnail_url,
        upload_date=transcript.upload_date,
        view_count=transcript.view_count,
        like_count=transcript.like_count,
        tags=transcript.tags,
        transcript_languages=transcript.transcript_languages,
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
        description=document.description,
        channel_id=document.channel_id,
        channel_name=document.channel_name,
        duration_seconds=document.duration_seconds,
        thumbnail_url=document.thumbnail_url,
        upload_date=document.upload_date,
        view_count=document.view_count,
        like_count=document.like_count,
        tags=document.tags,
        transcript_languages=document.transcript_languages,
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
    body: dict[str, object] = {
        "segments": [segment.model_dump(mode="json") for segment in document.segments]
    }
    if document.summary_embedding is not None:
        body["summary_embedding"] = document.summary_embedding
    if document.description is not None:
        body["description"] = document.description
    if document.tags:
        body["tags"] = document.tags
    if document.transcript_languages:
        body["transcript_languages"] = document.transcript_languages
    return json.dumps(body, separators=(",", ":"))


def _raw_document_metadata(document: RawTranscriptDocument) -> dict[str, str | int | float]:
    metadata: dict[str, str | int | float] = {
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
    if document.summary is not None:
        metadata["summary"] = document.summary
    if document.summary_model is not None:
        metadata["summary_model"] = document.summary_model
    if document.summary_generated_at is not None:
        metadata["summary_generated_at"] = document.summary_generated_at
    if document.summary_embedding_model is not None:
        metadata["summary_embedding_model"] = document.summary_embedding_model
    if document.summary_embedded_at is not None:
        metadata["summary_embedded_at"] = document.summary_embedded_at
    if document.description is not None:
        metadata["description"] = _truncate_metadata_text(document.description)
    if document.channel_id is not None:
        metadata["channel_id"] = document.channel_id
    if document.channel_name is not None:
        metadata["channel_name"] = document.channel_name
    if document.duration_seconds is not None:
        metadata["duration_seconds"] = document.duration_seconds
    if document.thumbnail_url is not None:
        metadata["thumbnail_url"] = str(document.thumbnail_url)
    if document.upload_date is not None:
        metadata["upload_date"] = document.upload_date
    if document.view_count is not None:
        metadata["view_count"] = document.view_count
    if document.like_count is not None:
        metadata["like_count"] = document.like_count
    return metadata


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
    if chunk.channel_id is not None:
        metadata["channel_id"] = chunk.channel_id
    if chunk.channel_name is not None:
        metadata["channel_name"] = chunk.channel_name
    if chunk.title is not None:
        metadata["title"] = chunk.title
    if chunk.upload_date is not None:
        metadata["upload_date"] = chunk.upload_date
    if chunk.context_header is not None:
        metadata["context_header"] = chunk.context_header
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


def _missing_video_metadata(document: RawTranscriptDocument) -> bool:
    return not document.title or not document.channel_name


def _raw_document_with_metadata(
    document: RawTranscriptDocument,
    metadata: dict[str, object],
) -> RawTranscriptDocument:
    if not metadata:
        return document
    author = metadata.get("author") if isinstance(metadata.get("author"), dict) else {}
    media = metadata.get("media") if isinstance(metadata.get("media"), dict) else {}
    additional = (
        metadata.get("additionalData")
        if isinstance(metadata.get("additionalData"), dict)
        else {}
    )
    tags = metadata.get("tags") if isinstance(metadata.get("tags"), list) else []
    transcript_languages = (
        metadata.get("transcriptLanguages")
        if isinstance(metadata.get("transcriptLanguages"), list)
        else additional.get("transcriptLanguages")
        if isinstance(additional.get("transcriptLanguages"), list)
        else []
    )
    return document.model_copy(
        update={
            "title": _none_if_empty(metadata.get("title")) or document.title,
            "description": _none_if_empty(metadata.get("description"))
            or document.description,
            "channel_id": _none_if_empty(additional.get("channelId"))
            or _none_if_empty(author.get("id"))
            or document.channel_id,
            "channel_name": _none_if_empty(author.get("displayName"))
            or _none_if_empty(author.get("username"))
            or document.channel_name,
            "duration_seconds": _float_or_none(media.get("duration"))
            or document.duration_seconds,
            "thumbnail_url": _http_url_or_none(media.get("thumbnailUrl"))
            or document.thumbnail_url,
            "upload_date": _none_if_empty(metadata.get("createdAt"))
            or document.upload_date,
            "view_count": _int_or_none(_nested(metadata, "stats", "views"))
            or document.view_count,
            "like_count": _int_or_none(_nested(metadata, "stats", "likes"))
            or document.like_count,
            "tags": [str(tag) for tag in tags] or document.tags,
            "transcript_languages": [str(lang) for lang in transcript_languages]
            or document.transcript_languages,
        }
    )


def _nested(data: dict[str, object], parent: str, child: str) -> object:
    value = data.get(parent)
    if not isinstance(value, dict):
        return None
    return value.get(child)


def _truncate_metadata_text(value: str, limit: int = 7000) -> str:
    return value[:limit]


def _float_or_none(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def _http_url_or_none(value: object) -> HttpUrl | None:
    text = _none_if_empty(value)
    if text is None:
        return None
    try:
        return HttpUrl(text)
    except (ValueError, ValidationError):
        return None


def _int_or_none(value: object) -> int | None:
    if value is None:
        return None
    return int(value)

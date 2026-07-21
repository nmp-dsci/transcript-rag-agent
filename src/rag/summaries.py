from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

import chromadb
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import HttpUrl

from src.rag.embeddings import EmbeddingModel, cosine_similarity
from src.rag.models import (
    RawTranscriptDocument,
    RetrievedTranscriptSummary,
    TranscriptSummaryRecord,
)
from src.rag.storage import RawTranscriptStore


SUMMARY_SYSTEM_PROMPT = """You summarize YouTube transcripts for RAG filtering.

Use only the provided transcript text. Produce a concise but specific summary of
the transcript's key topics, claims, entities, dates, policy names, and domain
terms that would help route future user questions to this transcript. Avoid
generic filler. Return only JSON in this shape:
{"summary":"key topics and claims..."}"""


class ChatModel(Protocol):
    def invoke(self, messages: list[SystemMessage | HumanMessage]) -> object:
        ...


class TranscriptSummaryGenerator:
    def __init__(
        self,
        llm: ChatModel,
        model_name: str,
        max_transcript_chars: int = 40_000,
    ) -> None:
        self.llm = llm
        self.model_name = model_name
        self.max_transcript_chars = max_transcript_chars

    def summarize(self, raw_document: RawTranscriptDocument) -> str:
        transcript_text = " ".join(
            segment.text for segment in raw_document.segments
        ).strip()
        if len(transcript_text) > self.max_transcript_chars:
            half = self.max_transcript_chars // 2
            transcript_text = f"{transcript_text[:half]}\n...\n{transcript_text[-half:]}"
        response = self.llm.invoke(
            [
                SystemMessage(content=SUMMARY_SYSTEM_PROMPT),
                HumanMessage(content=transcript_text),
            ]
        )
        content = getattr(response, "content", response)
        data = _json_object(str(content))
        summary = str(data.get("summary", "")).strip()
        if not summary:
            raise ValueError("Transcript summary LLM returned an empty summary")
        return summary


class TranscriptSummaryStore:
    collection_name = "transcript_summaries"

    def __init__(
        self,
        path: Path | str,
        embedding_model: EmbeddingModel,
        embedding_model_name: str,
        raw_store: RawTranscriptStore | None = None,
        collection_name: str | None = None,
    ) -> None:
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.embedding_model = embedding_model
        self.embedding_model_name = embedding_model_name
        self.raw_store = raw_store
        self.collection_name = collection_name or self.collection_name
        self.client = chromadb.PersistentClient(path=str(self.path))
        self.collection = self.client.get_or_create_collection(self.collection_name)

    def upsert_summary(self, record: TranscriptSummaryRecord) -> None:
        self.collection.upsert(
            ids=[record.summary_id],
            documents=[record.summary],
            embeddings=[record.summary_embedding],
            metadatas=[_summary_metadata(record)],
        )

    def get_summary(self, video_id: str) -> TranscriptSummaryRecord | None:
        result = self.collection.get(
            ids=[_summary_id(video_id)],
            include=["documents", "metadatas", "embeddings"],
        )
        if not result.get("ids"):
            return None
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []
        embeddings = result.get("embeddings")
        if not documents or not metadatas:
            return None
        embedding = _first_embedding(embeddings)
        return _record_from_parts(documents[0], metadatas[0] or {}, embedding)

    def ensure_summary(
        self,
        raw_document: RawTranscriptDocument,
        generator: TranscriptSummaryGenerator,
        refresh: bool = False,
        chunk_count: int | None = None,
    ) -> tuple[TranscriptSummaryRecord, str]:
        if not refresh:
            existing = self.get_summary(raw_document.video_id)
            if (
                existing is not None
                and existing.summary_embedding_model == self.embedding_model_name
            ):
                self._backfill_raw_document(raw_document, existing)
                return existing, "hit"
            if (
                raw_document.summary
                and raw_document.summary_embedding
                and raw_document.summary_embedding_model == self.embedding_model_name
            ):
                record = _record_from_raw(raw_document, chunk_count)
                self.upsert_summary(record)
                return record, "indexed"

        status = "refresh" if refresh else "created"
        summary = (
            generator.summarize(raw_document)
            if refresh or not raw_document.summary
            else raw_document.summary
        )
        if not summary.strip():
            raise ValueError("Cannot create summary embedding from an empty summary")
        embedding = self.embedding_model.embed_query(summary)
        now = _now_iso()
        updated = raw_document.model_copy(
            update={
                "summary": summary,
                "summary_model": generator.model_name,
                "summary_generated_at": now
                if refresh or not raw_document.summary_generated_at
                else raw_document.summary_generated_at,
                "summary_embedding": embedding,
                "summary_embedding_model": self.embedding_model_name,
                "summary_embedded_at": now,
            }
        )
        if self.raw_store is not None:
            self.raw_store.upsert_raw_document(updated)
        record = _record_from_raw(updated, chunk_count)
        self.upsert_summary(record)
        return record, status

    def _backfill_raw_document(
        self,
        raw_document: RawTranscriptDocument,
        record: TranscriptSummaryRecord,
    ) -> None:
        if self.raw_store is None:
            return
        if (
            raw_document.summary == record.summary
            and raw_document.summary_embedding == record.summary_embedding
            and raw_document.summary_embedding_model == record.summary_embedding_model
        ):
            return
        self.raw_store.upsert_raw_document(
            raw_document.model_copy(
                update={
                    "summary": record.summary,
                    "summary_model": record.summary_model,
                    "summary_generated_at": record.summary_generated_at,
                    "summary_embedding": record.summary_embedding,
                    "summary_embedding_model": record.summary_embedding_model,
                    "summary_embedded_at": record.summary_embedded_at,
                }
            )
        )

    def query_relevant_transcripts(
        self,
        question: str,
        top_k: int,
        min_score: float,
    ) -> list[RetrievedTranscriptSummary]:
        embedding = self.embedding_model.embed_query(question)
        result = self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            # Relevance is scored below with cosine_similarity over the returned
            # embeddings, so Chroma's own distances are never read.
            include=["documents", "metadatas", "embeddings"],
        )
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        raw_embeddings = result.get("embeddings")
        embeddings = raw_embeddings[0] if raw_embeddings is not None else []
        summaries: list[RetrievedTranscriptSummary] = []
        for index, document in enumerate(documents):
            metadata = metadatas[index] or {}
            embedding_value = embeddings[index] if index < len(embeddings) else []
            if hasattr(embedding_value, "tolist"):
                embedding_value = embedding_value.tolist()
            score = cosine_similarity(embedding, list(embedding_value))
            if score < min_score:
                continue
            record = _record_from_parts(document, metadata, list(embedding_value))
            summaries.append(
                RetrievedTranscriptSummary(
                    **record.model_dump(mode="json"),
                    score=score,
                )
            )
        return summaries


def _record_from_raw(
    raw_document: RawTranscriptDocument,
    chunk_count: int | None,
) -> TranscriptSummaryRecord:
    if not raw_document.summary or not raw_document.summary_embedding:
        raise ValueError("Raw transcript does not have a summary embedding")
    return TranscriptSummaryRecord(
        transcript_id=raw_document.transcript_id,
        video_id=raw_document.video_id,
        source_url=raw_document.source_url,
        summary=raw_document.summary,
        summary_model=raw_document.summary_model or "",
        summary_generated_at=raw_document.summary_generated_at or "",
        summary_embedding=raw_document.summary_embedding,
        summary_embedding_model=raw_document.summary_embedding_model or "",
        summary_embedded_at=raw_document.summary_embedded_at or "",
        title=raw_document.title,
        language=raw_document.language,
        segment_count=len(raw_document.segments),
        chunk_count=chunk_count,
    )


def _record_from_parts(
    document: str,
    metadata: dict[str, object],
    embedding: list[float],
) -> TranscriptSummaryRecord:
    return TranscriptSummaryRecord(
        transcript_id=str(metadata["transcript_id"]),
        video_id=str(metadata["video_id"]),
        source_url=HttpUrl(str(metadata["source_url"])),
        summary=document,
        summary_model=str(metadata.get("summary_model", "")),
        summary_generated_at=str(metadata.get("summary_generated_at", "")),
        summary_embedding=embedding,
        summary_embedding_model=str(metadata.get("summary_embedding_model", "")),
        summary_embedded_at=str(metadata.get("summary_embedded_at", "")),
        title=_none_if_empty(metadata.get("title")),
        language=_none_if_empty(metadata.get("language")),
        segment_count=int(metadata.get("segment_count", 0)),
        chunk_count=_int_or_none(metadata.get("chunk_count")),
    )


def _summary_metadata(record: TranscriptSummaryRecord) -> dict[str, str | int]:
    metadata: dict[str, str | int] = {
        "transcript_id": record.transcript_id,
        "video_id": record.video_id,
        "source_url": str(record.source_url),
        "provider": "supadata",
        "title": record.title or "",
        "language": record.language or "",
        "summary_model": record.summary_model,
        "summary_generated_at": record.summary_generated_at,
        "summary_embedding_model": record.summary_embedding_model,
        "summary_embedded_at": record.summary_embedded_at,
        "segment_count": record.segment_count,
    }
    if record.chunk_count is not None:
        metadata["chunk_count"] = record.chunk_count
    return metadata


def _json_object(content: str) -> dict[str, object]:
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM response was not valid JSON: {content}") from exc
    if not isinstance(value, dict):
        raise ValueError("LLM response JSON must be an object")
    return value


def _summary_id(video_id: str) -> str:
    return f"summary:{video_id}"


def _first_embedding(value: object) -> list[float]:
    if value is None:
        return []
    try:
        if hasattr(value, "tolist"):
            value = value.tolist()
        first = value[0]  # type: ignore[index]
        if hasattr(first, "tolist"):
            first = first.tolist()
        return [float(item) for item in first]
    except (IndexError, TypeError):
        return []


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _none_if_empty(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text or None


def _int_or_none(value: object) -> int | None:
    if value is None or value == "":
        return None
    return int(value)

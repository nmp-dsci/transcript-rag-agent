from __future__ import annotations

from pydantic import BaseModel, Field, HttpUrl


class RawTranscriptSegment(BaseModel):
    text: str
    offset_ms: int | None = None
    duration_ms: int | None = None
    start_seconds: float | None = None
    end_seconds: float | None = None
    language: str | None = None


class RawTranscriptDocument(BaseModel):
    transcript_id: str
    video_id: str
    source_url: HttpUrl
    provider: str = "supadata"
    title: str | None = None
    description: str | None = None
    channel_id: str | None = None
    channel_name: str | None = None
    duration_seconds: float | None = None
    thumbnail_url: HttpUrl | None = None
    upload_date: str | None = None
    view_count: int | None = None
    like_count: int | None = None
    tags: list[str] = Field(default_factory=list)
    transcript_languages: list[str] = Field(default_factory=list)
    language: str | None = None
    segments: list[RawTranscriptSegment] = Field(default_factory=list)
    fetched_at: str
    source_collection: str = "raw_transcripts"
    summary: str | None = None
    summary_model: str | None = None
    summary_generated_at: str | None = None
    summary_embedding: list[float] | None = None
    summary_embedding_model: str | None = None
    summary_embedded_at: str | None = None


class TranscriptChunk(BaseModel):
    transcript_id: str
    video_id: str
    source_url: HttpUrl
    chunk_index: int
    text: str
    start_seconds: float | None = None
    end_seconds: float | None = None
    start_segment_index: int | None = None
    end_segment_index: int | None = None
    segment_count: int = 0

    @property
    def chunk_id(self) -> str:
        return f"chunk:{self.video_id}:{self.chunk_index}"


class RetrievedChunk(TranscriptChunk):
    score: float | None = None


class TranscriptSummaryRecord(BaseModel):
    transcript_id: str
    video_id: str
    source_url: HttpUrl
    summary: str
    summary_model: str
    summary_generated_at: str
    summary_embedding: list[float]
    summary_embedding_model: str
    summary_embedded_at: str
    title: str | None = None
    language: str | None = None
    segment_count: int = 0
    chunk_count: int | None = None

    @property
    def summary_id(self) -> str:
        return f"summary:{self.video_id}"


class RetrievedTranscriptSummary(TranscriptSummaryRecord):
    score: float | None = None


class RagContextResult(BaseModel):
    video_id: str
    source_url: HttpUrl
    query: str
    top_k: int
    chunks: list[RetrievedChunk] = Field(default_factory=list)


class ContextComparisonResult(BaseModel):
    question: str
    raw_answer: str
    rag_answer: str
    semantic_similarity: float
    raw_prompt_tokens_estimate: int
    rag_prompt_tokens_estimate: int
    token_savings_percent: float

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
    language: str | None = None
    segments: list[RawTranscriptSegment] = Field(default_factory=list)
    fetched_at: str
    source_collection: str = "raw_transcripts"


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

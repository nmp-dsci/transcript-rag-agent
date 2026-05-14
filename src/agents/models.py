from __future__ import annotations

from pydantic import BaseModel, Field, HttpUrl


class TranscriptSummary(BaseModel):
    summary: str
    top_findings: list[str] = Field(min_length=3, max_length=3)


class TranscriptAnswer(BaseModel):
    question: str
    answer: str
    source_video_id: str


class SummaryRequest(BaseModel):
    video_id: str
    source_url: str
    message: str = "Summarize this transcript."


class QuestionRequest(BaseModel):
    video_id: str
    source_url: str
    question: str


class RagQuestionRequest(BaseModel):
    question: str
    source_url: HttpUrl | None = None
    top_k: int = 10


class RagAnswerReference(BaseModel):
    label: str
    source_url: HttpUrl
    timestamp_url: HttpUrl
    start_seconds: float | None = None
    end_seconds: float | None = None
    chunk_index: int
    video_id: str


class RagTranscriptAnswer(BaseModel):
    question: str
    answer: str
    references: list[RagAnswerReference] = Field(default_factory=list)

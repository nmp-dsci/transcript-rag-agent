from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, HttpUrl

from src.rag.models import RetrievedChunk


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


class RagAnswerReference(BaseModel):
    label: str
    source_url: HttpUrl
    timestamp_url: HttpUrl
    start_seconds: float | None = None
    end_seconds: float | None = None
    chunk_index: int
    video_id: str


class RecursionOptions(BaseModel):
    max_depth: int = 1
    max_followups: int = 3
    followup_top_k: int | None = None
    novelty_min_chunks: int = 2
    max_total_followups: int | None = None


class RagQuestionRequest(BaseModel):
    question: str
    source_url: HttpUrl | None = None
    top_k: int = 10
    filter_transcripts: bool = False
    transcript_filter_top_k: int = 5
    transcript_filter_min_score: float = 0.25
    recursive: bool = False
    recursion_options: RecursionOptions | None = None
    # Scope retrieval to one channel. Ignored when source_url pins a single
    # video, which is already the narrower scope.
    channel_id: str | None = None
    # Per-request override of the provider's configured retrieval strategy, so
    # the scoreboard can compare semantic and hybrid under the same judge.
    retrieval_mode: Literal["semantic", "hybrid"] | None = None
    # Prior turns, condensed, for follow-up questions that depend on context.
    history: list[str] = Field(default_factory=list)


class FollowupSubtopic(BaseModel):
    topic: str
    rationale: str = ""
    followup_query: str
    confidence: float = 0.0


class SubtopicEvidence(BaseModel):
    subtopic_index: int
    subtopic: FollowupSubtopic
    chunks: list[RetrievedChunk] = Field(default_factory=list)
    outcome: str


class SubtopicAnswer(BaseModel):
    subtopic_index: int
    topic: str
    followup_query: str
    answer: str
    references: list[RagAnswerReference] = Field(default_factory=list)


class RecursionStage(BaseModel):
    name: str
    llm_calls: int
    retrievals: int


class RecursionTrace(BaseModel):
    stages: list[RecursionStage] = Field(default_factory=list)
    subtopic_evidence: list[SubtopicEvidence] = Field(default_factory=list)
    subtopic_answers: list[SubtopicAnswer] = Field(default_factory=list)
    preserved_first_answer: str | None = None
    terminated_reason: str
    total_followups_proposed: int = 0
    total_followups_executed: int = 0


class RagTranscriptAnswer(BaseModel):
    question: str
    answer: str
    references: list[RagAnswerReference] = Field(default_factory=list)
    subtopics: list[FollowupSubtopic] = Field(default_factory=list)
    followups_requested: bool = False
    answer_confidence: float | None = None
    recursion: RecursionTrace | None = None


class AgentProgressEvent(BaseModel):
    iteration: int = Field(description="1-based retrieval counter.")
    event_type: Literal["retrieval_start", "retrieval_complete", "answer_start"]
    query: str | None = Field(
        default=None, description="The retrieval query for this iteration."
    )
    chunk_count: int | None = Field(
        default=None, description="Populated on retrieval_complete."
    )

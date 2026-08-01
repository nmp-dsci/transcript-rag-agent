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
    # A document the user shared, already fetched, extracted and formatted with
    # [§N] section markers (see :mod:`src.documents.review`). Its presence is
    # what turns an answer into a review: the corpus stops being the subject and
    # becomes the source of criteria, and the answering prompt changes to match.
    document_context: str | None = None


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


class QueryRewrite(BaseModel):
    """What a history-aware query rewrite actually produced for one answer.

    Recorded only when the rewrite call was made, so its absence means no call
    happened. ``degraded`` marks a rewrite that fell back to the raw question —
    the trace must never read a failed rewrite as a successful one.
    """

    query: str
    degraded: bool = False
    elapsed_ms: int | None = None


class TraceStep(BaseModel):
    """One step of an answer path's execution, persisted with the answer.

    Built only from data the executing code actually recorded — a step never
    describes what *should* have happened, so an empty ``chunk_ids`` or a
    ``None`` ``elapsed_ms`` means "not measured", never a guess.
    """

    phase: Literal["route", "filter", "retrieve", "rerank", "merge", "llm"]
    label: str
    detail: str = ""
    #: Chunk identities this step produced/kept, in order ("chunk:<video>:<i>").
    chunk_ids: list[str] = Field(default_factory=list)
    model: str | None = None
    elapsed_ms: int | None = None
    iteration: int | None = None


class RetrievalPass(BaseModel):
    """One retrieval an answer ran, with the stages the provider measured for it.

    A recursive answer retrieves several times — a first pass plus one query per
    follow-up subtopic — and each one overwrites the agent's ``last_context``.
    Keeping the passes in order, each tagged with what it retrieved for, is what
    lets a persisted trace attribute measured stages to the retrieval that
    produced them instead of reading a follow-up's stages as the first retrieval.
    """

    #: What this retrieval was for, e.g. ``first retrieval``/``follow-up 2 ...``.
    label: str
    steps: list[TraceStep] = Field(default_factory=list)


def chunk_id(chunk: object) -> str | None:
    """The canonical ``chunk:<video_id>:<index>`` id, or None if unidentifiable."""
    video_id = getattr(chunk, "video_id", None)
    index = getattr(chunk, "chunk_index", None)
    if video_id is None or index is None:
        return None
    return f"chunk:{video_id}:{index}"


def chunk_ids_for(chunks: list | None) -> list[str]:
    return [cid for chunk in (chunks or []) if (cid := chunk_id(chunk)) is not None]


class AgentProgressEvent(BaseModel):
    iteration: int = Field(description="1-based retrieval counter.")
    event_type: Literal["retrieval_start", "retrieval_complete", "answer_start"]
    query: str | None = Field(default=None, description="The retrieval query for this iteration.")
    chunk_count: int | None = Field(default=None, description="Populated on retrieval_complete.")

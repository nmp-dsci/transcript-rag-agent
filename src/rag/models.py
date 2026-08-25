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
    #: Enrichment state, recorded rather than inferred. Before these existed,
    #: "this video has no summary" was read off ``summary is None`` — which
    #: cannot tell a video nobody has summarised yet from one whose summary
    #: provider returned 402 and lost the attempt. Repairing the second needs
    #: to find it first.
    #:
    #: ``pending`` nothing has run | ``done`` succeeded | ``failed`` attempted
    #: and raised. ``None`` on documents written before this field existed,
    #: which read as ``pending`` — see :func:`summary_state`.
    summary_status: str | None = None
    #: Which generator wrote :attr:`summary`: ``description`` (the creator's
    #: own YouTube blurb, no LLM) or ``llm``. A corpus mixing both puts two
    #: very different registers into one embedding space, so the router's
    #: similarity scores stop meaning the same thing video to video — worth
    #: being able to detect.
    summary_source: str | None = None
    #: Same three states, for knowledge-graph extraction.
    graph_status: str | None = None


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
    # Video-level identity copied onto every chunk so retrieval can filter on it
    # natively. Optional because chunks indexed before the backfill lack them.
    channel_id: str | None = None
    channel_name: str | None = None
    title: str | None = None
    upload_date: str | None = None
    # The channel/title/timestamp preamble prepended before embedding, kept
    # separate so the UI and the LLM prompt can show the spoken text alone.
    context_header: str | None = None

    @property
    def chunk_id(self) -> str:
        return f"chunk:{self.video_id}:{self.chunk_index}"

    @property
    def embedding_text(self) -> str:
        """What gets embedded: the contextual header plus the spoken text.

        Transcript chunks are conversational fragments that often lose their
        subject ("had. So, I'm going to just copy…"), which embeds poorly. The
        header restores the video-level context the speaker left implicit.
        """
        if not self.context_header:
            return self.text
        return f"{self.context_header}\n{self.text}"


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


#: Enrichment states, in the order a video moves through them.
PENDING = "pending"
DONE = "done"
FAILED = "failed"


def summary_state(document: RawTranscriptDocument) -> str:
    """The summary's enrichment state, tolerant of pre-field documents.

    Documents written before ``summary_status`` existed carry ``None``. Those
    are not unknowable: a stored summary means it succeeded, and its absence
    means nothing has produced one yet. Only ``failed`` is genuinely new
    information, which is exactly why the field had to exist.
    """
    if document.summary_status:
        return document.summary_status
    return DONE if (document.summary or "").strip() else PENDING


def graph_state(document: RawTranscriptDocument) -> str:
    """The graph's enrichment state. Unset means nothing has run yet."""
    return document.graph_status or PENDING

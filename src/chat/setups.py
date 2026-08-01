"""RAG setup registry and a runner that answers one question several ways.

A "setup" is one of the comparable ``rag-ask`` agent configurations. The runner
builds the shared retrieval stack and agents once, then answers a question with
each selected setup so the interactive chat can show them side by side — the
same three configurations the evaluation report compares.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from src.agents.models import (
    AgentProgressEvent,
    QueryRewrite,
    RagQuestionRequest,
    RecursionOptions,
    RecursionTrace,
    TraceStep,
    chunk_ids_for,
)
from src.agents.rag_agent import RagAgent
from src.agents.rag_transcript_agent import RagTranscriptAgent
from src.config import Settings
from src.rag.context import MultiTranscriptRagContextProvider, RetrievalError
from src.rag.embeddings import HuggingFaceEmbeddingModel
from src.rag.eval import estimate_tokens
from src.rag.indexing import RagIndexer
from src.rag.storage import RawTranscriptStore, TranscriptChunkStore
from src.rag.summaries import TranscriptSummaryStore
from src.transcripts.fetcher import SuperdataTranscriptFetcher


@dataclass(frozen=True)
class SetupSpec:
    """Static metadata for one comparable RAG setup."""

    key: str
    title: str
    description: str


# Order defines the 1-based menu numbering used by ``select_setups``.
SETUP_SPECS: list[SetupSpec] = [
    SetupSpec(
        key="rag_llm",
        title="rag_llm (single-hop)",
        description=("One retrieval across all indexed transcripts, then a single LLM answer."),
    ),
    SetupSpec(
        key="rag_llm_recursive",
        title="rag_llm (recursive)",
        description=(
            "Multi-hop retrieval: follow-up queries fan out, then a final synthesis call."
        ),
    ),
    SetupSpec(
        key="rag_agent",
        title="rag_agent (agentic)",
        description=(
            "LangGraph ReAct loop that retrieves across sub-topics until it "
            "judges it has enough evidence."
        ),
    ),
    SetupSpec(
        key="graph_rag",
        title="graph_rag (knowledge graph)",
        description=(
            "GraphRAG (P4): routes local/global/temporal, answers over the "
            "Neo4j entity/claim graph — plus vector retrieval on local "
            "questions. Requires index-graph."
        ),
    ),
]

SETUP_KEYS: list[str] = [spec.key for spec in SETUP_SPECS]
_SPECS_BY_KEY: dict[str, SetupSpec] = {spec.key: spec for spec in SETUP_SPECS}


def setup_spec(key: str) -> SetupSpec:
    return _SPECS_BY_KEY[key]


@dataclass
class AskScope:
    """Everything that narrows or shapes retrieval for one question.

    Grouped into one object because it has to travel unchanged through the
    runner, both agents, and the recursive follow-up loop — passing five loose
    keyword arguments down that chain is how they drift apart.
    """

    channel_id: str | None = None
    retrieval_mode: str | None = None
    filter_transcripts: bool = False
    # Condensed prior turns for follow-up questions.
    history: list[str] = field(default_factory=list)


@dataclass
class SetupResult:
    """One setup's answer to a question, with the metadata the UI displays."""

    key: str
    title: str
    command: str
    answer: str
    references: list[Any] = field(default_factory=list)
    token_estimate: int = 0
    chunk_count: int = 0
    llm_calls: int | None = None
    iterations: int | None = None
    terminated_reason: str | None = None
    elapsed_seconds: float = 0.0
    error: str | None = None
    # Retrieved chunk texts, persisted so RAGAS can judge the answer later.
    contexts: list[str] = field(default_factory=list)
    # Identity of every retrieved chunk, in retrieval order. References cover
    # only the chunks the LLM chose to cite, so recall against a golden set
    # needs this separate record of what retrieval actually returned.
    retrieved_chunk_ids: list[str] = field(default_factory=list)
    # Identity of the stack that produced the answer. Scores from different
    # models must never be averaged together, so the scoreboard groups on these.
    model: str | None = None
    embedding_model: str | None = None
    top_k: int | None = None
    # Retrieval scope and strategy, recorded so the scoreboard never averages a
    # channel-scoped hybrid run together with a whole-corpus semantic one.
    channel_id: str | None = None
    retrieval_mode: str | None = None
    # Follow-up subtopics the answering LLM proposed. The agent contract always
    # returns these; surfacing them lets the UI offer them as next questions.
    followups: list[dict[str, Any]] = field(default_factory=list)
    # Ordered execution steps (serialized TraceSteps): what this answer's path
    # actually did — route decisions, retrievals with chunk ids, rerank passes,
    # LLM calls — persisted so the trace survives beyond the live SSE stream.
    trace: list[dict[str, Any]] = field(default_factory=list)


def select_setups(raw: str) -> list[str]:
    """Parse a user selection into ordered, de-duplicated setup keys.

    Accepts ``a``/``all`` for every setup, or a comma/space separated list of
    1-based menu indices and/or setup keys. Unknown tokens raise ``ValueError``.
    """
    text = raw.strip().lower()
    if not text:
        raise ValueError("No setup selected")
    if text in {"a", "all"}:
        return list(SETUP_KEYS)
    selected: list[str] = []
    for token in (t for t in re.split(r"[,\s]+", text) if t):
        key: str | None = None
        if token.isdigit():
            index = int(token)
            if 1 <= index <= len(SETUP_KEYS):
                key = SETUP_KEYS[index - 1]
        elif token in _SPECS_BY_KEY:
            key = token
        if key is None:
            raise ValueError(f"Unknown setup: {token}")
        if key not in selected:
            selected.append(key)
    return selected


def command_for(key: str, url: str | None = None) -> str:
    """Reconstruct the equivalent ``rag-ask`` command for display and history."""
    url_flag = f' --url "{url}"' if url else ""
    flags = {
        "rag_llm": "--rag_llm",
        "rag_llm_recursive": "--rag_llm --recursive",
        "rag_agent": "--rag_agent",
        "graph_rag": "--graph_rag",
    }[key]
    return f'uv run python -m src.cli rag-ask "$question" {flags}{url_flag}'


ProgressFn = Callable[[str], None]
AgentEventFn = Callable[[AgentProgressEvent], None]


class RagSetupRunner:
    """Answer a question with one or more setups using a shared retrieval stack."""

    def __init__(
        self,
        settings: Settings,
        provider: MultiTranscriptRagContextProvider,
    ) -> None:
        self._settings = settings
        self._provider = provider
        self._rag_llm_agent: RagTranscriptAgent | None = None
        self._rag_agent: RagAgent | None = None
        self._graph_rag_agent = None  # GraphRagAgent, built lazily (needs Neo4j)

    @property
    def provider(self) -> MultiTranscriptRagContextProvider:
        """The shared retrieval provider, reused by the ranking endpoint."""
        return self._provider

    @classmethod
    def from_settings(cls, settings: Settings) -> "RagSetupRunner":
        fetcher = SuperdataTranscriptFetcher(
            settings.superdata_api_key,
            timeout_seconds=settings.supadata_timeout_seconds,
            poll_interval_seconds=settings.supadata_poll_interval_seconds,
            max_poll_seconds=settings.supadata_max_poll_seconds,
        )
        raw_store = RawTranscriptStore(
            settings.chroma_path,
            fetcher=fetcher,
            collection_name=settings.raw_transcript_collection,
        )
        embedding_model = HuggingFaceEmbeddingModel(settings.embedding_model)
        chunk_store = TranscriptChunkStore(
            settings.chroma_path,
            embedding_model=embedding_model,
            collection_name=settings.chunk_collection,
        )
        indexer = RagIndexer(
            raw_store=raw_store,
            chunk_store=chunk_store,
            target_chars=settings.chunk_target_chars,
            overlap_chars=settings.chunk_overlap_chars,
        )
        summary_store = TranscriptSummaryStore(
            settings.chroma_path,
            embedding_model=embedding_model,
            embedding_model_name=settings.embedding_model,
            raw_store=raw_store,
            collection_name=settings.transcript_summary_collection,
        )
        reranker = None
        if settings.rerank_enabled:
            from src.rag.rerank import CrossEncoderReranker

            reranker = CrossEncoderReranker.from_model_name(settings.rerank_model)
        provider = MultiTranscriptRagContextProvider(
            raw_store=raw_store,
            chunk_store=chunk_store,
            indexer=indexer,
            summary_store=summary_store,
            retrieval_mode=settings.retrieval_mode,
            retrieval_candidates=settings.retrieval_candidates,
            reranker=reranker,
            neighbor_span=settings.neighbor_span,
        )
        return cls(settings, provider)

    def _rag_llm(self) -> RagTranscriptAgent:
        if self._rag_llm_agent is None:
            self._rag_llm_agent = RagTranscriptAgent.from_settings(self._settings, self._provider)
        return self._rag_llm_agent

    def _agentic(self) -> RagAgent:
        if self._rag_agent is None:
            self._rag_agent = RagAgent.from_settings(self._settings, self._provider)
        return self._rag_agent

    def _graph(self):
        if self._graph_rag_agent is None:
            from src.agents.graph_agent import GraphRagAgent

            self._graph_rag_agent = GraphRagAgent.from_settings(self._settings, self._provider)
        return self._graph_rag_agent

    def _agent_for(self, key: str) -> Any:
        """The (cached) agent one setup answers with."""
        if key == "rag_agent":
            return self._agentic()
        if key == "graph_rag":
            return self._graph()
        return self._rag_llm()

    def run_many(
        self,
        keys: list[str],
        question: str,
        *,
        url: str | None = None,
        top_k: int | None = None,
        on_progress: ProgressFn | None = None,
    ) -> list[SetupResult]:
        results: list[SetupResult] = []
        for key in keys:
            if on_progress is not None:
                on_progress(f"Running {setup_spec(key).title} ...")
            results.append(self.run(key, question, url=url, top_k=top_k))
        return results

    def run(
        self,
        key: str,
        question: str,
        *,
        url: str | None = None,
        top_k: int | None = None,
        on_agent_event: AgentEventFn | None = None,
        scope: "AskScope | None" = None,
    ) -> SetupResult:
        """Answer with one setup.

        ``on_agent_event`` receives per-iteration research events and only ever
        fires for ``rag_agent`` — the other setups make a single retrieval pass
        and have no intermediate steps to report. Passing it also selects the
        streaming agent call, so it is what decides whether ``rag_agent``'s
        persisted trace carries a step per iteration or one summary step: a
        caller that does not stream (CLI, evals) observes no per-iteration
        events, and the trace never claims steps it did not observe.
        """
        spec = setup_spec(key)
        effective_top_k = top_k or self._settings.rag_top_k
        scope = scope or AskScope()
        model = self._settings.deepseek_model
        started = time.monotonic()
        # Bound before the call that can raise, so the failure path can persist
        # whatever the run had already recorded instead of an empty trace.
        agent: Any = None
        events: list[AgentProgressEvent] = []
        try:
            agent = self._agent_for(key)
            # Agents are cached across questions and clear their own per-answer
            # record only once ``answer`` runs. Clearing it here, ahead of
            # everything that can fail, is what stops a failure before that
            # point — a malformed URL rejected while the request is built — from
            # persisting the previous question's steps as this answer's trace.
            _clear_per_answer_state(agent)
            if key == "rag_agent":
                # Only the streaming path has per-iteration events to collect;
                # passing None is what makes the CLI/eval path call answer().
                collect: AgentEventFn | None = None
                if on_agent_event is not None:
                    stream = on_agent_event

                    def collect_event(event: AgentProgressEvent) -> None:
                        events.append(event)
                        stream(event)

                    collect = collect_event
                answer = self._run_rag_agent(agent, question, url, effective_top_k, collect, scope)
                return self._build_result(
                    spec,
                    url,
                    answer,
                    agent.last_context,
                    top_k=effective_top_k,
                    iterations=agent.last_iteration_count,
                    terminated_reason=agent.last_terminated_reason,
                    elapsed=time.monotonic() - started,
                    scope=scope,
                    trace=_agent_event_steps(events, agent.last_iteration_count, model),
                )
            if key == "graph_rag":
                answer = agent.answer(self._request(question, url, effective_top_k, scope))
                return self._build_result(
                    spec,
                    url,
                    answer,
                    agent.last_context,
                    top_k=effective_top_k,
                    llm_calls=agent.last_llm_calls,
                    terminated_reason=agent.last_route,
                    elapsed=time.monotonic() - started,
                    scope=scope,
                    trace=list(getattr(agent, "last_trace", None) or []),
                )
            answer, llm_calls = self._run_rag_llm(agent, key, question, url, effective_top_k, scope)
            rewrite = getattr(agent, "last_rewrite", None)
            if answer.recursion is not None:
                trace = _recursion_steps(
                    answer.recursion, model, rewrite, _first_retrieval_steps(agent)
                )
            else:
                trace = [
                    *_rewrite_steps(rewrite, model),
                    *_retrieval_steps(agent),
                    TraceStep(
                        phase="llm",
                        label="Answer",
                        detail="one answer call over the retrieved chunks, citing chunk ids",
                        model=model,
                    ),
                ]
            return self._build_result(
                spec,
                url,
                answer,
                agent.last_context,
                top_k=effective_top_k,
                llm_calls=llm_calls,
                terminated_reason=(
                    answer.recursion.terminated_reason if answer.recursion else None
                ),
                elapsed=time.monotonic() - started,
                scope=scope,
                trace=trace,
            )
        except Exception as exc:  # one failing setup must not abort the comparison
            partial = _partial_trace(key, agent, events, model, exc)
            return SetupResult(
                key=spec.key,
                title=spec.title,
                command=command_for(spec.key, url),
                answer="",
                elapsed_seconds=time.monotonic() - started,
                error=str(exc),
                model=self._settings.deepseek_model,
                embedding_model=self._settings.embedding_model,
                top_k=effective_top_k,
                channel_id=scope.channel_id,
                retrieval_mode=scope.retrieval_mode or self._settings.retrieval_mode,
                trace=[step.model_dump(mode="json") for step in partial],
            )

    def _request(self, question, url, top_k, scope: "AskScope", **extra):
        return RagQuestionRequest(
            question=question,
            source_url=url,
            top_k=top_k,
            channel_id=scope.channel_id,
            retrieval_mode=scope.retrieval_mode,
            filter_transcripts=scope.filter_transcripts,
            transcript_filter_top_k=self._settings.transcript_filter_top_k,
            transcript_filter_min_score=self._settings.transcript_filter_min_score,
            history=list(scope.history),
            **extra,
        )

    def _run_rag_llm(self, agent, key, question, url, top_k, scope: "AskScope"):
        if key == "rag_llm_recursive":
            request = self._request(
                question,
                url,
                top_k,
                scope,
                recursive=True,
                recursion_options=RecursionOptions(
                    max_depth=self._settings.rag_max_depth,
                    max_followups=self._settings.rag_max_followups,
                    followup_top_k=self._settings.rag_followup_top_k,
                    novelty_min_chunks=self._settings.rag_novelty_min_chunks,
                    max_total_followups=self._settings.rag_max_total_followups,
                ),
            )
            answer = agent.answer(request)
            recursion = answer.recursion
            llm_calls = sum(stage.llm_calls for stage in recursion.stages) if recursion else 1
            return answer, llm_calls + _rewrite_calls(agent)
        answer = agent.answer(self._request(question, url, top_k, scope))
        return answer, 1 + _rewrite_calls(agent)

    def _run_rag_agent(self, agent, question, url, top_k, on_agent_event=None, scope=None):
        request = self._request(question, url, top_k, scope or AskScope())
        if on_agent_event is None:
            return agent.answer(request)
        return agent.answer_streaming(request, on_agent_event)

    def _build_result(
        self,
        spec: SetupSpec,
        url: str | None,
        answer: Any,
        context: Any,
        *,
        top_k: int | None = None,
        llm_calls: int | None = None,
        iterations: int | None = None,
        terminated_reason: str | None = None,
        elapsed: float = 0.0,
        scope: "AskScope | None" = None,
        trace: list[TraceStep] | None = None,
    ) -> SetupResult:
        scope = scope or AskScope()
        context_text = context.context_text if context is not None else ""
        chunks = context.retrieved_chunks if context is not None else []
        return SetupResult(
            key=spec.key,
            title=spec.title,
            command=command_for(spec.key, url),
            answer=answer.answer,
            references=list(answer.references or []),
            token_estimate=estimate_tokens(context_text or ""),
            chunk_count=len(chunks or []),
            llm_calls=llm_calls,
            iterations=iterations,
            terminated_reason=terminated_reason,
            elapsed_seconds=round(elapsed, 2),
            contexts=[
                chunk.text
                for chunk in (chunks or [])
                if isinstance(getattr(chunk, "text", None), str)
            ],
            retrieved_chunk_ids=[
                f"chunk:{chunk.video_id}:{chunk.chunk_index}"
                for chunk in (chunks or [])
                if getattr(chunk, "video_id", None) is not None
            ],
            model=self._settings.deepseek_model,
            embedding_model=self._settings.embedding_model,
            top_k=top_k,
            channel_id=scope.channel_id,
            retrieval_mode=scope.retrieval_mode or self._settings.retrieval_mode,
            followups=_followups_to_dicts(getattr(answer, "subtopics", [])),
            trace=[step.model_dump(mode="json") for step in (trace or [])],
        )


def _clear_per_answer_state(agent: Any) -> None:
    """Discard the previous answer's record from a cached agent.

    Exactly the fields ``_partial_trace`` reads back, so a persisted trace can
    only ever contain steps this answer recorded. Attributes an agent does not
    have are left alone — the agentic setup keeps its events in the runner.
    """
    if hasattr(agent, "last_context"):
        agent.last_context = None
    if hasattr(agent, "last_rewrite"):
        agent.last_rewrite = None
    if hasattr(agent, "last_retrievals"):
        agent.last_retrievals = []
    if hasattr(agent, "last_trace"):
        agent.last_trace = []


def _partial_trace(
    key: str,
    agent: Any,
    events: list[AgentProgressEvent],
    model: str,
    error: BaseException,
) -> list[TraceStep]:
    """What a failed run had already recorded, and nothing beyond it.

    Retrieval often succeeds and the answer call is what fails, so the steps
    collected up to that point are exactly the diagnostic the error entry
    needs. Nothing is reconstructed and nothing is inherited: the runner cleared
    this state before the run started, so an absent record means the stage never
    ran on this question and no step for it is emitted.
    """
    if key == "graph_rag":
        return [*(getattr(agent, "last_trace", None) or []), *_error_steps(error)]
    if key == "rag_agent":
        # Iterations are deliberately not passed: a count read off a half-run
        # agent would summarise a loop that never finished.
        return [*_agent_event_steps(events, None, model), *_error_steps(error)]
    steps = _rewrite_steps(getattr(agent, "last_rewrite", None), model)
    steps.extend(_retrieval_steps(agent))
    if getattr(agent, "last_retrievals", None) is None:
        # An agent that keeps no per-retrieval record only has the failed
        # retrieval's stages on the error itself. One that does already recorded
        # them under the pass they belong to, and the error's copy of the same
        # stages would report that retrieval twice.
        steps.extend(_error_steps(error))
    return steps


def _error_steps(error: BaseException) -> list[TraceStep]:
    """The stages a failed retrieval measured before raising, if it carries any.

    Only ``RetrievalError`` promises this, and only ``TraceStep``s are taken
    from it: anything else spliced in here would blow up on serialisation and
    replace the error the setup is actually reporting.
    """
    if not isinstance(error, RetrievalError):
        return []
    return [step for step in error.trace if isinstance(step, TraceStep)]


def _retrieval_steps(agent: Any) -> list[TraceStep]:
    """Every retrieval this answer ran, in order, attributed to its own pass.

    A recursive answer retrieves more than once, and every pass records the same
    stage labels, so without the pass label a follow-up's stages read as the
    answer's first retrieval. One pass has nothing to disambiguate, so its
    stages pass through exactly as the provider measured them.
    """
    passes = getattr(agent, "last_retrievals", None)
    if passes is None:
        return list(getattr(getattr(agent, "last_context", None), "trace", None) or [])
    if len(passes) == 1:
        return list(passes[0].steps)
    return [
        step.model_copy(update={"detail": _attributed(retrieval.label, step.detail)})
        for retrieval in passes
        for step in retrieval.steps
    ]


def _attributed(pass_label: str, detail: str) -> str:
    return f"{pass_label} — {detail}" if detail else pass_label


def _first_retrieval_steps(agent: Any) -> list[TraceStep]:
    """The stages the first retrieval measured, or none if it recorded none."""
    passes = getattr(agent, "last_retrievals", None) or []
    return list(passes[0].steps) if passes else []


def _rewrite_calls(agent: Any) -> int:
    """The LLM calls the history-aware query rewrite cost this answer.

    ``retrieval_query`` only reaches the model when the request carries prior
    turns, and records the result on ``last_rewrite``; no record means no call,
    so a question asked without history counts exactly what it did before.
    """
    return 1 if getattr(agent, "last_rewrite", None) is not None else 0


def _rewrite_steps(rewrite: QueryRewrite | None, model: str) -> list[TraceStep]:
    """The history-aware query rewrite, when one was actually attempted.

    ``retrieval_query`` only calls the LLM when the request carries prior
    turns, so no record means no call was made and no step belongs in the
    trace. A rewrite that fell back to the raw question reads as the failure it
    was rather than as a successful rewrite.
    """
    if rewrite is None:
        return []
    return [
        TraceStep(
            phase="llm",
            label="Rewrite query",
            detail=(
                f'rewrite failed; retrieval ran on the question as asked: "{rewrite.query}"'
                if rewrite.degraded
                else f'follow-up rewritten to stand alone: "{rewrite.query}"'
            ),
            model=model,
            elapsed_ms=rewrite.elapsed_ms,
        )
    ]


def _agent_event_steps(
    events: list[AgentProgressEvent], iterations: int | None, model: str
) -> list[TraceStep]:
    """The agentic setup's streamed research events, as persistable TraceSteps.

    Events only flow on the streaming path; when none were collected (CLI and
    eval runs call ``answer()`` directly), a single summary step records the
    iteration count so the trace never claims steps it did not observe.
    """
    steps: list[TraceStep] = []
    for event in events:
        if event.event_type == "retrieval_complete":
            found = f"{event.chunk_count} chunks" if event.chunk_count is not None else "chunks"
            steps.append(
                TraceStep(
                    phase="retrieve",
                    label=f"Retrieve (iteration {event.iteration})",
                    detail=(f'"{event.query}" — {found}' if event.query else found),
                    iteration=event.iteration,
                )
            )
        elif event.event_type == "answer_start":
            steps.append(
                TraceStep(
                    phase="llm",
                    label="Answer",
                    detail="the model judged it had enough evidence",
                    model=model,
                    iteration=event.iteration,
                )
            )
    if not steps and iterations:
        steps.append(
            TraceStep(
                phase="retrieve",
                label="ReAct retrievals",
                detail=(
                    f"{iterations} retrieval iteration"
                    f"{'s' if iterations != 1 else ''} (per-step events not "
                    "streamed on this path)"
                ),
            )
        )
    return steps


def _recursion_steps(
    recursion: RecursionTrace,
    model: str,
    rewrite: QueryRewrite | None = None,
    first_retrieval: list[TraceStep] | None = None,
) -> list[TraceStep]:
    """A recursive answer's own RecursionTrace, flattened into TraceSteps.

    Converts rather than re-records: every fact here (subtopic queries, the
    chunks each follow-up retrieved, merge/skip outcomes, whether synthesis
    ran) is read off the trace the agent already built while answering. Two
    things the RecursionTrace does not carry are passed in: the rewrite, and
    the stages the provider measured for the first retrieval — those stand in
    for a description of it wherever they exist, so a recursive answer reports
    the same measurements a single-hop one does for the identical retrieval.
    """
    steps: list[TraceStep] = [
        *_rewrite_steps(rewrite, model),
        *(
            list(first_retrieval)
            if first_retrieval
            else [
                TraceStep(
                    phase="retrieve",
                    label="First retrieval",
                    detail=_first_retrieval_detail(rewrite),
                )
            ]
        ),
        TraceStep(
            phase="llm",
            label="First-pass answer",
            detail=(
                f"answered and proposed {recursion.total_followups_proposed} follow-up subtopics"
            ),
            model=model,
        ),
    ]
    for evidence in recursion.subtopic_evidence:
        steps.append(
            TraceStep(
                phase="retrieve",
                label=f"Follow-up: {evidence.subtopic.topic}"[:80],
                detail=f'"{evidence.subtopic.followup_query}" — {evidence.outcome}',
                chunk_ids=chunk_ids_for(evidence.chunks),
                iteration=evidence.subtopic_index,
            )
        )
    ran_synthesis = any(stage.name == "final_synthesis" for stage in recursion.stages)
    if ran_synthesis:
        steps.append(
            TraceStep(
                phase="merge",
                label="Merge evidence",
                detail=(
                    f"{recursion.total_followups_executed} of "
                    f"{recursion.total_followups_proposed} follow-ups executed; "
                    "contexts deduplicated"
                ),
            )
        )
        steps.append(
            TraceStep(
                phase="llm",
                label="Synthesize",
                detail=f"final synthesis over the merged context ({recursion.terminated_reason})",
                model=model,
            )
        )
    else:
        steps.append(
            TraceStep(
                phase="merge",
                label="First-pass answer kept",
                detail=f"no synthesis ran ({recursion.terminated_reason})",
            )
        )
    return steps


def _first_retrieval_detail(rewrite: QueryRewrite | None) -> str:
    """What the first retrieval actually embedded."""
    if rewrite is None:
        return "initial retrieval for the question as asked"
    if rewrite.degraded:
        return "rewrite failed; initial retrieval for the question as asked"
    return f'initial retrieval on the rewritten query "{rewrite.query}"'


def _followups_to_dicts(subtopics: list[Any]) -> list[dict[str, Any]]:
    """Serialise proposed follow-ups, tolerating plain dicts from fakes."""
    followups: list[dict[str, Any]] = []
    for subtopic in subtopics or []:
        if hasattr(subtopic, "model_dump"):
            followups.append(subtopic.model_dump(mode="json"))
        elif isinstance(subtopic, dict):
            followups.append(subtopic)
    return followups

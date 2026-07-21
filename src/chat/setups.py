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
    RagQuestionRequest,
    RecursionOptions,
)
from src.agents.rag_agent import RagAgent
from src.agents.rag_transcript_agent import RagTranscriptAgent
from src.config import Settings
from src.rag.context import MultiTranscriptRagContextProvider
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
        description=(
            "One retrieval across all indexed transcripts, then a single LLM answer."
        ),
    ),
    SetupSpec(
        key="rag_llm_recursive",
        title="rag_llm (recursive)",
        description=(
            "Multi-hop retrieval: follow-up queries fan out, then a final "
            "synthesis call."
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
]

SETUP_KEYS: list[str] = [spec.key for spec in SETUP_SPECS]
_SPECS_BY_KEY: dict[str, SetupSpec] = {spec.key: spec for spec in SETUP_SPECS}


def setup_spec(key: str) -> SetupSpec:
    return _SPECS_BY_KEY[key]


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
    # Identity of the stack that produced the answer. Scores from different
    # models must never be averaged together, so the scoreboard groups on these.
    model: str | None = None
    embedding_model: str | None = None
    top_k: int | None = None


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
        provider = MultiTranscriptRagContextProvider(
            raw_store=raw_store,
            chunk_store=chunk_store,
            indexer=indexer,
            summary_store=summary_store,
        )
        return cls(settings, provider)

    def _rag_llm(self) -> RagTranscriptAgent:
        if self._rag_llm_agent is None:
            self._rag_llm_agent = RagTranscriptAgent.from_settings(
                self._settings, self._provider
            )
        return self._rag_llm_agent

    def _agentic(self) -> RagAgent:
        if self._rag_agent is None:
            self._rag_agent = RagAgent.from_settings(self._settings, self._provider)
        return self._rag_agent

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
    ) -> SetupResult:
        """Answer with one setup.

        ``on_agent_event`` receives per-iteration research events and only ever
        fires for ``rag_agent`` — the other setups make a single retrieval pass
        and have no intermediate steps to report.
        """
        spec = setup_spec(key)
        effective_top_k = top_k or self._settings.rag_top_k
        started = time.monotonic()
        try:
            if key == "rag_agent":
                answer, agent = self._run_rag_agent(
                    question, url, effective_top_k, on_agent_event
                )
                context = agent.last_context
                return self._build_result(
                    spec,
                    url,
                    answer,
                    context,
                    top_k=effective_top_k,
                    iterations=agent.last_iteration_count,
                    terminated_reason=agent.last_terminated_reason,
                    elapsed=time.monotonic() - started,
                )
            answer, agent, llm_calls = self._run_rag_llm(
                key, question, url, effective_top_k
            )
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
            )
        except Exception as exc:  # one failing setup must not abort the comparison
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
            )

    def _run_rag_llm(self, key, question, url, top_k):
        agent = self._rag_llm()
        if key == "rag_llm_recursive":
            request = RagQuestionRequest(
                question=question,
                source_url=url,
                top_k=top_k,
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
            llm_calls = (
                sum(stage.llm_calls for stage in recursion.stages) if recursion else 1
            )
            return answer, agent, llm_calls
        request = RagQuestionRequest(question=question, source_url=url, top_k=top_k)
        return agent.answer(request), agent, 1

    def _run_rag_agent(self, question, url, top_k, on_agent_event=None):
        agent = self._agentic()
        request = RagQuestionRequest(question=question, source_url=url, top_k=top_k)
        if on_agent_event is None:
            return agent.answer(request), agent
        return agent.answer_streaming(request, on_agent_event), agent

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
    ) -> SetupResult:
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
            model=self._settings.deepseek_model,
            embedding_model=self._settings.embedding_model,
            top_k=top_k,
        )

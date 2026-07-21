from __future__ import annotations

import logging
from typing import Callable, Protocol

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from pydantic import ValidationError

from src.agents.context import TranscriptContext
from src.agents.models import (
    AgentProgressEvent,
    RagAnswerReference,
    RagQuestionRequest,
    RagTranscriptAnswer,
)
from src.agents.prompts import AGENTIC_RAG_SYSTEM_PROMPT
from src.agents.rag_transcript_agent import (
    _fallback_references,
    _json_object,
)
from src.config import Settings
from src.rag.context import MultiTranscriptRagContextProvider
from src.rag.embeddings import HuggingFaceEmbeddingModel
from src.rag.indexing import RagIndexer
from src.rag.storage import RawTranscriptStore, TranscriptChunkStore
from src.transcripts.fetcher import SuperdataTranscriptFetcher

logger = logging.getLogger(__name__)


class ChatModel(Protocol):
    def bind_tools(self, tools: list) -> "ChatModel": ...

    def invoke(self, messages: list) -> object: ...


class RagAgent:
    """Agentic RAG agent driven by a LangGraph ReAct research loop.

    The graph has exactly three elements: a ``generate_query_or_respond`` LLM
    node, a ``retrieve`` ToolNode, and a ``route_on_tool_calls`` conditional
    edge. The LLM decides what to research next and when to stop; the only hard
    cap is ``max_iterations`` enforced via LangGraph's ``recursion_limit``.
    """

    def __init__(
        self,
        llm: ChatModel,
        context_provider: MultiTranscriptRagContextProvider,
        max_context_chars: int = 40_000,
        max_iterations: int = 10,
    ) -> None:
        self.llm = llm
        self.context_provider = context_provider
        self.max_context_chars = max_context_chars
        self.max_iterations = max_iterations
        self.last_context: TranscriptContext | None = None
        self.last_iteration_count: int = 0
        self.last_terminated_reason: str = "completed"

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        context_provider: MultiTranscriptRagContextProvider | None = None,
    ) -> "RagAgent":
        kwargs: dict[str, object] = {
            "api_key": settings.deepseek_api_key,
            "model": settings.deepseek_model,
        }
        if settings.deepseek_base_url:
            kwargs["base_url"] = settings.deepseek_base_url
        if context_provider is None:
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
            context_provider = MultiTranscriptRagContextProvider(
                raw_store=raw_store,
                chunk_store=chunk_store,
                indexer=indexer,
            )
        return cls(
            ChatOpenAI(**kwargs),
            context_provider,
            max_iterations=settings.rag_agent_max_iterations,
        )

    def answer(self, request: RagQuestionRequest) -> RagTranscriptAnswer:
        """Run the research loop via ``graph.invoke()`` and return the answer.

        No streaming output. Used by tests and programmatic callers. The graph is
        driven with ``stream_mode="values"`` so that, if the ``max_iterations``
        guard fires, the partial message history is still available to parse the
        best answer the agent produced before termination.
        """
        graph, retrieved_contexts = self._build_graph(request)
        inputs = self._initial_state(request)
        messages: list = list(inputs["messages"])
        try:
            for state in graph.stream(
                inputs, config=self._run_config(), stream_mode="values"
            ):
                if state.get("messages"):
                    messages = state["messages"]
            self.last_terminated_reason = "completed"
        except GraphRecursionError:
            self.last_terminated_reason = "max_iterations_reached"
        self._finalize(request, retrieved_contexts)
        return self._parse_answer({"messages": messages}, request)

    def answer_streaming(
        self,
        request: RagQuestionRequest,
        on_event: Callable[[AgentProgressEvent], None] | None = None,
    ) -> RagTranscriptAnswer:
        """Run the research loop via ``graph.stream()``, calling ``on_event``.

        ``on_event`` is invoked synchronously per node update. Returns the same
        ``RagTranscriptAnswer`` as :meth:`answer` for identical inputs.
        """
        graph, retrieved_contexts = self._build_graph(request)
        inputs = self._initial_state(request)
        # In "updates" mode each node yields only its delta messages. Track the
        # last AI message emitted by the LLM node (with no tool calls) so the
        # final answer can be parsed even though the trailing stream event is a
        # ToolNode result.
        answer_messages: list = []
        iteration = 0
        pending_query: str | None = None
        try:
            for update in graph.stream(
                inputs, config=self._run_config(), stream_mode="updates"
            ):
                for node_name, node_output in update.items():
                    node_messages = (node_output or {}).get("messages", [])
                    if node_name == "generate_query_or_respond":
                        answer_messages = node_messages or answer_messages
                        pending_query = self._emit_generate_event(
                            node_messages, iteration, on_event
                        )
                        if pending_query is not None:
                            iteration += 1
                    elif node_name == "retrieve":
                        self._emit_retrieve_event(
                            node_messages, iteration, pending_query, on_event
                        )
                        pending_query = None
            self.last_terminated_reason = "completed"
        except GraphRecursionError:
            self.last_terminated_reason = "max_iterations_reached"
        self._finalize(request, retrieved_contexts)
        return self._parse_answer({"messages": answer_messages}, request)

    def _emit_generate_event(
        self,
        node_messages: list,
        iteration: int,
        on_event: Callable[[AgentProgressEvent], None] | None,
    ) -> str | None:
        last = node_messages[-1] if node_messages else None
        tool_calls = getattr(last, "tool_calls", None) if last is not None else None
        if tool_calls:
            query = str(tool_calls[0].get("args", {}).get("query", ""))
            if on_event is not None:
                on_event(
                    AgentProgressEvent(
                        iteration=iteration + 1,
                        event_type="retrieval_start",
                        query=query,
                    )
                )
            return query
        if on_event is not None:
            on_event(AgentProgressEvent(iteration=iteration, event_type="answer_start"))
        return None

    def _emit_retrieve_event(
        self,
        node_messages: list,
        iteration: int,
        query: str | None,
        on_event: Callable[[AgentProgressEvent], None] | None,
    ) -> None:
        if on_event is None:
            return
        chunk_count = self._chunk_count_from_tool_message(node_messages)
        on_event(
            AgentProgressEvent(
                iteration=iteration,
                event_type="retrieval_complete",
                query=query,
                chunk_count=chunk_count,
            )
        )

    @staticmethod
    def _chunk_count_from_tool_message(node_messages: list) -> int:
        if not node_messages:
            return 0
        content = getattr(node_messages[-1], "content", "")
        text = content if isinstance(content, str) else "\n".join(map(str, content))
        if not text.strip():
            return 0
        return len([block for block in text.split("\n\n") if block.strip()])

    def _build_graph(
        self, request: RagQuestionRequest
    ) -> tuple[CompiledStateGraph, list[TranscriptContext]]:
        """Construct and compile the LangGraph graph for this request.

        The ``retrieve_transcript_chunks`` tool is a closure capturing
        ``context_provider``, ``source_url``, ``top_k`` and
        ``filter_transcripts`` from the request; only ``query`` varies per call.
        Every retrieved ``TranscriptContext`` is appended to ``retrieved_contexts``
        so the union can be assembled once the loop exits.
        """
        retrieved_contexts: list[TranscriptContext] = []
        source_url = str(request.source_url) if request.source_url else None
        top_k = request.top_k
        filter_transcripts = request.filter_transcripts
        transcript_filter_top_k = request.transcript_filter_top_k
        transcript_filter_min_score = request.transcript_filter_min_score
        channel_id = request.channel_id
        retrieval_mode = request.retrieval_mode
        context_provider = self.context_provider

        @tool
        def retrieve_transcript_chunks(query: str) -> str:
            """Search indexed YouTube transcript chunks for content relevant to the query.

            Returns formatted chunk references with timestamps and source URLs.

            Call this tool whenever you need evidence from the transcript corpus.
            You should call it multiple times - once for the original question to
            identify the key topics, then once per sub-topic that needs deeper
            evidence. Keep calling it until you have enough information to write a
            comprehensive answer.
            """
            context = context_provider.get_context(
                question=query,
                source_url=source_url,
                top_k=top_k,
                filter_transcripts=filter_transcripts,
                transcript_filter_top_k=transcript_filter_top_k,
                transcript_filter_min_score=transcript_filter_min_score,
                channel_id=channel_id,
                retrieval_mode=retrieval_mode,
            )
            retrieved_contexts.append(context)
            return context.context_text or ""

        tools = [retrieve_transcript_chunks]
        bound_llm = self.llm.bind_tools(tools)

        def generate_query_or_respond(state: MessagesState) -> dict:
            response = bound_llm.invoke(state["messages"])
            return {"messages": [response]}

        def route_on_tool_calls(state: MessagesState) -> str:
            last = state["messages"][-1]
            if getattr(last, "tool_calls", None):
                return "retrieve"
            return END

        workflow = StateGraph(MessagesState)
        workflow.add_node("generate_query_or_respond", generate_query_or_respond)
        workflow.add_node("retrieve", ToolNode(tools))
        workflow.add_edge(START, "generate_query_or_respond")
        workflow.add_conditional_edges(
            "generate_query_or_respond",
            route_on_tool_calls,
            {"retrieve": "retrieve", END: END},
        )
        workflow.add_edge("retrieve", "generate_query_or_respond")
        return workflow.compile(), retrieved_contexts

    def _initial_state(self, request: RagQuestionRequest) -> dict:
        return {
            "messages": [
                SystemMessage(content=AGENTIC_RAG_SYSTEM_PROMPT),
                HumanMessage(content=request.question),
            ]
        }

    def _run_config(self) -> dict:
        # Each retrieve iteration costs two graph steps (LLM node + ToolNode);
        # the final answer adds one more LLM node. Allow up to max_iterations
        # retrieve calls before LangGraph raises GraphRecursionError.
        return {"recursion_limit": 2 * self.max_iterations + 1}

    def _finalize(
        self,
        request: RagQuestionRequest,
        retrieved_contexts: list[TranscriptContext],
    ) -> None:
        self.last_iteration_count = len(retrieved_contexts)
        self.last_context = _merge_contexts(retrieved_contexts, request)

    def _parse_answer(
        self, final_state: MessagesState, request: RagQuestionRequest
    ) -> RagTranscriptAnswer:
        """Deserialise the last AI message into a ``RagTranscriptAnswer``.

        Reuses the JSON-parse + fallback pattern from ``RagTranscriptAgent``.
        """
        content = self._last_ai_content(final_state["messages"])
        context = self.last_context or _empty_context(request)
        try:
            data = _json_object(content)
            data.setdefault("question", request.question)
            answer_text = str(data.get("answer", content)).strip()
            references = _parse_references(data.get("references"))
            if not references:
                references = _fallback_references(answer_text, context)
            return RagTranscriptAnswer(
                question=str(data.get("question", request.question)),
                answer=answer_text,
                references=references,
            )
        except (ValueError, ValidationError):
            answer_text = content.strip()
            return RagTranscriptAnswer(
                question=request.question,
                answer=answer_text,
                references=_fallback_references(answer_text, context),
            )

    @staticmethod
    def _last_ai_content(messages: list) -> str:
        for message in reversed(messages):
            is_ai = (
                isinstance(message, AIMessage) or getattr(message, "type", None) == "ai"
            )
            if not is_ai:
                continue
            if getattr(message, "tool_calls", None):
                continue
            content = getattr(message, "content", message)
            if isinstance(content, list):
                return "\n".join(str(item) for item in content)
            text = str(content)
            if text.strip():
                return text
        # Fall back to the last message of any kind with content.
        if messages:
            content = getattr(messages[-1], "content", messages[-1])
            if isinstance(content, list):
                return "\n".join(str(item) for item in content)
            return str(content)
        return ""


def _parse_references(raw) -> list[RagAnswerReference]:
    if not isinstance(raw, list):
        return []
    references: list[RagAnswerReference] = []
    for item in raw:
        try:
            references.append(RagAnswerReference.model_validate(item))
        except ValidationError:
            continue
    return references


def _chunk_key(chunk) -> tuple[str, int]:
    return (chunk.video_id, chunk.chunk_index)


def _merge_contexts(
    contexts: list[TranscriptContext], request: RagQuestionRequest
) -> TranscriptContext:
    if not contexts:
        return _empty_context(request)
    base = contexts[0]
    chunks = []
    selected_transcripts = []
    seen_chunks: set = set()
    seen_transcripts: set = set()
    for context in contexts:
        for chunk in context.retrieved_chunks or []:
            key = _chunk_key(chunk)
            if key in seen_chunks:
                continue
            seen_chunks.add(key)
            chunks.append(chunk)
        for transcript in context.selected_transcripts or []:
            key = getattr(transcript, "summary_id", transcript.video_id)
            if key in seen_transcripts:
                continue
            seen_transcripts.add(key)
            selected_transcripts.append(transcript)
    return TranscriptContext(
        transcript=base.transcript,
        cache_status=base.cache_status,
        context_text="\n\n".join(context.context_text or "" for context in contexts),
        context_mode=base.context_mode,
        retrieved_chunks=chunks,
        selected_transcripts=selected_transcripts,
        top_k=base.top_k,
    )


def _empty_context(request: RagQuestionRequest) -> TranscriptContext:
    from datetime import datetime, timezone

    from src.transcripts.models import Transcript

    return TranscriptContext(
        transcript=Transcript(
            video_id="all",
            url=str(request.source_url)
            if request.source_url
            else "https://www.youtube.com/watch?v=unknown",
            provider="rag",
            raw_text="",
            fetched_at=datetime.now(timezone.utc),
        ),
        cache_status="hit",
        context_text="",
        context_mode="rag",
        retrieved_chunks=[],
        selected_transcripts=[],
        top_k=request.top_k,
    )

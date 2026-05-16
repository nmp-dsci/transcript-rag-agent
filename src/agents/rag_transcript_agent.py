from __future__ import annotations

import json
import re
from typing import Protocol

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.agents.context import TranscriptContext
from src.agents.models import (
    RagAnswerReference,
    RagQuestionRequest,
    RagTranscriptAnswer,
)
from src.agents.prompts import (
    RAG_SYSTEM_PROMPT,
    build_rag_question_prompt,
    build_transcript_context_prompt,
)
from src.config import Settings
from src.rag.context import MultiTranscriptRagContextProvider
from src.rag.embeddings import HuggingFaceEmbeddingModel
from src.rag.indexing import RagIndexer
from src.rag.references import youtube_timestamp_url
from src.rag.storage import RawTranscriptStore, TranscriptChunkStore
from src.transcripts.fetcher import SuperdataTranscriptFetcher


class RagContextTooLongError(RuntimeError):
    pass


class ChatModel(Protocol):
    def invoke(self, messages: list[SystemMessage | HumanMessage]) -> object:
        ...


class RagTranscriptAgent:
    def __init__(
        self,
        llm: ChatModel,
        context_provider: MultiTranscriptRagContextProvider,
        max_context_chars: int = 40_000,
    ) -> None:
        self.llm = llm
        self.context_provider = context_provider
        self.max_context_chars = max_context_chars
        self.last_context: TranscriptContext | None = None

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        context_provider: MultiTranscriptRagContextProvider | None = None,
    ) -> "RagTranscriptAgent":
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
        return cls(ChatOpenAI(**kwargs), context_provider)

    def answer(self, request: RagQuestionRequest) -> RagTranscriptAnswer:
        context = self.context_provider.get_context(
            question=request.question,
            source_url=str(request.source_url) if request.source_url else None,
            top_k=request.top_k,
            filter_transcripts=request.filter_transcripts,
            transcript_filter_top_k=request.transcript_filter_top_k,
            transcript_filter_min_score=request.transcript_filter_min_score,
        )
        self.last_context = context
        context_text = context.context_text or ""
        if len(context_text) > self.max_context_chars:
            raise RagContextTooLongError("Retrieved RAG context is too long")
        content = self._invoke(
            context_text=context_text,
            user_prompt=build_rag_question_prompt(request.question),
        )
        data = _json_object(content)
        data.setdefault("question", request.question)
        answer = RagTranscriptAnswer.model_validate(data)
        if not answer.references:
            answer = answer.model_copy(
                update={"references": _fallback_references(answer.answer, context)}
            )
        return answer

    def _invoke(self, context_text: str, user_prompt: str) -> str:
        response = self.llm.invoke(
            [
                SystemMessage(content=RAG_SYSTEM_PROMPT),
                SystemMessage(content=build_transcript_context_prompt(context_text)),
                HumanMessage(content=user_prompt),
            ]
        )
        content = getattr(response, "content", response)
        if isinstance(content, list):
            return "\n".join(str(item) for item in content)
        return str(content)


def _fallback_references(
    answer_text: str, context: TranscriptContext
) -> list[RagAnswerReference]:
    cited = {
        int(match)
        for match in re.findall(r"\[(\d+)\]", answer_text)
        if match.isdigit()
    }
    chunks = context.retrieved_chunks or []
    if not cited:
        cited = set(range(1, len(chunks) + 1))
    references: list[RagAnswerReference] = []
    for label_index in sorted(cited):
        chunk_index = label_index - 1
        if chunk_index < 0 or chunk_index >= len(chunks):
            continue
        chunk = chunks[chunk_index]
        references.append(
            RagAnswerReference(
                label=f"[{label_index}]",
                source_url=chunk.source_url,
                timestamp_url=youtube_timestamp_url(
                    str(chunk.source_url), chunk.start_seconds
                ),
                start_seconds=chunk.start_seconds,
                end_seconds=chunk.end_seconds,
                chunk_index=chunk.chunk_index,
                video_id=chunk.video_id,
            )
        )
    return references


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

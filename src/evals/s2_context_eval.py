from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.agents.context import RawTranscriptContextProvider, TranscriptContext
from src.agents.models import QuestionRequest, TranscriptAnswer
from src.agents.transcript_agent import TranscriptAgent
from src.config import ConfigError, load_settings
from src.rag.context import RagTranscriptContextProvider
from src.rag.embeddings import HuggingFaceEmbeddingModel
from src.rag.eval import compare_answers
from src.rag.indexing import RagIndexer
from src.rag.storage import RawTranscriptStore, TranscriptChunkStore
from src.transcripts.fetcher import SuperdataTranscriptFetcher
from src.transcripts.youtube import extract_video_id


DEFAULT_VIDEO_URL = "https://www.youtube.com/watch?v=3hk7nO_q0a8"
DEFAULT_QUESTION = (
    "what does this video say  for capital gains tax, is it being grandfathered "
    "or every now under new rules, does that mean if I sell before 30 June 2027 "
    "I can still access 50% discount "
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="s2-context-eval",
        description="Compare raw transcript and top-k RAG answers for the S2 CGT question.",
    )
    parser.add_argument("--url", default=DEFAULT_VIDEO_URL)
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        payload = run_evaluation(
            source_url=args.url,
            question=args.question,
            top_k=args.top_k,
        )
    except (ConfigError, Exception) as exc:
        parser.exit(1, f"Error: {exc}\n")

    output = json.dumps(payload, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    print(output)
    return 0


def run_evaluation(
    source_url: str = DEFAULT_VIDEO_URL,
    question: str = DEFAULT_QUESTION,
    top_k: int | None = None,
) -> dict[str, Any]:
    settings = load_settings(require_keys=True)
    video_id = extract_video_id(source_url)
    resolved_top_k = top_k or settings.rag_top_k

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

    raw_provider = RawTranscriptContextProvider(raw_store, fetcher)
    rag_provider = RagTranscriptContextProvider(
        raw_store=raw_store,
        chunk_store=chunk_store,
        indexer=indexer,
        top_k=resolved_top_k,
    )
    raw_agent = TranscriptAgent.from_settings(settings, raw_provider)
    rag_agent = TranscriptAgent.from_settings(settings, rag_provider)
    request = QuestionRequest(video_id=video_id, source_url=source_url, question=question)

    raw_answer = raw_agent.answer(request)
    rag_answer = rag_agent.answer(request)
    if raw_agent.last_context is None or rag_agent.last_context is None:
        raise RuntimeError("Evaluation did not capture both context payloads")

    comparison = compare_answers(
        question=question,
        raw_answer=raw_answer.answer,
        rag_answer=rag_answer.answer,
        raw_prompt_context=raw_agent.last_context.context_text or "",
        rag_prompt_context=rag_agent.last_context.context_text or "",
        embedding_model=embedding_model,
    )
    return build_payload(
        video_id=video_id,
        source_url=source_url,
        top_k=resolved_top_k,
        raw_answer=raw_answer,
        rag_answer=rag_answer,
        raw_context=raw_agent.last_context,
        rag_context=rag_agent.last_context,
        comparison=comparison.model_dump(mode="json"),
    )


def build_payload(
    video_id: str,
    source_url: str,
    top_k: int,
    raw_answer: TranscriptAnswer,
    rag_answer: TranscriptAnswer,
    raw_context: TranscriptContext,
    rag_context: TranscriptContext,
    comparison: dict[str, Any],
) -> dict[str, Any]:
    return {
        "eval_name": "s2_raw_vs_rag_cgt_question",
        "video_id": video_id,
        "source_url": source_url,
        "question": raw_answer.question,
        "top_k": top_k,
        "raw": {
            "answer": raw_answer.answer,
            "cache_status": raw_context.cache_status,
            "prompt_tokens_estimate": comparison["raw_prompt_tokens_estimate"],
        },
        "rag": {
            "answer": rag_answer.answer,
            "cache_status": rag_context.cache_status,
            "prompt_tokens_estimate": comparison["rag_prompt_tokens_estimate"],
            "retrieved_chunks": [
                {
                    "chunk_index": chunk.chunk_index,
                    "score": chunk.score,
                    "start_seconds": chunk.start_seconds,
                    "end_seconds": chunk.end_seconds,
                    "text": chunk.text,
                }
                for chunk in (rag_context.retrieved_chunks or [])
            ],
        },
        "comparison": {
            "semantic_similarity": comparison["semantic_similarity"],
            "token_savings_percent": comparison["token_savings_percent"],
        },
    }


if __name__ == "__main__":
    raise SystemExit(main())

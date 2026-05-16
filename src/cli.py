from __future__ import annotations

import argparse
import sys
from pathlib import Path

from langchain_openai import ChatOpenAI

from src.agents.context import RawTranscriptContextProvider
from src.agents.models import QuestionRequest, RagQuestionRequest, SummaryRequest
from src.agents.rag_transcript_agent import RagTranscriptAgent
from src.agents.transcript_agent import TranscriptAgent
from src.config import ConfigError, load_settings
from src.dashboard.rag_pipeline import (
    collect_filter_test_rows,
    collect_pipeline_rows,
    write_dashboard,
)
from src.observability import (
    cli_run,
    log_answer,
    log_context_comparison,
    log_context_details,
    log_raw_transcript_metadata,
    log_summary,
    log_transcript,
    log_transcript_filter_details,
)
from src.rag.context import MultiTranscriptRagContextProvider, RagTranscriptContextProvider
from src.rag.embeddings import HuggingFaceEmbeddingModel
from src.rag.eval import compare_answers, estimate_tokens
from src.rag.indexing import RagIndexer
from src.rag.storage import RawTranscriptStore, TranscriptChunkStore
from src.rag.summaries import TranscriptSummaryGenerator, TranscriptSummaryStore
from src.transcripts.fetcher import SuperdataTranscriptFetcher
from src.transcripts.youtube import extract_video_id


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="yt-agent")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch = subparsers.add_parser("fetch", help="Fetch and cache a transcript")
    fetch.add_argument("url")
    fetch.add_argument("--no-refresh", action="store_true")

    fetch_raw = subparsers.add_parser("fetch-raw", help="Fetch and cache raw segments")
    fetch_raw.add_argument("url")
    fetch_raw.add_argument("--no-refresh", action="store_true")

    index_rag = subparsers.add_parser("index-rag", help="Index a transcript for RAG")
    index_rag.add_argument("url")
    index_rag.add_argument("--refresh", action="store_true")
    index_rag.add_argument("--refresh-summary", action="store_true")

    summarize = subparsers.add_parser("summarize", help="Summarize a transcript")
    summarize.add_argument("url")

    ask = subparsers.add_parser("ask", help="Ask a question about a transcript")
    ask.add_argument("url")
    ask.add_argument("question")
    ask.add_argument("--context", choices=["raw", "rag"], default="raw")
    ask.add_argument("--top-k", type=int, default=None)

    compare = subparsers.add_parser(
        "compare-context", help="Compare raw and RAG answers"
    )
    compare.add_argument("url")
    compare.add_argument("question")
    compare.add_argument("--top-k", type=int, default=None)

    rag_ask = subparsers.add_parser(
        "rag-ask", help="Ask across all indexed transcript chunks"
    )
    rag_ask.add_argument("question")
    rag_ask.add_argument("--url")
    rag_ask.add_argument("--top-k", type=int, default=None)
    rag_ask.add_argument("--filter-transcripts", action="store_true")
    rag_ask.add_argument("--transcript-filter-top-k", type=int, default=None)
    rag_ask.add_argument("--transcript-filter-min-score", type=float, default=None)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        settings = load_settings(require_keys=True)
        source_url = getattr(args, "url", None)
        video_id = extract_video_id(source_url) if source_url else None
        with cli_run(args.command, settings, video_id):
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
            raw_provider = RawTranscriptContextProvider(raw_store, fetcher)

            if args.command in {"fetch", "fetch-raw"}:
                context = raw_provider.get_or_refresh_transcript(
                    video_id, args.url, no_refresh=args.no_refresh
                )
                log_transcript(context.transcript, context.cache_status, settings)
                print(_format_fetch(context.transcript, context.cache_status))
                return 0

            if args.command == "index-rag":
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
                    summary_store=_build_summary_store(
                        settings, embedding_model, raw_store
                    ),
                    summary_generator=_build_summary_generator(settings),
                )
                result = indexer.index(
                    args.url,
                    refresh=args.refresh,
                    refresh_summary=args.refresh_summary,
                )
                log_raw_transcript_metadata(result.raw_document)
                print(
                    _format_index(
                        raw_collection=settings.raw_transcript_collection,
                        chunk_collection=settings.chunk_collection,
                        summary_collection=settings.transcript_summary_collection,
                        chunk_count=len(result.chunks),
                        summary_status=result.summary_status,
                        chroma_path=settings.chroma_path,
                    )
                )
                dashboard_path = _refresh_rag_pipeline_dashboard(settings)
                print(f"RAG pipeline dashboard: {dashboard_path}")
                return 0

            if args.command == "summarize":
                agent = TranscriptAgent.from_settings(settings, raw_provider)
                summary = agent.summarize(
                    SummaryRequest(video_id=video_id, source_url=args.url)
                )
                _log_last_context(agent, settings)
                log_summary(summary)
                print(_format_summary(summary.summary, summary.top_findings))
                return 0

            if args.command == "ask":
                context_mode = args.context
                top_k = args.top_k or settings.rag_top_k
                context_provider = raw_provider
                if context_mode == "rag":
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
                    context_provider = RagTranscriptContextProvider(
                        raw_store=raw_store,
                        chunk_store=chunk_store,
                        indexer=indexer,
                        top_k=top_k,
                    )
                agent = TranscriptAgent.from_settings(settings, context_provider)
                answer = agent.answer(
                    QuestionRequest(
                        video_id=video_id,
                        source_url=args.url,
                        question=args.question,
                    )
                )
                _log_last_context(agent, settings)
                if agent.last_context is not None:
                    log_context_details(
                        context_mode=agent.last_context.context_mode,
                        top_k=agent.last_context.top_k,
                        retrieved_chunks=agent.last_context.retrieved_chunks,
                        raw_prompt_tokens_estimate=(
                            estimate_tokens(agent.last_context.context_text or "")
                            if agent.last_context.context_mode == "raw"
                            else None
                        ),
                        rag_prompt_tokens_estimate=(
                            estimate_tokens(agent.last_context.context_text or "")
                            if agent.last_context.context_mode == "rag"
                            else None
                        ),
                    )
                log_answer(answer)
                print(answer.answer)
                return 0

            if args.command == "compare-context":
                top_k = args.top_k or settings.rag_top_k
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
                raw_agent = TranscriptAgent.from_settings(settings, raw_provider)
                rag_agent = TranscriptAgent.from_settings(
                    settings,
                    RagTranscriptContextProvider(
                        raw_store=raw_store,
                        chunk_store=chunk_store,
                        indexer=indexer,
                        top_k=top_k,
                    ),
                )
                request = QuestionRequest(
                    video_id=video_id,
                    source_url=args.url,
                    question=args.question,
                )
                raw_answer = raw_agent.answer(request)
                rag_answer = rag_agent.answer(request)
                comparison = compare_answers(
                    question=args.question,
                    raw_answer=raw_answer.answer,
                    rag_answer=rag_answer.answer,
                    raw_prompt_context=raw_agent.last_context.context_text
                    if raw_agent.last_context
                    else "",
                    rag_prompt_context=rag_agent.last_context.context_text
                    if rag_agent.last_context
                    else "",
                    embedding_model=embedding_model,
                )
                log_context_comparison(comparison)
                if rag_agent.last_context is not None:
                    log_context_details(
                        context_mode=rag_agent.last_context.context_mode,
                        top_k=rag_agent.last_context.top_k,
                        retrieved_chunks=rag_agent.last_context.retrieved_chunks,
                        rag_prompt_tokens_estimate=comparison.rag_prompt_tokens_estimate,
                        raw_prompt_tokens_estimate=comparison.raw_prompt_tokens_estimate,
                    )
                print(_format_comparison(comparison))
                return 0

            if args.command == "rag-ask":
                top_k = args.top_k or settings.rag_top_k
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
                    summary_store=_build_summary_store(
                        settings, embedding_model, raw_store
                    ),
                )
                agent = RagTranscriptAgent.from_settings(settings, context_provider)
                filter_top_k = (
                    args.transcript_filter_top_k or settings.transcript_filter_top_k
                )
                filter_min_score = (
                    args.transcript_filter_min_score
                    if args.transcript_filter_min_score is not None
                    else settings.transcript_filter_min_score
                )
                answer = agent.answer(
                    RagQuestionRequest(
                        question=args.question,
                        source_url=args.url,
                        top_k=top_k,
                        filter_transcripts=(
                            args.filter_transcripts and args.url is None
                        ),
                        transcript_filter_top_k=filter_top_k,
                        transcript_filter_min_score=filter_min_score,
                    )
                )
                if agent.last_context is not None:
                    log_context_details(
                        context_mode=agent.last_context.context_mode,
                        top_k=agent.last_context.top_k,
                        retrieved_chunks=agent.last_context.retrieved_chunks,
                        rag_prompt_tokens_estimate=estimate_tokens(
                            agent.last_context.context_text or ""
                        ),
                    )
                    log_transcript_filter_details(
                        enabled=args.filter_transcripts and args.url is None,
                        selected_transcripts=agent.last_context.selected_transcripts,
                        filter_top_k=filter_top_k,
                        min_score=filter_min_score,
                        retrieved_chunks=agent.last_context.retrieved_chunks,
                    )
                print(
                    _format_rag_answer(
                        answer,
                        selected_transcripts=agent.last_context.selected_transcripts
                        if agent.last_context is not None
                        else [],
                    )
                )
                return 0

        parser.error(f"Unknown command: {args.command}")
        return 2
    except (ConfigError, Exception) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


def _log_last_context(agent: TranscriptAgent, settings) -> None:
    if agent.last_context is None:
        return
    log_transcript(
        agent.last_context.transcript,
        agent.last_context.cache_status,
        settings,
    )


def _format_fetch(transcript, cache_status: str) -> str:
    return "\n".join(
        [
            f"Transcript cached: {transcript.video_id}",
            f"Cache status: {cache_status}",
            f"Characters: {len(transcript.raw_text)}",
        ]
    )


def _format_summary(summary: str, top_findings: list[str]) -> str:
    lines = ["Summary", summary, "", "Top 3 findings"]
    lines.extend(f"{index}. {finding}" for index, finding in enumerate(top_findings, 1))
    return "\n".join(lines)


def _format_index(
    raw_collection: str,
    chunk_collection: str,
    summary_collection: str,
    chunk_count: int,
    summary_status: str | None,
    chroma_path,
) -> str:
    return "\n".join(
        [
            "RAG index updated",
            f"Raw transcript collection: {raw_collection}",
            f"Chunk collection: {chunk_collection}",
            f"Transcript summary collection: {summary_collection}",
            f"Chunks: {chunk_count}",
            f"Summary: {summary_status or 'not configured'}",
            f"Chroma path: {chroma_path}",
        ]
    )


def _format_comparison(comparison) -> str:
    return "\n".join(
        [
            "Raw answer",
            comparison.raw_answer,
            "",
            "RAG answer",
            comparison.rag_answer,
            "",
            f"Semantic similarity: {comparison.semantic_similarity:.3f}",
            f"Raw prompt tokens estimate: {comparison.raw_prompt_tokens_estimate}",
            f"RAG prompt tokens estimate: {comparison.rag_prompt_tokens_estimate}",
            f"Token savings percent: {comparison.token_savings_percent:.1f}",
        ]
    )


def _format_rag_answer(answer, selected_transcripts=None) -> str:
    lines = []
    if selected_transcripts:
        lines.append("Selected transcripts")
        for index, transcript in enumerate(selected_transcripts, 1):
            score = (
                "unknown"
                if transcript.score is None
                else f"{transcript.score:.3f}"
            )
            lines.append(
                f"{index}. score={score} video={transcript.video_id} "
                f"url={transcript.source_url}"
            )
        lines.append("")
        lines.append("Answer")
    lines.append(answer.answer)
    if answer.references:
        lines.extend(["", "References"])
        for reference in answer.references:
            start = (
                "unknown"
                if reference.start_seconds is None
                else str(int(reference.start_seconds))
            )
            end = (
                "unknown"
                if reference.end_seconds is None
                else str(int(reference.end_seconds))
            )
            lines.append(
                f"{reference.label} {reference.timestamp_url} "
                f"{start}-{end}s video={reference.video_id}"
            )
    return "\n".join(lines)


def _build_summary_store(settings, embedding_model, raw_store):
    return TranscriptSummaryStore(
        settings.chroma_path,
        embedding_model=embedding_model,
        embedding_model_name=settings.embedding_model,
        raw_store=raw_store,
        collection_name=settings.transcript_summary_collection,
    )


def _build_summary_generator(settings):
    kwargs: dict[str, object] = {
        "api_key": settings.deepseek_api_key,
        "model": settings.deepseek_model,
    }
    if settings.deepseek_base_url:
        kwargs["base_url"] = settings.deepseek_base_url
    return TranscriptSummaryGenerator(
        ChatOpenAI(**kwargs),
        model_name=settings.deepseek_model,
    )


def _refresh_rag_pipeline_dashboard(settings) -> Path:
    output = Path("dashboard/rag_pipeline.html")
    rows = collect_pipeline_rows(settings)
    filter_test_rows = collect_filter_test_rows(settings, rows)
    write_dashboard(
        output=output,
        rows=rows,
        settings=settings,
        filter_test_rows=filter_test_rows,
    )
    return output


if __name__ == "__main__":
    raise SystemExit(main())

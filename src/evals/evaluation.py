from __future__ import annotations

import argparse
import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.agents.context import RawTranscriptContextProvider
from src.agents.models import QuestionRequest, RagQuestionRequest
from src.agents.rag_transcript_agent import RagTranscriptAgent
from src.agents.transcript_agent import TranscriptAgent
from src.config import ConfigError, load_settings
from src.rag.context import MultiTranscriptRagContextProvider, RagTranscriptContextProvider
from src.rag.embeddings import HuggingFaceEmbeddingModel, cosine_similarity
from src.rag.eval import estimate_tokens
from src.rag.indexing import RagIndexer
from src.rag.references import youtube_timestamp_url
from src.rag.storage import RawTranscriptStore, TranscriptChunkStore
from src.transcripts.fetcher import SuperdataTranscriptFetcher
from src.transcripts.youtube import extract_video_id


DEFAULT_VIDEO_URL = "https://www.youtube.com/watch?v=3hk7nO_q0a8"
DEFAULT_QUESTION = (
    "what does this video say  for capital gains tax, is it being grandfathered "
    "or every now under new rules, does that mean if I sell before 30 June 2027 "
    "I can still access 50% discount "
)


@dataclass(frozen=True)
class EvaluationRun:
    name: str
    input_type: str
    answer: str
    context_text: str
    retrieved_chunks: list[Any]
    source_url: str | None = None

    @property
    def token_estimate(self) -> int:
        return estimate_tokens(self.context_text)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evaluation",
        description=(
            "Compare raw single-transcript, RAG single-transcript, and RAG "
            "all-transcripts answers."
        ),
    )
    parser.add_argument("--url", default=DEFAULT_VIDEO_URL)
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--output", type=Path, default=Path("evaluation/evaluation.html"))
    parser.add_argument("--json-output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = run_evaluation(
            source_url=args.url,
            question=args.question,
            top_k=args.top_k,
        )
    except (ConfigError, Exception) as exc:
        parser.exit(1, f"Error: {exc}\n")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report["html"], encoding="utf-8")
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(report["json"], indent=2) + "\n", encoding="utf-8"
        )
    print(f"Wrote {args.output}")
    return 0


def run_evaluation(
    source_url: str = DEFAULT_VIDEO_URL,
    question: str = DEFAULT_QUESTION,
    top_k: int | None = None,
) -> dict[str, Any]:
    settings = load_settings(require_keys=True)
    video_id = extract_video_id(source_url)
    resolved_top_k = top_k or settings.rag_top_k

    fetcher = SuperdataTranscriptFetcher(settings.superdata_api_key)
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

    raw_agent = TranscriptAgent.from_settings(
        settings,
        RawTranscriptContextProvider(raw_store, fetcher),
    )
    rag_single_agent = TranscriptAgent.from_settings(
        settings,
        RagTranscriptContextProvider(
            raw_store=raw_store,
            chunk_store=chunk_store,
            indexer=indexer,
            top_k=resolved_top_k,
        ),
    )
    rag_all_agent = RagTranscriptAgent.from_settings(
        settings,
        MultiTranscriptRagContextProvider(
            raw_store=raw_store,
            chunk_store=chunk_store,
            indexer=indexer,
        ),
    )

    raw_answer = raw_agent.answer(
        QuestionRequest(video_id=video_id, source_url=source_url, question=question)
    )
    rag_single_answer = rag_single_agent.answer(
        QuestionRequest(video_id=video_id, source_url=source_url, question=question)
    )
    rag_all_answer = rag_all_agent.answer(
        RagQuestionRequest(question=question, top_k=resolved_top_k)
    )
    if (
        raw_agent.last_context is None
        or rag_single_agent.last_context is None
        or rag_all_agent.last_context is None
    ):
        raise RuntimeError("Evaluation did not capture all context payloads")

    runs = [
        EvaluationRun(
            name="raw_single",
            input_type="raw",
            source_url=source_url,
            answer=raw_answer.answer,
            context_text=raw_agent.last_context.context_text or "",
            retrieved_chunks=[],
        ),
        EvaluationRun(
            name="rag_single",
            input_type="rag single",
            source_url=source_url,
            answer=rag_single_answer.answer,
            context_text=rag_single_agent.last_context.context_text or "",
            retrieved_chunks=rag_single_agent.last_context.retrieved_chunks or [],
        ),
        EvaluationRun(
            name="rag_all",
            input_type="rag all",
            source_url=None,
            answer=rag_all_answer.answer,
            context_text=rag_all_agent.last_context.context_text or "",
            retrieved_chunks=rag_all_agent.last_context.retrieved_chunks or [],
        ),
    ]
    embeddings = embedding_model.embed_documents([run.answer for run in runs])
    similarities = _pairwise_similarities(runs, embeddings)
    payload = _json_payload(
        question=question,
        source_url=source_url,
        top_k=resolved_top_k,
        runs=runs,
        similarities=similarities,
    )
    return {
        "html": render_html_report(
            question=question,
            source_url=source_url,
            top_k=resolved_top_k,
            runs=runs,
            similarities=similarities,
        ),
        "json": payload,
    }


def _pairwise_similarities(
    runs: list[EvaluationRun],
    embeddings: list[list[float]],
) -> dict[str, float]:
    values: dict[str, float] = {}
    for left_index, left in enumerate(runs):
        for right_index in range(left_index + 1, len(runs)):
            right = runs[right_index]
            values[f"{left.name}__{right.name}"] = cosine_similarity(
                embeddings[left_index], embeddings[right_index]
            )
    return values


def render_html_report(
    question: str,
    source_url: str,
    top_k: int,
    runs: list[EvaluationRun],
    similarities: dict[str, float],
) -> str:
    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="en">',
            "<head>",
            '<meta charset="utf-8">',
            "<title>Transcript Agent Evaluation</title>",
            "<style>",
            "body{font-family:Arial,sans-serif;line-height:1.45;margin:32px;color:#18202a}",
            "table{border-collapse:collapse;width:100%;margin:16px 0}",
            "th,td{border:1px solid #ccd3dd;padding:8px;text-align:left;vertical-align:top}",
            "th{background:#eef2f6} article{border-top:2px solid #ccd3dd;margin-top:28px;padding-top:16px}",
            "pre{white-space:pre-wrap;background:#f7f8fa;border:1px solid #d8dee8;padding:12px;overflow:auto}",
            "details{margin:8px 0;padding:8px;border:1px solid #d8dee8;background:#fbfcfd}",
            "summary{cursor:pointer;font-weight:bold}",
            ".metric{font-family:ui-monospace,Menlo,monospace}",
            "</style>",
            "</head>",
            "<body>",
            "<h1>Transcript Agent Evaluation</h1>",
            f"<p><strong>Question:</strong> {html.escape(question)}</p>",
            f"<p><strong>Single transcript URL:</strong> {html.escape(source_url)}</p>",
            f"<p><strong>RAG top K:</strong> {top_k}</p>",
            "<h2>Summary</h2>",
            _summary_table(runs),
            "<h2>Pairwise Similarity</h2>",
            _similarity_table(similarities),
            "<h2>Answers</h2>",
            *[_run_section(run) for run in runs],
            "</body>",
            "</html>",
        ]
    )


def _summary_table(runs: list[EvaluationRun]) -> str:
    rows = [
        "<tr><th>Run</th><th>Transcript input type</th><th>Filter</th><th>Token estimate</th><th>Retrieved chunks</th><th>Answer chars</th></tr>"
    ]
    for run in runs:
        rows.append(
            "<tr>"
            f"<td>{html.escape(run.name)}</td>"
            f"<td>{html.escape(run.input_type)}</td>"
            f"<td>{html.escape(run.source_url or 'all indexed transcripts')}</td>"
            f"<td class=\"metric\">{run.token_estimate}</td>"
            f"<td class=\"metric\">{len(run.retrieved_chunks)}</td>"
            f"<td class=\"metric\">{len(run.answer)}</td>"
            "</tr>"
        )
    return "<table>" + "".join(rows) + "</table>"


def _similarity_table(similarities: dict[str, float]) -> str:
    rows = ["<tr><th>Pair</th><th>Embedding cosine similarity</th></tr>"]
    for pair, score in similarities.items():
        rows.append(
            "<tr>"
            f"<td>{html.escape(pair)}</td>"
            f"<td class=\"metric\">{score:.3f}</td>"
            "</tr>"
        )
    return "<table>" + "".join(rows) + "</table>"


def _run_section(run: EvaluationRun) -> str:
    chunks = "\n".join(
        _chunk_details(index, chunk)
        for index, chunk in enumerate(run.retrieved_chunks, 1)
    )
    if not chunks:
        chunks = "<p>No retrieved chunks for raw transcript input.</p>"
    return "\n".join(
        [
            "<article>",
            f"<h3>{html.escape(run.name)} ({html.escape(run.input_type)})</h3>",
            f"<p><strong>Filter:</strong> {html.escape(run.source_url or 'all indexed transcripts')}</p>",
            f"<p><strong>Token estimate:</strong> <span class=\"metric\">{run.token_estimate}</span></p>",
            "<h4>Answer</h4>",
            f"<pre>{html.escape(run.answer)}</pre>",
            "<h4>Retrieved chunks</h4>",
            chunks,
            "</article>",
        ]
    )


def _chunk_details(index: int, chunk) -> str:
    timestamp_url = youtube_timestamp_url(str(chunk.source_url), chunk.start_seconds)
    summary = (
        f"[{index}] {chunk.video_id} "
        f"{chunk.start_seconds}-{chunk.end_seconds}s "
        f"score={chunk.score}"
    )
    return "\n".join(
        [
            "<details>",
            f"<summary>{html.escape(summary)}</summary>",
            f'<p><a href="{html.escape(timestamp_url)}">Open video at timestamp</a></p>',
            f"<p>chunk_index={chunk.chunk_index}</p>",
            f"<pre>{html.escape(chunk.text)}</pre>",
            "</details>",
        ]
    )


def _json_payload(
    question: str,
    source_url: str,
    top_k: int,
    runs: list[EvaluationRun],
    similarities: dict[str, float],
) -> dict[str, Any]:
    return {
        "eval_name": "raw_vs_rag_single_vs_rag_all",
        "question": question,
        "source_url": source_url,
        "top_k": top_k,
        "runs": [
            {
                "name": run.name,
                "input_type": run.input_type,
                "source_url": run.source_url,
                "answer": run.answer,
                "token_estimate": run.token_estimate,
                "retrieved_chunks": [
                    {
                        "rank": index,
                        "score": chunk.score,
                        "video_id": chunk.video_id,
                        "source_url": str(chunk.source_url),
                        "timestamp_url": youtube_timestamp_url(
                            str(chunk.source_url), chunk.start_seconds
                        ),
                        "start_seconds": chunk.start_seconds,
                        "end_seconds": chunk.end_seconds,
                        "chunk_index": chunk.chunk_index,
                        "text": chunk.text,
                    }
                    for index, chunk in enumerate(run.retrieved_chunks, 1)
                ],
            }
            for run in runs
        ],
        "similarities": similarities,
    }


if __name__ == "__main__":
    raise SystemExit(main())

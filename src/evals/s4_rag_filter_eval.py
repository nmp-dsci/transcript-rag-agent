from __future__ import annotations

import argparse
import html
import json
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow

from src.agents.models import RagQuestionRequest
from src.agents.rag_transcript_agent import RagTranscriptAgent
from src.config import ConfigError, load_settings
from src.dashboard.theme import dark_style_block
from src.observability import setup_mlflow
from src.rag.context import MultiTranscriptRagContextProvider
from src.rag.embeddings import HuggingFaceEmbeddingModel, cosine_similarity
from src.rag.eval import estimate_tokens
from src.rag.indexing import RagIndexer
from src.rag.references import youtube_timestamp_url
from src.rag.storage import RawTranscriptStore, TranscriptChunkStore
from src.rag.summaries import TranscriptSummaryStore
from src.transcripts.fetcher import SuperdataTranscriptFetcher


DEFAULT_QUESTION = (
    "what does this video say  for capital gains tax, is it being grandfathered "
    "or every now under new rules, does that mean if I sell before 30 June 2027 "
    "I can still access 50% discount "
)


@dataclass(frozen=True)
class S4Run:
    name: str
    filter_enabled: bool
    answer: str
    context_text: str
    retrieved_chunks: list[Any]
    selected_transcripts: list[Any]
    time_seconds: float

    @property
    def token_estimate(self) -> int:
        return estimate_tokens(self.context_text)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="s4_rag_filter_eval",
        description="Compare all-chunk RAG with transcript-summary filtered RAG.",
    )
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--transcript-filter-top-k", type=int, default=None)
    parser.add_argument("--transcript-filter-min-score", type=float, default=None)
    parser.add_argument("--output", type=Path, default=Path("dashboard/s4_rag_filter.html"))
    parser.add_argument("--json-output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = run_evaluation(
            question=args.question,
            top_k=args.top_k,
            transcript_filter_top_k=args.transcript_filter_top_k,
            transcript_filter_min_score=args.transcript_filter_min_score,
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
    question: str = DEFAULT_QUESTION,
    top_k: int | None = None,
    transcript_filter_top_k: int | None = None,
    transcript_filter_min_score: float | None = None,
) -> dict[str, Any]:
    settings = load_settings(require_keys=True)
    resolved_top_k = top_k or settings.rag_top_k
    filter_top_k = transcript_filter_top_k or settings.transcript_filter_top_k
    filter_min_score = (
        transcript_filter_min_score
        if transcript_filter_min_score is not None
        else settings.transcript_filter_min_score
    )

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
    summary_store = TranscriptSummaryStore(
        settings.chroma_path,
        embedding_model=embedding_model,
        embedding_model_name=settings.embedding_model,
        raw_store=raw_store,
        collection_name=settings.transcript_summary_collection,
    )
    indexer = RagIndexer(
        raw_store=raw_store,
        chunk_store=chunk_store,
        target_chars=settings.chunk_target_chars,
        overlap_chars=settings.chunk_overlap_chars,
    )
    provider = MultiTranscriptRagContextProvider(
        raw_store=raw_store,
        chunk_store=chunk_store,
        indexer=indexer,
        summary_store=summary_store,
    )
    agent = RagTranscriptAgent.from_settings(settings, provider)

    setup_mlflow(settings)
    with mlflow.start_run(run_name="s4_rag_filter_eval"):
        mlflow.log_param("evaluation", "s4_rag_filter")
        mlflow.log_param("question", question)
        mlflow.log_param("top_k", resolved_top_k)
        mlflow.log_param("transcript_filter_top_k", filter_top_k)
        mlflow.log_param("transcript_filter_min_score", filter_min_score)

        unfiltered = _run_agent(
            agent,
            RagQuestionRequest(question=question, top_k=resolved_top_k),
            "rag_all_unfiltered",
        )
        filtered = _run_agent(
            agent,
            RagQuestionRequest(
                question=question,
                top_k=resolved_top_k,
                filter_transcripts=True,
                transcript_filter_top_k=filter_top_k,
                transcript_filter_min_score=filter_min_score,
            ),
            "rag_all_filtered",
        )
        answer_embeddings = embedding_model.embed_documents(
            [unfiltered.answer, filtered.answer]
        )
        answer_similarity = cosine_similarity(answer_embeddings[0], answer_embeddings[1])
        token_delta = filtered.token_estimate - unfiltered.token_estimate
        time_delta = filtered.time_seconds - unfiltered.time_seconds
        token_percent = _percent_change(unfiltered.token_estimate, filtered.token_estimate)
        time_percent = _percent_change(unfiltered.time_seconds, filtered.time_seconds)
        chunk_overlap = len(
            {chunk.video_id for chunk in unfiltered.retrieved_chunks}
            & {chunk.video_id for chunk in filtered.retrieved_chunks}
        )
        runs = [unfiltered, filtered]
        payload = _json_payload(
            question=question,
            top_k=resolved_top_k,
            filter_top_k=filter_top_k,
            filter_min_score=filter_min_score,
            runs=runs,
            answer_similarity=answer_similarity,
            token_delta=token_delta,
            token_percent=token_percent,
            time_delta=time_delta,
            time_percent=time_percent,
            chunk_video_overlap=chunk_overlap,
        )
        report_html = render_html_report(payload)
        mlflow.log_metric("rag_all_unfiltered_prompt_tokens", unfiltered.token_estimate)
        mlflow.log_metric("rag_all_filtered_prompt_tokens", filtered.token_estimate)
        mlflow.log_metric("rag_all_unfiltered_time_seconds", unfiltered.time_seconds)
        mlflow.log_metric("rag_all_filtered_time_seconds", filtered.time_seconds)
        mlflow.log_metric("selected_transcript_count", len(filtered.selected_transcripts))
        mlflow.log_metric("rag_all_unfiltered_chunk_count", len(unfiltered.retrieved_chunks))
        mlflow.log_metric("rag_all_filtered_chunk_count", len(filtered.retrieved_chunks))
        mlflow.log_metric("answer_similarity", answer_similarity)
        mlflow.log_metric("token_delta", token_delta)
        mlflow.log_metric("token_percent_change", token_percent)
        mlflow.log_metric("time_delta", time_delta)
        mlflow.log_metric("time_percent_change", time_percent)
        _log_artifact(report_html, "s4_rag_filter.html")
        _log_artifact(json.dumps(payload, indent=2), "s4_rag_filter.json")
        _log_artifact(
            json.dumps(
                {
                    "selected_transcripts": [
                        transcript.model_dump(mode="json")
                        for transcript in filtered.selected_transcripts
                    ]
                },
                indent=2,
            ),
            "s4_selected_transcripts.json",
        )
        for run in runs:
            _log_artifact(
                json.dumps(
                    {
                        "chunks": [
                            chunk.model_dump(mode="json")
                            for chunk in run.retrieved_chunks
                        ]
                    },
                    indent=2,
                ),
                f"s4_{run.name}_chunks.json",
            )

    return {"html": report_html, "json": payload}


def _run_agent(
    agent: RagTranscriptAgent,
    request: RagQuestionRequest,
    name: str,
) -> S4Run:
    started = time.perf_counter()
    answer = agent.answer(request)
    elapsed = time.perf_counter() - started
    if agent.last_context is None:
        raise RuntimeError(f"{name} did not capture context")
    return S4Run(
        name=name,
        filter_enabled=request.filter_transcripts,
        answer=answer.answer,
        context_text=agent.last_context.context_text or "",
        retrieved_chunks=agent.last_context.retrieved_chunks or [],
        selected_transcripts=agent.last_context.selected_transcripts or [],
        time_seconds=elapsed,
    )


def _json_payload(
    question: str,
    top_k: int,
    filter_top_k: int,
    filter_min_score: float,
    runs: list[S4Run],
    answer_similarity: float,
    token_delta: int,
    token_percent: float,
    time_delta: float,
    time_percent: float,
    chunk_video_overlap: int,
) -> dict[str, Any]:
    return {
        "question": question,
        "top_k": top_k,
        "transcript_filter_top_k": filter_top_k,
        "transcript_filter_min_score": filter_min_score,
        "runs": [
            {
                "name": run.name,
                "filter_enabled": run.filter_enabled,
                "answer": run.answer,
                "prompt_tokens_estimate": run.token_estimate,
                "time_seconds": run.time_seconds,
                "selected_transcripts": [
                    transcript.model_dump(mode="json")
                    for transcript in run.selected_transcripts
                ],
                "retrieved_chunks": [
                    chunk.model_dump(mode="json") for chunk in run.retrieved_chunks
                ],
            }
            for run in runs
        ],
        "comparison": {
            "answer_similarity": answer_similarity,
            "token_delta": token_delta,
            "token_percent_change": token_percent,
            "time_delta": time_delta,
            "time_percent_change": time_percent,
            "chunk_video_overlap": chunk_video_overlap,
            "selected_transcript_ids": [
                transcript.transcript_id for transcript in runs[1].selected_transcripts
            ],
        },
    }


def render_html_report(payload: dict[str, Any]) -> str:
    runs = payload["runs"]
    comparison = payload["comparison"]
    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="en">',
            "<head>",
            '<meta charset="utf-8">',
            "<title>S4 RAG Filter Evaluation</title>",
            *dark_style_block(),
            "</head>",
            "<body>",
            "<header><h1>S4 RAG Filter Evaluation</h1></header>",
            "<main>",
            f"<p><strong>Question:</strong> {html.escape(payload['question'])}</p>",
            "<h2>Summary</h2>",
            _summary_table(runs),
            "<h2>Comparison</h2>",
            _comparison_table(comparison),
            "<h2>Answers</h2>",
            *[_run_section(run) for run in runs],
            "</main>",
            "</body>",
            "</html>",
        ]
    )


def _summary_table(runs: list[dict[str, Any]]) -> str:
    rows = [
        "<tr><th>Run</th><th>Transcript filter</th><th>Selected transcripts</th>"
        "<th>Retrieved chunks</th><th>Prompt tokens</th><th>Time seconds</th></tr>"
    ]
    for run in runs:
        rows.append(
            "<tr>"
            f"<td>{html.escape(run['name'])}</td>"
            f"<td>{run['filter_enabled']}</td>"
            f"<td>{len(run['selected_transcripts'])}</td>"
            f"<td>{len(run['retrieved_chunks'])}</td>"
            f"<td>{run['prompt_tokens_estimate']}</td>"
            f"<td>{run['time_seconds']:.3f}</td>"
            "</tr>"
        )
    return f"<table>{''.join(rows)}</table>"


def _comparison_table(comparison: dict[str, Any]) -> str:
    rows = "".join(
        f"<tr><th>{html.escape(key)}</th><td>{html.escape(str(value))}</td></tr>"
        for key, value in comparison.items()
    )
    return f"<table>{rows}</table>"


def _run_section(run: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"<article><h3>{html.escape(run['name'])}</h3>",
            f"<pre>{html.escape(run['answer'])}</pre>",
            _selected_transcripts_section(run["selected_transcripts"]),
            _chunks_section(run["retrieved_chunks"]),
            "</article>",
        ]
    )


def _selected_transcripts_section(transcripts: list[dict[str, Any]]) -> str:
    if not transcripts:
        return "<p>No selected transcript summaries.</p>"
    details = []
    for transcript in transcripts:
        score = transcript.get("score")
        details.append(
            "<details>"
            f"<summary>video={html.escape(transcript['video_id'])} "
            f"score={html.escape(str(score))}</summary>"
            f"<p><strong>URL:</strong> {html.escape(transcript['source_url'])}</p>"
            f"<pre>{html.escape(transcript['summary'])}</pre>"
            "</details>"
        )
    return "<h4>Selected Transcript Summaries</h4>" + "".join(details)


def _chunks_section(chunks: list[dict[str, Any]]) -> str:
    if not chunks:
        return "<p>No retrieved chunks.</p>"
    details = []
    for chunk in chunks:
        timestamp_url = youtube_timestamp_url(
            chunk["source_url"], chunk.get("start_seconds")
        )
        details.append(
            "<details>"
            f"<summary>video={html.escape(chunk['video_id'])} "
            f"chunk={chunk['chunk_index']} score={html.escape(str(chunk.get('score')))}</summary>"
            f"<p><strong>URL:</strong> {html.escape(chunk['source_url'])}</p>"
            f"<p><strong>Timestamp:</strong> {html.escape(str(timestamp_url))}</p>"
            f"<pre>{html.escape(chunk['text'])}</pre>"
            "</details>"
        )
    return "<h4>Retrieved Chunks</h4>" + "".join(details)


def _percent_change(before: float, after: float) -> float:
    if before == 0:
        return 0.0
    return ((after - before) / before) * 100.0


def _log_artifact(content: str, artifact_name: str) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / artifact_name
        path.write_text(content, encoding="utf-8")
        mlflow.log_artifact(str(path))


if __name__ == "__main__":
    raise SystemExit(main())

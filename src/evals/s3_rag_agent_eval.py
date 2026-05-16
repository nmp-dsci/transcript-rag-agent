from __future__ import annotations

import argparse
import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.agents.models import RagQuestionRequest, RagTranscriptAnswer
from src.agents.rag_transcript_agent import RagTranscriptAgent
from src.config import ConfigError, load_settings
from src.dashboard.theme import dark_style_block
from src.rag.context import MultiTranscriptRagContextProvider
from src.rag.embeddings import HuggingFaceEmbeddingModel, cosine_similarity
from src.rag.eval import estimate_tokens
from src.rag.indexing import RagIndexer
from src.rag.references import youtube_timestamp_url
from src.rag.storage import RawTranscriptStore, TranscriptChunkStore
from src.transcripts.fetcher import SuperdataTranscriptFetcher


DEFAULT_QUESTION = (
    "what does this video say  for capital gains tax, is it being grandfathered "
    "or every now under new rules, does that mean if I sell before 30 June 2027 "
    "I can still access 50% discount "
)


@dataclass(frozen=True)
class EvalRun:
    name: str
    source_url: str | None
    answer: RagTranscriptAnswer
    context_text: str
    retrieved_chunks: list[Any]

    @property
    def token_estimate(self) -> int:
        return estimate_tokens(self.context_text)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="s3-rag-agent-eval",
        description="Render S3 multi-transcript RAG diagnostics as HTML.",
    )
    parser.add_argument("--url", action="append", required=True)
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--json-output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if len(args.url) < 2:
        parser.error("S3 evaluation requires at least two --url values")
    try:
        report = run_evaluation(
            urls=args.url,
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
    urls: list[str],
    question: str = DEFAULT_QUESTION,
    top_k: int | None = None,
) -> dict[str, Any]:
    settings = load_settings(require_keys=True)
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
    provider = MultiTranscriptRagContextProvider(
        raw_store=raw_store,
        chunk_store=chunk_store,
        indexer=indexer,
    )
    agent = RagTranscriptAgent.from_settings(settings, provider)

    runs: list[EvalRun] = []
    for index, url in enumerate(urls[:2], 1):
        runs.append(
            _run_agent(
                agent=agent,
                name=f"url_{index}_only",
                question=question,
                source_url=url,
                top_k=resolved_top_k,
            )
        )
    runs.append(
        _run_agent(
            agent=agent,
            name="all_indexed",
            question=question,
            source_url=None,
            top_k=resolved_top_k,
        )
    )

    answer_embeddings = embedding_model.embed_documents(
        [run.answer.answer for run in runs]
    )
    similarities = _similarities(runs, answer_embeddings)
    payload = _json_payload(question, resolved_top_k, runs, similarities)
    return {
        "html": render_html_report(question, resolved_top_k, runs, similarities),
        "json": payload,
    }


def _run_agent(
    agent: RagTranscriptAgent,
    name: str,
    question: str,
    source_url: str | None,
    top_k: int,
) -> EvalRun:
    answer = agent.answer(
        RagQuestionRequest(question=question, source_url=source_url, top_k=top_k)
    )
    if agent.last_context is None:
        raise RuntimeError(f"{name} did not capture context")
    return EvalRun(
        name=name,
        source_url=source_url,
        answer=answer,
        context_text=agent.last_context.context_text or "",
        retrieved_chunks=agent.last_context.retrieved_chunks or [],
    )


def _similarities(
    runs: list[EvalRun], embeddings: list[list[float]]
) -> dict[str, float]:
    by_name = {run.name: index for index, run in enumerate(runs)}
    pairs = [
        ("url_1_only", "all_indexed"),
        ("url_2_only", "all_indexed"),
        ("url_1_only", "url_2_only"),
    ]
    values: dict[str, float] = {}
    for left, right in pairs:
        if left in by_name and right in by_name:
            values[f"{left}__{right}"] = cosine_similarity(
                embeddings[by_name[left]], embeddings[by_name[right]]
            )
    return values


def render_html_report(
    question: str,
    top_k: int,
    runs: list[EvalRun],
    similarities: dict[str, float],
) -> str:
    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="en">',
            "<head>",
            '<meta charset="utf-8">',
            "<title>S3 RAG Agent Evaluation</title>",
            *dark_style_block(),
            "</head>",
            "<body>",
            "<header><h1>S3 RAG Agent Evaluation</h1></header>",
            "<main>",
            f"<p><strong>Question:</strong> {html.escape(question)}</p>",
            f"<p><strong>Top K:</strong> {top_k}</p>",
            _summary_table(runs, similarities),
            "<h2>Answers And Retrieved Chunks</h2>",
            *[_run_section(run) for run in runs],
            "<h2>Pairwise Similarity</h2>",
            _similarity_list(similarities),
            "</main>",
            "</body>",
            "</html>",
        ]
    )


def _summary_table(runs: list[EvalRun], similarities: dict[str, float]) -> str:
    rows = [
        "<tr><th>Run</th><th>Filter</th><th>Answer chars</th><th>Token estimate</th><th>Chunks</th><th>Similarity to all</th></tr>"
    ]
    for run in runs:
        similarity = ""
        key = f"{run.name}__all_indexed"
        if key in similarities:
            similarity = f"{similarities[key]:.3f}"
        rows.append(
            "<tr>"
            f"<td>{html.escape(run.name)}</td>"
            f"<td>{html.escape(run.source_url or 'all indexed transcripts')}</td>"
            f"<td class=\"metric\">{len(run.answer.answer)}</td>"
            f"<td class=\"metric\">{run.token_estimate}</td>"
            f"<td class=\"metric\">{len(run.retrieved_chunks)}</td>"
            f"<td class=\"metric\">{similarity}</td>"
            "</tr>"
        )
    return "<table>" + "".join(rows) + "</table>"


def _run_section(run: EvalRun) -> str:
    references = "\n".join(
        f"{reference.label} {reference.timestamp_url}"
        for reference in run.answer.references
    )
    chunks = "\n".join(_chunk_details(index, chunk) for index, chunk in enumerate(run.retrieved_chunks, 1))
    return "\n".join(
        [
            "<article>",
            f"<h3>{html.escape(run.name)}</h3>",
            f"<p><strong>Filter:</strong> {html.escape(run.source_url or 'all indexed transcripts')}</p>",
            "<h4>Answer</h4>",
            f"<pre>{html.escape(run.answer.answer)}</pre>",
            "<h4>References</h4>",
            f"<pre>{html.escape(references or 'No references returned')}</pre>",
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


def _similarity_list(similarities: dict[str, float]) -> str:
    if not similarities:
        return "<p>No similarity metrics.</p>"
    items = "".join(
        f"<li><span class=\"metric\">{html.escape(name)}: {value:.3f}</span></li>"
        for name, value in similarities.items()
    )
    return f"<ul>{items}</ul>"


def _json_payload(
    question: str,
    top_k: int,
    runs: list[EvalRun],
    similarities: dict[str, float],
) -> dict[str, Any]:
    return {
        "eval_name": "s3_rag_agent_eval",
        "question": question,
        "top_k": top_k,
        "runs": [
            {
                "name": run.name,
                "source_url": run.source_url,
                "answer": run.answer.model_dump(mode="json"),
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

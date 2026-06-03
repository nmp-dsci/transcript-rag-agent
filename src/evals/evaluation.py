from __future__ import annotations

import argparse
import html
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.agents.models import RagQuestionRequest, RecursionOptions
from src.agents.rag_agent import RagAgent
from src.agents.rag_transcript_agent import RagTranscriptAgent
from src.config import ConfigError, load_settings
from src.dashboard.theme import dark_style_block
from src.rag.context import MultiTranscriptRagContextProvider
from src.rag.embeddings import HuggingFaceEmbeddingModel
from src.rag.eval import estimate_tokens
from src.rag.indexing import RagIndexer
from src.rag.storage import RawTranscriptStore, TranscriptChunkStore
from src.rag.summaries import TranscriptSummaryStore
from src.transcripts.fetcher import SuperdataTranscriptFetcher


DEFAULT_QUESTION = (
    "what does this corpus say about how AI engineers leverage agentic coding "
    "to fully develop features, what is the best workflow for agentic coding"
)

# The three agent setups, expressed as the exact bash commands a user would run.
# The column title is derived from the flags in each command (see _title_from_command).
SETUP_COMMANDS = [
    'uv run python -m src.cli rag-ask "$question" --rag_llm --top-k 30',
    'uv run python -m src.cli rag-ask "$question" --rag_llm --recursive --top-k 10',
    'uv run python -m src.cli rag-ask "$question" --rag_agent --top-k 10',
]


@dataclass(frozen=True)
class AgentRun:
    """One agent setup's answer plus the command that produced it."""

    title: str
    command: str
    answer: str
    references: list[Any] = field(default_factory=list)
    token_estimate: int = 0
    chunk_count: int = 0
    llm_calls: int | None = None
    iterations: int | None = None
    terminated_reason: str | None = None


def _title_from_command(command: str) -> str:
    """Derive a column title from the bash command: the flags after the question."""
    marker = '"$question"'
    flags = command.split(marker, 1)[1].strip() if marker in command else command
    return flags or command


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evaluation",
        description=(
            "Compare one question across three agent setups: rag_llm single-hop, "
            "rag_llm recursive, and the agentic rag_agent."
        ),
    )
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument(
        "--output", type=Path, default=Path("dashboard/evaluation.html")
    )
    parser.add_argument("--json-output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = run_evaluation(question=args.question)
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


def run_evaluation(question: str = DEFAULT_QUESTION) -> dict[str, Any]:
    settings = load_settings(require_keys=True)

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

    rag_llm_agent = RagTranscriptAgent.from_settings(settings, provider)
    rag_agent = RagAgent.from_settings(settings, provider)

    recursion_options = RecursionOptions(
        max_depth=settings.rag_max_depth,
        max_followups=settings.rag_max_followups,
        followup_top_k=settings.rag_followup_top_k,
        novelty_min_chunks=settings.rag_novelty_min_chunks,
        max_total_followups=settings.rag_max_total_followups,
    )

    # Setup 1: rag_llm single-hop, top-k 30.
    answer_1 = rag_llm_agent.answer(RagQuestionRequest(question=question, top_k=30))
    context_1 = rag_llm_agent.last_context
    run_1 = _build_run(
        SETUP_COMMANDS[0],
        answer_1,
        context_1,
        llm_calls=1,
    )

    # Setup 2: rag_llm recursive, top-k 10.
    answer_2 = rag_llm_agent.answer(
        RagQuestionRequest(
            question=question,
            top_k=10,
            recursive=True,
            recursion_options=recursion_options,
        )
    )
    context_2 = rag_llm_agent.last_context
    recursion = answer_2.recursion
    run_2 = _build_run(
        SETUP_COMMANDS[1],
        answer_2,
        context_2,
        llm_calls=(
            sum(stage.llm_calls for stage in recursion.stages) if recursion else 1
        ),
        terminated_reason=recursion.terminated_reason if recursion else None,
    )

    # Setup 3: agentic rag_agent, top-k 10.
    answer_3 = rag_agent.answer(RagQuestionRequest(question=question, top_k=10))
    context_3 = rag_agent.last_context
    run_3 = _build_run(
        SETUP_COMMANDS[2],
        answer_3,
        context_3,
        iterations=rag_agent.last_iteration_count,
        terminated_reason=rag_agent.last_terminated_reason,
    )

    runs = [run_1, run_2, run_3]
    return {
        "html": render_html_report(question=question, runs=runs),
        "json": _json_payload(question=question, runs=runs),
    }


def _build_run(
    command: str,
    answer: Any,
    context: Any,
    *,
    llm_calls: int | None = None,
    iterations: int | None = None,
    terminated_reason: str | None = None,
) -> AgentRun:
    context_text = context.context_text if context is not None else ""
    chunks = context.retrieved_chunks if context is not None else []
    return AgentRun(
        title=_title_from_command(command),
        command=command,
        answer=answer.answer,
        references=list(answer.references or []),
        token_estimate=estimate_tokens(context_text or ""),
        chunk_count=len(chunks or []),
        llm_calls=llm_calls,
        iterations=iterations,
        terminated_reason=terminated_reason,
    )


def _layout_style_block() -> list[str]:
    return [
        "<style>",
        ".question-box{background:#151c26;border:1px solid #2d3745;"
        "padding:16px 20px;margin-bottom:20px}",
        ".question-box p{margin:8px 0 0;font-size:16px;color:#e7edf5}",
        ".answer-columns{display:grid;grid-template-columns:repeat(3,1fr);"
        "gap:16px;align-items:start}",
        ".answer-col{border:1px solid #2d3745;background:#151c26;"
        "padding:16px;min-width:0}",
        ".answer-col h2{margin:0 0 10px;font-size:14px;color:#f6f8fb;"
        "font-family:ui-monospace,Menlo,monospace;word-break:break-word}",
        ".answer-col .answer{white-space:pre-wrap;background:#10161f;"
        "border:1px solid #2d3745;color:#e7edf5;padding:12px;overflow:auto}",
        ".col-meta{display:flex;flex-wrap:wrap;gap:8px;margin:10px 0;"
        "font-family:ui-monospace,Menlo,monospace;font-size:12px;color:#9fb3c8}",
        ".col-meta span{background:#10161f;border:1px solid #2d3745;padding:4px 8px}",
        "@media(max-width:980px){.answer-columns{grid-template-columns:1fr}}",
        "</style>",
    ]


def render_html_report(question: str, runs: list[AgentRun]) -> str:
    columns = "\n".join(_run_column(run) for run in runs)
    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="en">',
            "<head>",
            '<meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            "<title>Agentic Coding Evaluation — 3 Agent Setups</title>",
            *dark_style_block(),
            *_layout_style_block(),
            "</head>",
            "<body>",
            "<header><h1>Agentic Coding Evaluation — 3 Agent Setups</h1></header>",
            "<main>",
            '<div class="question-box">',
            "<strong>Question</strong>",
            f"<p>{html.escape(question)}</p>",
            "</div>",
            '<div class="answer-columns">',
            columns,
            "</div>",
            "</main>",
            "</body>",
            "</html>",
        ]
    )


def _run_column(run: AgentRun) -> str:
    parts = [
        '<section class="answer-col">',
        f"<h2>{html.escape(run.title)}</h2>",
        "<details><summary>Command</summary>",
        f"<pre><code>{html.escape(run.command)}</code></pre>",
        "</details>",
        '<div class="col-meta">',
        f"<span>tokens ~{run.token_estimate}</span>",
        f"<span>chunks {run.chunk_count}</span>",
        f"<span>answer {len(run.answer)} chars</span>",
    ]
    if run.llm_calls is not None:
        parts.append(f"<span>LLM calls {run.llm_calls}</span>")
    if run.iterations is not None:
        parts.append(f"<span>iterations {run.iterations}</span>")
    if run.terminated_reason:
        parts.append(f"<span>{html.escape(run.terminated_reason)}</span>")
    parts.append("</div>")
    parts.append("<h3>Answer</h3>")
    parts.append(f'<pre class="answer">{html.escape(run.answer)}</pre>')
    if run.references:
        parts.append(_references_details(run.references))
    parts.append("</section>")
    return "\n".join(parts)


def _references_details(references: list[Any]) -> str:
    items = []
    for ref in references:
        label = html.escape(str(getattr(ref, "label", "")))
        timestamp_url = html.escape(str(getattr(ref, "timestamp_url", "")))
        video_id = html.escape(str(getattr(ref, "video_id", "")))
        items.append(
            f"<li>{label} "
            f'<a href="{timestamp_url}">open at timestamp</a> '
            f'<span class="metric">{video_id}</span></li>'
        )
    return (
        f"<details><summary>References ({len(references)})</summary>"
        f"<ul>{''.join(items)}</ul></details>"
    )


def _json_payload(question: str, runs: list[AgentRun]) -> dict[str, Any]:
    return {
        "eval_name": "rag_llm_vs_rag_llm_recursive_vs_rag_agent",
        "question": question,
        "runs": [
            {
                "title": run.title,
                "command": run.command,
                "answer": run.answer,
                "token_estimate": run.token_estimate,
                "chunk_count": run.chunk_count,
                "llm_calls": run.llm_calls,
                "iterations": run.iterations,
                "terminated_reason": run.terminated_reason,
                "references": [
                    ref.model_dump(mode="json") if hasattr(ref, "model_dump") else ref
                    for ref in run.references
                ],
            }
            for run in runs
        ],
    }


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from langchain_openai import ChatOpenAI

from src.agents.context import RawTranscriptContextProvider
from src.agents.llm import chat_model_kwargs
from src.agents.models import (
    AgentProgressEvent,
    QuestionRequest,
    RagQuestionRequest,
    RecursionOptions,
    SummaryRequest,
)
from src.agents.rag_agent import RagAgent
from src.agents.rag_transcript_agent import RagTranscriptAgent
from src.agents.transcript_agent import TranscriptAgent
from src.config import ConfigError, load_settings
from src.dashboard.rag_pipeline import (
    DEFAULT_FILTER_TEST_QUESTION,
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
    log_recursion_trace,
    log_summary,
    log_transcript,
    log_transcript_filter_details,
)
from src.rag.context import (
    MultiTranscriptRagContextProvider,
    RagTranscriptContextProvider,
)
from src.rag.embeddings import HuggingFaceEmbeddingModel
from src.rag.eval import compare_answers, estimate_tokens
from src.rag.indexing import RagIndexer
from src.rag.ingestion import (
    candidate_record,
    ingestion_runs_dir,
    start_ingestion_run,
    write_ingestion_run,
)
from src.rag.deep_research import DEFAULT_GAP_PROBES, DEFAULT_ROUND_ONE_PROBES
from src.rag.packs import PACK_ARMS
from src.rag.storage import RawTranscriptStore, TranscriptChunkStore
from src.rag.summaries import TranscriptSummaryGenerator, TranscriptSummaryStore
from src.transcripts.discovery import (
    SupadataDiscoveryClient,
    discover_channel_videos,
    discover_latest_channel_videos,
    discover_search_results,
)
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

    bulk = subparsers.add_parser("bulk-index", help="Discover and index many videos")
    bulk_subparsers = bulk.add_subparsers(dest="bulk_mode", required=True)
    bulk_channel = bulk_subparsers.add_parser("channel", help="Index videos from a channel")
    bulk_channel.add_argument("--channel", required=True)
    channel_window = bulk_channel.add_mutually_exclusive_group()
    channel_window.add_argument("--latest", type=int)
    channel_window.add_argument("--since", type=_parse_date_arg)
    bulk_channel.add_argument("--until", type=_parse_date_arg)
    bulk_channel.add_argument("--max-results", type=int, default=50)
    _add_bulk_common_args(bulk_channel)

    bulk_search = bulk_subparsers.add_parser("search", help="Index YouTube search results")
    bulk_search.add_argument("--query", required=True)
    bulk_search.add_argument("--top-n", type=int, default=10)
    _add_bulk_common_args(bulk_search)

    subparsers.add_parser(
        "chat",
        help="Interactive menu: ask questions across RAG setups or fetch new URLs",
    )

    serve = subparsers.add_parser("serve", help="Run the live web chat app (FastAPI + uvicorn)")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)

    golden = subparsers.add_parser(
        "eval-golden",
        help="Run the golden question set under the current config and snapshot it",
    )
    golden.add_argument(
        "--setup",
        default="rag_llm",
        choices=["rag_llm", "rag_llm_recursive", "rag_agent", "graph_rag"],
        help="Which RAG setup answers the golden questions",
    )
    golden.add_argument(
        "--retrieval",
        choices=["semantic", "hybrid"],
        default=None,
        help="Override the configured retrieval mode for this run",
    )
    golden.add_argument("--top-k", type=int, default=None)
    golden.add_argument(
        "--no-judge",
        action="store_true",
        help="Skip RAGAS scoring and report reference metrics only (much faster)",
    )
    golden.add_argument(
        "--reference-metrics",
        action="store_true",
        help="Also run LLM reference metrics (answer correctness/similarity, recall)",
    )
    golden.add_argument(
        "--diff",
        action="store_true",
        help="Diff the two most recent saved runs instead of running a new one",
    )

    ablation = subparsers.add_parser(
        "eval-ablation",
        help=(
            "Sweep retrieval configs over the golden set — semantic/semantic+rerank/"
            "hybrid/hybrid+rerank, or --sweep extended for the HyDE, multi-query and "
            "contextual-retrieval variants too"
        ),
    )
    ablation.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Final chunk count each configuration retrieves (default: YT_AGENT_RAG_TOP_K)",
    )
    ablation.add_argument(
        "--sweep",
        choices=["default", "extended"],
        default="default",
        help=(
            "Which configurations to measure. 'default' is semantic/semantic+rerank/"
            "hybrid/hybrid+rerank — deterministic and offline. 'extended' adds the "
            "HyDE, multi-query and contextual-retrieval columns, which need "
            "DEEPSEEK_API_KEY (query expansions are cached per question) and, for "
            "the contextual ones, a prior index-contextual"
        ),
    )

    critique = subparsers.add_parser(
        "eval-critique",
        help=(
            "Held-out critique eval: hold one expert's video out of every retrieval "
            "path, review a document with the rest of the corpus, and score how many "
            "of that expert's criteria the system reached without having seen them"
        ),
    )
    critique.add_argument(
        "--setups",
        default=None,
        help=(
            "Comma-separated setup keys to measure, baseline first "
            "(default: rag_llm_filtered, the summary-filtered chunk dump). "
            "rag_conflict_aware appends both sides of a known disagreement; "
            "rubric_packs is the rubric-driven reviewer, which runs no retrieval "
            "and judges the artifact against the shipped expert packs"
        ),
    )
    critique.add_argument(
        "--rescore",
        default=None,
        metavar="RUN_ID_OR_PATH",
        help=(
            "Re-score a committed run's stored findings under the current scorer "
            "instead of retrieving and critiquing again. Use when the scoring rule "
            "changed and the old number is no longer comparable — re-running the "
            "whole thing would change the findings too, so the two effects could "
            "not be told apart. A bare id names a file in evals/runs/; a path "
            "ending in .json is read as written, so a committed run that lives "
            "elsewhere (experts/ablation.json) re-scores under the same rule"
        ),
    )
    critique.add_argument(
        "--in-place",
        action="store_true",
        help=(
            "Write the re-scored run back over its source instead of committing a "
            "second file, keeping its run id, timestamp and envelope. Use when the "
            "old file's numbers are ones the project no longer stands behind: a "
            "stale run left in evals/runs/ is read by the app as a current "
            "measurement. Nothing is lost — the published figures stay on every "
            "cell as criteria_recall_ungated / evidence_precision_ungated"
        ),
    )
    critique.add_argument(
        "--repeats",
        type=int,
        default=None,
        help=(
            "Matcher repeats resolved by per-criterion majority vote. The matcher "
            "does not agree with itself run to run, so the score is a vote and the "
            "run reports the spread (default: 5). Pairings are cached, so a re-run "
            "over an unchanged matrix is free and returns the identical number"
        ),
    )

    matrix = subparsers.add_parser(
        "eval-matrix",
        help=(
            "Head-to-head: every RAG setup answers the same golden questions, "
            "scored by the same RAGAS + reference metrics (s05 §06)"
        ),
    )
    matrix.add_argument(
        "--setups",
        default=None,
        help=(
            "Comma-separated setup keys. Defaults to every comparable setup, "
            "since the Scoreboard ranks the newest committed run; unchanged "
            "cells come from the cache, so a narrower list saves little"
        ),
    )
    matrix.add_argument(
        "--questions",
        default=None,
        help=(
            "Comma-separated golden question ids to score (default: the whole "
            "golden set). A judged cell is the expensive unit of a matrix run, "
            "so naming a sample is how you trade coverage for turnaround — the "
            "ids are recorded in the run, and the sample must cover every "
            "question type for the by-type pivot to mean anything"
        ),
    )
    matrix.add_argument("--top-k", type=int, default=None)
    matrix.add_argument(
        "--no-judge",
        action="store_true",
        help="Skip the RAGAS judge (faithfulness/relevancy/precision); much faster",
    )
    matrix.add_argument(
        "--no-reference-metrics",
        action="store_true",
        help=(
            "Skip answer_correctness/answer_similarity/llm_context_recall. On by "
            "default for the matrix: answer_correctness is the primary verdict "
            "on global/temporal questions"
        ),
    )
    matrix.add_argument(
        "--refresh",
        action="store_true",
        help=(
            "Bypass the per-cell cache and rescore every cell, even ones already "
            "scored under an unchanged configuration. Off by default: a cell is "
            "cached by a fingerprint of the question plus the exact answering/"
            "judging config, so adding one new --setups entry only scores that "
            "one, and edited questions or reconfigured engines invalidate "
            "automatically"
        ),
    )

    rejudge = subparsers.add_parser(
        "rejudge",
        help=(
            "Re-score a committed matrix run under a different rubric, reusing "
            "its stored answers and grounding scores (only the new metrics are "
            "judged), and commit the result as a second run"
        ),
    )
    rejudge.add_argument(
        "--run",
        required=True,
        help="Run id (or path) of the committed matrix-*.json to re-score",
    )
    rejudge.add_argument(
        "--rubric",
        default="depth-v2",
        choices=["depth-v2"],
        help=(
            "Rubric to score under. depth-v2 keeps grounding at 40%% of the "
            "composite and adds five LLM-judged depth metrics at 60%%, capped "
            "at 0.5 when faithfulness < 0.6"
        ),
    )
    rejudge.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help=(
            "Concurrent depth-judge calls (default: 4). One call per answer. "
            "Measured, not guessed: 4 workers cleared 80 cells cleanly, while "
            "6-8 way concurrency against this endpoint collapsed to no "
            "completions at all for ~15 minutes. Raise it only if you have "
            "measured your own provider"
        ),
    )

    index_graph = subparsers.add_parser(
        "index-graph",
        help=(
            "Build the GraphRAG knowledge graph: extract entities/claims per "
            "chunk (cached by chunk hash), then Leiden communities + summaries"
        ),
    )
    index_graph.add_argument(
        "--refresh",
        action="store_true",
        help="Wipe the graph before rebuilding (extraction cache still applies)",
    )
    index_graph.add_argument(
        "--skip-communities",
        action="store_true",
        help="Extraction only; skip community detection and summaries",
    )
    index_graph.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Only process the first N chunks (smoke-testing)",
    )

    index_contextual = subparsers.add_parser(
        "index-contextual",
        help=(
            "Build the Contextual Retrieval index: one LLM-written situating "
            "sentence per chunk (cached by chunk hash), embedded into a parallel "
            "collection so it can be compared against the baseline index"
        ),
    )
    index_contextual.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Only situate the first N chunks (smoke-testing)",
    )
    index_contextual.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Concurrent situating calls (default: 8)",
    )

    index_themes = subparsers.add_parser(
        "index-themes",
        help=(
            "Build the cross-video theme layer (RAPTOR level 2): cluster the "
            "stored chunk embeddings across every video, then one LLM call per "
            "cluster. Reads the corpus; writes only the theme artifact"
        ),
    )
    index_themes.add_argument(
        "--dry-run",
        action="store_true",
        help="Cluster and report the numbers only — no LLM calls, nothing written",
    )
    index_themes.add_argument(
        "--excerpts",
        type=int,
        default=12,
        help="Excerpts per theme handed to the summarizer (default: 12)",
    )

    index_conflicts = subparsers.add_parser(
        "index-conflicts",
        help=(
            "Build the disagreement layer: pair claims from different creators "
            "about the same thing, ask of each pair whether one person could "
            "hold both views, and keep only those where they could not. Reads "
            "the corpus and the cached graph extractions; writes only the "
            "conflict artifact"
        ),
    )
    index_conflicts.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Generate and report candidate pairs only — no adjudication calls, "
            "nothing written. Candidate generation is deterministic, so this "
            "settles the search half of the layer for free"
        ),
    )
    index_conflicts.add_argument(
        "--probes-only",
        action="store_true",
        help=(
            "Run the calibration probes against the live adjudicator and stop. "
            "Five pairs, two planted contradictions and three complementary, so "
            "an adjudicator that calls everything a conflict is caught before "
            "the corpus sweep is paid for"
        ),
    )
    index_conflicts.add_argument(
        "--probe-repeats",
        type=int,
        default=1,
        help=(
            "Times each probe is put to the adjudicator. The adjudicator does "
            "not have to agree with itself, so >1 turns a tick into a rate "
            "(default: 1)"
        ),
    )
    index_conflicts.add_argument(
        "--max-candidates",
        type=int,
        default=None,
        help=(
            "Adjudication budget, highest-similarity pairs first. Raising it "
            "cannot raise conflict_precision — that is conflicts over pairs "
            "adjudicated. The default is set above the pool this corpus "
            "produces, so every candidate is adjudicated and none is excluded "
            "by budget (default: 600)"
        ),
    )
    index_conflicts.add_argument(
        "--allow-within-channel",
        action="store_true",
        help=(
            "Also adjudicate pairs of videos from the same channel. Off by "
            "default: one person qualifying themselves is not a corpus "
            "disagreement"
        ),
    )
    index_conflicts.add_argument(
        "--adjudicate-repeats",
        type=int,
        default=None,
        help=(
            "Times each corpus pair is put to the adjudicator before it is "
            "believed; a strict majority carries and the tally ships on the "
            "card. 1 is a single draw, which is what the first sweep did and "
            "why three of its four conflicts did not reproduce (default: 9)"
        ),
    )
    index_conflicts.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Concurrent adjudication calls (default: 8)",
    )
    index_conflicts.add_argument(
        "--resume-key",
        default=None,
        help=(
            "Cache each (pair, look) under this name so a killed sweep resumes "
            "instead of restarting. At 9 looks a sweep is thousands of calls "
            "and the artifact is only written at the end, so a kill near the "
            "finish costs everything. Reuse the key to resume a run; use a NEW "
            "key to measure again — reusing one would replay the first run's "
            "answers and manufacture agreement between two runs that never "
            "independently happened. Off by default"
        ),
    )

    subparsers.add_parser(
        "eval-graph-extraction",
        help=(
            "Score cached graph extractions against the hand-labelled sample "
            "(entity/claim recall; no LLM calls)"
        ),
    )

    summarize = subparsers.add_parser("summarize", help="Summarize a transcript")
    summarize.add_argument("url")

    ask = subparsers.add_parser("ask", help="Ask a question about a transcript")
    ask.add_argument("url")
    ask.add_argument("question")
    ask.add_argument("--context", choices=["raw", "rag"], default="raw")
    ask.add_argument("--top-k", type=int, default=None)

    compare = subparsers.add_parser("compare-context", help="Compare raw and RAG answers")
    compare.add_argument("url")
    compare.add_argument("question")
    compare.add_argument("--top-k", type=int, default=None)

    rag_ask = subparsers.add_parser("rag-ask", help="Ask across all indexed transcript chunks")
    rag_ask.add_argument("question")
    rag_ask.add_argument("--url")
    rag_ask.add_argument("--top-k", type=int, default=None)
    rag_ask.add_argument("--filter-transcripts", action="store_true")
    rag_ask.add_argument("--transcript-filter-top-k", type=int, default=None)
    rag_ask.add_argument("--transcript-filter-min-score", type=float, default=None)
    recursive_group = rag_ask.add_mutually_exclusive_group()
    recursive_group.add_argument(
        "--recursive",
        dest="recursive",
        action="store_true",
        default=None,
        help="Enable recursive multi-hop RAG",
    )
    recursive_group.add_argument(
        "--no-recursive",
        dest="recursive",
        action="store_false",
        help="Disable recursive RAG even when enabled by env default",
    )
    rag_ask.add_argument("--max-depth", type=int, default=None)
    rag_ask.add_argument("--max-followups", type=int, default=None)
    rag_ask.add_argument("--followup-top-k", type=int, default=None)
    rag_ask.add_argument("--novelty-min-chunks", type=int, default=None)
    rag_ask.add_argument("--max-total-followups", type=int, default=None)
    rag_ask.add_argument("--show-followups", action="store_true")
    rag_ask.add_argument("--print-trace", action="store_true")
    rag_ask.add_argument(
        "--query-transform",
        choices=["hyde", "multi_query"],
        default=None,
        help=(
            "Rewrite the question before embedding it: 'hyde' retrieves on a "
            "hypothetical answer passage, 'multi_query' retrieves on several "
            "paraphrases and RRF-fuses them (default: YT_AGENT_QUERY_TRANSFORM)"
        ),
    )
    rag_ask.add_argument(
        "--contextual",
        action="store_true",
        help=(
            "Retrieve against the Contextual Retrieval index — chunks embedded "
            "with an LLM-written situating sentence. Requires index-contextual"
        ),
    )
    agent_group = rag_ask.add_mutually_exclusive_group()
    agent_group.add_argument(
        "--rag_agent",
        dest="rag_agent",
        action="store_true",
        default=False,
        help=(
            "Use the agentic LangGraph RAG agent (rag_agent) instead of the "
            "pipeline agent (rag_llm)."
        ),
    )
    agent_group.add_argument(
        "--rag_llm",
        dest="rag_llm",
        action="store_true",
        default=False,
        help=(
            "Use the pipeline RAG agent (rag_llm). This is the default when "
            "neither --rag_llm nor --rag_agent is passed."
        ),
    )
    agent_group.add_argument(
        "--graph_rag",
        dest="graph_rag",
        action="store_true",
        default=False,
        help=(
            "Use the GraphRAG agent (graph_rag): routes local/global/temporal "
            "and answers over the knowledge graph. Requires index-graph."
        ),
    )
    rag_ask.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help=(
            "Max ReAct loop iterations for --rag_agent mode "
            "(default: YT_AGENT_RAG_AGENT_MAX_ITERATIONS or 10)."
        ),
    )

    build_packs = subparsers.add_parser(
        "build-packs",
        help=(
            "Build the declared rubric packs (experts/packs.json) three ways — "
            "raptor-only, communities-only and merged — routing each pack's "
            "membership through the summary index. Reads the corpus; writes "
            "only under experts/"
        ),
    )
    build_packs.add_argument(
        "--topic",
        action="append",
        default=None,
        help="Only this pack (repeatable). Default: every declared pack.",
    )
    build_packs.add_argument(
        "--arm",
        action="append",
        choices=list(PACK_ARMS),
        default=None,
        help="Only this arm (repeatable). Default: all three.",
    )
    build_packs.add_argument(
        "--units",
        type=int,
        default=8,
        help=(
            "Source units per arm, capped to the smaller pool of units that "
            "clear the creator floor, so every arm spends the same number of "
            "LLM calls (default: 8)"
        ),
    )
    build_packs.add_argument(
        "--ship",
        default="merged",
        help=(
            "Which arm to copy to experts/<topic>/pack.json, the one the app "
            "renders (default: merged)"
        ),
    )
    build_packs.add_argument(
        "--dry-run",
        action="store_true",
        help="Route and assemble units only — no LLM calls, nothing written",
    )

    score_packs = subparsers.add_parser(
        "score-packs",
        help=(
            "D2: score a pack's three arms on the held-out critique harness and "
            "write experts/ablation.json"
        ),
    )
    score_packs.add_argument(
        "--topic",
        default="resume-design",
        help=(
            "Pack to score. Only a pack whose artifact the held-out expert "
            "reviews can be scored (default: resume-design)"
        ),
    )
    score_packs.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Matcher repeats resolved by per-criterion vote (default: 5)",
    )

    deep_research = subparsers.add_parser(
        "deep-research",
        help=(
            "Build one pack the slow way — plan, execute, criticise, republish — "
            "and write the build report to experts/<topic>/research.json. Also "
            "builds the one-shot control that spends the same executor budget "
            "without the critic, which is what the loop has to beat."
        ),
    )
    deep_research.add_argument(
        "--topic",
        default="resume-design",
        help=(
            "Pack to build. Only a pack whose artifact the held-out expert "
            "reviews can be scored (default: resume-design)"
        ),
    )
    deep_research.add_argument(
        "--probes",
        type=int,
        default=DEFAULT_ROUND_ONE_PROBES,
        help=(
            "Executor calls in round one. Defaults to the unit budget the "
            "hand-built arms spent, so round one and the shipped pack cost the "
            f"same (default: {DEFAULT_ROUND_ONE_PROBES})"
        ),
    )
    deep_research.add_argument(
        "--gap-probes",
        type=int,
        default=DEFAULT_GAP_PROBES,
        help=(
            "Probes the critic's round gets, and equally the extra probes the "
            f"one-shot control gets (default: {DEFAULT_GAP_PROBES})"
        ),
    )
    deep_research.add_argument(
        "--frontier",
        action="store_true",
        help=(
            "Skip the three-arm build and add V8's frontier round instead: a "
            "coverage-aware gap critic, round-two retrieval forbidden to re-read "
            "round one's ground, and rules admitted only when they rest on a "
            "passage no admitted rule rests on. Also writes the deep-r2-admit "
            "ablation, which spends no call"
        ),
    )
    deep_research.add_argument(
        "--score",
        action="store_true",
        help=(
            "After building, score every arm and the hand-built pack on the "
            "held-out critique harness and fold the table into the report"
        ),
    )
    deep_research.add_argument(
        "--score-only",
        action="store_true",
        help="Skip the build and score the arms already on disk",
    )
    deep_research.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Matcher repeats resolved by per-criterion vote (default: 5)",
    )

    return parser


def _add_bulk_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", dest="skip_existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    parser.add_argument("--refresh-summary", action="store_true")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--label")
    parser.add_argument("--no-discovery-cache", action="store_true")


def _parse_date_arg(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid date: {value}") from exc


def _run_eval_golden(args, settings) -> int:
    """Snapshot the golden set under the current config, or diff two snapshots."""
    from src.chat.setups import AskScope, RagSetupRunner
    from src.evals.regression import (
        diff_runs,
        list_runs,
        load_run,
        run_golden_eval,
        save_run,
    )

    if args.diff:
        runs = list_runs()
        if len(runs) < 2:
            print(f"Need two saved runs to diff; found {len(runs)}.")
            return 1
        before, after = load_run(runs[-2]), load_run(runs[-1])
        diff = diff_runs(before, after)
        print(f"{diff['before_run']}  →  {diff['after_run']}\n")
        for move in diff["metrics"]:
            arrow = {"better": "▲", "worse": "▼", "unchanged": "·"}[move["direction"]]
            print(
                f"  {arrow} {move['metric']:<22} "
                f"{move['before']:.3f} → {move['after']:.3f}  ({move['delta']:+.3f})"
            )
        if diff["entries"]:
            print("\nquestions that moved:")
            for entry in diff["entries"]:
                metrics = ", ".join(
                    f"{name} {change['delta']:+.2f}" for name, change in entry["changes"].items()
                )
                print(f"  {entry['id']}: {metrics}")
        print(
            f"\nregressed: {', '.join(diff['regressed']) or 'none'}"
            f"   improved: {', '.join(diff['improved']) or 'none'}"
        )
        return 1 if diff["regressed"] else 0

    from src.evals.golden import answer_correctness_fns
    from src.evals.judge import RagasJudge

    runner = RagSetupRunner.from_settings(settings)
    judge = None if args.no_judge else RagasJudge.from_settings(settings)
    reference_fns = answer_correctness_fns(settings) if args.reference_metrics else None
    run = run_golden_eval(
        runner,
        settings,
        setup=args.setup,
        judge=judge,
        reference_fns=reference_fns,
        scope=AskScope(retrieval_mode=args.retrieval),
        top_k=args.top_k,
        on_progress=print,
    )
    path = save_run(run)
    summary = run["summary"]
    print(f"\n{run['run_id']} — {summary['scored']}/{summary['entries']} scored")
    for metric, value in sorted(summary["averages"].items()):
        print(f"  {metric:<22} {value:.3f}")
    if summary["failed"]:
        print(f"  {summary['failed']} question(s) failed and were excluded")
    print(f"\nsaved {path}")
    print("compare against the previous run with: eval-golden --diff")
    return 0


def _run_eval_matrix(args, settings) -> int:
    """Every setup × the same golden questions, one committed comparison run."""
    from src.chat.setups import RagSetupRunner, SETUP_KEYS
    from src.evals.golden import answer_correctness_fns, load_golden
    from src.evals.judge import RagasJudge
    from src.evals.matrix import (
        DEFAULT_MATRIX_SETUPS,
        format_matrix_table,
        run_matrix,
    )
    from src.evals.regression import save_run

    setups = (
        [token.strip() for token in args.setups.split(",") if token.strip()]
        if args.setups
        else list(DEFAULT_MATRIX_SETUPS)
    )
    unknown = [setup for setup in setups if setup not in SETUP_KEYS]
    if unknown:
        print(f"Unknown setups: {', '.join(unknown)}", file=sys.stderr)
        return 2

    entries = None
    if args.questions:
        wanted = [token.strip() for token in args.questions.split(",") if token.strip()]
        by_id = {entry.id: entry for entry in load_golden()}
        missing = [question_id for question_id in wanted if question_id not in by_id]
        if missing:
            print(f"Unknown golden question ids: {', '.join(missing)}", file=sys.stderr)
            return 2
        # Ordered by the request, de-duplicated: the ids are what the run is
        # scoped to, so scoring one twice would double its weight in the
        # averages without that being visible anywhere in the snapshot.
        entries = [by_id[question_id] for question_id in dict.fromkeys(wanted)]

    if "rag_llm_contextual" in setups:
        contextual_store = TranscriptChunkStore(
            settings.chroma_path,
            embedding_model=HuggingFaceEmbeddingModel(settings.embedding_model),
            collection_name=settings.contextual_chunk_collection,
        )
        if not contextual_store.has_any_chunks():
            print(
                "The contextual index is empty. Run "
                "`uv run python -m src.cli index-contextual` before running "
                "eval-matrix with rag_llm_contextual (or pass --setups to "
                "exclude it).",
                file=sys.stderr,
            )
            return 1

    if "rag_llm_filtered" in setups:
        # The filter routes on per-video summaries; with none stored every
        # question raises "no transcript summaries matched" and the whole
        # column comes back as errors rather than as a comparison.
        summary_store = _build_summary_store(
            settings, HuggingFaceEmbeddingModel(settings.embedding_model), raw_store=None
        )
        if summary_store.collection.count() == 0:
            print(
                "No transcript summaries are stored, so rag_llm_filtered has "
                "nothing to route on. Run `uv run python -m src.cli index-rag` "
                "over the corpus first (or pass --setups to exclude it).",
                file=sys.stderr,
            )
            return 1

    runner = RagSetupRunner.from_settings(settings)
    judge = None if args.no_judge else RagasJudge.from_settings(settings)
    reference_fns = None if args.no_reference_metrics else answer_correctness_fns(settings)
    result = run_matrix(
        runner,
        settings,
        setups=setups,
        judge=judge,
        reference_fns=reference_fns,
        entries=entries,
        top_k=args.top_k,
        on_progress=print,
        refresh=args.refresh,
    )
    path = save_run(result)
    print(f"\n{result['run_id']} — {result['entry_count']} questions × {len(setups)} setups")
    print(f"question types: {result['question_types']}\n")
    print(format_matrix_table(result))
    print(f"\nsaved {path}")
    return 0


def _run_rejudge(args, settings) -> int:
    """Re-score one committed matrix run under a second rubric."""
    from src.evals.judge import RUBRICS, DepthJudge
    from src.evals.matrix import format_matrix_table
    from src.evals.regression import save_run
    from src.evals.rejudge import chunk_context_lookup, find_run, rejudge_run

    rubric = RUBRICS[args.rubric]
    try:
        source = find_run(args.run)
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    judge = DepthJudge.from_settings(settings)
    result = rejudge_run(
        source,
        depth_fn=judge.score,
        contexts_fn=chunk_context_lookup(settings),
        rubric=rubric,
        judge_model=judge.judge_model,
        on_progress=print,
        max_workers=args.max_workers,
    )
    path = save_run(result)
    capped = sum(
        1 for run in result["runs"].values() for cell in run["entries"] if cell.get("cap_applied")
    )
    print(
        f"\n{result['run_id']} — {rubric.version} over {result['entry_count']} "
        f"questions × {len(result['setups'])} setups "
        f"(rejudged from {result['rejudged_from']}, {capped} cell(s) capped)\n"
    )
    print(format_matrix_table(result))
    print(f"\nsaved {path}")
    return 0


def _run_index_graph(args, settings) -> int:
    """Build the knowledge graph: extraction (cached) → Leiden → summaries."""
    from src.rag.embeddings import HuggingFaceEmbeddingModel
    from src.rag.graph_pipeline import build_graph
    from src.rag.storage import TranscriptChunkStore

    embedding_model = HuggingFaceEmbeddingModel(settings.embedding_model)
    chunk_store = TranscriptChunkStore(
        settings.chroma_path,
        embedding_model=embedding_model,
        collection_name=settings.chunk_collection,
    )
    chunks = chunk_store.all_chunks()
    if args.max_chunks is not None:
        chunks = chunks[: args.max_chunks]
    if not chunks:
        print("No chunks indexed. Run index-rag / bulk-index first.", file=sys.stderr)
        return 1

    stats = build_graph(
        settings,
        chunks,
        refresh=args.refresh,
        skip_communities=args.skip_communities,
        on_progress=print,
    )
    counts = stats["counts"]
    print("\nGraph built")
    print(f"  Entities:    {counts.get('entities', 0)}")
    print(f"  Relations:   {counts.get('relations', 0)}")
    print(f"  Claims:      {counts.get('claims', 0)}")
    print(
        f"  Communities: {counts.get('communities', 0)} ({stats['communities_summarized']} summarized)"
    )
    print(f"  Chunks:      {len(chunks)} ({stats['failed']} extraction failures)")
    print(f"  Neo4j:       {settings.neo4j_uri}")
    if stats["failed_details"]:
        print("\nFailed chunks:")
        for chunk_id, error in stats["failed_details"]:
            print(f"  {chunk_id}: {error}")
    return 0


def _run_index_themes(args, settings) -> int:
    """Build RAPTOR level 2: cluster chunk embeddings, then summarize each cluster.

    Clustering reads the vectors already in Chroma, so it costs nothing and can
    be checked with ``--dry-run`` before any LLM call is made — the gate
    numbers (how many themes span more than one video) are settled by the
    clustering alone.
    """
    from langchain_openai import ChatOpenAI

    from src.agents.llm import chat_model_kwargs
    from src.api.corpus import load_chunk_embeddings
    from src.rag.themes import ThemeStore, ThemeSummarizer, build_themes

    records = load_chunk_embeddings(settings.chroma_path, settings.chunk_collection)
    if not records:
        print("No chunk embeddings found. Run index-rag / bulk-index first.", file=sys.stderr)
        return 1

    summarizer = (
        None
        if args.dry_run
        else ThemeSummarizer(
            ChatOpenAI(**chat_model_kwargs(settings)),
            model_name=settings.deepseek_model,
        )
    )
    index = build_themes(
        records,
        summarizer,
        embedding_model=settings.embedding_model,
        chunk_collection=settings.chunk_collection,
        excerpts_per_theme=args.excerpts,
        on_progress=print,
    )

    stats = index.stats
    print("\nTheme layer built")
    print(f"  Themes:        {stats['themes']}")
    print(f"  Cross-video:   {stats['cross_video_themes']} (spanning 2+ videos)")
    print(f"  Single-video:  {stats['single_video_themes']}")
    print(f"  Videos:        {stats['videos_covered']} covered, {stats['chunks_clustered']} chunks")
    print(f"  Videos/theme:  {stats['video_count_distribution']}")
    print(
        "  Purity:        max property share in a job-search theme "
        f"{stats['max_property_share_in_job_search_theme']:.0%}"
    )
    if args.dry_run:
        print("\nDry run — nothing written.")
        return 0
    path = ThemeStore(settings.theme_path).save(index)
    print(f"  Written:       {path}")
    return 0


def _run_index_conflicts(args, settings) -> int:
    """Build the disagreement layer: candidates from claims, then one call each.

    Claims come from the **cached** GraphRAG extractions rather than from Neo4j,
    for two reasons. The cache is keyed on chunk id plus a hash of the chunk
    text, so it is exactly as current as the graph and cannot silently serve a
    claim for transcript that has since been re-chunked; and it needs no
    container, so this layer rebuilds on a machine where the graph is down. A
    chunk with no cached extraction contributes no claims and is reported, not
    guessed at.
    """
    from src.rag.conflicts import (
        DEFAULT_ADJUDICATE_REPEATS,
        ConflictConfig,
        AdjudicationCache,
        ConflictStore,
        LlmAdjudicator,
        build_conflicts,
        candidate_pairs,
        claims_from_chunks,
        corpus_fingerprint,
        probes_passed,
        run_probes,
    )
    from src.rag.embeddings import HuggingFaceEmbeddingModel
    from src.rag.graph_extract import extraction_cache_key
    from src.rag.graph_models import ChunkExtraction
    from src.rag.storage import TranscriptChunkStore

    embedding_model = HuggingFaceEmbeddingModel(settings.embedding_model)
    adjudicator = LlmAdjudicator(
        ChatOpenAI(**chat_model_kwargs(settings)),
        model_name=settings.deepseek_model,
    )

    if args.probes_only:
        results = run_probes(adjudicator, repeats=max(1, args.probe_repeats))
        for row in results:
            mark = "pass" if row["passed"] else "FAIL"
            print(f"  [{mark}] {row['probe_id']}: expected {row['expect']}, got {row['verdicts']}")
            if row["axis"]:
                print(f"         axis: {row['axis']}")
        print(f"\nProbes: {sum(1 for r in results if r['passed'])}/{len(results)} passed")
        return 0 if probes_passed(results) else 1

    cache_dir = settings.graph_cache_dir
    if cache_dir is None or not cache_dir.is_dir():
        print(
            "No graph extraction cache found. Run index-graph first.",
            file=sys.stderr,
        )
        return 1

    chunks = TranscriptChunkStore(
        settings.chroma_path,
        embedding_model=embedding_model,
        collection_name=settings.chunk_collection,
    ).all_chunks()
    if not chunks:
        print("No chunks found. Run index-rag / bulk-index first.", file=sys.stderr)
        return 1

    missing = 0

    def claim_texts(chunk) -> list[str]:
        nonlocal missing
        path = cache_dir / f"{extraction_cache_key(chunk)}.json"
        if not path.exists():
            missing += 1
            return []
        try:
            extraction = ChunkExtraction.model_validate_json(path.read_text(encoding="utf-8"))
        except ValueError:
            missing += 1
            return []
        return [claim.text for claim in extraction.claims]

    claims = claims_from_chunks(chunks, claim_texts)
    corpus = corpus_fingerprint(chunks)
    print(
        f"{corpus['videos']} videos, {len(chunks)} chunks (digest {corpus['digest']}), "
        f"{len(claims)} claims ({missing} chunks with no cached extraction)"
    )

    config = ConflictConfig(
        cross_channel_only=not args.allow_within_channel,
        adjudicate_repeats=max(1, args.adjudicate_repeats or DEFAULT_ADJUDICATE_REPEATS),
        **({"max_candidates": args.max_candidates} if args.max_candidates else {}),
    )

    if args.dry_run:
        pairs = candidate_pairs(claims, embedding_model.embed_documents, config)
        print(f"\n{len(pairs)} candidate chunk pairs (deterministic; no calls made)")
        for pair in pairs[:20]:
            print(
                f"  {pair.similarity:.3f}  {pair.left.channel_name} vs {pair.right.channel_name}\n"
                f"         A: {pair.left.text[:110]}\n"
                f"         B: {pair.right.text[:110]}"
            )
        print("\nDry run — nothing written.")
        return 0

    index = build_conflicts(
        claims,
        embedding_model.embed_documents,
        adjudicator,
        config=config,
        embedding_model=settings.embedding_model,
        adjudicator_model=settings.deepseek_model,
        chunk_collection=settings.chunk_collection,
        corpus=corpus,
        probe_repeats=max(1, args.probe_repeats),
        cache=(
            AdjudicationCache(settings.chroma_path.parent / "conflict_cache", args.resume_key)
            if args.resume_key
            else None
        ),
        max_workers=max(1, args.max_workers),
        on_progress=print,
    )

    stats = index.stats
    print("\nDisagreement layer built")
    print(
        f"  Conflicts:     {stats['conflicts']} "
        f"({stats['unanimous_conflicts']} unanimous, {stats['split_conflicts']} split; "
        f"{stats['factual_conflicts']} factual)"
    )
    print(
        f"  Adjudicated:   {stats['candidates_adjudicated']} candidate pairs "
        f"x {stats['adjudicate_repeats']} = {stats['adjudications']} calls"
    )
    print(
        f"  Agreement:     {stats['verdict_agreement']} "
        f"({stats['pairs_with_split_verdicts']} pairs drew more than one verdict)"
    )
    print(f"  Precision:     {stats['conflict_precision']} (conflicts / pairs adjudicated)")
    print(f"  Channels:      {stats['channels_involved']} across {stats['videos_involved']} videos")
    print(
        f"  Corpus:        {corpus['videos']} videos / {corpus['chunks']} chunks "
        f"(digest {corpus['digest']}); {index.corpus['chunks_with_claims']} chunks had claims"
    )
    if stats.get("subsample_counts_at_3") is not None:
        print(
            f"  Spread:        ±{stats['count_sd_estimate']} "
            f"({stats['firm_conflicts']} firm, {stats['undecided_pairs']} pairs undecided); "
            f"3-look sub-runs of this data: {stats['subsample_counts_at_3']}"
        )
    print(f"  Rejected:      {stats['rejected']}")
    print(f"  Probes passed: {stats['probes_passed']}")
    if not stats["probes_passed"]:
        print(
            "\n  The adjudicator failed its own calibration. The conflicts above "
            "are not trustworthy — see the probes in the artifact.",
            file=sys.stderr,
        )
    path = ConflictStore(settings.conflict_path).save(index)
    print(f"  Written:       {path}")
    return 0


def _run_build_packs(args, settings) -> int:
    """Build every declared rubric pack, three ways, from one shared workspace.

    A dry run stops after routing and unit assembly: membership, the two unit
    pools and the shared budget are all deterministic and cost nothing, so they
    are worth reading before any LLM call is paid for — and a pack whose
    membership is wrong is a pack whose rubrics cannot be right.
    """
    from src.rag.pack_build import build_all, open_workspace, route_members, shared_budget
    from src.rag.packs import (
        PackStore,
        community_units,
        included_video_ids,
        raptor_units,
    )

    store = PackStore("experts")
    catalog = store.catalog()
    print(f"Blocked from every pack: {', '.join(catalog.blocked())}")

    if args.dry_run:
        workspace = open_workspace(settings, catalog, with_model=False)
        provenance = workspace.provenance()
        print(
            f"corpus {provenance.corpus_digest} — {provenance.chunk_count} chunks, "
            f"{provenance.video_count} videos, {provenance.theme_count} themes, "
            f"{provenance.graph_extractions} extractions"
        )
        for declaration in catalog.packs:
            if args.topic and declaration.topic not in args.topic:
                continue
            members = route_members(workspace, declaration, store.overrides(declaration.topic))
            videos = included_video_ids(members)
            raptor = raptor_units(workspace.themes, videos, workspace.by_chunk)
            communities = community_units(workspace.extractions, videos, workspace.by_chunk)
            print(
                f"\n{declaration.topic}: {len(videos)} videos · "
                f"raptor {len(raptor)} units · communities {len(communities)} units · "
                f"budget {shared_budget(args.units, raptor, communities)}"
            )
            for member in members:
                mark = "+" if member.included else "-"
                print(
                    f"  {mark} {member.score:.3f} {member.video_id:<13} "
                    f"{(member.channel_name or '?')[:28]:<30} {(member.title or '')[:52]}"
                )
        return 0

    built = build_all(
        settings,
        topics=args.topic,
        arms=args.arm or PACK_ARMS,
        budget=args.units,
        on_progress=print,
    )
    print("\nPacks built")
    for topic, arms in built.items():
        for arm, pack in arms.items():
            stats = pack.stats
            print(
                f"  {topic:<18} {arm:<12} {stats['rubrics']:>3} rubrics · "
                f"{stats['citations']:>3} citations · "
                f"{stats['multi_creator_share']:.0%} multi-creator · "
                f"{stats['gaps']} gaps · "
                f"{stats['excluded_video_citations']} excluded-video citations"
            )
        shipped = arms.get(args.ship) or next(iter(arms.values()))
        path = store.save_pack(shipped, shipped=True)
        print(f"  {topic:<18} shipped {shipped.arm} → {path}")
    return 0


def _verdict_weight(verdict: dict) -> str:
    """How much weight a metric's lead carries, in the words the verdict earned.

    Kept next to its two callers rather than in ``pack_ablation`` because it is
    presentation: the verdict itself carries ``basis``, and this only names it.
    The fallback covers runs committed before ``basis`` existed.
    """
    basis = verdict.get("basis")
    if basis is None:
        return "decisive" if verdict.get("decisive") else "inside the scorer noise"
    return {
        "cleared-spread": "decisive",
        "inside-spread": "inside the scorer noise",
        "unrepeated": "scored once — reliability unmeasured",
        "tied": "tied",
        "single-arm": "no comparison",
        "no-scored-arm": "not scored",
    }.get(basis, basis)


def _run_score_packs(args, settings) -> int:
    """D2: score one pack's three arms on the held-out critique harness."""
    from src.api.corpus import load_chunk_embeddings
    from src.evals.critique import format_table
    from src.evals.pack_ablation import run_pack_ablation
    from src.rag.packs import PackStore

    records = load_chunk_embeddings(settings.chroma_path, settings.chunk_collection)
    run = run_pack_ablation(
        settings,
        topic=args.topic,
        repeats=args.repeats,
        records=records,
        on_progress=print,
    )
    print()
    print(format_table(run))
    for metric, verdict in run["verdicts"].items():
        print(
            f"  {metric}: {verdict.get('leader')} leads "
            f"({_verdict_weight(verdict)}) "
            f"— {verdict.get('reason')}"
        )
    path = PackStore("experts").save_ablation(run)
    print(f"\nsaved {path}")
    return 0


def _run_deep_research(args, settings) -> int:
    """Build one pack through the plan → execute → criticise → republish loop.

    The report is written whether or not the arms are scored, because the thing
    worth reading first is not a number: it is what the gap critic said was
    missing and which rules its probes actually produced. ``--score`` adds the
    held-out table underneath, including the hand-built pack — which costs
    nothing to include, since the matcher cache is keyed on the findings text
    and the shipped pack's have not moved.
    """
    from src.api.corpus import load_chunk_embeddings
    from src.evals.critique import format_table
    from src.evals.regression import save_run
    from src.rag.deep_research import (
        attach_scores,
        load_report,
        run_deep_research,
        run_frontier_round,
        score_research,
    )
    from src.rag.packs import PackStore

    store = PackStore("experts")
    records = load_chunk_embeddings(settings.chroma_path, settings.chunk_collection)

    if args.score_only:
        report = load_report(store, args.topic)
        if report is None:
            print(f"No build report for {args.topic}. Run without --score-only first.")
            return 1
    elif args.frontier:
        # Additive on purpose: the frontier arm has to be scored *beside* the
        # arms it claims to beat, so it is built into the existing report rather
        # than replacing a build whose three arms are already committed.
        report = run_frontier_round(
            settings,
            topic=args.topic,
            gap_probes_count=args.gap_probes,
            records=records,
            on_progress=print,
        )
        block = report["frontier"]
        print("\nGaps the coverage-aware critic named after round 1")
        for gap in block["gaps"]:
            print(f"  {gap['gap_id']}: {gap['missing']}")
            print(f"      probe: {gap['probe']}")
        print("\nRules the frontier round added")
        for row in block["diff"]["added"]:
            print(f"  {row['rubric_id']} [{row['unit_id']}] {row['criterion']}")
        print("\nRules refused for resting on no passage of their own")
        for row in block["refused"]:
            print(f"  {row['rubric_id']} {row['criterion']}")
        print("\nRediscovery against the shipped pack")
        for arm, row in block["rediscovery"].items():
            print(
                f"  {arm:<16} {row['rediscovered']}/{row['added']} added rules cite a chunk "
                f"{row.get('shipped_arm')} already cites"
            )
    else:
        report = run_deep_research(
            settings,
            topic=args.topic,
            round_one_probes=args.probes,
            gap_probes_count=args.gap_probes,
            records=records,
            on_progress=print,
        )
        print("\nGaps the critic named after round 1")
        for gap in report["gaps"]:
            print(f"  {gap['gap_id']}: {gap['missing']}")
            print(f"      probe: {gap['probe']}")
        print("\nRules round 2 added because of them")
        for row in report["diff"]["added"]:
            print(f"  {row['rubric_id']} [{row['unit_id']}] {row['criterion']}")
        print("\nCall budget")
        for row in report["budget"]:
            print(
                f"  {row['arm']:<14} planner {row['planner_calls']} · "
                f"executor {row['executor_calls']}/{row['probes_budgeted']} · "
                f"critic {row['critic_calls']} · total {row['total_llm_calls']} · "
                f"{row['rubrics']} rubrics · {row['citations']} quotes"
            )

    if not (args.score or args.score_only):
        print(f"\nsaved {store.topic_dir(args.topic) / 'research.json'}")
        return 0

    run = score_research(
        settings, topic=args.topic, repeats=args.repeats, records=records, on_progress=print
    )
    print()
    print(format_table(run))
    for metric, verdict in run["verdicts"].items():
        print(
            f"  {metric}: {verdict.get('leader')} leads "
            f"({_verdict_weight(verdict)}) "
            f"— {verdict.get('reason')}"
        )
    attach_scores(store, args.topic, run)
    # Committed beside the other held-out runs as well as folded into the build
    # report: the report is the loop's own account of itself, and a score that
    # only exists inside the artifact it is arguing for is not a score a
    # reviewer can put next to anything. This is the file the Critique panel
    # reads.
    run_path = save_run(run)
    print(f"\nsaved {store.topic_dir(args.topic) / 'research.json'}")
    print(f"saved {run_path}")
    return 0


def _run_eval_graph_extraction(settings) -> int:
    """Score cached extractions against the hand-labelled sample. Cache-only."""
    from src.evals.graph_extraction import load_labelled, score_all
    from src.rag.embeddings import HuggingFaceEmbeddingModel
    from src.rag.graph_extract import GraphExtractor
    from src.rag.storage import TranscriptChunkStore

    class _CacheOnly:
        """Fails extraction for any chunk the cache does not already cover."""

        def invoke(self, messages: list) -> object:
            raise ValueError("not in extraction cache; run index-graph first")

    embedding_model = HuggingFaceEmbeddingModel(settings.embedding_model)
    chunk_store = TranscriptChunkStore(
        settings.chroma_path,
        embedding_model=embedding_model,
        collection_name=settings.chunk_collection,
    )
    labelled = load_labelled()
    wanted = {label.chunk_id for label in labelled}
    extractor = GraphExtractor(_CacheOnly(), cache_dir=settings.graph_cache_dir)
    extractions = {
        chunk.chunk_id: extractor.extract(chunk)
        for chunk in chunk_store.all_chunks()
        if chunk.chunk_id in wanted
    }
    report = score_all(extractions, labelled)
    print(
        f"graph extraction eval — {report['labelled_chunks']} labelled chunks\n"
        f"  entity_recall  {report['entity_recall']}\n"
        f"  claim_recall   {report['claim_recall']}"
    )
    for result in report["results"]:
        line = f"  {result['chunk_id']}: entities {result['entity_recall']}"
        if result["claim_recall"] is not None:
            line += f", claims {result['claim_recall']}"
        if result["missed_entities"]:
            line += f" — missed {result['missed_entities']}"
        if result["error"]:
            line += f" — ERROR: {result['error']}"
        print(line)
    return 0


def _run_index_contextual(args, settings) -> int:
    """Situate every chunk with an LLM sentence and index the result separately."""
    from langchain_openai import ChatOpenAI

    from src.rag.contextualize import ChunkContextualizer, build_contextual_index
    from src.rag.embeddings import HuggingFaceEmbeddingModel
    from src.rag.storage import TranscriptChunkStore

    embedding_model = HuggingFaceEmbeddingModel(settings.embedding_model)
    chunk_store = TranscriptChunkStore(
        settings.chroma_path,
        embedding_model=embedding_model,
        collection_name=settings.chunk_collection,
    )
    if not chunk_store.has_any_chunks():
        print("No chunks indexed. Run index-rag / bulk-index first.", file=sys.stderr)
        return 1
    contextual_store = TranscriptChunkStore(
        settings.chroma_path,
        embedding_model=embedding_model,
        collection_name=settings.contextual_chunk_collection,
    )

    kwargs: dict[str, object] = {
        "api_key": settings.deepseek_api_key,
        "model": settings.deepseek_model,
        "timeout": settings.llm_timeout_seconds,
    }
    if settings.deepseek_base_url:
        kwargs["base_url"] = settings.deepseek_base_url
    contextualizer = ChunkContextualizer(ChatOpenAI(**kwargs), cache_dir=settings.context_cache_dir)

    result = build_contextual_index(
        chunk_store,
        contextual_store,
        contextualizer,
        on_progress=print,
        max_chunks=args.max_chunks,
        max_workers=args.max_workers,
    )
    print("\nContextual index built")
    print(f"  {result.summary()}")
    print(f"  Collection: {settings.contextual_chunk_collection}")
    return 0


def _run_eval_ablation(args, settings) -> int:
    """Sweep retrieval configurations over the golden set and snapshot the table.

    Retrieval-only: no answer is generated and no judge runs. The default sweep
    is fully deterministic and needs no API key — only the local embedding, BM25
    and cross-encoder models. ``--sweep extended`` adds the query-side transforms,
    which do call an LLM once per question (cached), and the contextual columns,
    which read the index ``index-contextual`` builds.
    """
    from src.evals.ablation import format_table, run_default_ablation
    from src.evals.regression import save_run

    result = run_default_ablation(settings, top_k=args.top_k, sweep=args.sweep, on_progress=print)
    path = save_run(result)
    print(f"\n{result['run_id']} — {result['entries']} golden questions\n")
    print(format_table(result))
    print(f"\nsaved {path}")
    return 0


def _with_published_contested(source: dict[str, Any], rescored: dict[str, Any]) -> dict[str, Any]:
    """Carry every cell's contested measurement over from the run being re-scored.

    ``contested_coverage`` sits deliberately outside the grounding gate — its
    denominator is fixed by retrieval before the answering call, so the padding
    attacks the gate exists for cannot move it (``src/evals/KNOWN_GAP_attack2``).
    A re-score under the gate therefore has no business changing it, and left to
    itself it would: the committed conflict corpus has been edited since these
    runs were published, so re-deriving the pairs turns "0 of 2 disagreements in
    context" into "0 of 1" — a movement with nothing to do with the rule under
    test, in a file whose whole purpose is to make one difference legible.

    So the published rows are kept verbatim. A run that wants a re-derived
    conflict layer wants a re-run, which is where that measurement is made.
    """
    cells: list[dict[str, Any]] = []
    for stored, fresh in zip(source.get("cells", []), rescored.get("cells", []), strict=True):
        cell = dict(fresh)
        cell["scores"] = {
            **(fresh.get("scores") or {}),
            "contested_coverage": (stored.get("scores") or {}).get("contested_coverage"),
        }
        for key in ("conflicts", "conflicts_in_context", "conflicts_named"):
            if key in stored:
                cell[key] = stored[key]
        cells.append(cell)
    return {**rescored, "cells": cells}


def _rescored_in_place(source: dict[str, Any], rescored: dict[str, Any]) -> dict[str, Any]:
    """The re-scored run written back over its source, identity and envelope intact.

    It is the *same run* — same findings, same matcher votes, same held-out
    video — read under a different scoring rule, so it keeps its ``run_id`` and
    ``created_at``: everything that cites this run by id (the app's expand
    request, the demo verdicts, the walkthrough) still resolves, and the file
    does not claim to be a measurement taken today.

    Merged rather than replaced, because two of the committed runs carry an
    envelope :func:`~src.evals.critique.build_run` knows nothing about — the pack
    ablation is ``kind: pack-ablation`` with a ``topic`` and per-arm ``rubrics``
    the packs tab reads, and re-scoring it into a bare critique run would take
    that tab down to make a metrics change.

    Nothing is lost by writing over the old numbers: the replay reproduces them
    exactly, so they survive on every cell as ``criteria_recall_ungated`` and
    ``evidence_precision_ungated``. What is gained is that no file in
    ``evals/runs/`` still presents an uncertified figure as a measured one.
    """
    cells = [
        {**stored, **fresh}
        for stored, fresh in zip(source.get("cells", []), rescored.get("cells", []), strict=True)
    ]
    merged = {**source, **rescored, "cells": cells}
    merged["run_id"] = source.get("run_id")
    merged["created_at"] = source.get("created_at")
    merged["kind"] = source.get("kind", rescored.get("kind"))
    # ``rescored_from`` would name this file itself, which reads as a second run
    # derived from a first. There is one run; it was re-scored on this date.
    merged.pop("rescored_from", None)
    merged["rescored_at"] = datetime.now(timezone.utc).isoformat()
    return merged


def _run_eval_critique(args, settings) -> int:
    """Measure whether the corpus reaches a held-out expert's criteria.

    Costs two LLM calls per setup plus the matcher's repeats — no RAGAS judge
    runs, because three of the four metrics are string and id arithmetic and the
    fourth is a cached vote. That is what makes it cheap enough for a later slice
    to re-run rather than cite this one's number.
    """
    from src.evals.critique import DEFAULT_MATCH_REPEATS, format_table
    from src.evals.critique_run import run_default_critique_eval
    from src.evals.regression import save_run

    repeats = args.repeats or DEFAULT_MATCH_REPEATS
    if args.rescore:
        import json

        from src.evals.critique import load_critique_dataset
        from src.evals.critique_run import (
            cached_matcher,
            chunk_text_lookup,
            llm_matcher,
            provenance_for_run,
            repeated_matcher,
            replay_matcher,
            rescore_committed_run,
        )
        from src.evals.regression import DEFAULT_RUNS_DIR

        # A bare id names a file in ``evals/runs/``; a path is read as written,
        # because one committed critique run lives outside that directory
        # (``experts/ablation.json``, which the packs tab reads) and it has to be
        # re-scorable under the same rule as the rest or the gate is applied to
        # the runs that happen to be conveniently placed.
        source_path = (
            Path(args.rescore)
            if args.rescore.endswith(".json")
            else DEFAULT_RUNS_DIR / f"{args.rescore}.json"
        )
        source = json.loads(source_path.read_text(encoding="utf-8"))
        dataset = load_critique_dataset()
        # A rescore re-applies an *arithmetic* rule to findings that have not
        # changed, so re-rolling the pairing would move the number for a reason
        # that is not the rule under test — and cost a call per cell. Every cell
        # stores the votes its matcher cast, so they are replayed when they are
        # there; the LLM matcher is the fallback for a run old enough not to
        # have them.
        has_votes = all(cell.get("match_runs") for cell in source.get("cells", []))
        match = (
            replay_matcher(source)
            if has_votes
            else cached_matcher(
                repeated_matcher(llm_matcher(settings), repeats=repeats), repeats=repeats
            )
        )
        rescore_config: dict[str, Any] = {
            # The source's own repeat count, not this invocation's default: the
            # votes are replayed, so the number of them is a fact about the run
            # being re-scored and not about the command re-scoring it.
            "match_repeats": source.get("match_repeats") or repeats,
            # Beside ``matcher`` rather than over it. The pairing really was cast
            # by the model named there; what changed is only that this file's
            # scores were re-derived from those stored votes.
            "rescored_matcher": "replayed-votes" if has_votes else "llm",
            "rescore_note": (
                "re-scored from this run's own stored findings and matcher votes — no "
                "model calls. Replaying under the gate the run published under reproduces "
                "every published score exactly, which is what makes any difference here "
                "attributable to the gate alone; criteria_recall_ungated and "
                "evidence_precision_ungated are the figures this run published"
            ),
        }
        result = _with_published_contested(
            source,
            rescore_committed_run(
                source,
                dataset,
                match,
                chunk_text_lookup(settings),
                # No conflict layer is loaded: the contested measurement is
                # carried over from the run being re-scored, because the gate
                # does not touch it and the committed conflict corpus has moved
                # since. See :func:`_with_published_contested`.
                conflicts=(),
                # Pack arms kept their per-finding provenance on disk even when
                # the run file predates the field; a retrieval arm never had one
                # and stays ungraded. See :func:`provenance_for_run`.
                provenance=provenance_for_run(source),
                config=rescore_config,
            ),
        )
        if args.in_place:
            merged = _rescored_in_place(source, result)
            source_path.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
            print(f"\n{merged['run_id']} — re-scored in place\n")
            print(format_table(merged))
            print(f"\nwrote {source_path}")
            return 0
        path = save_run(result)
        print(f"\n{result['run_id']} — rescored from {args.rescore}\n")
        print(format_table(result))
        print(f"\nsaved {path}")
        return 0

    setups = [s.strip() for s in (args.setups or "").split(",") if s.strip()] or None
    result = run_default_critique_eval(
        settings,
        setups=setups,
        repeats=repeats,
        on_progress=print,
    )
    path = save_run(result)
    print(f"\n{result['run_id']} — held out {result['held_out_video_id']}\n")
    print(format_table(result))
    print(f"\nsaved {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        settings = load_settings(require_keys=args.command != "eval-ablation")
        if args.command == "chat":
            from src.chat.session import run_session

            return run_session(settings)
        if args.command == "serve":
            import uvicorn

            from src.api.main import create_app

            uvicorn.run(create_app(settings), host=args.host, port=args.port)
            return 0
        if args.command == "eval-golden":
            return _run_eval_golden(args, settings)
        if args.command == "eval-ablation":
            return _run_eval_ablation(args, settings)
        if args.command == "eval-matrix":
            return _run_eval_matrix(args, settings)
        if args.command == "eval-critique":
            return _run_eval_critique(args, settings)
        if args.command == "rejudge":
            return _run_rejudge(args, settings)
        if args.command == "index-graph":
            return _run_index_graph(args, settings)
        if args.command == "index-contextual":
            return _run_index_contextual(args, settings)
        if args.command == "index-themes":
            return _run_index_themes(args, settings)
        if args.command == "index-conflicts":
            return _run_index_conflicts(args, settings)
        if args.command == "build-packs":
            return _run_build_packs(args, settings)
        if args.command == "score-packs":
            return _run_score_packs(args, settings)
        if args.command == "deep-research":
            return _run_deep_research(args, settings)
        if args.command == "eval-graph-extraction":
            return _run_eval_graph_extraction(settings)
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
                    summary_store=_build_summary_store(settings, embedding_model, raw_store),
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
                        removed_chunk_ids=result.removed_chunk_ids,
                    )
                )
                dashboard_path = _refresh_rag_pipeline_dashboard(settings)
                print(f"RAG pipeline dashboard: {dashboard_path}")
                return 0

            if args.command == "bulk-index":
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
                    summary_store=_build_summary_store(settings, embedding_model, raw_store),
                    summary_generator=_build_summary_generator(settings),
                )
                output = _run_bulk_index(
                    args=args,
                    settings=settings,
                    raw_store=raw_store,
                    chunk_store=chunk_store,
                    indexer=indexer,
                )
                print(output)
                dashboard_path = _refresh_rag_pipeline_dashboard(settings)
                print(f"RAG pipeline dashboard: {dashboard_path}")
                return 0

            if args.command == "summarize":
                agent = TranscriptAgent.from_settings(settings, raw_provider)
                summary = agent.summarize(SummaryRequest(video_id=video_id, source_url=args.url))
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
                from src.rag.query_transform import build_query_transform

                top_k = args.top_k or settings.rag_top_k
                embedding_model = HuggingFaceEmbeddingModel(settings.embedding_model)
                chunk_store = TranscriptChunkStore(
                    settings.chroma_path,
                    embedding_model=embedding_model,
                    collection_name=(
                        settings.contextual_chunk_collection
                        if args.contextual
                        else settings.chunk_collection
                    ),
                )
                indexer = RagIndexer(
                    raw_store=raw_store,
                    # The contextual collection is derived from the baseline one
                    # by index-contextual, so auto-indexing must never write into
                    # it — a chunk added here would carry no situating sentence
                    # and quietly weaken the index it is measured against.
                    chunk_store=(
                        TranscriptChunkStore(
                            settings.chroma_path,
                            embedding_model=embedding_model,
                            collection_name=settings.chunk_collection,
                        )
                        if args.contextual
                        else chunk_store
                    ),
                    target_chars=settings.chunk_target_chars,
                    overlap_chars=settings.chunk_overlap_chars,
                )
                context_provider = MultiTranscriptRagContextProvider(
                    raw_store=raw_store,
                    chunk_store=chunk_store,
                    indexer=indexer,
                    summary_store=_build_summary_store(settings, embedding_model, raw_store),
                    query_transform=build_query_transform(
                        args.query_transform or settings.query_transform, settings
                    ),
                )
                if args.rag_agent:
                    return _run_rag_agent(args, settings, context_provider, top_k)
                if args.graph_rag:
                    return _run_graph_rag_ask(args, settings, context_provider, top_k)
                agent = RagTranscriptAgent.from_settings(settings, context_provider)
                filter_top_k = args.transcript_filter_top_k or settings.transcript_filter_top_k
                filter_min_score = (
                    args.transcript_filter_min_score
                    if args.transcript_filter_min_score is not None
                    else settings.transcript_filter_min_score
                )
                recursive = (
                    settings.rag_recursive_default if args.recursive is None else args.recursive
                )
                recursion_options = None
                if recursive:
                    recursion_options = RecursionOptions(
                        max_depth=(
                            args.max_depth if args.max_depth is not None else settings.rag_max_depth
                        ),
                        max_followups=(
                            args.max_followups
                            if args.max_followups is not None
                            else settings.rag_max_followups
                        ),
                        followup_top_k=(
                            args.followup_top_k
                            if args.followup_top_k is not None
                            else settings.rag_followup_top_k
                        ),
                        novelty_min_chunks=(
                            args.novelty_min_chunks
                            if args.novelty_min_chunks is not None
                            else settings.rag_novelty_min_chunks
                        ),
                        max_total_followups=(
                            args.max_total_followups
                            if args.max_total_followups is not None
                            else settings.rag_max_total_followups
                        ),
                    )
                answer = agent.answer(
                    RagQuestionRequest(
                        question=args.question,
                        source_url=args.url,
                        top_k=top_k,
                        filter_transcripts=(args.filter_transcripts and args.url is None),
                        transcript_filter_top_k=filter_top_k,
                        transcript_filter_min_score=filter_min_score,
                        recursive=recursive,
                        recursion_options=recursion_options,
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
                log_recursion_trace(answer.recursion)
                print(
                    _format_rag_answer(
                        answer,
                        selected_transcripts=agent.last_context.selected_transcripts
                        if agent.last_context is not None
                        else [],
                        show_followups=args.show_followups or recursive,
                        print_trace=args.print_trace,
                    )
                )
                return 0

        parser.error(f"Unknown command: {args.command}")
        return 2
    except (ConfigError, Exception) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


# ANSI color cycle keyed by (iteration - 1) % 6. Iteration 1 -> bright cyan,
# 2 -> bright yellow, 3 -> bright green, 4 -> bright magenta, 5 -> bright blue,
# 6 -> bright white, then the cycle repeats (iteration 7 matches iteration 1).
_ITERATION_COLORS = (
    "\033[96m",  # bright cyan   (iteration 1)
    "\033[93m",  # bright yellow (iteration 2)
    "\033[92m",  # bright green  (iteration 3)
    "\033[95m",  # bright magenta(iteration 4)
    "\033[94m",  # bright blue   (iteration 5)
    "\033[97m",  # bright white  (iteration 6)
)
_ANSI_RESET = "\033[0m"


def _run_graph_rag_ask(args, settings, context_provider, top_k: int) -> int:
    """Answer one question with the GraphRAG agent and print route + citations."""
    from src.agents.graph_agent import GraphRagAgent

    agent = GraphRagAgent.from_settings(settings, context_provider)
    answer = agent.answer(
        RagQuestionRequest(
            question=args.question,
            source_url=args.url,
            top_k=top_k,
        )
    )
    if agent.last_context is not None:
        log_context_details(
            context_mode=agent.last_context.context_mode,
            top_k=agent.last_context.top_k,
            retrieved_chunks=agent.last_context.retrieved_chunks,
            rag_prompt_tokens_estimate=estimate_tokens(agent.last_context.context_text or ""),
        )
    print(f"Route: {agent.last_route}")
    print("")
    print(_format_rag_answer(answer))
    return 0


def _run_rag_agent(args, settings, context_provider, top_k: int) -> int:
    max_iterations = (
        args.max_iterations
        if args.max_iterations is not None
        else settings.rag_agent_max_iterations
    )
    agent = RagAgent.from_settings(settings, context_provider)
    agent.max_iterations = max_iterations

    is_tty = sys.stdout.isatty()
    print("Researching...")
    print("")
    answer_header_state = {"printed": False}
    on_event = _make_agent_progress_printer(is_tty, answer_header_state)
    answer = agent.answer_streaming(
        RagQuestionRequest(
            question=args.question,
            source_url=args.url,
            top_k=top_k,
            filter_transcripts=(args.filter_transcripts and args.url is None),
        ),
        on_event=on_event,
    )
    if not answer_header_state["printed"]:
        # max_iterations may terminate the loop before an answer_start event.
        print("")
        print("Answer")
    if agent.last_context is not None:
        log_context_details(
            context_mode=agent.last_context.context_mode,
            top_k=agent.last_context.top_k,
            retrieved_chunks=agent.last_context.retrieved_chunks,
            rag_prompt_tokens_estimate=estimate_tokens(agent.last_context.context_text or ""),
        )
    # The Answer / References body is only available from the streaming return
    # value (the parsed RagTranscriptAnswer); the on_event callback already
    # printed the blank line and the "Answer" header on the answer_start event.
    print(_format_rag_agent_answer(answer))
    print("")
    print(f"Agent: {agent.last_iteration_count} iterations (rag_agent)")
    return 0


def _make_agent_progress_printer(is_tty: bool, header_state: dict):
    """Build an on_event callback that prints streamed agent progress lines.

    On a TTY each retrieval line is colored by its iteration cycle and the chunk
    count overwrites the query line via a carriage return. On non-TTY stdout the
    query and chunk count print as two plain lines with no ANSI codes. The
    Answer / References / Agent footer lines never contain ANSI codes.
    ``header_state`` records whether the Answer header was printed so the caller
    can emit it if the loop terminates before an answer_start event.
    """

    def on_event(event: AgentProgressEvent) -> None:
        if event.event_type == "retrieval_start":
            line = f'[{event.iteration}] Retrieving: "{event.query}"'
            if is_tty:
                color = _color_for(event.iteration)
                print(f"{color}{line}", end="", flush=True)
            else:
                print(line)
        elif event.event_type == "retrieval_complete":
            suffix = f"  →  {event.chunk_count} chunks"
            if is_tty:
                color = _color_for(event.iteration)
                line = f'[{event.iteration}] Retrieving: "{event.query}"'
                print(f"\r{color}{line}{suffix}{_ANSI_RESET}", flush=True)
            else:
                print(suffix.strip())
        elif event.event_type == "answer_start":
            # Blank line then the Answer header in the default terminal color.
            print("")
            print("Answer")
            header_state["printed"] = True

    return on_event


def _color_for(iteration: int) -> str:
    return _ITERATION_COLORS[(iteration - 1) % 6]


def _format_rag_agent_answer(answer) -> str:
    """Format the agent's Answer / References body with no ANSI codes.

    Matches the existing rag-ask reference format. The "Answer" header is
    emitted by the on_event answer_start handler, so this returns only the
    answer text and the References block.
    """
    lines = [answer.answer]
    if answer.references:
        lines.extend(["", "References"])
        for reference in answer.references:
            start = (
                "unknown" if reference.start_seconds is None else str(int(reference.start_seconds))
            )
            end = "unknown" if reference.end_seconds is None else str(int(reference.end_seconds))
            lines.append(
                f"{reference.label} {reference.timestamp_url} "
                f"{start}-{end}s video={reference.video_id}"
            )
    return "\n".join(lines)


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
    removed_chunk_ids: list[str] | None = None,
) -> str:
    lines = [
        "RAG index updated",
        f"Raw transcript collection: {raw_collection}",
        f"Chunk collection: {chunk_collection}",
        f"Transcript summary collection: {summary_collection}",
        f"Chunks: {chunk_count}",
    ]
    # Only printed when a re-index actually shrank the video. Silence means
    # nothing was removed, which is the normal case; a number here means chunks
    # that would otherwise still be retrievable are gone.
    if removed_chunk_ids:
        lines.append(
            f"Removed stale chunks: {len(removed_chunk_ids)} "
            f"({', '.join(removed_chunk_ids[:5])}"
            f"{', …' if len(removed_chunk_ids) > 5 else ''})"
        )
    lines.extend(
        [
            f"Summary: {summary_status or 'not configured'}",
            f"Chroma path: {chroma_path}",
        ]
    )
    return "\n".join(lines)


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


def _format_rag_answer(
    answer,
    selected_transcripts=None,
    show_followups: bool = False,
    print_trace: bool = False,
) -> str:
    lines = []
    if selected_transcripts:
        lines.append("Selected transcripts")
        for index, transcript in enumerate(selected_transcripts, 1):
            score = "unknown" if transcript.score is None else f"{transcript.score:.3f}"
            lines.append(
                f"{index}. score={score} video={transcript.video_id} url={transcript.source_url}"
            )
        lines.append("")
        lines.append("Answer")
    lines.append(answer.answer)
    if answer.references:
        lines.extend(["", "References"])
        for reference in answer.references:
            start = (
                "unknown" if reference.start_seconds is None else str(int(reference.start_seconds))
            )
            end = "unknown" if reference.end_seconds is None else str(int(reference.end_seconds))
            lines.append(
                f"{reference.label} {reference.timestamp_url} "
                f"{start}-{end}s video={reference.video_id}"
            )
    if show_followups and answer.subtopics:
        lines.extend(["", "Proposed follow-ups"])
        for index, subtopic in enumerate(answer.subtopics, 1):
            lines.append(f"{index}. {subtopic.topic} (confidence {subtopic.confidence:.2f})")
            lines.append(f'   query: "{subtopic.followup_query}"')
    if answer.recursion is not None:
        trace = answer.recursion
        lines.extend(["", "Recursion trace"])
        stage_text = ", ".join(
            f"{stage.name} ({stage.llm_calls} LLM, {stage.retrievals} retrievals)"
            for stage in trace.stages
        )
        lines.append(f"Stages: {stage_text}")
        lines.append(f"Terminated: {trace.terminated_reason}")
        lines.append(f"Follow-ups proposed: {trace.total_followups_proposed}")
        lines.append(f"Follow-ups executed: {trace.total_followups_executed}")
        lines.append(f"Total LLM calls: {sum(stage.llm_calls for stage in trace.stages)}")
        if print_trace and trace.subtopic_evidence:
            lines.append("")
            lines.append("Trace chunks")
            for item in trace.subtopic_evidence:
                lines.append(
                    f"{item.subtopic_index}. {item.subtopic.topic} "
                    f"outcome={item.outcome} chunks={len(item.chunks)}"
                )
                lines.append(f'   query: "{item.subtopic.followup_query}"')
                for chunk in item.chunks:
                    preview = (chunk.text or "").replace("\n", " ")[:120]
                    lines.append(
                        f"   - video={chunk.video_id} chunk={chunk.chunk_index}: {preview}"
                    )
    return "\n".join(lines)


def _run_bulk_index(args, settings, raw_store, chunk_store, indexer) -> str:
    if args.concurrency != 1:
        raise ValueError("bulk-index currently supports --concurrency 1 only")
    mode = args.bulk_mode
    run = start_ingestion_run(
        mode=mode,
        label=args.label,
        query=getattr(args, "query", None),
        channel=getattr(args, "channel", None),
        since=str(getattr(args, "since", "") or "") or None,
        until=str(getattr(args, "until", "") or "") or None,
    )
    run_path = None
    try:
        discovery_client = SupadataDiscoveryClient(
            settings.superdata_api_key,
            timeout_seconds=settings.supadata_timeout_seconds,
            cache_dir=settings.chroma_path.parent / "discovery_cache",
            cache_ttl_hours=settings.discovery_cache_ttl_hours,
            use_cache=not args.no_discovery_cache,
        )
        if mode == "channel":
            if args.latest is not None and (args.since is not None or args.until is not None):
                raise ValueError("--latest cannot be combined with --since or --until")
            if args.latest is None and args.since is None and args.until is None:
                raise ValueError("channel mode requires --latest, --since, or --until")
            if args.latest is not None:
                videos = discover_latest_channel_videos(
                    args.channel,
                    limit=args.latest,
                    client=discovery_client,
                )
            else:
                videos = discover_channel_videos(
                    args.channel,
                    published_after=args.since,
                    published_before=args.until,
                    max_results=args.max_results,
                    client=discovery_client,
                )
        elif mode == "search":
            videos = discover_search_results(
                args.query,
                top_n=args.top_n,
                client=discovery_client,
            )
        else:
            raise ValueError(f"Unsupported bulk-index mode: {mode}")
    except Exception as exc:
        run.status = "failed"
        run.stage = "discovery"
        run.error = str(exc)
        run.complete()
        run_path = write_ingestion_run(run, ingestion_runs_dir(settings.chroma_path))
        return _format_bulk_summary(run, run_path)

    for video in videos:
        record = candidate_record(video)
        started = __import__("time").monotonic()
        try:
            fully_indexed = raw_store.get_raw_document(
                video.video_id
            ) is not None and chunk_store.has_chunks(video.video_id)
            if args.dry_run:
                record.outcome = "discovered"
                record.chunk_count = chunk_store.count_chunks(video.video_id)
            elif fully_indexed and args.skip_existing and not args.refresh_summary:
                record.outcome = "skipped_existing"
                record.chunk_count = chunk_store.count_chunks(video.video_id)
            else:
                result = indexer.index(
                    str(video.source_url),
                    refresh=not args.skip_existing,
                    refresh_summary=args.refresh_summary,
                )
                record.outcome = (
                    "summary_refreshed"
                    if fully_indexed and args.skip_existing and args.refresh_summary
                    else "indexed"
                )
                record.chunk_count = len(result.chunks)
                record.title = result.raw_document.title or record.title
                record.channel_name = result.raw_document.channel_name or record.channel_name
                record.published_at = result.raw_document.upload_date or record.published_at
        except Exception as exc:  # per-candidate failure should not abort the run
            record.outcome = "failed"
            record.error = str(exc)
        record.duration_seconds = round(__import__("time").monotonic() - started, 3)
        run.candidates.append(record)

    run.complete()
    run_path = write_ingestion_run(run, ingestion_runs_dir(settings.chroma_path))
    return _format_bulk_summary(run, run_path)


def _format_bulk_summary(run, run_path: Path | None) -> str:
    lines = [
        "Bulk index run",
        f"Run ID: {run.run_id}",
        f"Mode: {run.mode}",
        f"Status: {run.status}",
        f"Discovered: {run.candidate_count}",
        f"Indexed: {run.indexed_count}",
        f"Skipped: {run.skipped_count}",
        f"Failed: {run.failed_count}",
    ]
    if run.error:
        lines.append(f"Error: {run.error}")
    if run_path is not None:
        lines.append(f"Run record: {run_path}")
    lines.extend(["", "Outcome table", "video_id outcome chunks title"])
    for candidate in run.candidates:
        lines.append(
            f"{candidate.video_id} {candidate.outcome or ''} "
            f"{candidate.chunk_count if candidate.chunk_count is not None else ''} "
            f"{candidate.title or ''}"
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
    return TranscriptSummaryGenerator(
        ChatOpenAI(**chat_model_kwargs(settings)),
        model_name=settings.deepseek_model,
    )


def _refresh_rag_pipeline_dashboard(settings) -> Path:
    output = Path("dashboard/rag_pipeline.html")
    rows = collect_pipeline_rows(settings)
    filter_test_rows = collect_filter_test_rows(
        settings,
        rows,
        DEFAULT_FILTER_TEST_QUESTION,
    )
    write_dashboard(
        output=output,
        rows=rows,
        settings=settings,
        filter_test_question=DEFAULT_FILTER_TEST_QUESTION,
        filter_test_rows=filter_test_rows,
    )
    return output


if __name__ == "__main__":
    raise SystemExit(main())

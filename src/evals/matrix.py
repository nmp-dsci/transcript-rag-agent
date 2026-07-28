"""The RAGAS head-to-head matrix: every engine × the same golden questions.

The s05 §06 design made concrete. Each engine in the setup registry answers
every golden entry; every answer is scored by the *same* pipeline — the
deterministic id/IR metrics where the entry declares expected chunks, the
RAGAS judge (faithfulness / answer_relevancy / context_precision) over the
engine's own retrieved contexts, and the reference-based metrics
(``answer_correctness`` against the hand-written reference answer — the
primary verdict for global/temporal questions, where no chunk list can be
"the" reference).

One matrix run produces one committed ``matrix-<timestamp>.json`` under
``evals/runs/``: the per-setup runs verbatim (so nothing is lost relative to
``eval-golden``), plus a comparison pivot — metric × setup overall and broken
down by question type, which is where "graph wins global, ties local, pays
latency" becomes a number instead of a claim.

Cost/latency are first-class columns: ``avg_elapsed_seconds`` and
``avg_token_estimate`` per cell, because an engine that wins quality while
tripling latency is a router argument, not a replacement.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from src.config import Settings
from src.evals.golden import GoldenEntry, load_golden
from src.evals.matrix_cache import DEFAULT_CACHE_DIR, cell_fingerprint, load_cell, save_cell
from src.evals.regression import run_golden_eval

DEFAULT_MATRIX_SETUPS = ["rag_llm", "rag_llm_recursive", "rag_agent", "graph_rag"]

#: The pivot rows. A subset of QUALITY_METRICS plus the ops columns — the
#: matrix is a comparison surface, so it shows the metrics that differ by
#: engine, not every raw number (the per-setup runs keep those).
MATRIX_METRICS = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "answer_correctness",
    "answer_similarity",
    "llm_context_recall",
    "context_recall",
    "recall@10",
    "ndcg@10",
]


def config_snapshot(
    settings: Settings,
    *,
    top_k: int | None = None,
    judge: Any | None = None,
    ragas_version: str | None = None,
) -> dict[str, Any]:
    """The same config shape :func:`~src.evals.regression.run_golden_eval`
    records, computed directly from settings — so it is available even when
    every cell for a setup came from the cache and no fresh call ran.

    ``ragas_version`` is passed in rather than read here because it only
    describes a run that actually judged: metric implementations change between
    releases, so a score is comparable only to one from the same version, and
    stamping a version onto an unjudged run would imply a judge that never ran.
    """
    return {
        "answer_model": settings.deepseek_model,
        "embedding_model": settings.embedding_model,
        "retrieval_mode": settings.retrieval_mode,
        "rerank_enabled": settings.rerank_enabled,
        "neighbor_span": settings.neighbor_span,
        "top_k": top_k or settings.rag_top_k,
        "judge_model": getattr(judge, "judge_model", None)
        or settings.judge_model
        or settings.deepseek_model,
        "judge_samples": settings.judge_samples,
        "ragas_version": ragas_version,
    }


def run_matrix(
    runner: Any,
    settings: Settings,
    *,
    setups: list[str] | None = None,
    judge: Any | None = None,
    reference_fns: dict[str, Any] | None = None,
    entries: list[GoldenEntry] | None = None,
    top_k: int | None = None,
    on_progress: Callable[[str], None] | None = None,
    on_cell: Callable[[dict[str, Any]], None] | None = None,
    now: datetime | None = None,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    refresh: bool = False,
) -> dict[str, Any]:
    """Run every setup over the same golden entries and pivot the results.

    Caches by default: each ``(setup, entry)`` cell is looked up by a
    fingerprint of the question plus the exact answering/judging
    configuration (see :mod:`src.evals.matrix_cache`), so re-running this
    after adding one new engine variant only pays for that variant's cells —
    every unchanged cell from a prior run is reused, not re-scored. Pass
    ``refresh=True`` to bypass the cache and rescore everything (still
    written back to the cache afterwards).

    ``on_progress`` receives human-readable lines (the CLI prints them);
    ``on_cell`` receives a structured record per finished cell — ``setup``,
    ``entry_id``, ``cached``, ``done`` and ``total`` — which is what a progress
    bar needs and what parsing the log lines would only approximate.
    """
    setups = setups or list(DEFAULT_MATRIX_SETUPS)
    entries = entries if entries is not None else load_golden()
    moment = now or datetime.now(timezone.utc)

    judge_model = getattr(judge, "judge_model", None) if judge is not None else None
    judge_samples = getattr(judge, "samples", None) if judge is not None else None
    ragas_version_str: str | None = None
    if judge is not None:
        from src.evals.judge import ragas_version

        ragas_version_str = ragas_version()

    setup_runs: dict[str, dict[str, Any]] = {}
    cache_hits = 0
    cache_misses = 0
    cells_done = 0
    cells_total = len(setups) * len(entries)
    config = config_snapshot(settings, top_k=top_k, judge=judge, ragas_version=ragas_version_str)

    def report_cell(setup: str, entry_id: str, cached: bool) -> None:
        if on_cell is not None:
            on_cell(
                {
                    "setup": setup,
                    "entry_id": entry_id,
                    "cached": cached,
                    "done": cells_done,
                    "total": cells_total,
                }
            )

    for setup in setups:
        if on_progress is not None:
            on_progress(f"── setup {setup} ({len(entries)} questions) ──")
        entry_results: list[dict[str, Any]] = []
        for entry in entries:
            fingerprint = cell_fingerprint(
                setup,
                entry,
                settings,
                top_k=top_k,
                judge_model=judge_model,
                judge_samples=judge_samples,
                ragas_version=ragas_version_str,
                reference_scored=reference_fns is not None,
            )
            cached = None if refresh else load_cell(fingerprint, cache_dir)
            if cached is not None:
                cache_hits += 1
                cells_done += 1
                entry_results.append(cached)
                if on_progress is not None:
                    on_progress(f"[{setup}] {entry.id}: cached")
                report_cell(setup, entry.id, cached=True)
                continue
            cache_misses += 1
            single = run_golden_eval(
                runner,
                settings,
                setup=setup,
                judge=judge,
                reference_fns=reference_fns,
                entries=[entry],
                top_k=top_k,
                on_progress=on_progress,
                now=moment,
            )
            result = single["entries"][0]
            save_cell(fingerprint, result, cache_dir)
            cells_done += 1
            entry_results.append(result)
            report_cell(setup, entry.id, cached=False)
        setup_runs[setup] = {
            "run_id": f"matrix-{moment.strftime('%Y%m%d-%H%M%S')}-{setup}",
            "created_at": moment.isoformat(),
            "setup": setup,
            "config": config,
            "entries": entry_results,
            "summary": _summarize_dicts(entry_results),
        }

    return {
        "run_id": f"matrix-{moment.strftime('%Y%m%d-%H%M%S')}",
        "created_at": moment.isoformat(),
        "kind": "matrix",
        "setups": setups,
        "config": config,
        "judged": judge is not None,
        "reference_scored": reference_fns is not None,
        "entry_count": len(entries),
        "question_types": _type_counts(entries),
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "runs": setup_runs,
        "comparison": build_comparison(setup_runs),
    }


def _summarize_dicts(entry_dicts: list[dict[str, Any]]) -> dict[str, Any]:
    """:func:`~src.evals.regression.summarize`, from the plain dicts a cached
    or freshly-scored cell is stored as, rather than ``EntryResult`` objects."""
    from src.evals.regression import EntryResult, summarize

    results = [
        EntryResult(
            id=d["id"],
            question=d["question"],
            domain=d["domain"],
            question_type=d.get("question_type", "local"),
            answer=d.get("answer", ""),
            error=d.get("error"),
            scores=d.get("scores", {}),
            retrieved_chunk_ids=d.get("retrieved_chunk_ids", []),
            elapsed_seconds=d.get("elapsed_seconds", 0.0),
            token_estimate=d.get("token_estimate", 0),
        )
        for d in entry_dicts
    ]
    return summarize(results)


def build_comparison(setup_runs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Pivot per-setup runs into metric × setup, overall and per question type."""
    overall: dict[str, dict[str, float]] = {}
    by_type: dict[str, dict[str, dict[str, float]]] = {}
    ops: dict[str, dict[str, Any]] = {}

    for setup, run in setup_runs.items():
        entries = [e for e in run.get("entries", []) if not e.get("error")]
        failed = [e for e in run.get("entries", []) if e.get("error")]
        ops[setup] = {
            "avg_elapsed_seconds": _mean([e.get("elapsed_seconds", 0.0) for e in entries]),
            "avg_token_estimate": _mean([e.get("token_estimate", 0) for e in entries]),
            "answered": len(entries),
            "failed": len(failed),
        }
        # Partition once per setup rather than once per (setup, metric): the
        # split by question type does not depend on which metric is being
        # averaged, so rebuilding it inside the metric loop repeats the same
        # filter for every entry in MATRIX_METRICS.
        typed_entries: dict[str, list[dict[str, Any]]] = {}
        for entry in entries:
            typed_entries.setdefault(entry.get("question_type", "local"), []).append(entry)

        for metric in MATRIX_METRICS:
            value = _mean(_metric_values(entries, metric))
            if value is not None:
                overall.setdefault(metric, {})[setup] = value
            for question_type in sorted(typed_entries):
                typed_value = _mean(_metric_values(typed_entries[question_type], metric))
                if typed_value is not None:
                    by_type.setdefault(question_type, {}).setdefault(metric, {})[setup] = (
                        typed_value
                    )

    return {"overall": overall, "by_question_type": by_type, "ops": ops}


def format_matrix_table(result: dict[str, Any]) -> str:
    """A terminal-readable pivot: rows metrics, columns setups."""
    setups = result["setups"]
    comparison = result["comparison"]
    lines: list[str] = []
    if "cache_hits" in result or "cache_misses" in result:
        hits = result.get("cache_hits", 0)
        misses = result.get("cache_misses", 0)
        total = hits + misses
        lines.append(
            f"cache: {hits}/{total} cells reused, {misses} scored fresh"
            if total
            else "cache: no cells"
        )
        lines.append("")

    def table(title: str, rows: dict[str, dict[str, float]]) -> None:
        if not rows:
            return
        lines.append(title)
        header = f"  {'metric':<22}" + "".join(f"{setup:>18}" for setup in setups)
        lines.append(header)
        for metric, cells in rows.items():
            row = f"  {metric:<22}"
            for setup in setups:
                value = cells.get(setup)
                row += f"{value:>18.3f}" if isinstance(value, float) else f"{'—':>18}"
            lines.append(row)
        lines.append("")

    table("overall", comparison.get("overall", {}))
    for question_type, rows in sorted(comparison.get("by_question_type", {}).items()):
        table(f"question_type = {question_type}", rows)

    ops = comparison.get("ops", {})
    if ops:
        lines.append("ops")
        lines.append(f"  {'':<22}" + "".join(f"{setup:>18}" for setup in setups))
        for key in ("avg_elapsed_seconds", "avg_token_estimate", "answered", "failed"):
            row = f"  {key:<22}"
            for setup in setups:
                value = ops.get(setup, {}).get(key)
                row += (
                    f"{value:>18.2f}"
                    if isinstance(value, float)
                    else f"{value if value is not None else '—':>18}"
                )
            lines.append(row)
    return "\n".join(lines)


def _metric_values(entries: list[dict[str, Any]], metric: str) -> list[float]:
    return [
        value
        for entry in entries
        if isinstance((value := (entry.get("scores") or {}).get(metric)), (int, float))
    ]


def _mean(values: list[Any]) -> float | None:
    numbers = [float(v) for v in values if isinstance(v, (int, float))]
    if not numbers:
        return None
    return round(sum(numbers) / len(numbers), 4)


def _type_counts(entries: list[GoldenEntry]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in entries:
        question_type = getattr(entry, "question_type", "local")
        counts[question_type] = counts.get(question_type, 0) + 1
    return counts

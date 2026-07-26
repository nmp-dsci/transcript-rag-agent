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
from typing import Any, Callable

from src.config import Settings
from src.evals.golden import GoldenEntry, load_golden
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
    now: datetime | None = None,
) -> dict[str, Any]:
    """Run every setup over the same golden entries and pivot the results."""
    setups = setups or list(DEFAULT_MATRIX_SETUPS)
    entries = entries if entries is not None else load_golden()
    moment = now or datetime.now(timezone.utc)

    setup_runs: dict[str, dict[str, Any]] = {}
    for setup in setups:
        if on_progress is not None:
            on_progress(f"── setup {setup} ({len(entries)} questions) ──")
        setup_runs[setup] = run_golden_eval(
            runner,
            settings,
            setup=setup,
            judge=judge,
            reference_fns=reference_fns,
            entries=entries,
            top_k=top_k,
            on_progress=on_progress,
            now=moment,
        )

    return {
        "run_id": f"matrix-{moment.strftime('%Y%m%d-%H%M%S')}",
        "created_at": moment.isoformat(),
        "kind": "matrix",
        "setups": setups,
        "config": next(iter(setup_runs.values()))["config"] if setup_runs else {},
        "judged": judge is not None,
        "reference_scored": reference_fns is not None,
        "entry_count": len(entries),
        "question_types": _type_counts(entries),
        "runs": setup_runs,
        "comparison": build_comparison(setup_runs),
    }


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
        for metric in MATRIX_METRICS:
            value = _mean(_metric_values(entries, metric))
            if value is not None:
                overall.setdefault(metric, {})[setup] = value
            for question_type in sorted({e.get("question_type", "local") for e in entries}):
                typed = [e for e in entries if e.get("question_type", "local") == question_type]
                typed_value = _mean(_metric_values(typed, metric))
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

"""Repeatable golden-set evaluation runs, and diffs between them.

The workbench scores answers as they are asked, which measures whatever was
configured at that moment. This runs the *same* curated questions through the
*current* configuration and snapshots the result, so a retrieval or prompt
change can be shown to have moved the numbers rather than assumed to have.

Each run writes one JSON file under ``.yt-agent/eval_runs/``. ``diff_runs``
compares two snapshots per question and per metric, which is what makes a
regression visible instead of merely recorded.

    uv run python -m src.cli eval-golden --setup rag_llm
    uv run python -m src.cli eval-golden --setup rag_llm --retrieval hybrid
    uv run python -m src.cli eval-golden --diff
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from src.config import Settings
from src.evals.golden import GoldenEntry, evaluate_entry, load_golden

DEFAULT_RUNS_DIR = Path(".yt-agent/eval_runs")

# Metrics where a drop is a regression. Latency and tokens move the other way
# and are reported separately rather than folded into a quality verdict.
QUALITY_METRICS = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "video_recall",
    "answer_correctness",
    "answer_similarity",
    "llm_context_recall",
]


@dataclass
class EntryResult:
    """One golden question, answered and scored under one configuration."""

    id: str
    question: str
    domain: str
    answer: str = ""
    error: str | None = None
    scores: dict[str, float | None] = field(default_factory=dict)
    retrieved_chunk_ids: list[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    token_estimate: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "question": self.question,
            "domain": self.domain,
            "answer": self.answer,
            "error": self.error,
            "scores": self.scores,
            "retrieved_chunk_ids": self.retrieved_chunk_ids,
            "elapsed_seconds": self.elapsed_seconds,
            "token_estimate": self.token_estimate,
        }


def run_golden_eval(
    runner: Any,
    settings: Settings,
    *,
    setup: str = "rag_llm",
    judge: Any | None = None,
    reference_fns: dict[str, Any] | None = None,
    entries: list[GoldenEntry] | None = None,
    scope: Any = None,
    top_k: int | None = None,
    on_progress: Callable[[str], None] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Answer and score every golden entry, returning a run snapshot.

    A question that fails is recorded with its error and excluded from the
    averages rather than scored as zero — a crash is missing data, and averaging
    it as zero would quietly understate the configuration being tested.
    """
    entries = entries if entries is not None else load_golden()
    results: list[EntryResult] = []

    for index, entry in enumerate(entries, start=1):
        if on_progress is not None:
            on_progress(f"[{index}/{len(entries)}] {entry.id}: {entry.question[:60]}")
        result = EntryResult(id=entry.id, question=entry.question, domain=entry.domain)
        try:
            answered = runner.run(setup, entry.question, top_k=top_k, scope=scope)
            if answered.error:
                raise RuntimeError(answered.error)
            result.answer = answered.answer
            result.retrieved_chunk_ids = list(answered.retrieved_chunk_ids)
            result.elapsed_seconds = answered.elapsed_seconds
            result.token_estimate = answered.token_estimate
            result.scores = dict(
                evaluate_entry(
                    entry,
                    answered.answer,
                    result.retrieved_chunk_ids,
                    score_fns=reference_fns,
                    contexts=answered.contexts,
                )
            )
            if judge is not None:
                evaluation = judge.score(
                    entry.question,
                    answered.answer,
                    answered.contexts,
                    answer_model=answered.model,
                )
                result.scores.update(evaluation.get("scores") or {})
                result.scores["composite"] = evaluation.get("composite")
        except Exception as exc:
            result.error = str(exc)
        results.append(result)

    moment = now or datetime.now(timezone.utc)
    return {
        "run_id": f"eval-{moment.strftime('%Y%m%d-%H%M%S')}",
        "created_at": moment.isoformat(),
        "setup": setup,
        "config": {
            "answer_model": settings.deepseek_model,
            "embedding_model": settings.embedding_model,
            "retrieval_mode": getattr(scope, "retrieval_mode", None)
            or settings.retrieval_mode,
            "rerank_enabled": settings.rerank_enabled,
            "neighbor_span": settings.neighbor_span,
            "top_k": top_k or settings.rag_top_k,
            "judge_model": settings.judge_model or settings.deepseek_model,
            "judge_samples": settings.judge_samples,
        },
        "entries": [result.to_dict() for result in results],
        "summary": summarize(results),
    }


def summarize(results: list[EntryResult]) -> dict[str, Any]:
    """Mean of each metric over the entries that produced one."""
    scored = [r for r in results if r.error is None]
    averages: dict[str, float] = {}
    for metric in QUALITY_METRICS + ["composite"]:
        values = [
            r.scores[metric]
            for r in scored
            if isinstance(r.scores.get(metric), (int, float))
        ]
        if values:
            averages[metric] = round(sum(values) / len(values), 4)
    return {
        "entries": len(results),
        "scored": len(scored),
        "failed": len(results) - len(scored),
        "averages": averages,
        "avg_elapsed_seconds": (
            round(sum(r.elapsed_seconds for r in scored) / len(scored), 2)
            if scored
            else None
        ),
        "avg_token_estimate": (
            round(sum(r.token_estimate for r in scored) / len(scored))
            if scored
            else None
        ),
    }


def save_run(run: dict[str, Any], runs_dir: Path = DEFAULT_RUNS_DIR) -> Path:
    runs_dir.mkdir(parents=True, exist_ok=True)
    path = runs_dir / f"{run['run_id']}.json"
    path.write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
    return path


def list_runs(runs_dir: Path = DEFAULT_RUNS_DIR) -> list[Path]:
    """Saved runs, oldest first. Ids are timestamps, so name order is time order."""
    if not runs_dir.exists():
        return []
    return sorted(runs_dir.glob("eval-*.json"))


def load_run(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def diff_runs(
    before: dict[str, Any], after: dict[str, Any], threshold: float = 0.02
) -> dict[str, Any]:
    """Per-metric and per-question movement between two runs.

    ``threshold`` is the movement below which a change is treated as noise. A
    single judged sample is not precise enough for every third decimal to mean
    something, so small drifts are reported as unchanged rather than dressed up
    as regressions.
    """
    metric_moves: list[dict[str, Any]] = []
    before_avg = before.get("summary", {}).get("averages", {})
    after_avg = after.get("summary", {}).get("averages", {})
    for metric in sorted(set(before_avg) | set(after_avg)):
        old, new = before_avg.get(metric), after_avg.get(metric)
        if old is None or new is None:
            continue
        delta = round(new - old, 4)
        metric_moves.append(
            {
                "metric": metric,
                "before": old,
                "after": new,
                "delta": delta,
                "direction": _direction(delta, threshold),
            }
        )

    before_entries = {e["id"]: e for e in before.get("entries", [])}
    entry_moves: list[dict[str, Any]] = []
    for entry in after.get("entries", []):
        old_entry = before_entries.get(entry["id"])
        if old_entry is None:
            continue
        changes = {}
        for metric in QUALITY_METRICS + ["composite"]:
            old, new = old_entry["scores"].get(metric), entry["scores"].get(metric)
            if not isinstance(old, (int, float)) or not isinstance(new, (int, float)):
                continue
            delta = round(new - old, 4)
            if _direction(delta, threshold) != "unchanged":
                changes[metric] = {"before": old, "after": new, "delta": delta}
        if changes:
            entry_moves.append(
                {
                    "id": entry["id"],
                    "question": entry["question"],
                    "changes": changes,
                }
            )

    regressions = [m for m in metric_moves if m["direction"] == "worse"]
    return {
        "before_run": before.get("run_id"),
        "after_run": after.get("run_id"),
        "threshold": threshold,
        "metrics": metric_moves,
        "entries": entry_moves,
        "regressed": [m["metric"] for m in regressions],
        "improved": [m["metric"] for m in metric_moves if m["direction"] == "better"],
    }


def _direction(delta: float, threshold: float) -> str:
    if delta > threshold:
        return "better"
    if delta < -threshold:
        return "worse"
    return "unchanged"

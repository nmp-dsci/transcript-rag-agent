"""Aggregate RAGAS evaluations per RAG setup for the Scoreboard view."""

from __future__ import annotations

from typing import Any

from src.chat.history import ChatEntry


def build_scoreboard(entries: list[ChatEntry]) -> dict[str, Any]:
    accumulators: dict[str, dict[str, Any]] = {}
    entries_judged = 0

    for entry in entries:
        judged = [
            answer
            for answer in entry.answers
            if answer.evaluation and answer.evaluation.get("composite") is not None
        ]
        if judged:
            entries_judged += 1
        # A "contest" needs at least two judged answers to compare; the winner
        # is the highest composite within the entry.
        winner_key = None
        if len(judged) >= 2:
            winner_key = max(
                judged, key=lambda answer: answer.evaluation["composite"]
            ).key

        for answer in entry.answers:
            acc = accumulators.setdefault(
                answer.key,
                {
                    "key": answer.key,
                    "title": answer.title,
                    "answers": 0,
                    "judged": 0,
                    "wins": 0,
                    "contests": 0,
                    "metric_sums": {},
                    "metric_counts": {},
                    "composite_sum": 0.0,
                    "latency_sum": 0.0,
                    "token_sum": 0,
                },
            )
            acc["answers"] += 1
            acc["latency_sum"] += answer.elapsed_seconds or 0.0
            acc["token_sum"] += answer.token_estimate or 0
            evaluation = answer.evaluation
            if evaluation and evaluation.get("composite") is not None:
                acc["judged"] += 1
                acc["composite_sum"] += evaluation["composite"]
                for metric, value in (evaluation.get("scores") or {}).items():
                    acc["metric_sums"][metric] = acc["metric_sums"].get(metric, 0.0) + value
                    acc["metric_counts"][metric] = acc["metric_counts"].get(metric, 0) + 1
            if winner_key is not None and any(a.key == answer.key for a in judged):
                acc["contests"] += 1
                if answer.key == winner_key:
                    acc["wins"] += 1

    rows = []
    for acc in accumulators.values():
        judged_count = acc["judged"]
        answer_count = acc["answers"]
        rows.append(
            {
                "key": acc["key"],
                "title": acc["title"],
                "answers": answer_count,
                "judged": judged_count,
                "avg_scores": {
                    metric: round(acc["metric_sums"][metric] / count, 4)
                    for metric, count in acc["metric_counts"].items()
                    if count
                },
                "avg_composite": (
                    round(acc["composite_sum"] / judged_count, 4)
                    if judged_count
                    else None
                ),
                "wins": acc["wins"],
                "contests": acc["contests"],
                "win_rate": (
                    round(acc["wins"] / acc["contests"], 4) if acc["contests"] else None
                ),
                "avg_latency_seconds": (
                    round(acc["latency_sum"] / answer_count, 2) if answer_count else None
                ),
                "avg_token_estimate": (
                    int(acc["token_sum"] / answer_count) if answer_count else None
                ),
            }
        )
    rows.sort(
        key=lambda row: (
            -(row["avg_composite"] if row["avg_composite"] is not None else -1.0),
            row["key"],
        )
    )
    return {
        "setups": rows,
        "entries_total": len(entries),
        "entries_judged": entries_judged,
    }

"""Aggregate RAGAS evaluations per RAG setup for the Scoreboard view.

Scores are only comparable when the stack that produced them matches, so
aggregation can group by setup alone (the historical view) or by setup and
answering model together. Win-rate contests are always scoped to a single
judge model: a self-graded answer and an independently-graded one never
compete against each other.
"""

from __future__ import annotations

from typing import Any

from src.chat.history import ChatAnswer, ChatEntry
from src.evals.judge import RUBRIC_VERSION, rubric_for

GroupBy = str
LEGACY_MODEL = "unknown"


def _group_key(answer: ChatAnswer, group_by: GroupBy) -> tuple[str, ...]:
    if group_by == "setup_model":
        return (answer.key, answer.model or LEGACY_MODEL)
    return (answer.key,)


def _judge_of(answer: ChatAnswer) -> str:
    return (answer.evaluation or {}).get("judge_model") or "unknown"


def _is_scored(answer: ChatAnswer) -> bool:
    evaluation = answer.evaluation
    return bool(evaluation and evaluation.get("composite") is not None)


def _winners_by_judge(answers: list[ChatAnswer]) -> dict[str, str | None]:
    """Winning answer key per judge model, for judges that scored 2+ answers.

    A contest needs at least two answers graded by the *same* judge; otherwise
    the comparison would rank two different graders' scales against each other.
    """
    by_judge: dict[str, list[ChatAnswer]] = {}
    for answer in answers:
        by_judge.setdefault(_judge_of(answer), []).append(answer)
    return {
        judge: (
            max(group, key=lambda a: a.evaluation["composite"]).key if len(group) >= 2 else None
        )
        for judge, group in by_judge.items()
    }


def build_scoreboard(
    entries: list[ChatEntry],
    *,
    group_by: GroupBy = "setup",
    judge_model: str | None = None,
) -> dict[str, Any]:
    accumulators: dict[tuple[str, ...], dict[str, Any]] = {}
    entries_judged = 0
    judge_models: set[str] = set()
    depth_judge_models: set[str] = set()
    ragas_versions: set[str] = set()
    embedding_models: set[str] = set()
    rubric_versions: set[str] = set()
    self_graded_answers = 0
    last_judged: str | None = None

    for entry in entries:
        scored = [a for a in entry.answers if _is_scored(a)]
        if judge_model:
            scored = [a for a in scored if _judge_of(a) == judge_model]
        if scored:
            entries_judged += 1
        winners = _winners_by_judge(scored)

        for answer in entry.answers:
            evaluation = answer.evaluation or {}
            judge_excluded = bool(judge_model and evaluation and _judge_of(answer) != judge_model)
            key = _group_key(answer, group_by)
            acc = accumulators.setdefault(
                key,
                {
                    "key": answer.key,
                    "title": answer.title,
                    "model": answer.model,
                    "legacy": answer.model is None,
                    "answers": 0,
                    "judged": 0,
                    "wins": 0,
                    "contests": 0,
                    "metric_sums": {},
                    "metric_counts": {},
                    "composite_sum": 0.0,
                    "capped": 0,
                    "ungrounded": 0,
                    "latency_sum": 0.0,
                    "token_sum": 0,
                },
            )
            acc["answers"] += 1
            acc["latency_sum"] += answer.elapsed_seconds or 0.0
            acc["token_sum"] += answer.token_estimate or 0
            if _is_scored(answer) and not judge_excluded:
                acc["judged"] += 1
                acc["composite_sum"] += evaluation["composite"]
                for metric, value in (evaluation.get("scores") or {}).items():
                    acc["metric_sums"][metric] = acc["metric_sums"].get(metric, 0.0) + value
                    acc["metric_counts"][metric] = acc["metric_counts"].get(metric, 0) + 1
                if evaluation.get("cap_applied"):
                    acc["capped"] += 1
                if evaluation.get("grounding_floor_breached"):
                    acc["ungrounded"] += 1
                if evaluation.get("self_graded"):
                    self_graded_answers += 1
                judge_models.add(_judge_of(answer))
                if evaluation.get("depth_judge_model"):
                    depth_judge_models.add(str(evaluation["depth_judge_model"]))
                rubric_versions.add(str(evaluation.get("rubric_version") or RUBRIC_VERSION))
                if evaluation.get("ragas_version"):
                    ragas_versions.add(str(evaluation["ragas_version"]))
                if evaluation.get("embedding_model"):
                    embedding_models.add(str(evaluation["embedding_model"]))
                scored_at = evaluation.get("scored_at")
                if scored_at and (last_judged is None or str(scored_at) > last_judged):
                    last_judged = str(scored_at)
                winner = winners.get(_judge_of(answer))
                if winner is not None:
                    acc["contests"] += 1
                    if answer.key == winner:
                        acc["wins"] += 1

    rows = []
    for acc in accumulators.values():
        judged_count = acc["judged"]
        answer_count = acc["answers"]
        rows.append(
            {
                "key": acc["key"],
                "title": acc["title"],
                "model": acc["model"],
                "legacy": acc["legacy"],
                "answers": answer_count,
                "judged": judged_count,
                "avg_scores": {
                    metric: round(acc["metric_sums"][metric] / count, 4)
                    for metric, count in acc["metric_counts"].items()
                    if count
                },
                "avg_composite": (
                    round(acc["composite_sum"] / judged_count, 4) if judged_count else None
                ),
                "wins": acc["wins"],
                "contests": acc["contests"],
                # How many of this row's judged answers hit the rubric's cap.
                # Always 0 under a rubric with no cap.
                "capped": acc["capped"],
                # How many breached the grounding floor at all — a superset of
                # ``capped``, since an answer can be ungrounded *and* have
                # scored below the cap on its own merits.
                "ungrounded": acc["ungrounded"],
                "win_rate": (round(acc["wins"] / acc["contests"], 4) if acc["contests"] else None),
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
            row["model"] or "",
        )
    )
    # The rubric these rows were composited under decides which metric columns
    # exist and what the composite means, so it is read off the records rather
    # than assumed. Mixed rubrics fall back to the original one: the columns it
    # names are the ones every record definitely has.
    rubric = rubric_for(next(iter(rubric_versions)) if len(rubric_versions) == 1 else None)
    return {
        "setups": rows,
        "entries_total": len(entries),
        "entries_judged": entries_judged,
        "group_by": group_by,
        "provenance": {
            "judge_models": sorted(judge_models),
            "depth_judge_models": sorted(depth_judge_models),
            "ragas_versions": sorted(ragas_versions),
            "embedding_models": sorted(embedding_models),
            "last_judged": last_judged,
            # The answering model also graded these answers. Stated rather than
            # left to be noticed: a leaderboard a model produced and then
            # scored is self-assessment, and reading it as an independent
            # verdict is the mistake this field exists to prevent.
            "self_graded": self_graded_answers > 0,
            "self_graded_answers": self_graded_answers,
            "rubric_version": rubric.version,
            "rubric_versions": sorted(rubric_versions),
            "metrics": list(rubric.metrics),
            "metric_weights": dict(rubric.weights),
            "metric_groups": [
                {
                    "key": group.key,
                    "label": group.label,
                    "weight": group.weight,
                    "metrics": list(group.metrics),
                }
                for group in rubric.groups
            ],
            "composite": rubric.composite_description,
        },
    }

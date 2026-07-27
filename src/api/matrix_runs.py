"""Committed matrix runs as the Scoreboard's data source.

The Scoreboard used to aggregate ``dashboard/chat_history.json`` — whichever
questions happened to be asked live through the Chat tab and judged. That made
the leaderboard depend on manual bookkeeping: a new engine stayed invisible
until someone re-asked every question through it, even when a matrix run had
already scored it against the whole golden set.

A matrix run is the controlled version of the same comparison: every setup
answering the *same* golden questions under one recorded configuration, graded
by one judge. This module adapts that on-disk shape into the
:class:`~src.chat.history.ChatEntry` objects
:func:`~src.api.scoreboard.build_scoreboard` already aggregates, so grouping,
win-rate contests and provenance keep working unchanged — only where the
numbers come from changes. Chat history is untouched; it stays the *live* set
behind the Chat tab rather than the leaderboard.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from src.api.experiments import DEFAULT_RUNS_DIR
from src.chat.history import ChatAnswer, ChatEntry
from src.chat.setups import setup_spec

#: Recorded on each adapted answer, so a Scoreboard row is traceable to the
#: command that produced it the way a live answer names its chat command.
MATRIX_COMMAND = "eval-matrix"


def _iter_matrix_files(runs_dir: Path | None = None) -> Iterator[dict[str, Any]]:
    """Every committed ``matrix-*.json``, skipping unreadable/foreign files."""
    directory = runs_dir or DEFAULT_RUNS_DIR
    if not directory.exists():
        return
    for path in sorted(directory.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            # A malformed snapshot must not take the whole tab down.
            continue
        if isinstance(data, dict) and data.get("kind") == "matrix":
            yield data


def _sort_key(data: dict[str, Any]) -> str:
    return str(data.get("created_at") or data.get("run_id") or "")


def list_matrix_runs(runs_dir: Path | None = None) -> list[dict[str, Any]]:
    """Descriptors for the Scoreboard's run picker, newest first.

    Deliberately lightweight — the picker only needs to label each run, not
    render it; the selected run is loaded separately.
    """
    runs = sorted(_iter_matrix_files(runs_dir), key=_sort_key, reverse=True)
    return [
        {
            "run_id": data.get("run_id"),
            "created_at": data.get("created_at"),
            "setups": data.get("setups", []),
            "entry_count": data.get("entry_count"),
            "judged": bool(data.get("judged", False)),
        }
        for data in runs
    ]


def load_matrix_run(
    run_id: str | None = None, runs_dir: Path | None = None
) -> dict[str, Any] | None:
    """One committed matrix run, or the newest when ``run_id`` is omitted.

    Returns ``None`` when nothing matches — an unknown ``run_id`` and an empty
    ``evals/runs/`` are the same situation for the caller: there is no run to
    score, so the Scoreboard renders its empty state rather than erroring.
    """
    runs = sorted(_iter_matrix_files(runs_dir), key=_sort_key, reverse=True)
    if run_id is not None:
        return next((data for data in runs if data.get("run_id") == run_id), None)
    return runs[0] if runs else None


def _title_for(setup: str) -> str:
    try:
        return setup_spec(setup).title
    except KeyError:
        # A run committed before a setup was renamed (or after it was removed)
        # still deserves a row — label it by key rather than dropping it.
        return setup


def _evaluation(
    cell: dict[str, Any], config: dict[str, Any], created_at: str
) -> dict[str, Any] | None:
    """The judge record for one cell, in the shape ``build_scoreboard`` reads.

    ``None`` when the cell carries no ``composite`` — an unjudged matrix run
    (``eval-matrix --no-judge``) has deterministic retrieval metrics but no
    RAGAS verdict, and must count as an answer that was never judged rather
    than as a zero.
    """
    scores = cell.get("scores") or {}
    composite = scores.get("composite")
    if not isinstance(composite, (int, float)):
        return None
    return {
        "judge": "ragas",
        "judge_model": config.get("judge_model"),
        "scores": {
            metric: value
            for metric, value in scores.items()
            if metric != "composite" and isinstance(value, (int, float))
        },
        "composite": composite,
        "ragas_version": config.get("ragas_version"),
        "embedding_model": config.get("embedding_model"),
        "judge_samples": config.get("judge_samples"),
        "scored_at": created_at,
        "error": None,
    }


def _answer(
    setup: str, cell: dict[str, Any], config: dict[str, Any], created_at: str
) -> ChatAnswer:
    return ChatAnswer(
        key=setup,
        title=_title_for(setup),
        command=MATRIX_COMMAND,
        answer=str(cell.get("answer") or ""),
        token_estimate=int(cell.get("token_estimate") or 0),
        elapsed_seconds=float(cell.get("elapsed_seconds") or 0.0),
        error=cell.get("error"),
        retrieved_chunk_ids=list(cell.get("retrieved_chunk_ids") or []),
        evaluation=_evaluation(cell, config, created_at),
        model=config.get("answer_model"),
        embedding_model=config.get("embedding_model"),
        top_k=config.get("top_k"),
        retrieval_mode=config.get("retrieval_mode"),
    )


def matrix_entries(data: dict[str, Any]) -> list[ChatEntry]:
    """One :class:`ChatEntry` per golden question, one answer per setup.

    A matrix run stores setup-major (every question under each setup); the
    scoreboard aggregates question-major, because a win-rate contest compares
    the setups that answered *the same* question. This pivots between the two.
    """
    run_config = data.get("config") or {}
    created_at = str(data.get("created_at") or "")
    by_question: dict[str, ChatEntry] = {}

    for setup, run in (data.get("runs") or {}).items():
        config = run.get("config") or run_config
        for cell in run.get("entries") or []:
            question_id = str(cell.get("id") or "")
            entry = by_question.get(question_id)
            if entry is None:
                entry = ChatEntry(
                    id=question_id,
                    question=str(cell.get("question") or ""),
                    url=None,
                    asked_at=created_at,
                )
                by_question[question_id] = entry
            entry.answers.append(_answer(setup, cell, config, created_at))

    return list(by_question.values())

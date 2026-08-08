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
import threading
from pathlib import Path
from typing import Any, Iterator

from src.api.experiments import DEFAULT_RUNS_DIR
from src.chat.history import ChatAnswer, ChatEntry
from src.chat.setups import setup_spec

#: Recorded on each adapted answer, so a Scoreboard row is traceable to the
#: command that produced it the way a live answer names its chat command.
MATRIX_COMMAND = "eval-matrix"

_Fingerprint = tuple[tuple[str, int, int], ...]
_cache_lock = threading.Lock()
_cache: dict[Path, tuple[_Fingerprint, list[dict[str, Any]]]] = {}


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


def _fingerprint(directory: Path) -> _Fingerprint:
    """Name, mtime and size of every ``*.json`` in the directory.

    Stat calls are cheap where parsing is not, and this catches a file being
    added, removed or rewritten in place — a run committed while the server is
    up must show up on the next request, not after a restart.
    """
    entries: list[tuple[str, int, int]] = []
    for path in sorted(directory.glob("*.json")):
        try:
            stat = path.stat()
        except OSError:
            continue
        entries.append((path.name, stat.st_mtime_ns, stat.st_size))
    return tuple(entries)


def load_matrix_runs(runs_dir: Path | None = None) -> list[dict[str, Any]]:
    """Every committed matrix run, newest first, from one pass over the
    directory.

    The Scoreboard needs both the picker's list *and* one selected run per
    request, and a committed run is a large document (hundreds of KB once a
    sweep covers four engines). Reading and parsing the directory once and
    deriving both from that pass keeps an ordinary interaction — changing the
    group-by, the judge filter or the selected run — from re-parsing every
    file twice.

    The parsed result is then memoized against the directory's stat
    fingerprint, so those interactions re-read nothing at all until a run is
    actually written. The list is fresh per call, but the run documents in it
    are the cached objects: callers read them (:func:`describe_matrix_run` and
    :func:`matrix_entries` both build new values) rather than editing them in
    place.
    """
    directory = runs_dir or DEFAULT_RUNS_DIR
    if not directory.exists():
        return []
    fingerprint = _fingerprint(directory)
    with _cache_lock:
        cached = _cache.get(directory)
        if cached is not None and cached[0] == fingerprint:
            return list(cached[1])
    runs = sorted(_iter_matrix_files(directory), key=_sort_key, reverse=True)
    with _cache_lock:
        _cache[directory] = (fingerprint, runs)
    return list(runs)


def describe_matrix_run(data: dict[str, Any]) -> dict[str, Any]:
    """One descriptor for the run picker, which only needs to label a run."""
    return {
        "run_id": data.get("run_id"),
        "created_at": data.get("created_at"),
        "setups": data.get("setups", []),
        "entry_count": data.get("entry_count"),
        "judged": bool(data.get("judged", False)),
    }


def select_matrix_run(
    runs: list[dict[str, Any]], run_id: str | None = None
) -> dict[str, Any] | None:
    """The named run out of ``runs`` (newest first), or the newest one.

    Returns ``None`` when nothing matches — an unknown ``run_id`` and an empty
    ``evals/runs/`` are the same situation for the caller: there is no run to
    score, so the Scoreboard renders its empty state rather than erroring.
    """
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


def matrix_questions(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Every golden question in this run, with each setup's composite score.

    ``build_scoreboard`` collapses a run down to per-setup averages; this
    keeps the per-question rows those averages are built from, so a reader can
    see exactly which golden questions were judged and how each setup scored
    on each one, not just the aggregate.

    Each cell also carries the **answer that was actually graded**, plus the
    model and cost facts for it. A score with no way to read the text behind it
    cannot be sanity-checked — "hybrid scored 0.71 here" only means something
    once you can see what it said — so the run's own answer text travels with
    its score rather than staying locked in the committed JSON.
    """
    by_question: dict[str, dict[str, Any]] = {}
    for setup, run in (data.get("runs") or {}).items():
        title = _title_for(setup)
        config = run.get("config") or data.get("config") or {}
        for cell in run.get("entries") or []:
            question_id = str(cell.get("id") or "")
            row = by_question.get(question_id)
            if row is None:
                row = {
                    "id": question_id,
                    "question": str(cell.get("question") or ""),
                    "domain": cell.get("domain"),
                    "question_type": cell.get("question_type"),
                    "setups": {},
                }
                by_question[question_id] = row
            scores = cell.get("scores") or {}
            composite = scores.get("composite")
            row["setups"][setup] = {
                "key": setup,
                "title": title,
                "composite": composite if isinstance(composite, (int, float)) else None,
                "judged": isinstance(composite, (int, float)),
                "error": cell.get("error"),
                "answer": str(cell.get("answer") or ""),
                "model": config.get("answer_model"),
                "elapsed_seconds": cell.get("elapsed_seconds"),
                "token_estimate": cell.get("token_estimate"),
                "chunk_count": len(cell.get("retrieved_chunk_ids") or []),
            }

    questions = list(by_question.values())
    questions.sort(key=lambda row: row["id"])
    for row in questions:
        row["setups"] = list(row["setups"].values())
    return questions

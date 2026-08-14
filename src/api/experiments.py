"""Read committed eval snapshots (``evals/runs/``) for the Experiments tab.

The workbench's other tabs read live state; this one reads the *committed* record
of what retrieval configurations have been measured — the ablation sweeps and the
end-to-end golden runs a reviewer can also open as JSON in the repo. It only reads,
and returns lightweight summaries (per-entry detail is dropped) so the tab renders
the comparison tables without shipping every retrieved-id list to the browser.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

#: ``evals/runs/`` at the repo root, independent of the server's working directory.
DEFAULT_RUNS_DIR = Path(__file__).resolve().parents[2] / "evals" / "runs"


def load_experiments(runs_dir: Path | None = None) -> dict[str, Any]:
    """Committed ablation and golden runs, newest first, as JSON-ready summaries."""
    directory = runs_dir or DEFAULT_RUNS_DIR
    ablations: list[dict[str, Any]] = []
    golden_runs: list[dict[str, Any]] = []
    matrix_runs: list[dict[str, Any]] = []
    critique_runs: list[dict[str, Any]] = []
    if directory.exists():
        for path in sorted(directory.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                # A malformed snapshot must not take the whole tab down.
                continue
            if not isinstance(data, dict):
                continue
            if data.get("kind") == "retrieval-ablation":
                ablations.append(_ablation_summary(data))
            elif data.get("kind") == "matrix":
                matrix_runs.append(_matrix_summary(data))
            # Before the duck-typed golden branch below, which matches on the
            # presence of ``setup``/``summary`` and would happily claim any run
            # kind that happens to carry both.
            elif data.get("kind") == "critique-eval":
                critique_runs.append(critique_summary(data))
            elif "setup" in data and "summary" in data:
                golden_runs.append(_golden_summary(data))
    ablations.sort(key=lambda run: run.get("created_at") or "", reverse=True)
    golden_runs.sort(key=lambda run: run.get("created_at") or "", reverse=True)
    matrix_runs.sort(key=lambda run: run.get("created_at") or "", reverse=True)
    critique_runs.sort(key=lambda run: run.get("created_at") or "", reverse=True)
    return {
        "ablations": ablations,
        "golden_runs": golden_runs,
        "matrix_runs": matrix_runs,
        "critique_runs": critique_runs,
    }


def _ablation_summary(data: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": data.get("run_id"),
        "created_at": data.get("created_at"),
        "entries": data.get("entries"),
        "metrics": data.get("metrics", []),
        "baseline": data.get("baseline"),
        # Which config set was measured. Absent on runs committed before the
        # extended sweep existed — those are all default sweeps, but the tab
        # says nothing rather than backfilling a label onto them.
        "sweep": data.get("sweep"),
        "cells": [
            {
                "label": cell.get("label"),
                "config": cell.get("config", {}),
                "averages": cell.get("averages", {}),
                "by_domain": cell.get("by_domain", {}),
            }
            for cell in data.get("cells", [])
            if isinstance(cell, dict)
        ],
        "deltas": data.get("deltas", []),
    }


def _matrix_summary(data: dict[str, Any]) -> dict[str, Any]:
    """A matrix run without the per-setup entry detail (the pivot is enough)."""
    return {
        "run_id": data.get("run_id"),
        "created_at": data.get("created_at"),
        "setups": data.get("setups", []),
        "entry_count": data.get("entry_count"),
        "judged": data.get("judged", False),
        "reference_scored": data.get("reference_scored", False),
        "question_types": data.get("question_types", {}),
        "comparison": data.get("comparison", {}),
    }


def critique_summary(data: dict[str, Any]) -> dict[str, Any]:
    """A held-out critique run without the per-finding detail.

    ``matches`` and ``findings`` are the bulk of one of these files and the tab
    only needs them once a row is opened, so they are dropped here and served by
    :func:`select_critique_run` from a separate request keyed by run id. This
    endpoint re-reads and re-parses every committed run on every call, and the
    detail of a run nobody expanded is pure weight on that path.
    """
    return {
        "run_id": data.get("run_id"),
        "created_at": data.get("created_at"),
        "held_out_video_id": data.get("held_out_video_id"),
        "held_out_title": data.get("held_out_title"),
        "artifact_url": data.get("artifact_url"),
        "artifact_kind": data.get("artifact_kind"),
        "criteria_total": data.get("criteria_total"),
        "criteria_applicable": data.get("criteria_applicable"),
        "metrics": data.get("metrics", []),
        "baseline": data.get("baseline"),
        "held_out_leaks": data.get("held_out_leaks"),
        "exclusion_version": data.get("exclusion_version"),
        "match_repeats": data.get("match_repeats"),
        "criteria_groups": data.get("criteria_groups"),
        # Which rule produced the scores below, and which arms it refused to
        # grade. Both belong on the summary rather than behind the expand: an
        # arm the gate could not certify must not render as a measured row, and
        # the tab cannot tell without being told. See src/evals/critique.py.
        "grounding_gate": data.get("grounding_gate"),
        "ungraded_cells": data.get("ungraded_cells", []),
        "config": data.get("config", {}),
        "cells": [
            {
                key: cell.get(key)
                for key in (
                    "setup",
                    "scores",
                    # The gate's verdict on this cell, and the figures the old
                    # rule produced. An arm the gate cannot grade scores None on
                    # criteria_recall and evidence_precision, and a panel that
                    # received only that would render "—" — indistinguishable
                    # from a blank, which a reader fills in with the number they
                    # last saw. The reason and the ungated pair travel with it so
                    # the row can say "not measured, and here is why, and here is
                    # what it read before".
                    "gradable",
                    "ungradable_reason",
                    "grounding_gate",
                    "criteria_recall_ungated",
                    "evidence_precision_ungated",
                    # How much per-finding retrieval there was to gate on. One
                    # distinct set across many findings is the shape the gate
                    # cannot tell from a shared pool — reported, never gated.
                    "findings_with_provenance",
                    "provenance_distinct_sets",
                    "citations_off_retrieval",
                    # The spread belongs on the list, not behind an expand: a
                    # reader comparing two runs has to be able to see that a
                    # recall difference is inside the scorer's own noise
                    # without opening either of them.
                    "score_spread",
                    "match_repeats",
                    "criteria_recall_all",
                    "criteria_recall_grouped",
                    "criteria_groups",
                    "criteria_applicable",
                    "criteria_matched",
                    "criteria_matched_ungrounded",
                    "findings_total",
                    "findings_grounded",
                    "findings_sharing_evidence",
                    "citations_total",
                    "citations_resolved",
                    "conflicts_in_context",
                    "conflicts_named",
                    "self_declared_contested",
                    "held_out_leaks",
                    "elapsed_seconds",
                    "token_estimate",
                    "error",
                )
            }
            | {
                "retrieved_video_ids": cell.get("retrieved_video_ids", []),
                # Small enough to ride along on the list, and it has to: a
                # recall of 0.000 that consensus produced out of pairings some
                # repeats did make is a different result from one nothing
                # reached, and a reader who never expands the row would read
                # them as the same number.
                "match_ballots": _match_ballots(cell),
            }
            for cell in data.get("cells", [])
            if isinstance(cell, dict)
        ],
    }


def _match_ballots(cell: dict[str, Any]) -> list[dict[str, Any]]:
    """Per-repeat matcher votes for every criterion any repeat paired.

    A join, not a computation. ``match_runs`` holds one ballot per matcher
    repeat and ``matches`` holds what :func:`src.evals.critique.consensus` made
    of those ballots; this puts the two beside each other so the tab can show a
    criterion two of five repeats paired and the majority vote then discarded.
    The consensus verdict is *read* from the run — nothing here re-decides it.

    Criteria no repeat ever paired are left out: their ballot is a row of blanks
    and the missed column already accounts for them.
    """
    runs = [run for run in cell.get("match_runs") or [] if isinstance(run, dict)]
    if not runs:
        return []
    matches = {m.get("id"): m for m in cell.get("matches") or [] if isinstance(m, dict)}
    finding_text = {
        f.get("id"): f.get("criterion") for f in cell.get("findings") or [] if isinstance(f, dict)
    }
    ballots: list[dict[str, Any]] = []
    # dict.fromkeys keeps first-seen order, which is criterion order.
    for criterion_id in dict.fromkeys(key for run in runs for key in run):
        draws = [run.get(criterion_id) for run in runs]
        if not any(draws):
            continue
        tally: dict[Any, int] = {}
        for draw in draws:
            tally[draw] = tally.get(draw, 0) + 1
        match = matches.get(criterion_id) or {}
        ballots.append(
            {
                "criterion_id": criterion_id,
                "criterion": match.get("criterion"),
                "applies_to": match.get("applies_to", []),
                "draws": draws,
                # Sorted by size, ties broken on id, so the order never depends
                # on the order the matcher repeats happened to run in.
                "votes": [
                    {
                        "finding_id": finding_id,
                        "finding_criterion": finding_text.get(finding_id),
                        "count": count,
                    }
                    for finding_id, count in sorted(
                        tally.items(), key=lambda item: (-item[1], item[0] or "")
                    )
                ],
                "consensus_finding_id": match.get("finding_id"),
                "consensus_finding_criterion": match.get("finding_criterion"),
                "agreement": match.get("agreement"),
            }
        )
    return ballots


def select_critique_run(run_id: str, runs_dir: Path | None = None) -> dict[str, Any] | None:
    """The full committed critique run with this id, detail included.

    ``None`` when there is no such run, which the caller turns into a 404 — an
    unknown id and an empty ``evals/runs/`` are the same situation for a tab
    that is trying to expand a row.
    """
    directory = runs_dir or DEFAULT_RUNS_DIR
    path = directory / f"{run_id}.json"
    # ``run_id`` reaches this from a URL path segment, so it is checked against
    # the directory listing rather than trusted to be a bare file name.
    if not path.exists() or path.parent.resolve() != directory.resolve():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(data, dict) or data.get("kind") != "critique-eval":
        return None
    return data


def _golden_summary(data: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": data.get("run_id"),
        "created_at": data.get("created_at"),
        "setup": data.get("setup"),
        "config": data.get("config", {}),
        "summary": data.get("summary", {}),
    }

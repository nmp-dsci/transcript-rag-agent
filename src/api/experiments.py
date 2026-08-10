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
        "config": data.get("config", {}),
        "cells": [
            {
                key: cell.get(key)
                for key in (
                    "setup",
                    "scores",
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
                    "contested_findings",
                    "held_out_leaks",
                    "elapsed_seconds",
                    "token_estimate",
                    "error",
                )
            }
            | {"retrieved_video_ids": cell.get("retrieved_video_ids", [])}
            for cell in data.get("cells", [])
            if isinstance(cell, dict)
        ],
    }


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

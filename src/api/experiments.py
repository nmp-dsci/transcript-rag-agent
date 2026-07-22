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
            elif "setup" in data and "summary" in data:
                golden_runs.append(_golden_summary(data))
    ablations.sort(key=lambda run: run.get("created_at") or "", reverse=True)
    golden_runs.sort(key=lambda run: run.get("created_at") or "", reverse=True)
    return {"ablations": ablations, "golden_runs": golden_runs}


def _ablation_summary(data: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": data.get("run_id"),
        "created_at": data.get("created_at"),
        "entries": data.get("entries"),
        "metrics": data.get("metrics", []),
        "baseline": data.get("baseline"),
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


def _golden_summary(data: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": data.get("run_id"),
        "created_at": data.get("created_at"),
        "setup": data.get("setup"),
        "config": data.get("config", {}),
        "summary": data.get("summary", {}),
    }

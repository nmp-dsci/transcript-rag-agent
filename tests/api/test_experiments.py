from __future__ import annotations

import json
from pathlib import Path

from src.api.experiments import load_experiments


def _write(directory: Path, name: str, payload: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text(json.dumps(payload), encoding="utf-8")


def _ablation(run_id: str, created_at: str) -> dict:
    return {
        "run_id": run_id,
        "created_at": created_at,
        "kind": "retrieval-ablation",
        "entries": 9,
        "metrics": ["context_recall", "recall@3"],
        "baseline": "semantic",
        "cells": [
            {
                "label": "semantic",
                "config": {"retrieval_mode": "semantic", "rerank": False},
                "averages": {"context_recall": 0.6, "recall@3": 0.26},
                "by_domain": {"property": {"context_recall": 0.7}},
                "entries": [{"id": "g1", "retrieved_chunk_ids": ["chunk:v:0"]}],
            }
        ],
        "deltas": [{"label": "hybrid", "vs_baseline": {"recall@3": 0.13}}],
    }


def _golden(run_id: str, created_at: str) -> dict:
    return {
        "run_id": run_id,
        "created_at": created_at,
        "setup": "rag_llm",
        "config": {"retrieval_mode": "hybrid", "rerank_enabled": True},
        "entries": [{"id": "g1"}],
        "summary": {"scored": 9, "averages": {"context_recall": 0.6}},
    }


def test_classifies_ablation_and_golden_runs(tmp_path: Path) -> None:
    _write(tmp_path, "ablation-1.json", _ablation("ablation-1", "2026-07-22T09:00:00+00:00"))
    _write(tmp_path, "eval-1.json", _golden("eval-1", "2026-07-22T10:00:00+00:00"))

    result = load_experiments(tmp_path)

    assert [run["run_id"] for run in result["ablations"]] == ["ablation-1"]
    assert [run["run_id"] for run in result["golden_runs"]] == ["eval-1"]


def test_ablation_summary_drops_heavy_per_entry_detail(tmp_path: Path) -> None:
    _write(tmp_path, "ablation-1.json", _ablation("ablation-1", "2026-07-22T09:00:00+00:00"))

    cell = load_experiments(tmp_path)["ablations"][0]["cells"][0]

    assert cell["averages"] == {"context_recall": 0.6, "recall@3": 0.26}
    assert cell["by_domain"] == {"property": {"context_recall": 0.7}}
    # Per-entry retrieved ids are not shipped to the browser.
    assert "entries" not in cell


def test_runs_are_newest_first(tmp_path: Path) -> None:
    _write(tmp_path, "a-old.json", _ablation("ablation-old", "2026-07-20T00:00:00+00:00"))
    _write(tmp_path, "a-new.json", _ablation("ablation-new", "2026-07-22T00:00:00+00:00"))

    ids = [run["run_id"] for run in load_experiments(tmp_path)["ablations"]]

    assert ids == ["ablation-new", "ablation-old"]


def test_malformed_snapshot_is_skipped_not_fatal(tmp_path: Path) -> None:
    _write(tmp_path, "ablation-1.json", _ablation("ablation-1", "2026-07-22T09:00:00+00:00"))
    (tmp_path / "broken.json").write_text("{not valid json", encoding="utf-8")

    result = load_experiments(tmp_path)

    assert [run["run_id"] for run in result["ablations"]] == ["ablation-1"]


def test_missing_directory_returns_empty(tmp_path: Path) -> None:
    result = load_experiments(tmp_path / "does-not-exist")

    assert result == {"ablations": [], "golden_runs": []}

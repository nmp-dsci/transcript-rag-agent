from __future__ import annotations

import json
from pathlib import Path

from src.api.experiments import load_experiments, select_critique_run


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

    assert result == {
        "ablations": [],
        "golden_runs": [],
        "matrix_runs": [],
        "critique_runs": [],
    }


def _critique(run_id: str, created_at: str) -> dict:
    return {
        "run_id": run_id,
        "created_at": created_at,
        "kind": "critique-eval",
        "held_out_video_id": "HELD",
        "held_out_title": "held out",
        "artifact_url": "https://example.test/",
        "artifact_kind": "portfolio",
        "criteria_total": 24,
        "criteria_applicable": 18,
        "metrics": ["criteria_recall", "evidence_precision"],
        "baseline": "rag_llm_filtered",
        "held_out_leaks": 0,
        "exclusion_version": "v1",
        "config": {"answer_model": "m"},
        # ``setup`` and ``summary`` together are what the golden branch
        # duck-types on, so a critique cell carrying both must still be
        # classified by its ``kind``.
        "summary": {"note": "not a golden run"},
        "setup": "rag_llm_filtered",
        "cells": [
            {
                "setup": "rag_llm_filtered",
                "scores": {"criteria_recall": 0.28, "evidence_precision": 1.0},
                "criteria_matched": 5,
                "findings_total": 9,
                "held_out_leaks": 0,
                "retrieved_video_ids": ["vidA"],
                "matches": [{"id": "c01", "matched": False}],
                "findings": [{"id": "f01", "criterion": "x"}],
                "answer": "a very long answer" * 500,
                "trace": [{"phase": "retrieve"}],
            }
        ],
    }


def test_a_critique_run_is_classified_by_kind_not_by_duck_typing(tmp_path: Path) -> None:
    """The golden branch matches on ``setup`` + ``summary`` being present.

    A critique run that happens to carry both must not be filed as a golden run,
    which is why its ``kind`` branch sits above that fallback.
    """
    _write(tmp_path, "critique-1.json", _critique("critique-1", "2026-08-10T00:00:00Z"))

    result = load_experiments(tmp_path)

    assert [run["run_id"] for run in result["critique_runs"]] == ["critique-1"]
    assert result["golden_runs"] == []


def test_the_critique_list_drops_the_per_finding_detail(tmp_path: Path) -> None:
    """``/api/experiments`` re-parses every run on every request.

    The matched-criteria detail is the bulk of a critique file and is only
    needed once a row is expanded, so it is served separately and must not ride
    along on the list.
    """
    _write(tmp_path, "critique-1.json", _critique("critique-1", "2026-08-10T00:00:00Z"))

    cell = load_experiments(tmp_path)["critique_runs"][0]["cells"][0]

    assert cell["scores"]["criteria_recall"] == 0.28
    assert "matches" not in cell
    assert "findings" not in cell
    assert "answer" not in cell


def test_selecting_a_critique_run_returns_the_full_detail(tmp_path: Path) -> None:
    _write(tmp_path, "critique-1.json", _critique("critique-1", "2026-08-10T00:00:00Z"))

    found = select_critique_run("critique-1", tmp_path)

    assert found is not None
    assert found["cells"][0]["matches"] == [{"id": "c01", "matched": False}]


def test_selecting_an_unknown_or_wrong_kind_run_returns_none(tmp_path: Path) -> None:
    """An id that names a real file of another kind is still not a critique run."""
    _write(tmp_path, "ablation-1.json", _ablation("ablation-1", "2026-08-10T00:00:00Z"))

    assert select_critique_run("nope", tmp_path) is None
    assert select_critique_run("ablation-1", tmp_path) is None


def test_a_critique_run_id_cannot_escape_the_runs_directory(tmp_path: Path) -> None:
    """The id arrives from a URL path segment and is never trusted as a path."""
    (tmp_path.parent / "secret.json").write_text('{"kind": "critique-eval"}', encoding="utf-8")

    assert select_critique_run("../secret", tmp_path) is None

"""Back-filling the eval cell cache from the legacy matrix checkpoint."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

from src.config import Settings
from src.evals.golden import GoldenEntry
from src.evals.matrix import config_snapshot
from src.evals.matrix_cache import cell_fingerprint, load_cell

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "migrate_matrix_checkpoint.py"
_spec = importlib.util.spec_from_file_location("migrate_matrix_checkpoint", SCRIPT)
assert _spec and _spec.loader
migrate_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(migrate_module)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        superdata_api_key="",
        deepseek_api_key="test-key",
        deepseek_model="deepseek-v4",
        deepseek_base_url=None,
        chroma_path=tmp_path / "chroma",
        mlflow_tracking_uri="",
        mlflow_experiment_name="test",
        log_transcript_artifacts=False,
    )


@pytest.fixture
def entry() -> GoldenEntry:
    return GoldenEntry(
        id="g001",
        question="What changed?",
        reference_answer="Quite a lot changed, in ways worth writing down.",
        expected_video_ids=["v1"],
        expected_chunk_ids=["chunk:v1:0"],
        domain="property",
        notes="fixture",
    )


def result(entry_id: str = "g001", error: str | None = None) -> dict[str, Any]:
    return {
        "id": entry_id,
        "question": "What changed?",
        "domain": "property",
        "question_type": "local",
        "answer": "An answer.",
        "error": error,
        "scores": {"faithfulness": 0.9, "composite": 0.8},
        "retrieved_chunk_ids": ["chunk:v1:0"],
        "elapsed_seconds": 1.0,
        "token_estimate": 10,
    }


def row(settings: Settings, *, setup: str = "rag_llm", **overrides: Any) -> dict[str, Any]:
    return {
        "setup": setup,
        "config": overrides.pop("config", config_snapshot(settings)),
        "entry": overrides.pop("entry", result()),
    }


def fingerprint_for(settings: Settings, entry: GoldenEntry, setup: str = "rag_llm") -> str:
    return cell_fingerprint(
        setup,
        entry,
        settings,
        judge_model=settings.judge_model or settings.deepseek_model,
        judge_samples=settings.judge_samples,
        ragas_version=migrate_module.ragas_version(),
        reference_scored=True,
    )


def test_read_checkpoint_skips_blank_and_malformed_lines(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"setup": "rag_llm", "config": {}, "entry": {"id": "g001"}}),
                "",
                "{not json",
                json.dumps({"missing": "required keys"}),
            ]
        ),
        encoding="utf-8",
    )
    rows = migrate_module.read_checkpoint(path)
    assert [r["setup"] for r in rows] == ["rag_llm"]


def test_read_checkpoint_of_a_missing_file_is_empty(tmp_path: Path) -> None:
    assert migrate_module.read_checkpoint(tmp_path / "nope.jsonl") == []


def test_migrate_writes_a_cell_the_matrix_driver_then_finds(
    tmp_path: Path, settings: Settings, entry: GoldenEntry
) -> None:
    cache_dir = tmp_path / "cache"
    counts = migrate_module.migrate([row(settings)], [entry], settings, cache_dir=cache_dir)
    assert counts["migrated"] == 1
    # The whole point: the fingerprint written here is the one the driver looks up.
    cached = load_cell(fingerprint_for(settings, entry), cache_dir)
    assert cached is not None
    assert cached["scores"]["composite"] == 0.8


def test_migrate_refuses_rows_recorded_under_a_different_config(
    tmp_path: Path, settings: Settings, entry: GoldenEntry
) -> None:
    stale = {**config_snapshot(settings), "answer_model": "some-older-model"}
    counts = migrate_module.migrate(
        [row(settings, config=stale)], [entry], settings, cache_dir=tmp_path / "cache"
    )
    # Writing this would claim the current config produced numbers it did not.
    assert counts == {
        "migrated": 0,
        "already_cached": 0,
        "config_mismatch": 1,
        "unknown_entry": 0,
        "errored": 0,
    }
    assert load_cell(fingerprint_for(settings, entry), tmp_path / "cache") is None


def test_migrate_tolerates_config_keys_the_old_format_never_wrote(
    tmp_path: Path, settings: Settings, entry: GoldenEntry
) -> None:
    """Config has gained fields since the checkpoint format was retired.

    A key the old rows never recorded is not evidence they were produced under
    a different configuration, so its absence must not block the migration.
    """
    older_shape = {
        key: value for key, value in config_snapshot(settings).items() if key != "ragas_version"
    }
    counts = migrate_module.migrate(
        [row(settings, config=older_shape)], [entry], settings, cache_dir=tmp_path / "cache"
    )
    assert counts["migrated"] == 1


def test_migrate_skips_entries_no_longer_in_the_golden_set(
    tmp_path: Path, settings: Settings, entry: GoldenEntry
) -> None:
    retired = row(settings, entry=result("g999"))
    counts = migrate_module.migrate([retired], [entry], settings, cache_dir=tmp_path / "cache")
    assert counts["unknown_entry"] == 1
    assert counts["migrated"] == 0


def test_migrate_never_caches_a_failed_cell(
    tmp_path: Path, settings: Settings, entry: GoldenEntry
) -> None:
    failed = row(settings, entry=result(error="deepseek 402"))
    counts = migrate_module.migrate([failed], [entry], settings, cache_dir=tmp_path / "cache")
    # A failure must be retried on the next run, not pinned as a result.
    assert counts["errored"] == 1
    assert load_cell(fingerprint_for(settings, entry), tmp_path / "cache") is None


def test_migrate_is_idempotent(tmp_path: Path, settings: Settings, entry: GoldenEntry) -> None:
    cache_dir = tmp_path / "cache"
    rows = [row(settings)]
    assert migrate_module.migrate(rows, [entry], settings, cache_dir=cache_dir)["migrated"] == 1
    second = migrate_module.migrate(rows, [entry], settings, cache_dir=cache_dir)
    assert second["migrated"] == 0
    assert second["already_cached"] == 1


def test_dry_run_reports_without_writing(
    tmp_path: Path, settings: Settings, entry: GoldenEntry
) -> None:
    cache_dir = tmp_path / "cache"
    counts = migrate_module.migrate(
        [row(settings)], [entry], settings, cache_dir=cache_dir, dry_run=True
    )
    assert counts["migrated"] == 1
    assert load_cell(fingerprint_for(settings, entry), cache_dir) is None

"""Golden-set regression runs: scoring, summarising, and diffing."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.config import Settings
from src.evals.golden import GoldenEntry
from src.evals.regression import (
    EntryResult,
    diff_runs,
    list_runs,
    load_run,
    run_golden_eval,
    save_run,
    summarize,
)


@pytest.fixture
def entries() -> list[GoldenEntry]:
    return [
        GoldenEntry(
            id="g001",
            question="What changed for investors?",
            reference_answer="Negative gearing was grandfathered.",
            expected_video_ids=["v1"],
            expected_chunk_ids=["chunk:v1:0", "chunk:v1:1"],
            domain="property",
        ),
        GoldenEntry(
            id="g002",
            question="How do agents use tools?",
            reference_answer="They call retrieval repeatedly.",
            expected_video_ids=["v2"],
            expected_chunk_ids=["chunk:v2:3"],
            domain="ai-coding",
        ),
    ]


class FakeResult:
    def __init__(self, answer="An answer.", chunk_ids=None, error=None):
        self.answer = answer
        self.error = error
        self.retrieved_chunk_ids = chunk_ids or []
        self.contexts = ["ctx"]
        self.elapsed_seconds = 2.0
        self.token_estimate = 1000
        self.model = "deepseek-v4-flash"


class FakeRunner:
    def __init__(self, by_question=None, raises=False):
        self.by_question = by_question or {}
        self.raises = raises
        self.calls: list[tuple] = []

    def run(self, setup, question, top_k=None, scope=None):
        self.calls.append((setup, question, top_k))
        if self.raises:
            raise RuntimeError("retrieval exploded")
        return self.by_question.get(question, FakeResult())


class FakeJudge:
    def score(self, question, answer, contexts, answer_model=None):
        return {
            "scores": {
                "faithfulness": 0.9,
                "answer_relevancy": 0.8,
                "context_precision": 0.7,
            },
            "composite": 0.8,
        }


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        superdata_api_key="k",
        deepseek_api_key="k",
        deepseek_model="deepseek-v4-flash",
        deepseek_base_url=None,
        chroma_path=tmp_path / "chroma",
        mlflow_tracking_uri="file:mlruns",
        mlflow_experiment_name="test",
        log_transcript_artifacts=False,
    )


def test_run_scores_every_entry_with_recall_and_judge(entries, settings):
    runner = FakeRunner(
        {
            "What changed for investors?": FakeResult(
                chunk_ids=["chunk:v1:0", "chunk:v1:1"]
            ),
            "How do agents use tools?": FakeResult(chunk_ids=["chunk:v9:0"]),
        }
    )
    run = run_golden_eval(
        runner, settings, setup="rag_llm", judge=FakeJudge(), entries=entries
    )
    assert run["summary"]["scored"] == 2
    first, second = run["entries"]
    # All expected chunks retrieved, versus none.
    assert first["scores"]["context_recall"] == 1.0
    assert second["scores"]["context_recall"] == 0.0
    assert first["scores"]["faithfulness"] == 0.9
    assert run["setup"] == "rag_llm"


def test_run_records_the_configuration_under_test(entries, settings):
    run = run_golden_eval(FakeRunner(), settings, entries=entries)
    assert run["config"]["answer_model"] == "deepseek-v4-flash"
    assert run["config"]["retrieval_mode"] == "semantic"
    assert "embedding_model" in run["config"]


def test_failed_entry_is_recorded_but_excluded_from_averages(entries, settings):
    run = run_golden_eval(FakeRunner(raises=True), settings, entries=entries)
    assert run["summary"]["failed"] == 2
    assert run["summary"]["scored"] == 0
    assert run["entries"][0]["error"] == "retrieval exploded"
    # A crash is missing data, not a zero score.
    assert run["summary"]["averages"] == {}


def test_summary_averages_only_the_entries_that_scored():
    results = [
        EntryResult(id="a", question="q", domain="d", scores={"faithfulness": 1.0}),
        EntryResult(id="b", question="q", domain="d", scores={"faithfulness": 0.5}),
        EntryResult(id="c", question="q", domain="d", error="boom"),
    ]
    summary = summarize(results)
    assert summary["averages"]["faithfulness"] == 0.75
    assert summary["scored"] == 2
    assert summary["failed"] == 1


def test_saved_runs_round_trip_and_list_oldest_first(tmp_path, settings, entries):
    for minute in (1, 2):
        run = run_golden_eval(
            FakeRunner(),
            settings,
            entries=entries,
            now=datetime(2026, 7, 21, 10, minute, tzinfo=timezone.utc),
        )
        save_run(run, tmp_path)
    runs = list_runs(tmp_path)
    assert len(runs) == 2
    assert load_run(runs[0])["run_id"] < load_run(runs[1])["run_id"]


def test_listing_runs_before_any_exist_is_empty(tmp_path):
    assert list_runs(tmp_path / "nothing-here") == []


def make_run(run_id, faithfulness, entry_score=None, metric="faithfulness"):
    score = faithfulness if entry_score is None else entry_score
    return {
        "run_id": run_id,
        "summary": {"averages": {metric: faithfulness}},
        "entries": [{"id": "g001", "question": "q", "scores": {metric: score}}],
    }


def test_diff_flags_a_drop_as_a_regression():
    diff = diff_runs(make_run("a", 0.90), make_run("b", 0.70))
    assert diff["regressed"] == ["faithfulness"]
    assert diff["metrics"][0]["delta"] == -0.2


def test_diff_flags_a_rise_as_an_improvement():
    diff = diff_runs(make_run("a", 0.60), make_run("b", 0.85))
    assert diff["improved"] == ["faithfulness"]


def test_movement_below_threshold_is_noise_not_a_regression():
    """One judged sample is not precise enough for every decimal to matter.

    ``faithfulness`` is LLM-judged, so the default noise threshold applies.
    """
    diff = diff_runs(
        make_run("a", 0.80, metric="faithfulness"),
        make_run("b", 0.79, metric="faithfulness"),
    )
    assert diff["regressed"] == []
    assert diff["metrics"][0]["direction"] == "unchanged"


def test_small_deterministic_recall_drop_is_still_a_regression():
    """context_recall is exact and id-based (see golden.py), not judged.

    A movement of 0.01 would be noise under the old uniform 0.02 threshold,
    but there is no sampling noise in a deterministic metric, so it must be
    flagged as a real regression.
    """
    diff = diff_runs(
        make_run("a", 0.80, metric="context_recall"),
        make_run("b", 0.79, metric="context_recall"),
    )
    assert "context_recall" in diff["regressed"]
    moves = {m["metric"]: m for m in diff["metrics"]}
    assert moves["context_recall"]["direction"] == "worse"


def test_diff_reports_which_questions_moved():
    diff = diff_runs(
        make_run("a", 0.80, entry_score=0.9), make_run("b", 0.80, entry_score=0.4)
    )
    assert diff["entries"][0]["id"] == "g001"
    assert diff["entries"][0]["changes"]["faithfulness"]["delta"] == -0.5


def test_questions_that_did_not_move_are_not_listed():
    diff = diff_runs(make_run("a", 0.80), make_run("b", 0.80))
    assert diff["entries"] == []


def test_metric_missing_from_one_run_is_skipped_not_guessed():
    before = {"run_id": "a", "summary": {"averages": {"faithfulness": 0.8}}, "entries": []}
    after = {
        "run_id": "b",
        "summary": {"averages": {"faithfulness": 0.8, "context_recall": 0.5}},
        "entries": [],
    }
    diff = diff_runs(before, after)
    assert [m["metric"] for m in diff["metrics"]] == ["faithfulness"]

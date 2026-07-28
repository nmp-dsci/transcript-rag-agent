"""Committed matrix runs adapted into the shape the Scoreboard aggregates."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.api import matrix_runs
from src.api.matrix_runs import (
    describe_matrix_run,
    load_matrix_runs,
    matrix_entries,
    matrix_questions,
    select_matrix_run,
)
from src.api.scoreboard import build_scoreboard
from tests.api.matrix_fixtures import matrix_cell as cell
from tests.api.matrix_fixtures import matrix_run


def write_run(runs_dir: Path, data: dict) -> None:
    runs_dir.mkdir(parents=True, exist_ok=True)
    (runs_dir / f"{data['run_id']}.json").write_text(json.dumps(data), encoding="utf-8")


def test_matrix_entries_pivots_setup_major_into_question_major() -> None:
    entries = matrix_entries(matrix_run())
    assert [entry.id for entry in entries] == ["g001", "g002"]
    # Every setup that answered a question lands on that question's entry —
    # this is what makes a win-rate contest possible.
    assert {answer.key for answer in entries[0].answers} == {"rag_llm", "rag_agent"}
    assert entries[0].question == "question g001?"


def test_adapted_answers_carry_config_provenance() -> None:
    answer = matrix_entries(matrix_run())[0].answers[0]
    assert answer.model == "deepseek-v4"
    assert answer.embedding_model == "fake-embeddings"
    assert answer.top_k == 10
    assert answer.retrieval_mode == "semantic"
    assert answer.command == "eval-matrix"
    assert answer.evaluation is not None
    assert answer.evaluation["judge_model"] == "deepseek-v4"
    # composite is lifted out of scores rather than double-counted as a metric.
    assert "composite" not in answer.evaluation["scores"]
    assert answer.evaluation["composite"] == 0.5


def test_unjudged_cells_have_no_evaluation() -> None:
    run = matrix_run(setups={"rag_llm": [cell("g001", composite=None)]}, judged=False)
    answer = matrix_entries(run)[0].answers[0]
    # A --no-judge run has retrieval metrics but no RAGAS verdict: that must
    # read as "never judged", not as a zero score.
    assert answer.evaluation is None


def test_unknown_setup_key_still_gets_a_row() -> None:
    run = matrix_run(setups={"retired_engine": [cell("g001")]})
    answer = matrix_entries(run)[0].answers[0]
    assert answer.key == "retired_engine"
    assert answer.title == "retired_engine"


def test_scoreboard_over_a_matrix_run_ranks_and_scores_contests() -> None:
    board = build_scoreboard(matrix_entries(matrix_run()))
    rows = {row["key"]: row for row in board["setups"]}
    assert board["entries_total"] == 2
    assert board["entries_judged"] == 2
    assert rows["rag_agent"]["avg_composite"] == 0.9
    assert rows["rag_llm"]["avg_composite"] == 0.5
    # Both setups answered both questions under the same judge, so every
    # question is a contest and the higher composite takes all of them.
    assert rows["rag_agent"]["win_rate"] == 1.0
    assert rows["rag_llm"]["win_rate"] == 0.0
    assert board["setups"][0]["key"] == "rag_agent"


def test_load_matrix_runs_is_newest_first_and_ignores_other_kinds(tmp_path: Path) -> None:
    write_run(tmp_path, matrix_run("matrix-older", created_at="2026-07-01T00:00:00+00:00"))
    write_run(tmp_path, matrix_run("matrix-newer", created_at="2026-07-27T00:00:00+00:00"))
    (tmp_path / "ablation-x.json").write_text(
        json.dumps({"kind": "retrieval-ablation", "run_id": "ablation-x"}), encoding="utf-8"
    )
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")

    runs = load_matrix_runs(tmp_path)
    assert [run["run_id"] for run in runs] == ["matrix-newer", "matrix-older"]
    # The same pass feeds the picker and the selected run — no second read.
    descriptor = describe_matrix_run(runs[0])
    assert descriptor["setups"] == ["rag_llm", "rag_agent"]
    assert descriptor["judged"] is True
    assert set(descriptor) == {"run_id", "created_at", "setups", "entry_count", "judged"}


def test_load_matrix_runs_reparses_only_when_the_directory_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Repeated reads of an unchanged directory are served from the memo.

    A committed run is a large document, and the Scoreboard re-reads the
    directory on every group-by toggle, judge filter and run-picker change. A
    newly written run must still appear without a restart, so the memo is keyed
    on the files' own stat fingerprint rather than on a process-lifetime flag.
    """
    reads: list[Path | None] = []
    real_iter = matrix_runs._iter_matrix_files

    def counting_iter(runs_dir: Path | None = None):
        reads.append(runs_dir)
        return real_iter(runs_dir)

    monkeypatch.setattr(matrix_runs, "_iter_matrix_files", counting_iter)

    write_run(tmp_path, matrix_run("matrix-older", created_at="2026-07-01T00:00:00+00:00"))
    assert [run["run_id"] for run in load_matrix_runs(tmp_path)] == ["matrix-older"]
    assert [run["run_id"] for run in load_matrix_runs(tmp_path)] == ["matrix-older"]
    assert len(reads) == 1

    write_run(tmp_path, matrix_run("matrix-newer", created_at="2026-07-27T00:00:00+00:00"))
    assert [run["run_id"] for run in load_matrix_runs(tmp_path)] == [
        "matrix-newer",
        "matrix-older",
    ]
    assert len(reads) == 2


def test_select_matrix_run_defaults_to_newest_and_selects_by_id(tmp_path: Path) -> None:
    write_run(tmp_path, matrix_run("matrix-older", created_at="2026-07-01T00:00:00+00:00"))
    write_run(tmp_path, matrix_run("matrix-newer", created_at="2026-07-27T00:00:00+00:00"))
    runs = load_matrix_runs(tmp_path)

    assert select_matrix_run(runs)["run_id"] == "matrix-newer"
    assert select_matrix_run(runs, "matrix-older")["run_id"] == "matrix-older"


def test_select_matrix_run_returns_none_when_missing(tmp_path: Path) -> None:
    assert select_matrix_run(load_matrix_runs(tmp_path)) is None
    write_run(tmp_path, matrix_run())
    assert select_matrix_run(load_matrix_runs(tmp_path), "no-such-run") is None


def test_matrix_questions_lists_every_question_with_each_setups_score() -> None:
    questions = matrix_questions(matrix_run())
    assert [q["id"] for q in questions] == ["g001", "g002"]
    first = questions[0]
    assert first["question"] == "question g001?"
    assert first["domain"] == "property"
    assert first["question_type"] == "local"
    by_key = {setup["key"]: setup for setup in first["setups"]}
    assert by_key["rag_llm"]["composite"] == 0.5
    assert by_key["rag_llm"]["judged"] is True
    assert by_key["rag_agent"]["composite"] == 0.9


def test_matrix_questions_marks_unjudged_and_errored_cells() -> None:
    run = matrix_run(
        setups={
            "rag_llm": [cell("g001", composite=None)],
            "rag_agent": [cell("g001", composite=None, error="timeout")],
        }
    )
    setups = {setup["key"]: setup for setup in matrix_questions(run)[0]["setups"]}
    assert setups["rag_llm"]["judged"] is False
    assert setups["rag_llm"]["composite"] is None
    assert setups["rag_agent"]["error"] == "timeout"

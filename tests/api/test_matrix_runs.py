"""Committed matrix runs adapted into the shape the Scoreboard aggregates."""

from __future__ import annotations

import json
from pathlib import Path

from src.api.matrix_runs import (
    list_matrix_runs,
    load_matrix_run,
    matrix_entries,
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


def test_list_matrix_runs_is_newest_first_and_ignores_other_kinds(tmp_path: Path) -> None:
    write_run(tmp_path, matrix_run("matrix-older", created_at="2026-07-01T00:00:00+00:00"))
    write_run(tmp_path, matrix_run("matrix-newer", created_at="2026-07-27T00:00:00+00:00"))
    (tmp_path / "ablation-x.json").write_text(
        json.dumps({"kind": "retrieval-ablation", "run_id": "ablation-x"}), encoding="utf-8"
    )
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")

    runs = list_matrix_runs(tmp_path)
    assert [run["run_id"] for run in runs] == ["matrix-newer", "matrix-older"]
    assert runs[0]["setups"] == ["rag_llm", "rag_agent"]
    assert runs[0]["judged"] is True


def test_load_matrix_run_defaults_to_newest_and_selects_by_id(tmp_path: Path) -> None:
    write_run(tmp_path, matrix_run("matrix-older", created_at="2026-07-01T00:00:00+00:00"))
    write_run(tmp_path, matrix_run("matrix-newer", created_at="2026-07-27T00:00:00+00:00"))

    assert load_matrix_run(None, tmp_path)["run_id"] == "matrix-newer"
    assert load_matrix_run("matrix-older", tmp_path)["run_id"] == "matrix-older"


def test_load_matrix_run_returns_none_when_missing(tmp_path: Path) -> None:
    assert load_matrix_run(None, tmp_path) is None
    write_run(tmp_path, matrix_run())
    assert load_matrix_run("no-such-run", tmp_path) is None

"""A depth-v2 run served through the Scoreboard's own data path.

The rubric decides which metric columns the tab renders and what "composite"
means, so the payload has to carry the rubric rather than the frontend
assuming one. These tests pin that, and that a ragas-v1 run served by the same
code is unchanged.
"""

from __future__ import annotations

from typing import Any

from src.api.matrix_runs import describe_matrix_run, matrix_entries, matrix_questions
from src.api.scoreboard import build_scoreboard
from tests.api.matrix_fixtures import MATRIX_CONFIG, matrix_cell, matrix_run

DEPTH_SCORES = {
    "insight_depth": 0.8,
    "specificity": 0.9,
    "coverage": 0.7,
    "evidence_breadth": 0.6,
    "calibration": 1.0,
}


def depth_cell(
    entry_id: str, *, faithfulness: float = 0.9, capped: bool = False, composite: float = 0.8
) -> dict[str, Any]:
    cell = matrix_cell(entry_id, faithfulness=faithfulness, composite=composite)
    cell["scores"].update(DEPTH_SCORES)
    cell["rubric_version"] = "depth-v2"
    cell["cap_applied"] = capped
    cell["cap_reason"] = (
        f"faithfulness {faithfulness:.2f} below 0.6 — depth cannot rescue an ungrounded answer"
        if capped
        else None
    )
    cell["composite_uncapped"] = 0.77 if capped else composite
    cell["depth_details"] = {
        metric: {"score": value, "rationale": f"{metric} rationale"}
        for metric, value in DEPTH_SCORES.items()
    }
    return cell


def depth_run() -> dict[str, Any]:
    run = matrix_run(
        "matrix-20260809-051901-depth-v2",
        created_at="2026-08-09T05:19:01+00:00",
        setups={
            "rag_llm": [
                depth_cell("g001"),
                depth_cell("g002", faithfulness=0.38, capped=True, composite=0.5),
            ],
            "rag_agent": [depth_cell("g001", composite=0.9), depth_cell("g002", composite=0.9)],
        },
        config={**MATRIX_CONFIG, "rubric_version": "depth-v2"},
    )
    run["rubric_version"] = "depth-v2"
    run["rejudged_from"] = "matrix-20260727-010101"
    return run


def _board(run: dict[str, Any]) -> dict[str, Any]:
    return build_scoreboard(matrix_entries(run), group_by="setup")


def test_the_run_picker_labels_the_rubric() -> None:
    assert describe_matrix_run(depth_run())["rubric_version"] == "depth-v2"
    assert describe_matrix_run(matrix_run())["rubric_version"] == "ragas-v1"


def test_provenance_names_the_eight_metrics_and_their_weights() -> None:
    provenance = _board(depth_run())["provenance"]
    assert provenance["rubric_version"] == "depth-v2"
    assert len(provenance["metrics"]) == 8
    assert sum(provenance["metric_weights"].values()) == 1.0
    groups = {group["key"]: group for group in provenance["metric_groups"]}
    assert groups["grounding"]["weight"] == 0.40
    assert groups["depth"]["weight"] == 0.60
    assert "capped at 0.5 when faithfulness < 0.6" in provenance["composite"]


def test_a_ragas_v1_run_still_reports_its_three_metrics() -> None:
    provenance = _board(matrix_run())["provenance"]
    assert provenance["rubric_version"] == "ragas-v1"
    assert provenance["metrics"] == [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
    ]
    assert provenance["metric_groups"] == []
    assert provenance["composite"] == "mean of the metric scores"


def test_a_row_counts_how_many_of_its_answers_were_capped() -> None:
    rows = {row["key"]: row for row in _board(depth_run())["setups"]}
    assert rows["rag_llm"]["capped"] == 1
    assert rows["rag_agent"]["capped"] == 0


def test_a_ragas_v1_row_reports_no_caps_rather_than_omitting_the_field() -> None:
    assert all(row["capped"] == 0 for row in _board(matrix_run())["setups"])


def test_the_cap_reason_travels_with_the_cell_for_the_answers_panel() -> None:
    questions = {row["id"]: row for row in matrix_questions(depth_run())}
    capped = next(s for s in questions["g002"]["setups"] if s["key"] == "rag_llm")
    assert capped["cap_applied"] is True
    assert "depth cannot rescue an ungrounded answer" in capped["cap_reason"]
    assert capped["composite_uncapped"] == 0.77
    assert capped["rubric_version"] == "depth-v2"


def test_every_depth_rationale_is_readable_per_cell() -> None:
    questions = {row["id"]: row for row in matrix_questions(depth_run())}
    cell = questions["g001"]["setups"][0]
    assert sorted(cell["rationales"]) == sorted(DEPTH_SCORES)
    assert cell["rationales"]["insight_depth"] == "insight_depth rationale"
    assert cell["scores"]["insight_depth"] == 0.8
    # The composite is reported separately, not as one of the metric scores.
    assert "composite" not in cell["scores"]


def test_a_ragas_v1_cell_has_no_rationales_and_is_not_capped() -> None:
    cell = matrix_questions(matrix_run())[0]["setups"][0]
    assert cell["rationales"] == {}
    assert cell["cap_applied"] is False
    assert cell["cap_reason"] is None
    assert cell["rubric_version"] == "ragas-v1"


def self_graded_run() -> dict[str, Any]:
    """Answer model, grounding judge and depth judge all the same model."""
    run = depth_run()
    for setup_run in run["runs"].values():
        setup_run["config"] = {
            **setup_run["config"],
            "answer_model": "deepseek-v4-flash",
            "judge_model": "deepseek-v4-flash",
            "depth_judge_model": "deepseek-v4-flash",
        }
    return run


def test_a_self_graded_ranking_is_declared_not_left_to_be_noticed() -> None:
    provenance = _board(self_graded_run())["provenance"]
    assert provenance["self_graded"] is True
    assert provenance["self_graded_answers"] == 4
    assert provenance["depth_judge_models"] == ["deepseek-v4-flash"]


def test_an_independent_depth_judge_is_not_reported_as_self_graded() -> None:
    run = self_graded_run()
    for setup_run in run["runs"].values():
        setup_run["config"] = {
            **setup_run["config"],
            "judge_model": "gpt-independent",
            "depth_judge_model": "gpt-independent",
        }
    provenance = _board(run)["provenance"]
    assert provenance["self_graded"] is False
    assert provenance["self_graded_answers"] == 0


def test_a_depth_judge_matching_the_answer_model_is_enough_to_be_self_graded() -> None:
    """60% of the composite came from the depth judge; an independent grounding
    judge does not make the ranking independent."""
    run = self_graded_run()
    for setup_run in run["runs"].values():
        setup_run["config"] = {**setup_run["config"], "judge_model": "gpt-independent"}
    assert _board(run)["provenance"]["self_graded"] is True


def test_the_depth_judge_is_named_even_when_only_the_run_config_records_it() -> None:
    """It is written once for the whole run; a cell reads its setup's config,
    so the two have to be merged or the tab omits the depth judge entirely."""
    run = depth_run()
    for setup_run in run["runs"].values():
        setup_run["config"] = {
            key: value for key, value in setup_run["config"].items() if key != "depth_judge_model"
        }
    run["config"] = {**run["config"], "depth_judge_model": "deepseek-v4-flash"}
    assert _board(run)["provenance"]["depth_judge_models"] == ["deepseek-v4-flash"]


def test_a_row_counts_ungrounded_answers_separately_from_capped_ones() -> None:
    run = depth_run()
    # An answer that breached the floor but scored below the cap on its own.
    cell = run["runs"]["rag_agent"]["entries"][0]
    cell["grounding_floor_breached"] = True
    cell["grounding_reason"] = "faithfulness 0.00 below 0.6 — depth cannot rescue it"
    cell["cap_applied"] = False

    rows = {row["key"]: row for row in _board(run)["setups"]}
    assert rows["rag_agent"]["ungrounded"] == 1
    assert rows["rag_agent"]["capped"] == 0
    # The capped one breached the floor too, so it counts in both.
    assert rows["rag_llm"]["capped"] == 1


def test_a_skipped_cell_is_not_averaged_into_the_rubrics_leaderboard() -> None:
    run = depth_run()
    skipped = run["runs"]["rag_agent"]["entries"][0]
    skipped["rejudged"] = False
    skipped["rejudge_skipped_reason"] = "no answer to grade for depth"
    skipped["source_composite"] = skipped["scores"]["composite"]
    skipped["scores"] = {**skipped["scores"], "composite": None}

    rows = {row["key"]: row for row in _board(run)["setups"]}
    assert rows["rag_agent"]["judged"] == 1
    assert rows["rag_agent"]["answers"] == 2
    assert rows["rag_agent"]["avg_composite"] == 0.9


def test_a_partly_resolved_context_is_readable_on_the_cell() -> None:
    run = depth_run()
    run["runs"]["rag_llm"]["entries"][0]["contexts_resolved"] = 24
    run["runs"]["rag_llm"]["entries"][0]["contexts_expected"] = 36
    questions = {q["id"]: q for q in matrix_questions(run)}
    cell = next(s for s in questions["g001"]["setups"] if s["key"] == "rag_llm")
    assert cell["contexts_resolved"] == 24
    assert cell["contexts_expected"] == 36


def test_the_run_descriptor_states_the_rubrics_coverage() -> None:
    run = depth_run()
    run["rejudged_cells"] = 79
    run["skipped_cells"] = 1
    descriptor = describe_matrix_run(run)
    assert descriptor["rejudged_cells"] == 79
    assert descriptor["skipped_cells"] == 1
    assert descriptor["rejudged_from"] == "matrix-20260727-010101"

"""Re-scoring a committed matrix run under depth-v2, with no LLM and no corpus.

The load-bearing property is that a rejudge changes *only* the rubric: the
answers, the retrieved ids and the three stored grounding scores must come
through byte-identical, or the depth-v2 vs ragas-v1 ranking comparison is
confounded by something other than the rubric.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from src.evals.judge import DEPTH_METRIC_NAMES, DEPTH_V2, MetricBreakdown
from src.evals.rejudge import find_run, rejudge_cell, rejudge_run

MOMENT = datetime(2026, 8, 9, 12, 0, 0, tzinfo=timezone.utc)


def _cell(setup: str, entry_id: str, **overrides: Any) -> dict[str, Any]:
    cell = {
        "id": entry_id,
        "question": f"question {entry_id}?",
        "domain": "property",
        "question_type": "local",
        "answer": f"{setup} answered {entry_id}",
        "error": None,
        "scores": {
            "faithfulness": 1.0,
            "answer_relevancy": 0.9,
            "context_precision": 0.5,
            "context_recall": 0.2,
            "composite": 0.8,
        },
        "retrieved_chunk_ids": [f"chunk:vid{entry_id}:0", f"chunk:vid{entry_id}:1"],
        "elapsed_seconds": 12.5,
        "token_estimate": 3000,
    }
    cell.update(overrides)
    return cell


def _run(**overrides: Any) -> dict[str, Any]:
    run = {
        "run_id": "matrix-20260729-061607",
        "created_at": "2026-07-29T06:16:07+00:00",
        "kind": "matrix",
        "setups": ["rag_llm", "graph_rag"],
        "config": {"answer_model": "deepseek-v4-flash", "judge_model": "deepseek-v4-flash"},
        "judged": True,
        "entry_count": 2,
        "question_ids": ["g001", "g002"],
        "runs": {
            "rag_llm": {
                "run_id": "matrix-20260729-061607-rag_llm",
                "setup": "rag_llm",
                "config": {"answer_model": "deepseek-v4-flash"},
                "entries": [_cell("rag_llm", "g001"), _cell("rag_llm", "g002")],
                "summary": {},
            },
            "graph_rag": {
                "run_id": "matrix-20260729-061607-graph_rag",
                "setup": "graph_rag",
                "config": {"answer_model": "deepseek-v4-flash"},
                "entries": [_cell("graph_rag", "g001"), _cell("graph_rag", "g002")],
                "summary": {},
            },
        },
        "comparison": {},
    }
    run.update(overrides)
    return run


def _depth(**scores: float):
    """A depth_fn returning fixed scores, recording what it was asked."""
    seen: list[tuple[str, str, list[str]]] = []
    values = {metric: scores.get(metric, 0.8) for metric in DEPTH_METRIC_NAMES}

    def depth_fn(question: str, answer: str, contexts: list[str]):
        seen.append((question, answer, list(contexts)))
        return {
            metric: MetricBreakdown(
                score=value, details={"score": value, "rationale": f"{metric} rationale"}
            )
            for metric, value in values.items()
        }

    depth_fn.seen = seen  # type: ignore[attr-defined]
    return depth_fn


def _contexts(chunk_ids):
    return [f"text for {chunk_id}" for chunk_id in chunk_ids]


# --- one cell ------------------------------------------------------------


def test_stored_grounding_scores_are_reused_untouched() -> None:
    cell = _cell("rag_llm", "g001")
    rejudged = rejudge_cell(cell, depth_fn=_depth(), contexts_fn=_contexts, rubric=DEPTH_V2)
    for metric in ("faithfulness", "answer_relevancy", "context_precision"):
        assert rejudged["scores"][metric] == cell["scores"][metric]
    assert rejudged["answer"] == cell["answer"]
    assert rejudged["retrieved_chunk_ids"] == cell["retrieved_chunk_ids"]


def test_the_composite_is_recomputed_under_the_new_rubric() -> None:
    rejudged = rejudge_cell(
        _cell("rag_llm", "g001"), depth_fn=_depth(), contexts_fn=_contexts, rubric=DEPTH_V2
    )
    expected = 1.0 * 0.20 + 0.5 * 0.10 + 0.9 * 0.10 + 0.8 * 0.60
    assert rejudged["scores"]["composite"] == pytest.approx(expected, abs=1e-4)
    assert rejudged["rubric_version"] == "depth-v2"


def test_the_five_depth_metrics_and_their_rationales_land_on_the_cell() -> None:
    rejudged = rejudge_cell(
        _cell("rag_llm", "g001"), depth_fn=_depth(), contexts_fn=_contexts, rubric=DEPTH_V2
    )
    for metric in DEPTH_METRIC_NAMES:
        assert rejudged["scores"][metric] == 0.8
        assert rejudged["depth_details"][metric]["rationale"] == f"{metric} rationale"


def test_the_depth_judge_sees_the_contexts_resolved_from_the_stored_ids() -> None:
    depth_fn = _depth()
    rejudge_cell(_cell("rag_llm", "g001"), depth_fn=depth_fn, contexts_fn=_contexts)
    question, answer, contexts = depth_fn.seen[0]  # type: ignore[attr-defined]
    assert question == "question g001?"
    assert answer == "rag_llm answered g001"
    assert contexts == ["text for chunk:vidg001:0", "text for chunk:vidg001:1"]


def test_a_cell_records_how_much_context_it_could_resolve() -> None:
    rejudged = rejudge_cell(
        _cell("rag_llm", "g001"), depth_fn=_depth(), contexts_fn=lambda ids: ["only one"]
    )
    assert rejudged["contexts_expected"] == 2
    assert rejudged["contexts_resolved"] == 1


def test_a_capped_cell_carries_the_reason_as_text() -> None:
    cell = _cell("rag_llm", "g001")
    cell["scores"]["faithfulness"] = 0.41
    rejudged = rejudge_cell(cell, depth_fn=_depth(), contexts_fn=_contexts)
    assert rejudged["cap_applied"] is True
    assert rejudged["scores"]["composite"] == 0.5
    assert "faithfulness 0.41 below 0.6" in rejudged["cap_reason"]
    assert rejudged["composite_uncapped"] > 0.5


def test_an_uncapped_cell_says_so_rather_than_leaving_the_field_off() -> None:
    rejudged = rejudge_cell(_cell("rag_llm", "g001"), depth_fn=_depth(), contexts_fn=_contexts)
    assert rejudged["cap_applied"] is False
    assert rejudged["cap_reason"] is None


def test_an_errored_cell_keeps_its_data_but_is_marked_unrejudged() -> None:
    cell = _cell("rag_llm", "g001", error="timeout", answer="", scores={})
    skipped = rejudge_cell(cell, depth_fn=_depth(), contexts_fn=_contexts)
    assert skipped["rejudged"] is False
    assert skipped["rejudge_skipped_reason"] == "cell errored"
    assert skipped["error"] == "timeout"
    assert skipped["scores"] == {}


def test_an_unjudged_cell_is_marked_rather_than_silently_passed_through() -> None:
    """A --no-judge run has no grounding scores to keep — nothing to rejudge."""
    cell = _cell("rag_llm", "g001", scores={"context_recall": 0.2})
    skipped = rejudge_cell(cell, depth_fn=_depth(), contexts_fn=_contexts)
    assert skipped["rejudged"] is False
    assert skipped["scores"]["context_recall"] == 0.2


def test_a_cell_with_no_answer_does_not_carry_its_old_rubrics_composite() -> None:
    """The defect this pins: a passed-through ragas-v1 composite averaged into
    a depth-v2 leaderboard mixes two scales in one mean with nothing marking
    it. Under depth-v2 such a cell was never scored, so it reports as such."""
    cell = _cell("rag_agent", "g003", answer="")
    cell["scores"]["composite"] = 0.0
    skipped = rejudge_cell(cell, depth_fn=_depth(), contexts_fn=_contexts)

    assert skipped["rejudged"] is False
    assert skipped["rejudge_skipped_reason"] == "no answer to grade for depth"
    # Cleared, so every consumer counts it as unjudged under this rubric...
    assert skipped["scores"]["composite"] is None
    # ...but not lost.
    assert skipped["source_composite"] == 0.0


def test_a_skipped_cell_is_excluded_from_the_depth_v2_averages() -> None:
    run = _run()
    run["runs"]["graph_rag"]["entries"][0]["answer"] = ""
    result = rejudge_run(run, depth_fn=_depth(), contexts_fn=_contexts, now=MOMENT)

    assert result["rejudged_cells"] == 3
    assert result["skipped_cells"] == 1
    summary = result["runs"]["graph_rag"]["summary"]
    # One of the two cells averaged, not two with a ragas-v1 zero dragging it.
    assert summary["averages"]["composite"] == pytest.approx(0.82, abs=1e-4)


def test_a_failing_depth_call_composites_the_grounding_half_and_says_why() -> None:
    def broken(question: str, answer: str, contexts: list[str]):
        raise RuntimeError("judge timeout")

    rejudged = rejudge_cell(_cell("rag_llm", "g001"), depth_fn=broken, contexts_fn=_contexts)
    assert rejudged["depth_error"] == "judge timeout"
    # Renormalised over the 0.4 of weight that was actually scored.
    assert rejudged["scores"]["composite"] == pytest.approx(
        (1.0 * 0.20 + 0.5 * 0.10 + 0.9 * 0.10) / 0.40, abs=1e-4
    )
    assert rejudged["composite_weight"] == pytest.approx(0.40)


# --- a whole run ---------------------------------------------------------


def test_the_rejudged_run_is_a_new_committed_run_that_names_its_source() -> None:
    result = rejudge_run(
        _run(), depth_fn=_depth(), contexts_fn=_contexts, judge_model="test-judge", now=MOMENT
    )
    assert result["run_id"] == "matrix-20260809-120000-depth-v2"
    assert result["kind"] == "matrix"
    assert result["rubric_version"] == "depth-v2"
    assert result["rejudged_from"] == "matrix-20260729-061607"
    assert result["config"]["rubric_version"] == "depth-v2"
    assert result["config"]["depth_judge_model"] == "test-judge"
    # The source run's own provenance survives.
    assert result["config"]["answer_model"] == "deepseek-v4-flash"


def test_every_cell_of_every_setup_is_rescored_in_place() -> None:
    depth_fn = _depth()
    result = rejudge_run(_run(), depth_fn=depth_fn, contexts_fn=_contexts, now=MOMENT)
    assert len(depth_fn.seen) == 4  # type: ignore[attr-defined]
    for setup in ("rag_llm", "graph_rag"):
        entries = result["runs"][setup]["entries"]
        assert [entry["id"] for entry in entries] == ["g001", "g002"]
        assert all(entry["rubric_version"] == "depth-v2" for entry in entries)


def test_the_run_summary_and_comparison_are_rebuilt_from_the_new_scores() -> None:
    result = rejudge_run(_run(), depth_fn=_depth(), contexts_fn=_contexts, now=MOMENT)
    averages = result["runs"]["rag_llm"]["summary"]["averages"]
    assert averages["insight_depth"] == pytest.approx(0.8)
    assert averages["composite"] == pytest.approx(0.82, abs=1e-4)
    assert result["comparison"]["overall"]["insight_depth"]["rag_llm"] == pytest.approx(0.8)


def test_concurrency_does_not_scramble_which_cell_got_which_score() -> None:
    """Cells come back in completion order; they must land back in run order."""

    def depth_fn(question: str, answer: str, contexts: list[str]):
        # Score keyed off the answer, so a mis-filed result is detectable.
        value = 0.9 if answer.startswith("graph_rag") else 0.1
        return {
            metric: MetricBreakdown(score=value, details={"score": value, "rationale": answer})
            for metric in DEPTH_METRIC_NAMES
        }

    result = rejudge_run(
        _run(), depth_fn=depth_fn, contexts_fn=_contexts, now=MOMENT, max_workers=4
    )
    for setup in ("rag_llm", "graph_rag"):
        for entry in result["runs"][setup]["entries"]:
            assert entry["depth_details"]["coverage"]["rationale"] == entry["answer"]


def test_progress_reports_every_cell() -> None:
    lines: list[str] = []
    rejudge_run(
        _run(), depth_fn=_depth(), contexts_fn=_contexts, now=MOMENT, on_progress=lines.append
    )
    assert len(lines) == 4
    assert "[4/4]" in lines[-1]


# --- finding the source run ----------------------------------------------


def test_find_run_reads_a_committed_matrix_run_by_id(tmp_path: Path) -> None:
    (tmp_path / "matrix-x.json").write_text(json.dumps(_run()), encoding="utf-8")
    assert find_run("matrix-x", tmp_path)["run_id"] == "matrix-20260729-061607"


def test_find_run_rejects_a_run_that_is_not_a_matrix(tmp_path: Path) -> None:
    (tmp_path / "eval-x.json").write_text(json.dumps({"kind": "golden"}), encoding="utf-8")
    with pytest.raises(ValueError, match="not a matrix run"):
        find_run("eval-x", tmp_path)


def test_find_run_reports_a_missing_run_rather_than_returning_nothing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        find_run("matrix-nope", tmp_path)

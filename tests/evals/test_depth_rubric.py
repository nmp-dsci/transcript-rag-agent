"""The depth-v2 rubric: its arithmetic, its cap, and ragas-v1 back-compat.

These are the numbers the Scoreboard ranks on, so they are pinned by hand
rather than by round-tripping the implementation: every expected composite
below is written out as the weighted sum it should be, so a weight edited in
``judge.py`` fails here instead of silently re-ranking the leaderboard.
"""

from __future__ import annotations

import math

import pytest

from src.evals.judge import (
    DEPTH_METRIC_NAMES,
    DEPTH_RUBRIC_VERSION,
    DEPTH_V2,
    RAGAS_V1,
    RUBRIC_VERSION,
    DepthJudge,
    MetricBreakdown,
    RagasJudge,
    build_depth_context_block,
    parse_depth_response,
    rubric_for,
)


def _full_scores(**overrides: float) -> dict[str, float]:
    """A score for every depth-v2 metric, overridable one at a time."""
    scores = {
        "faithfulness": 1.0,
        "context_precision": 1.0,
        "answer_relevancy": 1.0,
        "insight_depth": 1.0,
        "specificity": 1.0,
        "coverage": 1.0,
        "evidence_breadth": 1.0,
        "calibration": 1.0,
    }
    scores.update(overrides)
    return scores


# --- weights -------------------------------------------------------------


def test_depth_v2_weights_sum_to_one() -> None:
    assert sum(DEPTH_V2.weights.values()) == pytest.approx(1.0)


def test_depth_v2_weights_are_the_agreed_split() -> None:
    assert DEPTH_V2.weights == {
        "faithfulness": 0.20,
        "context_precision": 0.10,
        "answer_relevancy": 0.10,
        "insight_depth": 0.20,
        "specificity": 0.15,
        "coverage": 0.10,
        "evidence_breadth": 0.10,
        "calibration": 0.05,
    }


def test_grounding_is_forty_percent_and_depth_is_sixty() -> None:
    groups = {group.key: group for group in DEPTH_V2.groups}
    assert groups["grounding"].weight == pytest.approx(0.40)
    assert groups["depth"].weight == pytest.approx(0.60)
    for group in DEPTH_V2.groups:
        assert sum(DEPTH_V2.weights[metric] for metric in group.metrics) == pytest.approx(
            group.weight
        )


def test_every_metric_belongs_to_exactly_one_group() -> None:
    grouped = [metric for group in DEPTH_V2.groups for metric in group.metrics]
    assert sorted(grouped) == sorted(DEPTH_V2.metrics)
    assert len(grouped) == len(set(grouped)) == 8


def test_the_five_depth_metrics_are_the_new_ones() -> None:
    grounding = {"faithfulness", "answer_relevancy", "context_precision"}
    assert set(DEPTH_METRIC_NAMES) == set(DEPTH_V2.metrics) - grounding


# --- composite arithmetic ------------------------------------------------


def test_all_ones_composites_to_one() -> None:
    assert DEPTH_V2.composite(_full_scores()).composite == 1.0


def test_hand_built_scores_produce_the_exact_weighted_sum() -> None:
    scores = {
        "faithfulness": 0.90,
        "context_precision": 0.40,
        "answer_relevancy": 0.80,
        "insight_depth": 0.60,
        "specificity": 0.50,
        "coverage": 0.70,
        "evidence_breadth": 0.20,
        "calibration": 1.00,
    }
    expected = (
        0.90 * 0.20
        + 0.40 * 0.10
        + 0.80 * 0.10
        + 0.60 * 0.20
        + 0.50 * 0.15
        + 0.70 * 0.10
        + 0.20 * 0.10
        + 1.00 * 0.05
    )
    assert expected == pytest.approx(0.635)
    result = DEPTH_V2.composite(scores)
    assert result.composite == pytest.approx(0.635, abs=1e-4)
    assert result.cap_applied is False
    assert result.cap_reason is None
    assert result.weight_used == pytest.approx(1.0)


def test_a_shallow_but_perfectly_grounded_answer_cannot_reach_the_top() -> None:
    """The whole point of the rubric: grounding alone buys 0.40, not 1.0."""
    grounded_only = _full_scores(
        insight_depth=0.0, specificity=0.0, coverage=0.0, evidence_breadth=0.0, calibration=0.0
    )
    assert DEPTH_V2.composite(grounded_only).composite == pytest.approx(0.40, abs=1e-4)


def test_a_missing_metric_renormalises_rather_than_scoring_zero() -> None:
    """A failed judge call is missing data, not a zero — it must not punish."""
    scores = _full_scores()
    del scores["calibration"]
    result = DEPTH_V2.composite(scores)
    assert result.composite == 1.0
    assert result.weight_used == pytest.approx(0.95)


def test_no_scores_at_all_composites_to_none() -> None:
    result = DEPTH_V2.composite({})
    assert result.composite is None
    assert result.uncapped is None
    assert result.cap_applied is False


def test_nan_is_treated_as_missing() -> None:
    result = DEPTH_V2.composite(_full_scores(calibration=math.nan))
    assert result.composite == 1.0
    assert result.weight_used == pytest.approx(0.95)


# --- an unverifiable faithfulness must not renormalise the cap away ---------


def test_a_missing_faithfulness_is_capped_not_renormalised_away() -> None:
    """The defect this pins: with faithfulness absent, the composite used to
    renormalise over the remaining 0.8 weight and reach 1.00 with the cap
    structurally unable to fire — exactly the ungrounded answer the cap
    exists to stop."""
    scores = _full_scores()
    del scores["faithfulness"]
    result = DEPTH_V2.composite(scores)

    assert result.uncapped == 1.0
    assert result.composite == 0.5
    assert result.cap_applied is True
    assert result.grounding_floor_breached is True
    assert "grounding is unverified" in result.cap_reason


def test_a_nan_faithfulness_is_capped_the_same_way() -> None:
    result = DEPTH_V2.composite(_full_scores(faithfulness=math.nan))
    assert result.composite == 0.5
    assert result.cap_applied is True
    assert result.grounding_floor_breached is True
    assert "faithfulness could not be scored" in result.grounding_reason


def test_the_unverifiable_reason_does_not_invent_a_number() -> None:
    """A reason quoting a faithfulness score there was none of would be worse
    than no reason at all."""
    scores = _full_scores()
    del scores["faithfulness"]
    assert "0.00" not in DEPTH_V2.composite(scores).cap_reason


def test_a_missing_faithfulness_still_only_lowers_a_high_composite() -> None:
    scores = {metric: 0.2 for metric in DEPTH_V2.metrics}
    del scores["faithfulness"]
    result = DEPTH_V2.composite(scores)
    assert result.composite == pytest.approx(0.2, abs=1e-4)
    assert result.cap_applied is False
    # Still ungrounded, though — the badge must not depend on the number.
    assert result.grounding_floor_breached is True


def test_ragas_v1_ignores_a_missing_faithfulness_entirely() -> None:
    """No floor on the old rubric, so nothing changes for a legacy record."""
    result = RAGAS_V1.composite({"answer_relevancy": 1.0, "context_precision": 1.0})
    assert result.composite == 1.0
    assert result.cap_applied is False
    assert result.grounding_floor_breached is False


# --- breaching the floor is reported separately from the cap firing --------


def test_a_flatly_ungrounded_answer_is_flagged_even_though_it_scored_low() -> None:
    """faithfulness 0.00 and a low composite: the cap changes nothing, so a
    "capped" badge alone would leave the worst answer in the run unmarked."""
    scores = {metric: 0.1 for metric in DEPTH_V2.metrics}
    scores["faithfulness"] = 0.0
    result = DEPTH_V2.composite(scores)

    assert result.cap_applied is False
    assert result.cap_reason is None
    assert result.grounding_floor_breached is True
    assert "faithfulness 0.00 below 0.6" in result.grounding_reason


def test_a_capped_answer_is_also_reported_as_breaching_the_floor() -> None:
    result = DEPTH_V2.composite(_full_scores(faithfulness=0.41))
    assert result.cap_applied is True
    assert result.grounding_floor_breached is True
    assert result.grounding_reason == result.cap_reason


def test_a_grounded_answer_breaches_nothing() -> None:
    result = DEPTH_V2.composite(_full_scores())
    assert result.grounding_floor_breached is False
    assert result.grounding_reason is None


# --- the faithfulness cap ------------------------------------------------


def test_cap_fires_below_the_floor_and_records_a_readable_reason() -> None:
    result = DEPTH_V2.composite(_full_scores(faithfulness=0.41))
    assert result.cap_applied is True
    assert result.composite == 0.5
    # The uncapped number survives, so the UI can say what was given up.
    assert result.uncapped == pytest.approx(0.882, abs=1e-4)
    assert result.cap_reason == (
        "faithfulness 0.41 below 0.6 — depth cannot rescue an ungrounded answer"
    )


def test_the_cap_reason_never_prints_a_number_that_is_not_below_the_floor() -> None:
    """0.5999 at two decimals reads "faithfulness 0.60 below 0.6" — a reason
    that contradicts itself on exactly the cells a reader is checking."""
    result = DEPTH_V2.composite(_full_scores(faithfulness=0.5999))
    assert result.cap_applied is True
    assert result.cap_reason == (
        "faithfulness 0.5999 below 0.6 — depth cannot rescue an ungrounded answer"
    )


def test_the_cap_reason_stays_at_two_decimals_when_two_are_honest() -> None:
    result = DEPTH_V2.composite(_full_scores(faithfulness=0.594))
    assert result.cap_reason == (
        "faithfulness 0.59 below 0.6 — depth cannot rescue an ungrounded answer"
    )


def test_cap_does_not_fire_at_the_floor() -> None:
    result = DEPTH_V2.composite(_full_scores(faithfulness=0.6))
    assert result.cap_applied is False
    # Everything else perfect, faithfulness 0.6: 1.0 - 0.20 x 0.4.
    assert result.composite == pytest.approx(0.92, abs=1e-4)


def test_cap_does_not_fire_above_the_floor() -> None:
    result = DEPTH_V2.composite(_full_scores(faithfulness=0.61))
    assert result.cap_applied is False
    assert result.cap_reason is None


def test_cap_leaves_an_already_low_composite_alone() -> None:
    """The cap is a ceiling, not a floor — it must never raise a score."""
    poor = {metric: 0.1 for metric in DEPTH_V2.metrics}
    result = DEPTH_V2.composite(poor)
    assert result.cap_applied is False
    assert result.composite == pytest.approx(0.1, abs=1e-4)


def test_ragas_v1_has_no_cap() -> None:
    scores = {"faithfulness": 0.1, "answer_relevancy": 1.0, "context_precision": 1.0}
    result = RAGAS_V1.composite(scores)
    assert result.cap_applied is False
    assert result.composite == pytest.approx(0.7, abs=1e-4)


# --- ragas-v1 back-compat ------------------------------------------------


def test_ragas_v1_composite_is_still_the_mean() -> None:
    scores = {"faithfulness": 1.0, "answer_relevancy": 0.9724, "context_precision": 0.1714}
    assert RAGAS_V1.composite(scores).composite == pytest.approx(
        (1.0 + 0.9724 + 0.1714) / 3, abs=1e-4
    )


def test_the_ragas_judge_still_stamps_ragas_v1() -> None:
    judge = RagasJudge(
        score_fns={
            "faithfulness": lambda q, a, c: 0.8,
            "answer_relevancy": lambda q, a, c: 0.6,
            "context_precision": lambda q, a, c: 0.4,
        },
        judge_model="test-judge",
    )
    evaluation = judge.score("q?", "a", ["ctx"])
    assert evaluation["rubric_version"] == RUBRIC_VERSION == "ragas-v1"
    assert evaluation["composite"] == pytest.approx(0.6, abs=1e-4)
    assert evaluation["cap_applied"] is False
    assert evaluation["cap_reason"] is None


def test_rubric_lookup_falls_back_to_ragas_v1() -> None:
    assert rubric_for(None) is RAGAS_V1
    assert rubric_for("ragas-v1") is RAGAS_V1
    assert rubric_for("depth-v2") is DEPTH_V2
    # A record naming a rubric this build does not know still renders.
    assert rubric_for("depth-v99") is RAGAS_V1


def test_rubric_versions_are_distinct() -> None:
    assert DEPTH_RUBRIC_VERSION == "depth-v2" != RUBRIC_VERSION


# --- the depth judge's one structured call --------------------------------


def _response(**overrides: object) -> str:
    import json

    body = {
        metric: {"score": 0.5, "rationale": f"because of {metric}"} for metric in DEPTH_METRIC_NAMES
    }
    body.update(overrides)  # type: ignore[arg-type]
    return json.dumps(body)


class _FakeLLM:
    def __init__(self, *responses: str) -> None:
        self.responses = list(responses)
        self.calls: list[list[dict[str, str]]] = []

    def invoke(self, messages, /, *args, **kwargs):  # type: ignore[no-untyped-def]
        self.calls.append(messages)
        body = self.responses.pop(0)
        if isinstance(body, Exception):
            raise body
        return type("Response", (), {"content": body})()


def test_one_call_scores_all_five_depth_metrics() -> None:
    llm = _FakeLLM(_response())
    judge = DepthJudge(llm=llm, judge_model="test-judge")
    breakdowns = judge.score("q?", "the answer", ["ctx one", "ctx two"])

    assert len(llm.calls) == 1, "five metrics must cost one call, not five"
    assert sorted(breakdowns) == sorted(DEPTH_METRIC_NAMES)
    assert all(isinstance(value, MetricBreakdown) for value in breakdowns.values())


def test_rationales_persist_in_the_same_details_shape_as_the_other_metrics() -> None:
    judge = DepthJudge(llm=_FakeLLM(_response()), judge_model="test-judge")
    breakdown = judge.score("q?", "a", ["ctx"])["insight_depth"]
    assert breakdown.score == 0.5
    assert breakdown.details == {"score": 0.5, "rationale": "because of insight_depth"}


def test_the_prompt_carries_the_question_answer_and_every_context() -> None:
    llm = _FakeLLM(_response())
    DepthJudge(llm=llm, judge_model="t").score("why?", "because", ["alpha", "beta"])
    user = llm.calls[0][1]["content"]
    assert "why?" in user and "because" in user
    assert "alpha" in user and "beta" in user


def test_a_fenced_response_still_parses() -> None:
    fenced = f"```json\n{_response()}\n```"
    assert sorted(parse_depth_response(fenced)) == sorted(DEPTH_METRIC_NAMES)


def test_out_of_range_scores_are_clamped_not_rejected() -> None:
    parsed = parse_depth_response(
        _response(
            insight_depth={"score": 1.4, "rationale": "r"},
            calibration={"score": -2, "rationale": "r"},
        )
    )
    assert parsed["insight_depth"].score == 1.0
    assert parsed["calibration"].score == 0.0


@pytest.mark.parametrize(
    "body",
    [
        "not json at all",
        '{"insight_depth": {"score": 0.5, "rationale": "r"}}',
        '["insight_depth"]',
    ],
)
def test_a_response_that_breaks_the_contract_is_rejected(body: str) -> None:
    with pytest.raises(ValueError):
        parse_depth_response(body)


def test_a_non_numeric_score_is_rejected_rather_than_coerced() -> None:
    with pytest.raises(ValueError):
        parse_depth_response(_response(coverage={"score": "high", "rationale": "r"}))


def test_a_malformed_response_is_retried_once() -> None:
    judge = DepthJudge(llm=_FakeLLM("junk", _response()), judge_model="t")
    assert sorted(judge.score("q", "a", [])) == sorted(DEPTH_METRIC_NAMES)


def test_persistent_failure_raises_so_the_cell_records_it() -> None:
    judge = DepthJudge(llm=_FakeLLM("junk", "still junk"), judge_model="t")
    with pytest.raises(RuntimeError, match="depth judging failed"):
        judge.score("q", "a", [])


def test_an_answer_with_no_resolvable_context_still_gets_a_prompt() -> None:
    block = build_depth_context_block([])
    assert "no retrieved context" in block


def test_long_chunks_are_truncated_rather_than_dropped() -> None:
    """Dropping chunks would silently lower evidence_breadth."""
    block = build_depth_context_block(["x" * 5000, "y" * 5000])
    assert "[1]" in block and "[2]" in block

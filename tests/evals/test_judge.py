from __future__ import annotations

import math

from src.evals.judge import RUBRIC_VERSION, RagasJudge, unjudgeable


def _judge(**fns) -> RagasJudge:
    return RagasJudge(score_fns=fns, judge_model="test-judge")


def test_score_all_metrics_succeed() -> None:
    judge = _judge(
        faithfulness=lambda q, a, c: 0.8,
        answer_relevancy=lambda q, a, c: 0.6,
    )
    evaluation = judge.score("q?", "answer", ["ctx"])

    assert evaluation["scores"] == {"faithfulness": 0.8, "answer_relevancy": 0.6}
    assert evaluation["composite"] == 0.7
    assert evaluation["error"] is None
    assert evaluation["judge"] == "ragas"
    assert evaluation["judge_model"] == "test-judge"
    assert evaluation["rubric_version"] == RUBRIC_VERSION
    assert evaluation["scored_at"]


def test_score_partial_failure_keeps_other_metrics() -> None:
    def broken(question: str, answer: str, contexts: list[str]) -> float:
        raise RuntimeError("llm timeout")

    judge = _judge(faithfulness=lambda q, a, c: 1.0, answer_relevancy=broken)
    evaluation = judge.score("q?", "answer", ["ctx"])

    assert evaluation["scores"] == {"faithfulness": 1.0}
    assert evaluation["composite"] == 1.0
    assert "answer_relevancy: llm timeout" in evaluation["error"]


def test_score_nan_treated_as_error() -> None:
    judge = _judge(faithfulness=lambda q, a, c: math.nan)
    evaluation = judge.score("q?", "answer", ["ctx"])

    assert evaluation["scores"] == {}
    assert evaluation["composite"] is None
    assert "faithfulness" in evaluation["error"]


def test_score_receives_inputs() -> None:
    seen: list[tuple[str, str, list[str]]] = []

    def record(question: str, answer: str, contexts: list[str]) -> float:
        seen.append((question, answer, contexts))
        return 0.5

    _judge(faithfulness=record).score("the q", "the a", ["c1", "c2"])
    assert seen == [("the q", "the a", ["c1", "c2"])]


def test_unjudgeable_record() -> None:
    record = unjudgeable("answer errored; not judged", "test-judge")
    assert record["composite"] is None
    assert record["scores"] == {}
    assert record["error"] == "answer errored; not judged"
    assert record["judge_model"] == "test-judge"

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

import pytest

from src.evals.judge import (
    RUBRIC_VERSION,
    MetricBreakdown,
    RagasJudge,
    average_precision,
    unjudgeable,
)


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
    assert record["details"] is None
    assert record["self_graded"] is None
    assert record["spread"] == {}


# --- details -------------------------------------------------------------


def _faithfulness_breakdown(score: float = 0.5) -> MetricBreakdown:
    return MetricBreakdown(
        score=score,
        details={
            "claims": [
                {"claim": "A is true.", "verdict": 1, "reason": "stated"},
                {"claim": "B is true.", "verdict": 0, "reason": "absent"},
            ],
            "supported": 1,
            "total": 2,
        },
    )


def test_details_default_to_none_without_breakdown_fns() -> None:
    """History written before details existed must keep loading unchanged."""
    evaluation = _judge(faithfulness=lambda q, a, c: 0.8).score("q?", "a", ["c"])
    assert evaluation["details"] is None


def test_breakdown_fn_supplies_both_score_and_details() -> None:
    judge = RagasJudge(
        score_fns={},
        judge_model="test-judge",
        breakdown_fns={"faithfulness": lambda q, a, c: _faithfulness_breakdown()},
    )
    evaluation = judge.score("q?", "a", ["c"])

    assert evaluation["scores"] == {"faithfulness": 0.5}
    details = evaluation["details"]["faithfulness"]
    assert details["supported"] == 1
    assert details["total"] == 2
    assert [claim["verdict"] for claim in details["claims"]] == [1, 0]
    # The headline number is arithmetic over the rows shown beneath it.
    assert (
        evaluation["scores"]["faithfulness"]
        == details["supported"] / details["total"]
    )


def test_breakdown_fn_preferred_over_score_fn() -> None:
    """A metric with both scorers runs once, via the path that keeps details."""
    calls: list[str] = []

    def score_fn(q: str, a: str, c: list[str]) -> float:
        calls.append("score_fn")
        return 0.9

    def breakdown_fn(q: str, a: str, c: list[str]) -> MetricBreakdown:
        calls.append("breakdown_fn")
        return _faithfulness_breakdown()

    judge = RagasJudge(
        score_fns={"faithfulness": score_fn},
        judge_model="test-judge",
        breakdown_fns={"faithfulness": breakdown_fn},
    )
    evaluation = judge.score("q?", "a", ["c"])

    assert calls == ["breakdown_fn"]
    assert evaluation["scores"]["faithfulness"] == 0.5


def test_failing_breakdown_degrades_to_score_without_details() -> None:
    def broken(q: str, a: str, c: list[str]) -> MetricBreakdown:
        raise RuntimeError("prompt drift")

    judge = RagasJudge(
        score_fns={"faithfulness": lambda q, a, c: 0.9},
        judge_model="test-judge",
        breakdown_fns={"faithfulness": broken},
    )
    evaluation = judge.score("q?", "a", ["c"])

    assert evaluation["scores"] == {"faithfulness": 0.9}
    assert evaluation["error"] is None
    assert evaluation["details"] is None


def test_failing_breakdown_without_fallback_records_error() -> None:
    def broken(q: str, a: str, c: list[str]) -> MetricBreakdown:
        raise RuntimeError("prompt drift")

    judge = RagasJudge(
        score_fns={},
        judge_model="test-judge",
        breakdown_fns={"faithfulness": broken},
    )
    evaluation = judge.score("q?", "a", ["c"])

    assert evaluation["scores"] == {}
    assert "faithfulness: prompt drift" in evaluation["error"]
    assert evaluation["details"] is None


def test_details_omit_metrics_that_could_not_be_captured() -> None:
    judge = RagasJudge(
        score_fns={"answer_relevancy": lambda q, a, c: 0.4},
        judge_model="test-judge",
        breakdown_fns={"faithfulness": lambda q, a, c: _faithfulness_breakdown()},
    )
    evaluation = judge.score("q?", "a", ["c"])

    assert evaluation["details"]["faithfulness"] is not None
    assert evaluation["details"]["answer_relevancy"] is None


def test_details_map_errored_metrics_to_none() -> None:
    """Every attempted metric gets a key, so the drawer never guesses."""

    def broken(q: str, a: str, c: list[str]) -> float:
        raise RuntimeError("llm timeout")

    judge = RagasJudge(
        score_fns={"answer_relevancy": broken},
        judge_model="test-judge",
        breakdown_fns={"faithfulness": lambda q, a, c: _faithfulness_breakdown()},
    )
    evaluation = judge.score("q?", "a", ["c"])

    assert evaluation["details"]["answer_relevancy"] is None
    assert evaluation["details"]["faithfulness"] is not None
    assert "answer_relevancy: llm timeout" in evaluation["error"]


# --- multi-sample scoring -------------------------------------------------


def test_default_samples_is_one_call_per_metric() -> None:
    calls: list[int] = []

    def once(q: str, a: str, c: list[str]) -> float:
        calls.append(1)
        return 0.5

    evaluation = _judge(faithfulness=once).score("q?", "a", ["c"])
    assert len(calls) == 1
    assert evaluation["spread"] == {"faithfulness": 0.0}
    assert evaluation["sample_scores"] == {"faithfulness": [0.5]}
    assert evaluation["judge_samples"] == 1


def test_multiple_samples_report_mean_and_spread() -> None:
    values = iter([0.2, 0.5, 0.8])
    judge = RagasJudge(
        score_fns={"faithfulness": lambda q, a, c: next(values)},
        judge_model="test-judge",
        samples=3,
    )
    evaluation = judge.score("q?", "a", ["c"])

    assert evaluation["scores"] == {"faithfulness": 0.5}
    assert evaluation["spread"] == {"faithfulness": pytest.approx(0.6)}
    assert evaluation["sample_scores"] == {"faithfulness": [0.2, 0.5, 0.8]}
    assert evaluation["judge_samples"] == 3


def test_multiple_samples_survive_one_failed_run() -> None:
    outcomes = iter([0.4, RuntimeError("timeout"), 0.6])

    def flaky(q: str, a: str, c: list[str]) -> float:
        value = next(outcomes)
        if isinstance(value, Exception):
            raise value
        return value

    judge = RagasJudge(
        score_fns={"faithfulness": flaky}, judge_model="test-judge", samples=3
    )
    evaluation = judge.score("q?", "a", ["c"])

    assert evaluation["scores"] == {"faithfulness": 0.5}
    assert evaluation["sample_scores"] == {"faithfulness": [0.4, 0.6]}
    assert evaluation["error"] is None


def test_details_reconcile_with_the_first_sample() -> None:
    """With samples > 1 the score is a mean, so details name the run they explain."""
    scores = iter([0.5, 1.0])

    def breakdown(q: str, a: str, c: list[str]) -> MetricBreakdown:
        return _faithfulness_breakdown(next(scores))

    judge = RagasJudge(
        score_fns={},
        judge_model="test-judge",
        breakdown_fns={"faithfulness": breakdown},
        samples=2,
    )
    evaluation = judge.score("q?", "a", ["c"])

    details = evaluation["details"]["faithfulness"]
    assert evaluation["scores"]["faithfulness"] == 0.75  # the mean
    assert evaluation["sample_scores"]["faithfulness"][0] == 0.5
    assert details["supported"] / details["total"] == 0.5


# --- self-grading ---------------------------------------------------------


def test_self_graded_true_when_judge_matches_answering_model() -> None:
    judge = RagasJudge(
        score_fns={"faithfulness": lambda q, a, c: 0.5},
        judge_model="deepseek-v4",
        answer_model="deepseek-v4",
    )
    assert judge.score("q?", "a", ["c"])["self_graded"] is True


def test_self_graded_false_for_independent_judge() -> None:
    judge = RagasJudge(
        score_fns={"faithfulness": lambda q, a, c: 0.5},
        judge_model="other-judge",
        answer_model="deepseek-v4",
    )
    assert judge.score("q?", "a", ["c"])["self_graded"] is False


def test_self_graded_none_when_answering_model_unknown() -> None:
    assert _judge(faithfulness=lambda q, a, c: 0.5).score("q?", "a", ["c"])[
        "self_graded"
    ] is None


def test_per_answer_model_overrides_the_judge_default() -> None:
    judge = RagasJudge(
        score_fns={"faithfulness": lambda q, a, c: 0.5},
        judge_model="deepseek-v4",
        answer_model="deepseek-v4",
    )
    evaluation = judge.score("q?", "a", ["c"], answer_model="gpt-legacy")
    assert evaluation["self_graded"] is False


# --- ragas arithmetic -----------------------------------------------------


@pytest.fixture
def ragas_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    """Importable ragas with analytics off, as ``from_settings`` arranges it."""
    monkeypatch.setenv("RAGAS_DO_NOT_TRACK", "true")
    from src.evals import _ragas_compat

    _ragas_compat.install()


@pytest.mark.parametrize(
    "verdicts",
    [[], [0], [1], [0, 0], [1, 1], [1, 0], [0, 1], [1, 0, 1], [0, 1, 1, 0, 1]],
)
def test_average_precision_matches_ragas(
    verdicts: list[int], ragas_ready: None
) -> None:
    """Pin our reimplementation against ragas' own average precision."""
    from ragas.metrics._context_precision import (
        LLMContextPrecisionWithoutReference,
        Verification,
    )

    metric = LLMContextPrecisionWithoutReference()
    expected = metric._calculate_average_precision(
        [Verification(reason="", verdict=verdict) for verdict in verdicts]
    )
    assert average_precision(verdicts) == expected


# --- reconciliation with ragas' own scoring -------------------------------

_STATEMENTS = json.dumps({"statements": ["A is true.", "B is true."]})
_NLI = json.dumps(
    {
        "statements": [
            {"statement": "A is true.", "reason": "stated in chunk one", "verdict": 1},
            {"statement": "B is true.", "reason": "not in any chunk", "verdict": 0},
        ]
    }
)
_RELEVANCE = json.dumps({"question": "what is A?", "noncommittal": 0})
_USEFUL = json.dumps({"reason": "answers the question", "verdict": 1})
_USELESS = json.dumps({"reason": "off topic", "verdict": 0})

_CONTEXTS = ["chunk one is about A" + "." * 200, "chunk two is about Z"]


def _fake_stack() -> Any:
    """Real ragas metrics wired to a deterministic canned-response LLM."""
    from langchain_core.outputs import Generation, LLMResult
    from ragas.llms.base import BaseRagasLLM
    from ragas.metrics import (
        AnswerRelevancy,
        Faithfulness,
        LLMContextPrecisionWithoutReference,
    )

    @dataclass
    class FakeLLM(BaseRagasLLM):
        def _text_for(self, prompt_text: str) -> str:
            # Context precision asks once per chunk; everything else is
            # identified by the output model named in the prompt's schema.
            if '"title": "Verification"' in prompt_text:
                return _USEFUL if _CONTEXTS[0] in prompt_text else _USELESS
            if '"title": "StatementGeneratorOutput"' in prompt_text:
                return _STATEMENTS
            if '"title": "NLIStatementOutput"' in prompt_text:
                return _NLI
            if '"title": "ResponseRelevanceOutput"' in prompt_text:
                return _RELEVANCE
            raise AssertionError(f"unexpected prompt: {prompt_text[:200]}")

        def generate_text(
            self, prompt, n=1, temperature=0.01, stop=None, callbacks=None
        ):
            text = self._text_for(prompt.to_string())
            return LLMResult(generations=[[Generation(text=text) for _ in range(n)]])

        async def agenerate_text(
            self, prompt, n=1, temperature=0.01, stop=None, callbacks=None
        ):
            return self.generate_text(prompt, n, temperature, stop, callbacks)

        def is_finished(self, response) -> bool:
            return True

    class FakeEmbeddings:
        def embed_query(self, text: str) -> list[float]:
            return [1.0, 0.0]

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [[1.0, 0.5] for _ in texts]

    llm = FakeLLM()
    return (
        llm,
        Faithfulness(llm=llm),
        AnswerRelevancy(llm=llm, embeddings=FakeEmbeddings(), strictness=1),
        LLMContextPrecisionWithoutReference(llm=llm),
    )


@pytest.fixture
def ragas_stack(ragas_ready: None) -> Any:
    return _fake_stack()


def test_breakdowns_reproduce_ragas_scores_exactly(ragas_stack: Any) -> None:
    """The captured breakdown must agree with ragas' own number, to the bit.

    A breakdown that disagrees with the score above it is worse than none, so
    this pins our arithmetic against ``single_turn_score`` for the same input.
    """
    from ragas import SingleTurnSample

    from src.evals.judge import _build_breakdown_fns

    llm, faithfulness, relevancy, precision = ragas_stack
    breakdown_fns = _build_breakdown_fns(llm, faithfulness, relevancy, precision)

    question, answer = "what is A?", "A is true. B is true."
    sample = SingleTurnSample(
        user_input=question, response=answer, retrieved_contexts=list(_CONTEXTS)
    )
    native = {
        "faithfulness": float(faithfulness.single_turn_score(sample)),
        "answer_relevancy": float(relevancy.single_turn_score(sample)),
        "context_precision": float(precision.single_turn_score(sample)),
    }

    for name, expected in native.items():
        result = breakdown_fns[name](question, answer, list(_CONTEXTS))
        assert result.score == expected, name
        assert result.details is not None, name


def test_faithfulness_details_shape(ragas_stack: Any) -> None:
    from src.evals.judge import _build_breakdown_fns

    fns = _build_breakdown_fns(*ragas_stack)
    result = fns["faithfulness"]("what is A?", "A is true. B is true.", list(_CONTEXTS))

    assert result.details == {
        "claims": [
            {"claim": "A is true.", "verdict": 1, "reason": "stated in chunk one"},
            {"claim": "B is true.", "verdict": 0, "reason": "not in any chunk"},
        ],
        "supported": 1,
        "total": 2,
    }
    assert result.score == 0.5


def test_answer_relevancy_details_shape(ragas_stack: Any) -> None:
    from src.evals.judge import _build_breakdown_fns

    fns = _build_breakdown_fns(*ragas_stack)
    result = fns["answer_relevancy"]("what is A?", "A is true.", list(_CONTEXTS))

    assert result.details["generated_questions"] == ["what is A?"]
    assert result.details["noncommittal"] is False
    assert len(result.details["similarities"]) == 1
    assert result.score == pytest.approx(result.details["similarities"][0], abs=1e-4)


def test_context_precision_details_shape(ragas_stack: Any) -> None:
    from src.evals.judge import CHUNK_PREVIEW_CHARS, _build_breakdown_fns

    fns = _build_breakdown_fns(*ragas_stack)
    result = fns["context_precision"]("what is A?", "A is true.", list(_CONTEXTS))

    verdicts = result.details["verdicts"]
    assert [v["rank"] for v in verdicts] == [1, 2]
    assert [v["verdict"] for v in verdicts] == [1, 0]
    assert verdicts[0]["reason"] == "answers the question"
    # The long first chunk is truncated; the short second one is not.
    assert verdicts[0]["chunk_preview"] == _CONTEXTS[0][:CHUNK_PREVIEW_CHARS]
    assert len(verdicts[0]["chunk_preview"]) == CHUNK_PREVIEW_CHARS
    assert verdicts[1]["chunk_preview"] == _CONTEXTS[1]
    assert result.details["average_precision"] == round(result.score, 4)


def test_faithfulness_without_statements_scores_nan(ragas_ready: None) -> None:
    """No decomposable claims is ragas' NaN case, and NaN is an error, not 0.0."""
    from langchain_core.outputs import Generation, LLMResult
    from ragas.llms.base import BaseRagasLLM
    from ragas.metrics import (
        AnswerRelevancy,
        Faithfulness,
        LLMContextPrecisionWithoutReference,
    )

    from src.evals.judge import _build_breakdown_fns

    @dataclass
    class EmptyLLM(BaseRagasLLM):
        def generate_text(
            self, prompt, n=1, temperature=0.01, stop=None, callbacks=None
        ):
            text = json.dumps({"statements": []})
            return LLMResult(generations=[[Generation(text=text) for _ in range(n)]])

        async def agenerate_text(
            self, prompt, n=1, temperature=0.01, stop=None, callbacks=None
        ):
            return self.generate_text(prompt, n, temperature, stop, callbacks)

        def is_finished(self, response) -> bool:
            return True

    llm = EmptyLLM()
    fns = _build_breakdown_fns(
        llm,
        Faithfulness(llm=llm),
        AnswerRelevancy(llm=llm, embeddings=None, strictness=1),
        LLMContextPrecisionWithoutReference(llm=llm),
    )
    result = fns["faithfulness"]("q?", "a", ["c"])
    assert math.isnan(result.score)
    assert result.details is None

    judge = RagasJudge(
        score_fns={},
        judge_model="test-judge",
        breakdown_fns={"faithfulness": fns["faithfulness"]},
    )
    evaluation = judge.score("q?", "a", ["c"])
    assert evaluation["scores"] == {}
    assert "faithfulness" in evaluation["error"]
    assert evaluation["details"] is None

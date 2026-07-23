"""RAGAS answer judging for the web workbench.

Every setup's answer to a question is scored with the same three RAGAS
metrics — faithfulness (is the answer supported by the retrieved chunks?),
answer relevancy (does it address the question?), and context precision
(were the retrieved chunks useful?) — so all retrieval methods are graded
under one eval process.

Each score is also *explained*. RAGAS computes rich intermediates on the way
to a number — the claims it broke the answer into and whether the context
supports each one, the question it reverse-engineered from the answer, the
per-chunk usefulness verdicts — and normally throws them away. This module
keeps them under ``details`` so the UI can show how a score was derived.

Detail capture drives the ragas prompt objects directly rather than sniffing
an evaluation run through callbacks. That choice is about *reconciliation*:
the breakdown and the score come from the same captured structured output,
so the number in the UI is arithmetic over the rows shown beneath it and the
two cannot drift apart. Callback sniffing would re-derive the breakdown from
a separate view of the run and could disagree with the score it annotates,
which is worse than showing no breakdown at all. The arithmetic reproduced
here is ragas' own; ``tests/evals/test_judge.py`` pins it against ragas'
``single_turn_score`` for identical input.

The judge LLM defaults to the DeepSeek chat model already configured for
answering; override with ``YT_AGENT_JUDGE_MODEL`` / ``YT_AGENT_JUDGE_API_KEY``
/ ``YT_AGENT_JUDGE_BASE_URL`` to grade with an independent provider.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Sequence

from src.config import Settings

logger = logging.getLogger(__name__)

RUBRIC_VERSION = "ragas-v1"
METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_precision"]

# How much of each retrieved chunk to store alongside its precision verdict.
# Enough to recognise the chunk in a drawer, not so much that every evaluation
# duplicates the contexts already persisted with the answer.
CHUNK_PREVIEW_CHARS = 160


def ragas_version() -> str:
    """The installed ragas version, stamped onto every evaluation record.

    Metric implementations change between releases, so a score is only
    comparable to another score produced by the same version.
    """
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("ragas")
    except PackageNotFoundError:
        return "unknown"


# (question, answer, contexts) -> score in [0, 1]
ScoreFn = Callable[[str, str, list[str]], float]


@dataclass(frozen=True)
class MetricBreakdown:
    """A metric score together with the intermediates it was computed from.

    ``score`` must be arithmetic over ``details``: whatever the UI renders
    from the breakdown has to add up to the number reported for the metric.
    """

    score: float
    details: dict[str, Any] | None = None


# (question, answer, contexts) -> score plus the intermediates behind it
BreakdownFn = Callable[[str, str, list[str]], MetricBreakdown]


def average_precision(verdicts: Sequence[int]) -> float:
    """Ragas' context-precision arithmetic, reproduced exactly.

    Mean of precision@k over the ranks that were judged useful. The ``1e-10``
    added to the denominator is ragas' own guard against an all-zero verdict
    list; it is reproduced rather than cleaned up because dropping it would
    shift scores in the fourth decimal place away from ragas' numbers.
    """
    verdict_list = [1 if verdict else 0 for verdict in verdicts]
    denominator = sum(verdict_list) + 1e-10
    numerator = sum(
        (sum(verdict_list[: i + 1]) / (i + 1)) * verdict_list[i]
        for i in range(len(verdict_list))
    )
    return numerator / denominator


def _preview(text: str) -> str:
    return text[:CHUNK_PREVIEW_CHARS]


def _build_breakdown_fns(
    llm: Any,
    faithfulness: Any,
    relevancy: Any,
    precision: Any,
) -> dict[str, BreakdownFn]:
    """Score each metric by driving its ragas prompts and keeping the workings.

    Every function here returns the score *derived from* the intermediates it
    reports, so a breakdown can never contradict the score above it.
    """
    # Imported here, not at module scope: ragas pulls in a slow model stack.
    from ragas.async_utils import run
    from ragas.metrics._answer_relevance import ResponseRelevanceInput
    from ragas.metrics._context_precision import QAC
    from ragas.metrics._faithfulness import (
        NLIStatementInput,
        StatementGeneratorInput,
    )

    def faithfulness_breakdown(
        question: str, answer: str, contexts: list[str]
    ) -> MetricBreakdown:
        async def _generate() -> Any:
            statements = await faithfulness.statement_generator_prompt.generate(
                llm=llm,
                data=StatementGeneratorInput(question=question, answer=answer),
            )
            if not statements.statements:
                return None
            return await faithfulness.nli_statements_prompt.generate(
                llm=llm,
                data=NLIStatementInput(
                    context="\n".join(contexts), statements=statements.statements
                ),
            )

        verdicts = run(_generate)
        if verdicts is None or not verdicts.statements:
            # Ragas scores an answer it could not decompose as NaN; match it.
            return MetricBreakdown(score=math.nan, details=None)

        claims = [
            {
                "claim": item.statement,
                "verdict": 1 if item.verdict else 0,
                "reason": item.reason,
            }
            for item in verdicts.statements
        ]
        supported = sum(int(claim["verdict"]) for claim in claims)
        total = len(claims)
        return MetricBreakdown(
            score=supported / total,
            details={"claims": claims, "supported": supported, "total": total},
        )

    def relevancy_breakdown(
        question: str, answer: str, contexts: list[str]
    ) -> MetricBreakdown:
        responses = run(
            lambda: relevancy.question_generation.generate_multiple(
                llm=llm,
                data=ResponseRelevanceInput(response=answer),
                n=relevancy.strictness,
            )
        )
        generated = [item.question for item in responses]
        noncommittal = all(bool(item.noncommittal) for item in responses)
        if all(text == "" for text in generated):
            # Ragas' signal that the judge returned no usable question.
            return MetricBreakdown(score=math.nan, details=None)

        # Ragas' own similarity routine, so the cosines are its cosines.
        similarities = relevancy.calculate_similarity(question, generated)
        score = float(similarities.mean()) * int(not noncommittal)
        return MetricBreakdown(
            score=score,
            details={
                "generated_questions": generated,
                "noncommittal": noncommittal,
                "similarities": [round(float(value), 4) for value in similarities],
            },
        )

    def precision_breakdown(
        question: str, answer: str, contexts: list[str]
    ) -> MetricBreakdown:
        async def _verify() -> list[Any]:
            results = []
            for context in contexts:
                results.append(
                    await precision.context_precision_prompt.generate(
                        llm=llm,
                        data=QAC(question=question, context=context, answer=answer),
                    )
                )
            return results

        verifications = run(_verify)
        verdicts = [
            {
                "rank": index + 1,
                "verdict": 1 if item.verdict else 0,
                "reason": item.reason,
                "chunk_preview": _preview(context),
            }
            for index, (item, context) in enumerate(zip(verifications, contexts))
        ]
        score = average_precision([int(entry["verdict"]) for entry in verdicts])
        return MetricBreakdown(
            score=score,
            details={"verdicts": verdicts, "average_precision": round(score, 4)},
        )

    return {
        "faithfulness": faithfulness_breakdown,
        "answer_relevancy": relevancy_breakdown,
        "context_precision": precision_breakdown,
    }


@dataclass
class RagasJudge:
    """Scores answers via injected metric callables (real RAGAS or fakes).

    Two injection points, both public so tests can pass fakes:

    ``score_fns``
        ``(question, answer, contexts) -> float``. The score only.
    ``breakdown_fns``
        ``(question, answer, contexts) -> MetricBreakdown``. Score plus the
        intermediates behind it. Preferred when present; ``score_fns`` is the
        fallback if a breakdown raises, so losing the workings never costs the
        score.

    ``samples`` > 1 runs each metric that many times as independent calls and
    reports the mean. DeepSeek's OpenAI-compatible endpoint rejects ``n > 1``,
    so the repeats are separate requests rather than one batched completion.
    """

    score_fns: dict[str, ScoreFn]
    judge_model: str
    embedding_model: str | None = None
    breakdown_fns: dict[str, BreakdownFn] = field(default_factory=dict)
    # The model that wrote the answers, when known. Enables the self-grading
    # flag; ``None`` means unknown, which is reported rather than guessed.
    answer_model: str | None = None
    samples: int = 1

    @classmethod
    def from_settings(cls, settings: Settings) -> "RagasJudge":
        # ragas and its model stack load slowly; keep them out of module import.
        from src.evals import _ragas_compat

        _ragas_compat.install()

        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_openai import ChatOpenAI
        from ragas import SingleTurnSample
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import (
            AnswerRelevancy,
            Faithfulness,
            LLMContextPrecisionWithoutReference,
        )

        model = settings.judge_model or settings.deepseek_model
        llm = LangchainLLMWrapper(
            ChatOpenAI(
                model=model,
                # langchain types api_key as SecretStr but accepts a plain str at runtime.
                api_key=settings.judge_api_key or settings.deepseek_api_key,  # type: ignore[arg-type]
                base_url=settings.judge_base_url or settings.deepseek_base_url,
                temperature=0.0,
            )
        )
        embeddings = LangchainEmbeddingsWrapper(
            HuggingFaceEmbeddings(model_name=settings.embedding_model)
        )

        faithfulness = Faithfulness(llm=llm)
        # strictness controls how many synthetic questions are generated per
        # sample via a single n>1 chat completion; DeepSeek's OpenAI-compatible
        # endpoint rejects n>1, so keep it at 1.
        relevancy = AnswerRelevancy(llm=llm, embeddings=embeddings, strictness=1)
        precision = LLMContextPrecisionWithoutReference(llm=llm)

        def sample(question: str, answer: str, contexts: list[str]) -> Any:
            return SingleTurnSample(
                user_input=question,
                response=answer,
                retrieved_contexts=list(contexts),
            )

        # Kept as the fallback path: if driving the prompts directly ever
        # fails, ragas' own scoring still produces a number (without details).
        score_fns: dict[str, ScoreFn] = {
            "faithfulness": lambda q, a, c: float(
                faithfulness.single_turn_score(sample(q, a, c))
            ),
            "answer_relevancy": lambda q, a, c: float(
                relevancy.single_turn_score(sample(q, a, c))
            ),
            "context_precision": lambda q, a, c: float(
                precision.single_turn_score(sample(q, a, c))
            ),
        }
        return cls(
            score_fns=score_fns,
            judge_model=model,
            embedding_model=settings.embedding_model,
            breakdown_fns=_build_breakdown_fns(
                llm, faithfulness, relevancy, precision
            ),
            answer_model=settings.deepseek_model,
            samples=max(1, settings.judge_samples),
        )

    def _metric_names(self) -> list[str]:
        names = list(self.score_fns)
        names.extend(name for name in self.breakdown_fns if name not in names)
        return names

    def _run_metric(
        self, name: str, question: str, answer: str, contexts: list[str]
    ) -> MetricBreakdown:
        """One sample of one metric, preferring the path that keeps details."""
        breakdown_fn = self.breakdown_fns.get(name)
        score_fn = self.score_fns.get(name)
        if breakdown_fn is not None:
            try:
                return breakdown_fn(question, answer, contexts)
            except Exception as exc:
                if score_fn is None:
                    raise
                # Losing the workings must not lose the score.
                logger.warning(
                    "detail capture failed for %s; scoring without details: %s",
                    name,
                    exc,
                )
        if score_fn is None:
            raise KeyError(f"no scorer registered for {name}")
        return MetricBreakdown(score=score_fn(question, answer, contexts))

    def score(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        answer_model: str | None = None,
    ) -> dict[str, Any]:
        """Run every metric; a failing metric records an error, not a crash.

        With ``samples`` > 1 the reported score is the mean of independent
        runs, ``spread`` is max - min, and ``sample_scores`` lists the runs.
        ``details`` describes the *first* sample, so it reconciles exactly
        with ``sample_scores[metric][0]`` — and, when ``samples`` is 1, with
        ``scores[metric]`` itself.

        ``judge_samples`` is the *requested* sample count, the same for every
        metric. ``sample_counts`` is the per-metric count of attempts that
        actually succeeded (``len(sample_scores[metric])``), which can be
        lower than ``judge_samples`` when some attempts errored.

        ``answer_model`` names the model that wrote ``answer``; it overrides
        the judge's configured default and decides the ``self_graded`` flag.
        """
        started = time.monotonic()
        scores: dict[str, float] = {}
        spread: dict[str, float] = {}
        sample_scores: dict[str, list[float]] = {}
        sample_counts: dict[str, int] = {}
        details: dict[str, dict[str, Any] | None] = {}
        errors: list[str] = []
        samples = max(1, self.samples)

        for name in self._metric_names():
            values: list[float] = []
            captured: dict[str, Any] | None = None
            failure: str | None = None
            for _ in range(samples):
                try:
                    result = self._run_metric(name, question, answer, contexts)
                    value = result.score
                    if value is None or math.isnan(value):
                        raise ValueError("metric returned no score")
                except Exception as exc:
                    failure = failure or str(exc)
                    continue
                if not values:
                    captured = result.details
                values.append(float(value))
            if not values:
                errors.append(f"{name}: {failure or 'metric returned no score'}")
                details[name] = None
                continue
            scores[name] = round(sum(values) / len(values), 4)
            sample_scores[name] = [round(value, 4) for value in values]
            sample_counts[name] = len(values)
            spread[name] = round(max(values) - min(values), 4)
            details[name] = captured

        composite = round(sum(scores.values()) / len(scores), 4) if scores else None
        graded_by = answer_model or self.answer_model
        # All-None details collapse to None, the same null default the other
        # provenance fields use, so records stay clean when nothing was captured.
        captured_any = any(value is not None for value in details.values())
        return {
            "judge": "ragas",
            "judge_model": self.judge_model,
            "rubric_version": RUBRIC_VERSION,
            "ragas_version": ragas_version(),
            "embedding_model": self.embedding_model,
            "scores": scores,
            "composite": composite,
            "spread": spread,
            "sample_scores": sample_scores,
            "sample_counts": sample_counts,
            "judge_samples": samples,
            "details": details if captured_any else None,
            "self_graded": None if not graded_by else graded_by == self.judge_model,
            "elapsed_seconds": round(time.monotonic() - started, 2),
            "scored_at": datetime.now(timezone.utc).isoformat(),
            "error": "; ".join(errors) if errors else None,
        }


def unjudgeable(reason: str, judge_model: str = "") -> dict[str, Any]:
    """An evaluation record for answers that cannot be scored at all."""
    return {
        "judge": "ragas",
        "judge_model": judge_model,
        "rubric_version": RUBRIC_VERSION,
        "ragas_version": ragas_version(),
        "embedding_model": None,
        "scores": {},
        "composite": None,
        "spread": {},
        "sample_scores": {},
        "sample_counts": {},
        "judge_samples": 0,
        "details": None,
        "self_graded": None,
        "elapsed_seconds": 0.0,
        "scored_at": datetime.now(timezone.utc).isoformat(),
        "error": reason,
    }

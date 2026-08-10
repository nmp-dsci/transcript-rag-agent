"""RAGAS answer judging for the web workbench.

Every setup's answer to a question is scored with the same three RAGAS
metrics — faithfulness (is the answer supported by the retrieved chunks?),
answer relevancy (does it address the question?), and context precision
(were the retrieved chunks useful?) — so all retrieval methods are graded
under one eval process.

Those three all measure *grounding*, which is why this module also defines a
second rubric. Under ``ragas-v1`` a faithful one-chunk restatement scores near
1.0 while an answer that synthesises four creators can score lower, because
nothing in the composite rewards depth. ``depth-v2`` keeps the same grounding
metrics at 40% of the composite and spends the other 60% on five LLM-judged
depth metrics, with a hard cap that stops depth rescuing an ungrounded answer.
Both rubrics are live: ``rubric_version`` on a record says which one produced
its composite, and old records keep loading and rendering unchanged.

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

import json
import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Protocol, Sequence

from src.config import Settings

logger = logging.getLogger(__name__)

RUBRIC_VERSION = "ragas-v1"
METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_precision"]

DEPTH_RUBRIC_VERSION = "depth-v2"
#: The five metrics ``depth-v2`` adds. All LLM-judged, all from one call.
DEPTH_METRIC_NAMES = [
    "insight_depth",
    "specificity",
    "coverage",
    "evidence_breadth",
    "calibration",
]

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


# ─── Rubrics ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class MetricGroup:
    """A named block of metrics and the share of the composite it owns."""

    key: str
    label: str
    weight: float
    metrics: tuple[str, ...]


@dataclass(frozen=True)
class CompositeResult:
    """One composite, plus everything needed to explain how it got there.

    Two separate facts, deliberately not collapsed into one:

    ``grounding_floor_breached``
        The answer failed the rubric's grounding floor — faithfulness came in
        below it, *or* faithfulness could not be scored at all. True regardless
        of what the composite ended up being.
    ``cap_applied``
        The cap actually *lowered* the number. Only true when the answer would
        otherwise have scored above ``capped_maximum``.

    Reporting only the second is how the worst answers escape notice: an answer
    with faithfulness 0.00 breaches the floor but scores under the cap on its
    own, so a "capped" badge alone would silently mean "ungrounded *and*
    otherwise good" while the flatly ungrounded answer beside it looked clean.

    ``uncapped`` is the weighted sum before the cap; when ``cap_applied`` is
    false it equals ``composite``.
    """

    composite: float | None
    uncapped: float | None
    weight_used: float
    cap_applied: bool
    cap_reason: str | None
    grounding_floor_breached: bool = False
    grounding_reason: str | None = None


def _below_floor(value: float, floor: float) -> str:
    """``value`` at the shortest precision that still reads as below ``floor``.

    Two decimals is the house style and is right for almost every score, but a
    faithfulness of 0.5999 rounds to "0.60" and the sentence then reads
    *"faithfulness 0.60 below 0.6"* — a cap reason that contradicts itself, on
    exactly the cells a reader is most likely to be checking. So the precision
    widens until the printed number is what the claim says it is; only when no
    precision can do that (a value that is not actually below the floor) does
    it fall back, and the caller never reaches here with one.
    """
    for places in (2, 3, 4):
        text = f"{value:.{places}f}"
        if float(text) < floor:
            return text
    return repr(value)


@dataclass(frozen=True)
class Rubric:
    """How a set of metric scores becomes one composite.

    ``weights`` must sum to 1.0 over ``metrics`` — a composite is only readable
    as "out of 1" if it is a true weighted average. When a metric is missing
    (its judge call failed), the composite is renormalised over the weights
    that *were* scored rather than treating the gap as a zero, which would
    punish an answer for the judge's failure.
    """

    version: str
    metrics: tuple[str, ...]
    weights: Mapping[str, float]
    groups: tuple[MetricGroup, ...]
    composite_description: str
    #: Below this faithfulness the composite is capped. ``None`` disables it.
    faithfulness_floor: float | None = None
    capped_maximum: float = 0.5

    def composite(self, scores: Mapping[str, Any]) -> CompositeResult:
        """The weighted composite of ``scores``, with the grounding cap applied.

        Renormalising over the weights that were actually scored is right for a
        depth metric whose judge call failed — that is missing data, not a
        zero. It is *wrong* for faithfulness, because faithfulness is the one
        metric the cap keys on: renormalising it away removes both the score
        and the check on it, and an answer whose grounding call errored could
        then composite to 1.00 with nothing able to stop it. That is the exact
        case the cap exists to prevent, so an unverifiable faithfulness
        breaches the floor rather than dropping out of it.
        """
        weighted = 0.0
        weight_used = 0.0
        for metric in self.metrics:
            value = scores.get(metric)
            if not isinstance(value, (int, float)) or math.isnan(float(value)):
                continue
            weight = self.weights[metric]
            weighted += float(value) * weight
            weight_used += weight
        if weight_used <= 0:
            return CompositeResult(None, None, 0.0, False, None)

        uncapped = round(weighted / weight_used, 4)
        floor = self.faithfulness_floor
        if floor is None:
            return CompositeResult(uncapped, uncapped, round(weight_used, 4), False, None)

        raw = scores.get("faithfulness")
        faithfulness: float | None = None
        if isinstance(raw, (int, float)) and not math.isnan(float(raw)):
            faithfulness = float(raw)
        if faithfulness is not None and faithfulness >= floor:
            return CompositeResult(uncapped, uncapped, round(weight_used, 4), False, None)

        reason = (
            f"faithfulness {_below_floor(faithfulness, floor)} below {floor} — "
            "depth cannot rescue an ungrounded answer"
            if faithfulness is not None
            else (
                "faithfulness could not be scored — grounding is unverified, "
                "so depth cannot certify this answer"
            )
        )
        composite = min(uncapped, self.capped_maximum)
        return CompositeResult(
            composite=composite,
            uncapped=uncapped,
            weight_used=round(weight_used, 4),
            # The cap only "applied" where it changed the number. The breach
            # below is what is always true of this answer.
            cap_applied=composite < uncapped,
            cap_reason=reason if composite < uncapped else None,
            grounding_floor_breached=True,
            grounding_reason=reason,
        )


#: The original rubric: three grounding metrics, equally weighted. Expressed as
#: weights rather than a mean so both rubrics go through the same arithmetic and
#: a ``ragas-v1`` record keeps reproducing the number it was committed with.
RAGAS_V1 = Rubric(
    version=RUBRIC_VERSION,
    metrics=tuple(METRIC_NAMES),
    weights={name: 1 / 3 for name in METRIC_NAMES},
    groups=(),
    composite_description="mean of the metric scores",
)

#: Grounding is 40% of ``depth-v2`` and depth is 60%. Faithfulness carries the
#: most grounding weight because it is the one metric the cap keys on; insight
#: depth carries the most depth weight because "did this synthesise or just
#: restate one chunk?" is the question the whole rubric exists to ask.
DEPTH_V2 = Rubric(
    version=DEPTH_RUBRIC_VERSION,
    metrics=(
        "faithfulness",
        "context_precision",
        "answer_relevancy",
        "insight_depth",
        "specificity",
        "coverage",
        "evidence_breadth",
        "calibration",
    ),
    weights={
        "faithfulness": 0.20,
        "context_precision": 0.10,
        "answer_relevancy": 0.10,
        "insight_depth": 0.20,
        "specificity": 0.15,
        "coverage": 0.10,
        "evidence_breadth": 0.10,
        "calibration": 0.05,
    },
    groups=(
        MetricGroup(
            key="grounding",
            label="grounding",
            weight=0.40,
            metrics=("faithfulness", "context_precision", "answer_relevancy"),
        ),
        MetricGroup(
            key="depth",
            label="depth",
            weight=0.60,
            metrics=(
                "insight_depth",
                "specificity",
                "coverage",
                "evidence_breadth",
                "calibration",
            ),
        ),
    ),
    composite_description=(
        "weighted sum — grounding 40%, depth 60%; capped at 0.5 when faithfulness < 0.6"
    ),
    faithfulness_floor=0.6,
    capped_maximum=0.5,
)

RUBRICS: dict[str, Rubric] = {rubric.version: rubric for rubric in (RAGAS_V1, DEPTH_V2)}


def rubric_for(version: str | None) -> Rubric:
    """The named rubric, falling back to ``ragas-v1``.

    Records written before ``rubric_version`` existed, and records naming a
    rubric this build does not know, are read as the original three-metric
    rubric — the shape their scores are actually in.
    """
    return RUBRICS.get(version or RUBRIC_VERSION, RAGAS_V1)


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
        (sum(verdict_list[: i + 1]) / (i + 1)) * verdict_list[i] for i in range(len(verdict_list))
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

    def faithfulness_breakdown(question: str, answer: str, contexts: list[str]) -> MetricBreakdown:
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

    def relevancy_breakdown(question: str, answer: str, contexts: list[str]) -> MetricBreakdown:
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

    def precision_breakdown(question: str, answer: str, contexts: list[str]) -> MetricBreakdown:
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


# ─── Depth metrics (depth-v2) ────────────────────────────────────────────────

#: One call scores all five. Five separate calls would re-read the same answer
#: and the same contexts five times for no extra signal, and the metrics are
#: deliberately relative to each other — "deep but vague" is a judgement made
#: once, not assembled from five independent readings.
DEPTH_JUDGE_SYSTEM_PROMPT = """You grade how much an answer is WORTH, given the transcript \
chunks it was built from.

Grounding is graded elsewhere. Your job is the quality a grounded answer can \
still lack: an answer that faithfully restates one chunk is accurate and \
shallow, and it must not score like one that reads four creators and says \
something none of them said alone.

Score each metric from 0.0 to 1.0:

- insight_depth — does the answer synthesise, or restate? 1.0 = it connects \
claims across sources, explains a mechanism, or resolves a tension into a \
conclusion the chunks only imply. 0.5 = it organises several points but adds \
no reasoning. 0.0 = a paraphrase of a single passage.
- specificity — is it checkable? 1.0 = named schemes, figures, thresholds, \
dates, people, positions attributed to whoever held them. 0.0 = advice that \
would read the same against any transcript on the topic.
- coverage — of what the context actually offers on this question, how much \
does the answer use? 1.0 = every substantive facet in the chunks is \
represented. 0.0 = one facet out of many available.
- evidence_breadth — how many DISTINCT sources or speakers the answer really \
draws on, relative to how many the context offered. 1.0 = it uses the range \
available and makes clear who said what. 0.0 = one source, or unattributed.
- calibration — does confidence match evidence? 1.0 = it asserts what the \
chunks support, flags where they disagree or are thin, and marks predictions \
as predictions. 0.0 = flat assertions the context does not carry, or hedging \
so total it commits to nothing.

Rules:
- Judge only the answer and the context given. Never use outside knowledge.
- Length is not depth. A long, repetitive answer scores low on insight_depth.
- If the context is thin, an answer that says so scores WELL on calibration.
- Each rationale is ONE sentence naming the specific thing that decided the score.

Return JSON only, with this exact shape and no other keys:
{
  "insight_depth": {"score": 0.0, "rationale": "one sentence"},
  "specificity": {"score": 0.0, "rationale": "one sentence"},
  "coverage": {"score": 0.0, "rationale": "one sentence"},
  "evidence_breadth": {"score": 0.0, "rationale": "one sentence"},
  "calibration": {"score": 0.0, "rationale": "one sentence"}
}
"""

DEPTH_JUDGE_USER_PROMPT = """Question:
{question}

Retrieved context ({context_count} chunk(s)):
{contexts_block}

Answer under review:
{answer}
"""

#: Per-chunk and total budgets for the context block. A depth judgement needs
#: to see the breadth of what was retrieved more than the full text of any one
#: chunk, so chunks are truncated individually rather than the list being cut
#: short — dropping chunks would silently lower ``evidence_breadth``.
DEPTH_CONTEXT_CHARS = 1200
DEPTH_CONTEXT_TOTAL_CHARS = 14000


def build_depth_context_block(contexts: Sequence[str]) -> str:
    """The numbered context block the depth prompt reads."""
    if not contexts:
        return "(no retrieved context was recorded for this answer)"
    lines: list[str] = []
    used = 0
    for index, context in enumerate(contexts, start=1):
        text = (context or "").strip()[:DEPTH_CONTEXT_CHARS]
        if used + len(text) > DEPTH_CONTEXT_TOTAL_CHARS:
            lines.append(f"[{index}] (omitted — context budget reached)")
            continue
        used += len(text)
        lines.append(f"[{index}] {text}")
    return "\n\n".join(lines)


def parse_depth_response(content: str) -> dict[str, MetricBreakdown]:
    """Parse one depth-judge response into a breakdown per depth metric.

    Raises ``ValueError`` on anything that does not satisfy the contract, so a
    malformed response fails the cell rather than being averaged in as a
    plausible-looking zero.
    """
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"depth response was not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("depth response must be a JSON object")

    breakdowns: dict[str, MetricBreakdown] = {}
    for metric in DEPTH_METRIC_NAMES:
        entry = data.get(metric)
        if not isinstance(entry, dict):
            raise ValueError(f"depth response is missing {metric!r}")
        raw = entry.get("score")
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError(f"depth response gave {metric!r} a non-numeric score")
        score = float(raw)
        if math.isnan(score):
            raise ValueError(f"depth response gave {metric!r} no score")
        # Clamped rather than rejected: a judge that answers 1.2 has ranked the
        # answer at the top of the scale, which is usable; only an unparseable
        # response is a failure.
        score = min(1.0, max(0.0, score))
        rationale = str(entry.get("rationale") or "").strip()
        breakdowns[metric] = MetricBreakdown(
            score=round(score, 4),
            details={"score": round(score, 4), "rationale": rationale},
        )
    return breakdowns


class ChatModel(Protocol):
    """The one method the depth judge needs — ``ChatOpenAI`` or a test fake.

    Positional-only so the parameter *name* is not part of the contract:
    ``ChatOpenAI.invoke`` calls its first argument ``input``.
    """

    def invoke(self, messages: Any, /, *args: Any, **kwargs: Any) -> Any: ...


@dataclass
class DepthJudge:
    """The five ``depth-v2`` metrics, from one structured call per answer.

    The client is built through :func:`~src.agents.llm.chat_model_kwargs` like
    every other LLM path here, so it inherits the read timeout: a judge call
    that hangs must fail its cell rather than stall a whole rejudge run.
    """

    llm: ChatModel
    judge_model: str
    #: One retry, because a single malformed JSON body is the common failure and
    #: re-asking is cheaper than losing the cell.
    retries: int = 1

    @classmethod
    def from_settings(cls, settings: Settings) -> "DepthJudge":
        from langchain_openai import ChatOpenAI

        from src.agents.llm import chat_model_kwargs

        model = settings.judge_model or settings.deepseek_model
        kwargs = chat_model_kwargs(
            settings,
            model=model,
            api_key=settings.judge_api_key or settings.deepseek_api_key,
            temperature=0.0,
        )
        if settings.judge_base_url:
            kwargs["base_url"] = settings.judge_base_url
        return cls(llm=ChatOpenAI(**kwargs), judge_model=model)

    def score(
        self, question: str, answer: str, contexts: Sequence[str]
    ) -> dict[str, MetricBreakdown]:
        """All five depth metrics for one answer, each with its rationale."""
        prompt = DEPTH_JUDGE_USER_PROMPT.format(
            question=question,
            context_count=len(contexts),
            contexts_block=build_depth_context_block(contexts),
            answer=answer,
        )
        messages = [
            {"role": "system", "content": DEPTH_JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        last_error: Exception | None = None
        for _ in range(max(1, self.retries + 1)):
            try:
                response = self.llm.invoke(messages)
                return parse_depth_response(str(getattr(response, "content", response)))
            except Exception as exc:
                last_error = exc
                logger.warning("depth judging attempt failed: %s", exc)
        raise RuntimeError(f"depth judging failed: {last_error}")


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
    #: Which rubric turns the metric scores into a composite. ``ragas-v1`` here;
    #: ``depth-v2`` composites are produced by the rejudge path, which reuses
    #: these stored grounding scores rather than re-deriving them.
    rubric: Rubric = RAGAS_V1

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
                # Without a read timeout a socket that is accepted and then never
                # answered blocks forever inside the request. That is not
                # theoretical here: it is what made a judged matrix run sail past
                # its own --max-seconds budget and have to be killed. A judge call
                # that has produced nothing in this long is not going to, so fail
                # it and let the cell be retried rather than hang the run.
                timeout=settings.llm_timeout_seconds,
                max_retries=2,
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
            "faithfulness": lambda q, a, c: float(faithfulness.single_turn_score(sample(q, a, c))),
            "answer_relevancy": lambda q, a, c: float(relevancy.single_turn_score(sample(q, a, c))),
            "context_precision": lambda q, a, c: float(
                precision.single_turn_score(sample(q, a, c))
            ),
        }
        return cls(
            score_fns=score_fns,
            judge_model=model,
            embedding_model=settings.embedding_model,
            breakdown_fns=_build_breakdown_fns(llm, faithfulness, relevancy, precision),
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

        composited = self.rubric.composite(scores)
        graded_by = answer_model or self.answer_model
        # All-None details collapse to None, the same null default the other
        # provenance fields use, so records stay clean when nothing was captured.
        captured_any = any(value is not None for value in details.values())
        return {
            "judge": "ragas",
            "judge_model": self.judge_model,
            "rubric_version": self.rubric.version,
            "ragas_version": ragas_version(),
            "embedding_model": self.embedding_model,
            "scores": scores,
            "composite": composited.composite,
            "cap_applied": composited.cap_applied,
            "cap_reason": composited.cap_reason,
            "grounding_floor_breached": composited.grounding_floor_breached,
            "grounding_reason": composited.grounding_reason,
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
        "cap_applied": False,
        "cap_reason": None,
        "grounding_floor_breached": False,
        "grounding_reason": None,
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

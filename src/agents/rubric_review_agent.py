"""The reviewer that judges a document against the shipped rubric packs.

One LLM call per pack, never one call for all of them. Sixty-one rubrics and a
nine-section page in a single prompt is a request for the model to skim, and
what it skims is the tail — which is exactly where the packs it thinks are less
relevant sit. Per pack, the model sees twelve to seventeen rules and is asked
for twelve to seventeen verdicts, and the count it has to return is small enough
that missing one is visible to it.

There is no retrieval in this path at all, and that is the design. The criteria
were retrieved once, at pack build time, over the whole corpus with the
disagreement and creator-coverage gates applied — so a review does not re-run a
top-k and hope the ten chunks it lands on happen to be the right rules. What
that buys is stated plainly in the run report rather than hidden: the pack
reviewer read the corpus through a build the chunk-dump baseline never had.

Failure is per pack. A pack whose call times out leaves its rubrics as
``unjudged`` with the error on the pack row, and the other three packs still
produce a review — a partial review that says which part is missing is worth
more than an exception.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

from src.agents.models import TraceStep
from src.agents.prompts import RUBRIC_REVIEW_SYSTEM_PROMPT, build_rubric_review_prompt
from src.config import Settings
from src.documents.models import Document
from src.documents.review import SectionSelection, classify_document, format_document_context
from src.documents.rubric_review import (
    PackOutcome,
    RubricReview,
    build_answer,
    build_references,
    format_rubrics,
    load_review_packs,
    parse_pack_verdicts,
    sort_verdicts,
)


class ChatModel(Protocol):
    def invoke(self, messages: list) -> object: ...


ProgressFn = Callable[[str], None]

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def _json_object(text: str) -> dict[str, Any]:
    """The JSON object in a model reply, fenced or not.

    Raises rather than returning ``{}``: an empty object here would be parsed
    into "the model skipped every rubric in this pack", which reads in the UI as
    a reviewer that considered them and declined, when what actually happened is
    that the reply was unusable. The pack row says ``error`` instead.
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z]*\n?", "", stripped)
        stripped = re.sub(r"\n?```$", "", stripped).strip()
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        match = _JSON_BLOCK.search(stripped)
        if match is None:
            raise ValueError(f"Rubric review reply was not JSON: {text[:200]}")
        parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("Rubric review reply must be a JSON object")
    return parsed


class RubricReviewResult:
    """A finished review, plus the shape the chat runner needs from it."""

    def __init__(self, review: RubricReview, trace: list[TraceStep], llm_calls: int) -> None:
        self.review = review
        self.trace = trace
        self.llm_calls = llm_calls
        self.verdicts = sort_verdicts(review.verdicts)
        self.references = build_references(self.verdicts)
        self.answer = build_answer(review)


class RubricReviewAgent:
    """Judge a document against every shipped pack, one call each."""

    def __init__(
        self,
        llm: ChatModel,
        packs: Sequence[Any],
        model_name: str = "",
    ) -> None:
        self.llm = llm
        self.packs = list(packs)
        self.model_name = model_name

    @classmethod
    def from_settings(
        cls, settings: Settings, packs_dir: Path | str | None = None
    ) -> "RubricReviewAgent":
        from langchain_openai import ChatOpenAI

        from src.agents.llm import chat_model_kwargs

        return cls(
            ChatOpenAI(**chat_model_kwargs(settings)),
            load_review_packs(packs_dir),
            model_name=settings.deepseek_model,
        )

    def review(
        self,
        document: Document,
        selection: SectionSelection,
        *,
        on_progress: ProgressFn | None = None,
    ) -> RubricReviewResult:
        if not self.packs:
            raise ValueError(
                "No rubric packs are built, so there is nothing to review against. "
                "Run `uv run python -m src.cli build-packs` first."
            )
        kind = classify_document(document)
        context = format_document_context(document, selection)
        section_indices = [section.index for section in selection.sections]
        review = RubricReview(
            document_id=document.id,
            document_url=document.url,
            document_kind=kind,
        )
        trace: list[TraceStep] = [
            TraceStep(
                phase="route",
                label="Load rubric packs",
                detail=(
                    f"{len(self.packs)} packs, "
                    f"{sum(len(pack.rubrics) for pack in self.packs)} rubrics — "
                    "criteria built over the whole corpus, not retrieved per question"
                ),
            ),
            TraceStep(
                phase="route",
                label="Read the document",
                detail=selection.detail(),
            ),
        ]
        calls = 0
        for pack in self.packs:
            if on_progress is not None:
                on_progress(f"Applying {pack.name} ({len(pack.rubrics)} rubrics)…")
            started = time.monotonic()
            prompt = build_rubric_review_prompt(context, format_rubrics(pack), len(pack.rubrics))
            try:
                from langchain_core.messages import HumanMessage, SystemMessage

                response = self.llm.invoke(
                    [
                        SystemMessage(content=RUBRIC_REVIEW_SYSTEM_PROMPT),
                        HumanMessage(content=prompt),
                    ]
                )
                calls += 1
                payload = _json_object(str(getattr(response, "content", response)))
            except Exception as exc:  # noqa: BLE001 - a failed pack is a reported row
                verdicts, outcome = parse_pack_verdicts({}, pack, section_indices)
                outcome.error = f"{type(exc).__name__}: {exc}"
                outcome.elapsed_seconds = time.monotonic() - started
                review.verdicts.extend(verdicts)
                review.packs.append(outcome)
                trace.append(_pack_step(outcome, verdicts))
                continue
            verdicts, outcome = parse_pack_verdicts(payload, pack, section_indices)
            outcome.elapsed_seconds = time.monotonic() - started
            review.verdicts.extend(verdicts)
            review.packs.append(outcome)
            trace.append(_pack_step(outcome, verdicts))
        return RubricReviewResult(review, trace, calls)


def _pack_step(outcome: PackOutcome, verdicts: Sequence[Any]) -> TraceStep:
    """One pack's row in the trace: what it decided, and what it could not.

    The rejections are in ``note`` rather than ``detail`` because the trace row
    clips ``detail`` at roughly a quarter of its length, and the quarter that
    survives has to be the counts.
    """
    if outcome.error:
        return TraceStep(
            phase="llm",
            label=f"{outcome.name} — failed",
            detail=f"{outcome.rubrics} rubrics unjudged",
            note=outcome.error,
            elapsed_ms=int(outcome.elapsed_seconds * 1000),
        )
    counts: dict[str, int] = {}
    for verdict in verdicts:
        counts[verdict.verdict] = counts.get(verdict.verdict, 0) + 1
    rejected = []
    if outcome.unanchored_failures:
        rejected.append(
            f"{len(outcome.unanchored_failures)} failure(s) named no section and were "
            f"not counted ({', '.join(outcome.unanchored_failures)})"
        )
    if outcome.unknown_rubric_ids:
        rejected.append(f"{len(outcome.unknown_rubric_ids)} unknown rubric id(s) dropped")
    if outcome.duplicate_rubric_ids:
        rejected.append(f"{len(outcome.duplicate_rubric_ids)} duplicate row(s) dropped")
    if outcome.missing_rubric_ids:
        rejected.append(f"{len(outcome.missing_rubric_ids)} rubric(s) got no verdict")
    return TraceStep(
        phase="llm",
        label=f"Apply {outcome.name}",
        detail=(
            f"{outcome.rubrics} rubrics — {counts.get('fail', 0)} fail, "
            f"{counts.get('pass', 0)} pass, {counts.get('n-a', 0)} n/a"
        ),
        note="; ".join(rejected) or None,
        elapsed_ms=int(outcome.elapsed_seconds * 1000),
    )

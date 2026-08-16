"""The corpus-wide question detector, and the near-misses it must not fire on.

The negatives matter more than the positives here. Lifting the filter's cap on
a genuinely local question is a mild degradation, but a detector that fires on
"every step of the workflow" or "across Australia's property markets" would
mean the cap effectively never applies, which is the filter turned off by
accident rather than by decision.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.evals.golden import DEFAULT_DATASET_PATH
from src.rag.question_scope import (
    CORPUS_WIDE_BEHAVIORS,
    DEFAULT_CORPUS_WIDE_BEHAVIOR,
    LEGACY_CORPUS_WIDE_BEHAVIOR,
    corpus_wide_behavior,
    corpus_wide_signal,
    is_corpus_wide_question,
)


@pytest.mark.parametrize(
    ("question", "signal"),
    [
        ("What are the main themes across this corpus of videos?", "corpus"),
        ("According to this corpus, how do senior engineers work?", "corpus"),
        ("What do all the videos say about negative gearing?", "all-sources"),
        ("What point does every transcript make about ATS?", "all-sources"),
        ("What are the recurring arguments the property videos make?", "all-sources"),
        ("What themes run throughout the transcripts?", "all-sources"),
        ("What advice is repeated across the episodes?", "all-sources"),
        ("What point comes up across your channels?", "across-sources"),
        ("What are the main themes here?", "aggregate"),
        ("Where do the speakers reach consensus on agentic coding?", "aggregate"),
        ("How did the stance on the budget evolve?", "over-time"),
        ("How has advice on resumes changed over the last year?", "over-time"),
    ],
)
def test_corpus_wide_questions_are_detected(question: str, signal: str) -> None:
    assert corpus_wide_signal(question) == signal


@pytest.mark.parametrize(
    "question",
    [
        # A quantifier over something that is not a source.
        "Describe the Agentic Engineering Workflow: every step and what it leverages.",
        "What are all the steps in the ingestion pipeline?",
        # "across" over something that is not a source.
        (
            "Summarise the key insights from the Suburb Data episode on what is "
            "happening across Australia's property markets."
        ),
        "How do prices vary across the eastern states?",
        # Ordinary specific questions.
        "How do I make my resume ATS-friendly?",
        "What should the professional summary at the top of a resume say?",
        "How should I prepare for behavioural interview questions?",
        "What are the key tax changes affecting property investors right now?",
        "Is the Gold Coast property market at risk of collapse, and why?",
        "How do I set up Herder, and what app do I run it in?",
        "What does this video say about negative gearing?",
    ],
)
def test_specific_questions_are_not_detected(question: str) -> None:
    assert corpus_wide_signal(question) is None
    assert not is_corpus_wide_question(question)


def test_empty_question_is_not_corpus_wide() -> None:
    assert corpus_wide_signal("") is None


def test_detector_agrees_with_the_golden_sets_own_labels() -> None:
    """``global``/``temporal`` entries are corpus-wide; ``local`` ones are not.

    Not a held-out measurement — the patterns were written with this file open,
    and it is twenty questions. It is here so that *editing* the golden set, or
    the patterns, surfaces a disagreement instead of quietly changing which
    questions get the wider filter.
    """
    entries = json.loads(Path(DEFAULT_DATASET_PATH).read_text(encoding="utf-8"))["entries"]
    disagreements = [
        (entry["question"], entry.get("question_type"), corpus_wide_signal(entry["question"]))
        for entry in entries
        if is_corpus_wide_question(entry["question"])
        != (entry.get("question_type") in {"global", "temporal"})
    ]

    assert disagreements == []


# --- the behaviour this detection drives, and its version --------------------


def test_the_shipped_names_are_pinned() -> None:
    """Growing the registry must be a deliberate edit, never a drive-by.

    Adding a name changes the eval cache's identity for every corpus-wide
    filtered cell — see :func:`src.evals.matrix_cache.behavior_material` — so
    this test exists to make that cost land on whoever adds one, in a diff that
    says so, rather than on whoever next runs the matrix.
    """
    assert set(CORPUS_WIDE_BEHAVIORS) == {"capped", "budget", "budget+diversity"}
    assert LEGACY_CORPUS_WIDE_BEHAVIOR == "capped"
    assert DEFAULT_CORPUS_WIDE_BEHAVIOR == "budget"


def test_every_registered_name_explains_itself() -> None:
    """A version nobody wrote down the meaning of is a number, not a version."""
    for name, description in CORPUS_WIDE_BEHAVIORS.items():
        assert description.strip(), name
        assert len(description) > 40, name


def test_the_behaviour_name_is_derived_from_the_switches() -> None:
    assert corpus_wide_behavior(cap_lifted=True, max_chunks_per_video=0) == "budget"
    assert corpus_wide_behavior(cap_lifted=True, max_chunks_per_video=2) == "budget+diversity"


def test_switching_the_corpus_wide_path_off_reports_the_historical_behaviour() -> None:
    """``corpus_wide_filter=False`` is the pre-change behaviour end to end, so
    the diversity cap it disables must not appear in the name."""
    assert corpus_wide_behavior(cap_lifted=False, max_chunks_per_video=0) == "capped"
    assert corpus_wide_behavior(cap_lifted=False, max_chunks_per_video=4) == "capped"


def test_every_derivable_name_is_registered() -> None:
    """A name the code can produce but the registry does not know would raise
    inside the fingerprint at the moment a run needed it."""
    derived = {
        corpus_wide_behavior(cap_lifted=lifted, max_chunks_per_video=cap)
        for lifted in (True, False)
        for cap in (0, 1, 3)
    }
    assert derived <= set(CORPUS_WIDE_BEHAVIORS)

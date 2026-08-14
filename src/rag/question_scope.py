"""Does a question ask about the corpus, or about something inside it?

The per-video summary filter routes a question to its ``top_k`` most relevant
videos before any chunk is searched. That cap is right for a question whose
answer lives in a few videos and wrong for one whose answer *is* the spread —
"what are the main themes across this corpus?" has no top 5, and answering it
from five videos produces a confident answer about a fifth of the evidence.

This module is the detection half of that. It is deliberately a deterministic
lexical rule, not an LLM classifier:

* it runs in front of every filtered retrieval, so it must not add an LLM call
  and a failure mode to the hot path;
* it must be testable without a provider, which matters more than usual here —
  the project's LLM credit is exhausted and an LLM-based detector could not be
  exercised at all right now;
* and a rule a reader can check by eye is auditable in a way a classifier's
  judgement is not. Every signal below can be traced back to the words that
  triggered it, which is why :func:`corpus_wide_signal` returns *which* family
  matched rather than a bare bool.

**On calibration.** These patterns were written with the twenty labelled
questions in ``src/evals/golden_dataset.json`` visible, and they classify all
twenty as that file labels them (four ``global`` and two ``temporal`` detected,
fourteen local/specific not detected). That is a sanity check, not a
measurement: twenty questions the author has read are not a held-out set, and
the honest number would come from unseen questions. What the rule is designed
for instead is a *benign* failure mode — see
:func:`~src.rag.context.MultiTranscriptRagContextProvider.get_context`, where a
false positive only lifts a cap and leaves the relevance threshold doing the
work.

**On versioning.** What the retriever *does* with a detected question is not
fixed, and every change to it is invisible to a config-field-based cache
fingerprint — the decision is derived from the question at query time, so no
setting moves when the behaviour does. :data:`CORPUS_WIDE_BEHAVIORS` at the
bottom of this module is that behaviour's identity, and
:func:`corpus_wide_behavior` derives it from a provider's switches so nothing
has to be kept in step by hand.
"""

from __future__ import annotations

import re

#: Nouns that name a *source* in this corpus. The scope patterns below only fire
#: on these, which is the whole reason "every **step** of the workflow" and
#: "across Australia's property **markets**" are not corpus-wide questions: both
#: carry a quantifier and a preposition that look global and neither is asking
#: about the videos.
_SOURCE_NOUN = r"(?:videos?|transcripts?|episodes?|channels?|creators?|speakers?|corpus|corpora)"

#: Ordered so the most specific explanation wins when several match.
_SIGNALS: tuple[tuple[str, re.Pattern[str]], ...] = (
    # The corpus named outright. No local question in this project's golden set
    # says "corpus", and one that does is asking about the whole of it.
    ("corpus", re.compile(r"\bcorp(?:us|ora)\b", re.IGNORECASE)),
    # A set of sources, quantified. "all/every/each/these/both the videos", and
    # "the property videos" / "the job-search videos" — a plural source noun
    # with a qualifier in front of it is a question about a group, not one item.
    (
        "all-sources",
        re.compile(
            rf"\b(?:all|every|each|both|these|those)\s+(?:of\s+)?(?:the\s+)?(?:\w+[- ])?{_SOURCE_NOUN}\b"
            rf"|\bthe\s+(?:\w+-\w+|\w+)\s+(?:videos|transcripts|episodes|channels)\b"
            rf"|\bthe\s+(?:videos|transcripts|episodes|channels)\b",
            re.IGNORECASE,
        ),
    ),
    # Spanning language, but only when what is spanned is a source. "across this
    # corpus", "throughout the videos" — never a bare "across", which is how
    # "across Australia's property markets" stays local.
    (
        "across-sources",
        re.compile(
            rf"\b(?:across|throughout|among|amongst|between|within|in)\s+"
            rf"(?:all\s+|both\s+|these\s+|those\s+|this\s+|the\s+|your\s+|our\s+)*{_SOURCE_NOUN}\b",
            re.IGNORECASE,
        ),
    ),
    # Aggregation over sources: the answer is a pattern, not a passage.
    (
        "aggregate",
        re.compile(
            r"\b(?:recurring|recurrent|repeatedly|in\s+common|common\s+(?:themes?|threads?|advice|ground)"
            r"|(?:main|key|central|overall|broad|overarching)\s+(?:themes?|threads?|takeaways?)"
            r"|consensus|(?:generally|widely)\s+agree|across\s+the\s+board"
            r"|(?:repeat|agree|disagree|converge|differ)s?\s+(?:on|about|across)"
            r"|how\s+often)\b",
            re.IGNORECASE,
        ),
    ),
    # Change over time. The golden set calls these ``temporal`` rather than
    # ``global``, but they span the corpus for the same reason: no five videos
    # are "the" answer to how a position moved.
    (
        "over-time",
        re.compile(
            r"\b(?:evolve[ds]?|evolving|evolution|over\s+time|change[ds]?\s+over"
            r"|shift(?:ed|s)?\s+(?:over|between|across)|trend(?:s|ed)?\s+(?:over|across)"
            r"|develop(?:ed|s)?\s+over)\b",
            re.IGNORECASE,
        ),
    ),
)


def corpus_wide_signal(question: str) -> str | None:
    """The name of the first signal marking ``question`` as corpus-wide, or ``None``.

    Returned as a name rather than a bool so the retrieval trace can say *why*
    the filter behaved differently — "corpus-wide (corpus)" is checkable against
    the question text; "corpus-wide" is a claim a reader has to take on trust.
    """
    text = question or ""
    for name, pattern in _SIGNALS:
        if pattern.search(text):
            return name
    return None


def is_corpus_wide_question(question: str) -> bool:
    """Whether ``question`` asks about the corpus as a whole."""
    return corpus_wide_signal(question) is not None


# ── the behaviour this detection drives, and its version ──────────────────────
#
# Detection is only half of the corpus-wide path; the other half is what the
# retriever *does* once a question is detected, and that half has now changed
# twice. Neither change moves a configuration field, so neither is visible to
# :func:`src.evals.matrix_cache.cell_fingerprint` — a cell scored under one
# behaviour would be handed back for a run of another, and a matrix column would
# silently mix them. The names below are the fingerprintable identity of that
# behaviour: a bump is what makes the change visible to the cache, and the
# registry is what makes an unrecognised name a loud failure instead of a
# silent match.

#: Every corpus-wide retrieval behaviour this project has shipped, oldest first,
#: each mapped to what it does. Add a name here whenever the corpus-wide path's
#: *behaviour* changes — not when its wording, its patterns' formatting, or its
#: comments do. The test suite pins the set of names, so growing it is a
#: deliberate edit and never a drive-by.
#:
#: ``capped`` is the historical behaviour and is therefore the one every cell in
#: the existing eval cache was scored under. That is why it is named rather than
#: merely implied: "the behaviour before we started tracking behaviour" has to
#: be a value you can write down, or the cache cannot say what it holds.
CORPUS_WIDE_BEHAVIORS: dict[str, str] = {
    "capped": (
        "The summary filter's top_k is a hard cap for every question, corpus-wide "
        "or not. Shipped until 2026-08; every cell in the pre-existing eval cache "
        "was scored under it."
    ),
    "budget": (
        "top_k is a budget: a question the corpus-wide detector fires on keeps "
        "min_score and lifts the cap to the number of summarised videos."
    ),
    "budget+diversity": (
        "budget, plus a per-video cap on how many chunks one video may contribute "
        "to the final top_k — so lifting the *video* cap can actually widen the "
        "*chunk* budget that bounds source breadth."
    ),
}

#: What the cache's untracked history was scored under. A cell fingerprinted
#: under this name is indistinguishable from one fingerprinted before behaviour
#: was tracked at all, which is the point: the pre-change arm of an A/B comes
#: back from the cache for free instead of costing a re-score.
LEGACY_CORPUS_WIDE_BEHAVIOR = "capped"

#: What a default-constructed provider implements today.
DEFAULT_CORPUS_WIDE_BEHAVIOR = "budget"


def corpus_wide_behavior(*, cap_lifted: bool, max_chunks_per_video: int = 0) -> str:
    """Name the corpus-wide behaviour a provider configuration implements.

    Derived from the switches rather than declared beside them, so a provider
    can report what it actually does and a caller never has to remember to keep
    a label in step with a flag. That is the whole difference between a version
    that describes the code and one that describes a guess about the code.

    ``cap_lifted`` is ``corpus_wide_filter``: off means the corpus-wide path is
    disabled outright, which is the pre-change behaviour end to end, so the
    diversity cap is not reported (it cannot fire — see
    :meth:`~src.rag.context.MultiTranscriptRagContextProvider._refine`).
    """
    if not cap_lifted:
        return LEGACY_CORPUS_WIDE_BEHAVIOR
    if max_chunks_per_video > 0:
        return "budget+diversity"
    return DEFAULT_CORPUS_WIDE_BEHAVIOR

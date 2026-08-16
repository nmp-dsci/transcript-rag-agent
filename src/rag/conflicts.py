"""Disagreements between creators, named rather than averaged.

Every other layer in this repo pushes the corpus towards *one* answer. Retrieval
picks the closest chunks, the theme layer writes a single summary per cluster,
and the answering agent produces one paragraph. When two creators genuinely
disagree, all three do the same thing to it: they blend. The blend reads
fluently and is the most misleading thing the system can produce, because the
reader cannot tell that a choice was made on their behalf.

This module finds those disagreements and keeps them apart. Its output is never
a verdict — a :class:`Conflict` has two sides, an **axis** (the one question the
two creators answer differently) and no winner field, deliberately, so nothing
downstream can render "the corpus says X" out of a case where the corpus says
both.

The test that decides what counts
---------------------------------
**"Could one person hold all of these views?"** If yes, this is complementary
detail — two people describing different parts of the same elephant — and
dressing it up as a conflict is a lie about the corpus. If no, there is an axis,
and the two positions belong on either side of it.

That test is the whole design constraint, because the failure mode here is not
missing a conflict, it is *inventing* one. A view stocked with manufactured
disagreements poisons every downstream use of it and cannot be detected by a
reader who has not watched the videos. So:

* the adjudicator is prompted to **default to complementary** and must state the
  single question the two sides answer differently before it may say conflict;
* :data:`PROBES` carries the calibration in both directions — flat
  contradictions that must surface, and complementary pairs (including two that
  are deliberately *about the same subject*, which is where a lazy adjudicator
  fails) that must not;
* every corpus pair is adjudicated :data:`DEFAULT_ADJUDICATE_REPEATS` times and
  carries only on a **strict majority**, with the tally on the record and on the
  card. This module shipped once without that and it was the worst thing in it:
  the probes were repeated, the corpus pairs were not, and three of the four
  cards that shipped turned out to be coin flips that did not reproduce;
* a quote has to be a *statement* (:func:`states_a_position`), because one of
  those four rested on an eight-word rhetorical question that resolved perfectly
  against the transcript and showed a reader nothing;
* a low count is a legitimate result and is reported as one. The statistics
  carry ``candidates_adjudicated`` beside ``conflicts`` so that a run which
  looked at 200 pairs and found 4 cannot be confused with one that looked at 6.

Axes and facts
--------------
Not every contradiction is a matter of judgement. "Should you cache?" has two
defensible answers and belongs side by side; "what is Brisbane's vacancy rate?"
has one, and two creators giving 8% and 0.4% two weeks apart is not an axis, it
is an error. :attr:`Conflict.kind` separates them so the second is surfaced
without even-handed framing. Neither names a winner — this layer can check a
claim against the corpus and not against the world — but a reader is told which
kind of thing they are looking at.

Why this cannot be gamed by emitting more
-----------------------------------------
Two mechanisms, borrowed from :func:`src.evals.critique.ground_findings`:

* **Exclusive evidence.** A chunk backs at most one conflict. Two conflicts that
  rest on the same pair of chunks are the same conflict said twice, and the
  second is dropped rather than counted — so the conflict count is bounded by
  how much distinct transcript the corpus actually contains, not by how
  talkative the adjudicator is.
* **A precision, not a rate.** ``conflict_precision`` is conflicts over
  *candidates adjudicated*. Widening the candidate net to inflate the count
  drives it down. There is deliberately no metric that rises when more
  candidates are proposed.

Provenance
----------
Nothing model-supplied is trusted for identity. The adjudicator sees two chunks
and returns a quote from each; :func:`verbatim_span` then locates that quote
**inside the stored chunk text** and the quote that ships is the corpus's own
words, cut from the store — not the model's transcription of them. The video id,
the chunk id and the timestamps come from the chunk record. This is the same
rule ``reconcile_agent_references`` applies in
:mod:`src.agents.rag_transcript_agent`, and it exists because 35 of 39
model-supplied chunk indices measured in this corpus were wrong, and because the
ASR renders "write skew" as "right skew" — a model quoting from memory silently
corrects that and the quote then resolves against nothing.

The pipeline
------------
``claims -> candidate pairs (deterministic) -> adjudication (LLM) -> resolve``

Candidates are generated over **claims**, not chunks or communities. A claim is
one declarative sentence, so two claims about the same subject sit close in
embedding space whether they agree or not — negation barely moves a
bi-encoder, which is a liability everywhere else in this repo and exactly the
property a conflict finder wants from its candidate generator. Chunks are 70
seconds of speech about several things at once and match on topic drift;
entity communities are fragmented in this corpus (452 of 841 hold one or two
entities) and are the wrong unit besides — a community is a neighbourhood, not
a proposition.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from pydantic import BaseModel, Field

# ─── Records ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ClaimRef:
    """One extracted claim, carrying the provenance of the chunk it came from.

    ``text`` is the LLM's paraphrase of what was said and is used **only** for
    candidate matching — it never reaches the output. Everything a reader is
    shown is cut from ``chunk_text``, which is the stored transcript verbatim.
    """

    claim_id: str
    text: str
    chunk_id: str
    chunk_text: str
    video_id: str
    channel_name: str
    title: str
    start_seconds: float
    end_seconds: float
    source_url: str = ""

    @property
    def watch_url(self) -> str:
        """The video at the moment this claim was spoken.

        Built from the chunk's own start time, floored to the second, because
        YouTube's ``t`` parameter is integer seconds and a float silently drops
        the whole parameter on some clients.
        """
        return f"https://www.youtube.com/watch?v={self.video_id}&t={int(self.start_seconds)}"


@dataclass(frozen=True)
class CandidatePair:
    """Two claims from different videos that the embedding put near each other.

    Near, here, means "about the same thing" and says nothing about agreement —
    see the module docstring. Whether this pair is a disagreement, complementary
    detail or noise is what the adjudicator decides.
    """

    left: ClaimRef
    right: ClaimRef
    similarity: float

    @property
    def key(self) -> tuple[str, str]:
        """Chunk-pair identity, order-independent — the unit of evidence."""
        return tuple(sorted((self.left.chunk_id, self.right.chunk_id)))  # type: ignore[return-value]

    @property
    def cross_channel(self) -> bool:
        """Different **channels** — which is not quite "different people".

        Named for what it actually measures. ``channel_name`` is the uploading
        channel, so a guest speaking in a cold-open montage, an interviewee, and
        the host are all attributed to the channel owner. Two chunks from
        different channels are therefore *at least* two videos and usually two
        people, but "two creators disagree" is a stronger claim than this field
        supports and calling it ``cross_creator`` made it silently.
        """
        return self.left.channel_name.strip().lower() != self.right.channel_name.strip().lower()


class Side(BaseModel):
    """One creator's end of an axis, with the provenance to check it."""

    video_id: str
    chunk_id: str
    channel_name: str
    title: str
    start_seconds: float
    end_seconds: float
    #: The stance in one line, written by the adjudicator. Not evidence.
    position: str
    #: Evidence: a span cut out of the stored transcript, so it is verbatim by
    #: construction rather than by the model's promise.
    quote: str
    #: How completely the adjudicator's quote was found in the chunk. 1.0 means
    #: it copied exactly; below :data:`QUOTE_MATCH_RATIO` the conflict is dropped.
    quote_ratio: float
    watch_url: str


class Conflict(BaseModel):
    """One disagreement: an axis and two sides. There is no winner field."""

    conflict_id: str
    #: The single question the two creators answer differently, as a question.
    axis: str
    #: Why one person could not hold both positions — the "could one person hold
    #: all these views?" test, answered in the adjudicator's own words so a
    #: reader can dispute it.
    why_incompatible: str
    left: Side
    right: Side
    similarity: float
    cross_channel: bool
    #: ``"axis"`` when reasonable people could land on either side, ``"factual"``
    #: when the two answer a question that has one true answer and one of them is
    #: simply wrong.
    #:
    #: The distinction exists because even-handed presentation is right for the
    #: first and misleading for the second. "Should you cache?" has two defensible
    #: answers and showing them side by side is the honest rendering; "what is
    #: Brisbane's vacancy rate?" has one, and framing 8% beside 0.4% as a matter
    #: of perspective would be a worse lie than picking one. This layer still
    #: names no winner in either case — it cannot check a fact against the world,
    #: only against the corpus — but it says which kind of thing the reader is
    #: looking at.
    kind: str = "axis"
    #: How many of :attr:`repeats` adjudications called this a conflict.
    #:
    #: On the record and on the card because the adjudicator does not agree with
    #: itself: the first sweep of this corpus adjudicated every pair **once**, and
    #: an independent re-run of its four shipped cards at three repeats each drew
    #: 1/3, 2/3, 3/3, 1/3. A count produced by a single draw is not a measurement
    #: of the corpus, and a reader cannot tell 3/3 from 1/3 unless the number is
    #: printed.
    votes: int = 1
    repeats: int = 1

    @property
    def unanimous(self) -> bool:
        return self.votes >= self.repeats


class ConflictIndex(BaseModel):
    """The artifact on disk: derived state, rebuilt by ``index-conflicts``."""

    version: int = 1
    generated_at: str
    embedding_model: str = ""
    adjudicator_model: str = ""
    chunk_collection: str = ""
    config: dict[str, Any] = Field(default_factory=dict)
    stats: dict[str, Any] = Field(default_factory=dict)
    #: The population this count was taken over — see
    #: :func:`corpus_fingerprint`. Without it two counts taken hours apart
    #: cannot be told apart from two counts taken over two different corpora.
    corpus: dict[str, Any] = Field(default_factory=dict)
    #: Calibration results — see :data:`PROBES`. Stored with the run so the
    #: artifact carries the evidence that its adjudicator was working in both
    #: directions on the day it produced these conflicts.
    probes: list[dict[str, Any]] = Field(default_factory=list)
    #: Every adjudicated pair with the rate it drew — see :func:`vote_ledger`.
    #: The run's raw measurement, kept so that any claim about the spread on the
    #: count can be recomputed from the artifact rather than taken on trust, and
    #: so two runs can be compared pair by pair.
    vote_ledger: list[dict[str, Any]] = Field(default_factory=list)
    #: Pairs that drew at least one conflict verdict without carrying a
    #: majority — see :func:`near_misses`. Not conflicts, and stored apart from
    #: them, but recorded so a reviewer can see what sat near the threshold
    #: rather than only how many did.
    near_misses: list[dict[str, Any]] = Field(default_factory=list)
    conflicts: list[Conflict] = Field(default_factory=list)


# ─── Configuration ───────────────────────────────────────────────────────────

#: Times every corpus pair is put to the adjudicator before it is believed.
#:
#: Nine, and odd so a strict majority always exists. The history is the
#: argument. One draw per pair shipped four cards, of which an independent
#: re-run at three draws each scored 1/3, 2/3, 3/3, 1/3 — three coin flips.
#: Three draws per pair then produced 4 conflicts and, on a second build over a
#: *provably identical* 478-pair pool, 2 — with two cards moving 3/3 to 0/3.
#:
#: An earlier version of this note claimed the spread was **irreducible** — that
#: more looks could not help — and gave a curve falling only from about 1.1 at
#: three looks to 0.7 at twenty-one. That claim was withdrawn: it was computed
#: by plugging ``p̂ = votes/3`` into the majority probability, and at three looks
#: that estimator can only place a pair at 0, 1/3, 2/3 or 1, of which the two
#: ends contribute no variance at all. The whole curve therefore collapses to
#: ``0.4382 * sqrt(pairs that split)`` — a property of the four-point grid, not
#: of any corpus. Simulation confirms it: three-look data reports the *same*
#: 4.3x fall from r=3 to r=45 whether the truth genuinely plateaus (1.3x) or
#: collapses to nothing (a factor of 10^8). It could not have told the
#: difference, so it was never evidence.
#:
#: Nine looks can tell the difference — the same simulation recovers 76x, 3.9x
#: and 2.0x for those corpora — which is the substantive reason to pay for them,
#: alongside two mechanical ones. Measured at nine, the spread does fall as
#: looks are added, slowly, and this corpus sits in the wide-ambiguous-band
#: case; :func:`stability_statistics` carries those figures and the reason even
#: they are only a floor.
#:
#: * **Resolution.** Three looks can only ever say 0, 1/3, 2/3 or 1, and the
#:   smallest majority is the same 2/3 that :data:`FIRM_VOTE_SHARE` calls firm —
#:   so a certainty and a coin flip are literally the same number. At nine, a
#:   pair the judge is sure of clears 6/9 essentially always and a coin flip
#:   clears it a quarter of the time.
#: * **A free second opinion.** Nine splits into three disjoint groups of three,
#:   so one run reports what three independent three-look builds of the same
#:   data would have said — see :func:`stability_statistics`. That is the
#:   4-then-2 discrepancy measured inside a single run instead of across two.
#:
#: The cost is real: on the shipped sweep 710 pairs became 6390 calls, better
#: than an hour of wall clock — which is why :class:`AdjudicationCache` exists.
#: There is no cheaper version
#: that keeps the guarantee — filtering on a first pass and re-checking only the
#: hits would be a precision filter over a *biased* sample, and would silently
#: miss every pair that would have carried on the majority but happened to draw
#: complementary first.
DEFAULT_ADJUDICATE_REPEATS = 9


@dataclass(frozen=True)
class ConflictConfig:
    """Knobs for the sweep. All deterministic; none reach the LLM as text.

    Travels into the artifact as ``config``, so a reader can see the settings a
    count was produced under without re-deriving them from the code that
    happened to be checked out.
    """

    #: Cosine below which two claims are not about the same thing at all. Chosen
    #: against this corpus by reading pairs at each band: below 0.62 the pairs
    #: are two different subjects that share a word ("scale" in sharding and in
    #: career growth), and adjudicating them is spend with no chance of a
    #: conflict in it.
    similarity_floor: float = 0.62
    #: Cosine above which two claims are the *same* claim restated. Those are
    #: agreements by construction, and they crowd out real candidates because
    #: near-duplicates are exactly what a bi-encoder ranks first.
    similarity_ceiling: float = 0.97
    #: Candidate pairs per claim, before global truncation.
    neighbors_per_claim: int = 4
    #: Chunk pairs sent to the adjudicator, highest similarity first. A budget,
    #: not a target: raising it cannot raise ``conflict_precision``.
    #:
    #: Set **above** the size of the pool this corpus produces (478 pairs after
    #: the caps below) so that nothing is excluded by budget and the run cannot
    #: be accused of having stopped just short of an inconvenient pair. That is
    #: not generosity — it is the only setting under which "the corpus contains
    #: N disagreements" is a statement about the corpus rather than about where
    #: the budget happened to fall. The first sweep used 240 and cut the pool in
    #: half at similarity 0.70; the sharpest dissent in the résumé material
    #: ("recruiters hunt for qualifications, not keywords" against the
    #: keyword-optimisation consensus) sits at rank 293 and was therefore
    #: invisible to it, because a bi-encoder ranks restatement above
    #: contradiction and the top of this list is agreements.
    max_candidates: int = 600
    #: Chunk pairs kept from any single pair of videos. Without this, one pair
    #: of talks on the same subject consumes the whole budget.
    max_per_video_pair: int = 12
    #: Candidate pairs any one chunk may appear in.
    #:
    #: This is the cap that makes the budget worth spending. The highest-cosine
    #: pairs in this corpus are *restatements* — six creators saying "keep your
    #: resume to one page" produced twenty-four of the top forty candidates on
    #: the first sweep, all of them the same agreement — because a bi-encoder
    #: ranks paraphrase above everything and agreement is what paraphrase looks
    #: like. Capping per chunk turns that cluster into a handful of pairs and
    #: spends the rest of the budget on subjects the sweep had not reached at
    #: all. It also matches what :func:`resolve_conflicts` does downstream: a
    #: chunk can only ever back one conflict, so proposing it ten times was
    #: always nine wasted calls.
    max_per_chunk: int = 2
    #: Only pairs from different **channels** are adjudicated by default. Two
    #: videos on one channel restating a position is not a corpus disagreement,
    #: and within-channel "conflicts" are overwhelmingly one person qualifying
    #: themselves ("usually X, but when Y, not X") — the textbook shape of
    #: something one person can hold entirely. See
    #: :attr:`CandidatePair.cross_channel` for why this is not "creator".
    cross_channel_only: bool = True
    #: Times each candidate pair is adjudicated before it is believed — see
    #: :data:`DEFAULT_ADJUDICATE_REPEATS` and :class:`AdjudicationVote`.
    #:
    #: A knob rather than a constant so a run can be made cheaper deliberately
    #: and *visibly*: it lands in the artifact's ``config`` beside the count it
    #: produced, so a 1-repeat sweep can never again be read as if it had been
    #: voted on. Setting it to 1 reproduces the behaviour this module shipped
    #: with, in which three of four cards were coin flips.
    adjudicate_repeats: int = DEFAULT_ADJUDICATE_REPEATS


#: Similarity above which the adjudicator's quote counts as present in the
#: chunk. Below 1.0 because a model retyping a quote drops a filler word or
#: normalises "gonna"; well above chance because the check exists to catch
#: invention. Matches :data:`src.evals.critique.QUOTE_MATCH_RATIO` on purpose —
#: a quote that ships from here must pass the same bar the critique scorer sets.
QUOTE_MATCH_RATIO = 0.80

#: Words a quote must have before it is worth showing. A three-word "it depends"
#: resolves perfectly and proves nothing.
#:
#: Raised from 8 after a shipped card rested on ``"like what is a technology free
#: domain model?"`` — exactly eight words, resolving perfectly, and stating no
#: position at all. Length alone was never going to catch that (see
#: :func:`states_a_position`), but eight words is short enough that a fragment
#: clears it by accident.
MIN_QUOTE_WORDS = 10

_WORD = re.compile(r"[a-z0-9']+")

#: Openers that make a span a question rather than a claim. Applied to the span
#: cut from the transcript, where ASR punctuation is unreliable, so the opener is
#: checked as well as the trailing "?".
_INTERROGATIVE = re.compile(
    r"^(?:so|and|but|well|like|now|i mean|you know|right|okay|ok)?[\s,]*"
    r"(?:what|who|whom|whose|which|when|where|why|how|is|are|was|were|do|does|did|"
    r"has|have|had|can|could|should|would|will|shall|am|any(?:one|body))\b",
    re.IGNORECASE,
)


def normalize(text: str) -> list[str]:
    """Lowercase word tokens — the comparable form of transcript text."""
    return _WORD.findall(str(text).lower())


def states_a_position(span: str) -> bool:
    """Whether this span asserts something, as opposed to asking something.

    A side of a conflict has to be the words in which that side *states* its
    position. A rhetorical question is not that, however well it resolves
    against the transcript and however clearly a human reading the surrounding
    minute can tell which way the speaker leans — the reader of a conflict card
    sees the span, not the minute.

    This is the check that was missing when a card shipped with *"like what is a
    technology free domain model?"* as one creator's position on ports and
    adapters. The speaker does hold that position; that span does not express
    it, and a reader could not verify it from what was on screen.

    Deliberately crude, and biased towards rejection: it costs a real conflict
    whose best sentence happens to be phrased as a question, which is the
    direction this module errs in everywhere else. A conflict dropped for a weak
    quote is a conflict that was going to be unfalsifiable on the card anyway.
    """
    stripped = str(span).strip()
    if not stripped:
        return False
    if stripped.rstrip().endswith("?"):
        return False
    return not _INTERROGATIVE.match(stripped)


# ─── Claims, with provenance stamped from the chunk ──────────────────────────


#: ``chunk -> the claim sentences extracted from it``. The CLI wires this to the
#: cached GraphRAG extractions; a list is enough for the tests. Typed on ``Any``
#: rather than a ``TranscriptChunk`` because the only thing this layer needs
#: from a chunk is the handful of attributes read below, and importing the
#: storage models here would drag chromadb into a module that is otherwise pure.
ClaimTextsFn = Callable[[Any], Sequence[str]]


def corpus_fingerprint(chunks: Sequence[Any]) -> dict[str, Any]:
    """What corpus this run actually saw, pinned so two runs can be compared.

    A conflict count is a measurement of a *population*, and this repo's corpus
    is a live thing — it grew 1372 -> 1460 -> 1736 chunks during a single
    afternoon's work, from ingests outside this layer entirely. Two counts taken
    hours apart are only comparable if they were taken over the same pairs, and
    without this a reader has no way to tell corpus drift from judge noise. That
    distinction mattered immediately: a 4-conflict run and a 2-conflict run
    looked like they might have been measuring different corpora, and were not —
    both saw 6196 claims and the same 478 pairs, because the chunks added in
    between had no cached extraction and so contributed nothing here.

    The digest covers chunk **identity and text**, so a re-chunk or an edit
    changes it even when the counts do not. It is deliberately not a hash of the
    claims: claims come from a cache that can be backfilled independently, and
    telling "the corpus changed" apart from "the extraction coverage changed" is
    exactly what this has to support — which is why ``chunks_with_claims`` is
    reported beside ``chunks`` rather than instead of it.
    """
    import hashlib

    digest = hashlib.sha256()
    videos: set[str] = set()
    count = 0
    for chunk in sorted(chunks, key=lambda item: str(getattr(item, "chunk_id", ""))):
        chunk_id = str(getattr(chunk, "chunk_id", ""))
        text = str(getattr(chunk, "text", "") or "")
        digest.update(chunk_id.encode("utf-8"))
        digest.update(hashlib.sha256(text.encode("utf-8")).digest())
        videos.add(str(getattr(chunk, "video_id", "")))
        count += 1
    return {
        "videos": len(videos),
        "chunks": count,
        "digest": digest.hexdigest()[:16],
    }


def claims_from_chunks(chunks: Sequence[Any], claim_texts: ClaimTextsFn) -> list[ClaimRef]:
    """Claims with every identity field taken from the **chunk**, not the model.

    The extraction that produced these sentences also emitted its own idea of
    which video and timestamp it was reading; that is thrown away here and the
    chunk record is used instead. See the module docstring for why nothing
    model-supplied is allowed to name a location in the corpus.
    """
    claims: list[ClaimRef] = []
    for chunk in chunks:
        text = str(getattr(chunk, "text", "") or "")
        if not text.strip():
            continue
        for index, claim in enumerate(claim_texts(chunk)):
            sentence = str(claim or "").strip()
            if not sentence:
                continue
            claims.append(
                ClaimRef(
                    claim_id=f"{chunk.chunk_id}#{index}",
                    text=sentence,
                    chunk_id=str(chunk.chunk_id),
                    chunk_text=text,
                    video_id=str(chunk.video_id),
                    channel_name=str(getattr(chunk, "channel_name", "") or "Unknown creator"),
                    title=str(getattr(chunk, "title", "") or ""),
                    start_seconds=float(getattr(chunk, "start_seconds", None) or 0.0),
                    end_seconds=float(getattr(chunk, "end_seconds", None) or 0.0),
                    source_url=str(getattr(chunk, "source_url", "") or ""),
                )
            )
    return sorted(claims, key=lambda claim: claim.claim_id)


# ─── Candidate generation (deterministic) ────────────────────────────────────

#: ``texts -> vectors``. The shipped MiniLM bi-encoder in practice — hence
#: ``list[str]`` rather than ``Sequence[str]``, which is what
#: :meth:`~src.rag.embeddings.HuggingFaceEmbeddingModel.embed_documents` accepts.
EmbedFn = Callable[[list[str]], list[list[float]]]


def candidate_pairs(
    claims: Sequence[ClaimRef],
    embed: EmbedFn,
    config: ConflictConfig | None = None,
) -> list[CandidatePair]:
    """Cross-video claim pairs about the same thing, best first.

    Wholly deterministic: the same corpus and the same embedding model produce
    the same list in the same order, so a change in the conflicts a run reports
    is attributable to the adjudicator rather than to the search. Ties break on
    claim id for the same reason.

    Deduplicated to one pair per **chunk pair** — several claims from the same
    two chunks are one piece of evidence, and adjudicating them separately would
    spend the budget re-reading the same seventy seconds of transcript.
    """
    settings = config or ConflictConfig()
    if len(claims) < 2:
        return []

    scores = _similarity_matrix(claims, embed, settings)
    scored: list[CandidatePair] = []
    for i, claim in enumerate(claims):
        neighbours = [
            (float(scores[i][j]), j)
            for j in range(len(claims))
            if settings.similarity_floor <= scores[i][j] <= settings.similarity_ceiling
        ]
        neighbours.sort(key=lambda item: (-item[0], claims[item[1]].claim_id))
        for score, j in neighbours[: settings.neighbors_per_claim]:
            left, right = (
                (claim, claims[j]) if claim.claim_id <= claims[j].claim_id else (claims[j], claim)
            )
            scored.append(CandidatePair(left=left, right=right, similarity=score))

    best: dict[tuple[str, str], CandidatePair] = {}
    for pair in scored:
        current = best.get(pair.key)
        if current is None or pair.similarity > current.similarity:
            best[pair.key] = pair

    ordered = sorted(best.values(), key=lambda pair: (-pair.similarity, pair.key))
    return _apply_caps(ordered, settings)[: settings.max_candidates]


def _similarity_matrix(
    claims: Sequence[ClaimRef],
    embed: EmbedFn,
    config: ConflictConfig,
) -> Any:
    """Cosine between every pair of claims, with ineligible pairs set to -1.

    A whole matrix rather than a nearest-neighbour index because the corpus has
    a few thousand claims and one matrix multiply settles it in under a second —
    an ANN index would add a dependency, an approximation and a tie-break that
    is not reproducible, to save nothing.

    Same-video and (by default) same-channel pairs are masked to ``-1`` rather
    than skipped in the caller so that "not eligible" and "not similar" are the
    same branch downstream. The diagonal goes with them: a claim is not evidence
    against itself.
    """
    import numpy as np

    raw = np.asarray(embed([claim.text for claim in claims]), dtype=np.float32)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    matrix = (raw / np.where(norms == 0, 1.0, norms)).astype(np.float32)
    scores = matrix @ matrix.T

    videos = np.array([claim.video_id for claim in claims])
    scores[videos[:, None] == videos[None, :]] = -1.0
    if config.cross_channel_only:
        channels = np.array([claim.channel_name.strip().lower() for claim in claims])
        scores[channels[:, None] == channels[None, :]] = -1.0
    np.fill_diagonal(scores, -1.0)
    return scores


def _apply_caps(pairs: Sequence[CandidatePair], config: ConflictConfig) -> list[CandidatePair]:
    """Spread the adjudication budget: cap per video pair and per chunk.

    Two conference talks on the same subject produce hundreds of near-neighbour
    claims, and one popular piece of advice produces dozens of restatements of
    itself; without these caps either would take the whole budget and the run
    could only ever find disagreements in one corner of the corpus. See
    :attr:`ConflictConfig.max_per_chunk` for what that looked like in practice.

    Applied to an already sorted list, so a chunk's allowance is spent on its
    strongest pairs, and the result is a deterministic function of the input
    order.
    """
    video_pairs: dict[tuple[str, str], int] = {}
    chunks: dict[str, int] = {}
    kept: list[CandidatePair] = []
    for pair in pairs:
        key: tuple[str, str] = tuple(sorted((pair.left.video_id, pair.right.video_id)))  # type: ignore[assignment]
        if video_pairs.get(key, 0) >= config.max_per_video_pair:
            continue
        if any(chunks.get(cid, 0) >= config.max_per_chunk for cid in pair.key):
            continue
        video_pairs[key] = video_pairs.get(key, 0) + 1
        for cid in pair.key:
            chunks[cid] = chunks.get(cid, 0) + 1
        kept.append(pair)
    return kept


# ─── Adjudication ────────────────────────────────────────────────────────────

ADJUDICATOR_SYSTEM_PROMPT = """You decide whether two people actually disagree.

You are given two excerpts from DIFFERENT videos by different creators, on a
similar subject. Apply exactly one test:

    COULD ONE PERSON HOLD BOTH OF THESE VIEWS AT THE SAME TIME?

If yes, this is NOT a conflict. Two people describing different parts of the
same subject, or the same advice in different words, or one adding detail the
other did not mention, are COMPLEMENTARY. So are a general rule and an
exception to it, and so is "do X" alongside "X is hard" — one person routinely
holds both of those.

Say "conflict" ONLY when you can write down a single question, and the two
excerpts give answers to that question that cannot both be followed. If you
cannot state that question in one sentence, it is not a conflict.

Default to complementary. A wrongly reported conflict is far worse than a
missed one: it tells a reader these creators fight about something they do not.

Beware two shapes that look like conflict and are not:
- Objecting to a REASON for a practice is not objecting to the practice. If one
  speaker mocks the justification usually given for X and then concedes X is
  still worth doing, that is not a conflict with someone who recommends X.
- A title, a topic, or a tone is not a position. Judge only what the excerpts
  actually say.

Then say which KIND of disagreement it is:
- "axis": a matter of judgement. Reasonable people land on either side and both
  answers can be defended.
- "factual": the question has ONE true answer and one of them is simply wrong —
  two different numbers for the same quantity, two different dates, two
  incompatible descriptions of the same thing.

Quotes: copy the words VERBATIM from the excerpt, including any transcription
errors. Do not tidy, do not paraphrase, do not join two separated sentences.
Each quote must be at least 10 words, must be a STATEMENT rather than a
question, and must be the words in which that side states its position. A
rhetorical question is not a position, however clearly it implies one.

Return only JSON:
{"verdict": "conflict" | "complementary" | "unrelated",
 "kind": "axis" | "factual",
 "axis": "the one question they answer differently, as a question",
 "why_incompatible": "why one person could not hold both",
 "position_a": "excerpt A's answer, one line",
 "position_b": "excerpt B's answer, one line",
 "quote_a": "verbatim statement from excerpt A",
 "quote_b": "verbatim statement from excerpt B"}

For a non-conflict, set verdict and leave the other fields as empty strings."""


@dataclass(frozen=True)
class Adjudication:
    """What the adjudicator said about one candidate pair."""

    verdict: str
    axis: str = ""
    why_incompatible: str = ""
    position_a: str = ""
    position_b: str = ""
    quote_a: str = ""
    quote_b: str = ""
    kind: str = "axis"
    error: str | None = None

    @property
    def is_conflict(self) -> bool:
        return self.verdict == "conflict"


@dataclass(frozen=True)
class AdjudicationVote:
    """Every verdict one pair drew, and the one that carries.

    The adjudicator is an LLM and does not agree with itself. :func:`run_probes`
    has said so since this module was written — *"a single pass proves nothing
    about the next one"* — and then the corpus sweep adjudicated every pair
    exactly once anyway, so the calibration strip certified a stability the cards
    themselves were never given. An independent re-run of the four cards that
    shipped from that sweep, three repeats each, drew **1/3, 2/3, 3/3, 1/3**:
    only one of the four survived its own re-examination.

    So a pair is a conflict when a **strict majority** of its repeats say so, and
    the tally travels with it onto the record and onto the card. Ties and splits
    resolve to *not* a conflict, which is the direction everything else in this
    module errs in: a disagreement nobody can reproduce is not one this layer
    should be asserting on a reader's behalf.
    """

    verdicts: tuple[Adjudication, ...]

    @property
    def repeats(self) -> int:
        return len(self.verdicts)

    @property
    def votes(self) -> int:
        """How many repeats said conflict."""
        return sum(1 for verdict in self.verdicts if verdict.is_conflict)

    @property
    def is_conflict(self) -> bool:
        """A strict majority, so 1/2 and 1/3 both fail and 2/3 carries."""
        return self.votes * 2 > self.repeats

    @property
    def unanimous(self) -> bool:
        return len({verdict.verdict for verdict in self.verdicts}) == 1

    @property
    def carried(self) -> Adjudication:
        """The first repeat that voted conflict — the one whose axis ships.

        First rather than best, because "best" would need a second judge and the
        repeats are supposed to be interchangeable draws from one. Falls back to
        the first verdict of any kind so the caller always has something to read
        a rejection reason off.
        """
        return next(
            (verdict for verdict in self.verdicts if verdict.is_conflict),
            self.verdicts[0] if self.verdicts else Adjudication(verdict="error"),
        )


#: ``(pair) -> an Adjudication``. The shipped implementation is
#: :class:`LlmAdjudicator`; the tests pass a dict-backed fake.
AdjudicateFn = Callable[[CandidatePair], Adjudication]


class AdjudicationCache:
    """One file per (pair, attempt), so a killed sweep resumes where it stopped.

    At nine looks this layer's sweep is thousands of calls and over an hour of
    wall clock, and it has twice been killed most of the way through with
    nothing to show — the artifact is only written at the end, so a run that
    dies at 4400 of 4896 calls costs 4400 calls and yields nothing. Every other
    expensive stage in this repo caches (``graph_cache``, ``context_cache``,
    ``critique_cache``, the matrix's per-cell cache); this one did not.

    **The attempt index is part of the key, and that is the whole design.** A
    cache keyed on the pair alone would serve one verdict to all nine looks and
    silently destroy the vote — the repeats would stop being independent draws
    and every pair would come back unanimous. Keyed on ``(pair, attempt)``,
    attempt 4 resumes as attempt 4 and the nine draws stay nine draws.

    ``run_id`` scopes the whole cache, and exists so that a *second* measurement
    cannot be accidentally served from the first one's answers. Reproducibility
    is the thing this layer is currently trying to establish; a cache that
    replayed the previous run would manufacture a perfect agreement between two
    runs that never independently happened. Resuming one run reuses its key;
    measuring again requires a new one, and there is deliberately no default.
    """

    def __init__(self, directory: Path | str, run_id: str) -> None:
        self.directory = Path(directory) / run_id
        self.run_id = run_id
        self.hits = 0
        self.writes = 0

    def _path(self, pair: CandidatePair, attempt: int) -> Path:
        import hashlib

        # The chunk *text* is in the key, not only the id: a re-chunk or an ASR
        # correction changes what the adjudicator was shown, and a cached
        # verdict about different words is not a verdict about these ones.
        digest = hashlib.sha256(
            "\x00".join(
                [
                    pair.left.chunk_id,
                    pair.right.chunk_id,
                    pair.left.chunk_text,
                    pair.right.chunk_text,
                    str(attempt),
                ]
            ).encode("utf-8")
        ).hexdigest()[:32]
        return self.directory / f"{digest}.json"

    def get(self, pair: CandidatePair, attempt: int) -> Adjudication | None:
        path = self._path(pair, attempt)
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        self.hits += 1
        return Adjudication(**data)

    def put(self, pair: CandidatePair, attempt: int, adjudication: Adjudication) -> None:
        # A failed call is never cached: it would pin a transient provider
        # outage into the record and be re-served as though it were a verdict.
        # Same rule as ``GraphExtractor._write_cache``.
        if adjudication.verdict == "error":
            return
        path = self._path(pair, attempt)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(adjudication)), encoding="utf-8")
        self.writes += 1


def format_pair(pair: CandidatePair) -> str:
    """The two excerpts as the adjudicator sees them.

    Whole chunks, not the claim sentences: the claims were only ever the search
    index, and asking the model to quote from a paraphrase would guarantee the
    quote does not resolve against the transcript.
    """
    return (
        f'EXCERPT A — {pair.left.channel_name}, "{pair.left.title}"\n'
        f"{pair.left.chunk_text.strip()}\n\n"
        f'EXCERPT B — {pair.right.channel_name}, "{pair.right.title}"\n'
        f"{pair.right.chunk_text.strip()}"
    )


class LlmAdjudicator:
    """One LLM call per candidate pair: two excerpts in, a verdict out."""

    def __init__(self, llm: Any, model_name: str = "") -> None:
        self.llm = llm
        self.model_name = model_name

    def __call__(self, pair: CandidatePair) -> Adjudication:
        from langchain_core.messages import HumanMessage, SystemMessage

        try:
            response = self.llm.invoke(
                [
                    SystemMessage(content=ADJUDICATOR_SYSTEM_PROMPT),
                    HumanMessage(content=format_pair(pair)),
                ]
            )
            data = _json_object(str(getattr(response, "content", response) or ""))
        except Exception as exc:  # noqa: BLE001 - one bad pair must not abort the sweep
            return Adjudication(verdict="error", error=str(exc)[:200])
        verdict = str(data.get("verdict", "")).strip().lower()
        return Adjudication(
            verdict=verdict
            if verdict in {"conflict", "complementary", "unrelated"}
            else "unrelated",
            axis=str(data.get("axis", "")).strip(),
            why_incompatible=str(data.get("why_incompatible", "")).strip(),
            position_a=str(data.get("position_a", "")).strip(),
            position_b=str(data.get("position_b", "")).strip(),
            quote_a=str(data.get("quote_a", "")).strip(),
            quote_b=str(data.get("quote_b", "")).strip(),
            kind="factual" if str(data.get("kind", "")).strip().lower() == "factual" else "axis",
        )


# ─── Provenance: cut the quote out of the stored transcript ──────────────────


@dataclass(frozen=True)
class QuoteMatch:
    """A verbatim span of the stored chunk, and how well the model's quote hit it."""

    text: str
    ratio: float


def verbatim_span(quote: str, chunk_text: str) -> QuoteMatch | None:
    """The words of ``chunk_text`` that the model was quoting — its words, not the model's.

    Returns a substring **cut from the stored transcript**, so what a reader sees
    is what the store holds even when the model silently corrected the ASR on
    the way past (this corpus renders "write skew" as "right skew" throughout,
    and a model quoting from understanding rather than from the page fixes that
    and produces a quote that resolves against nothing).

    The ratio is the one :func:`src.evals.critique.quote_ratio` computes — the
    share of the *quote*'s words that appear, in order, in the chunk — so a
    quote that ships from here would pass the critique scorer's provenance
    check unchanged. It is one-sided on purpose: the question is whether the
    speaker said this, not whether this is all they said.

    ``None`` when the quote falls below :data:`QUOTE_MATCH_RATIO` or is too
    short to mean anything (:data:`MIN_QUOTE_WORDS`). That is the signal to drop
    the conflict: a side whose evidence cannot be found in the transcript is not
    evidence.
    """
    needle = normalize(quote)
    if len(needle) < MIN_QUOTE_WORDS:
        return None
    words = chunk_text.split()
    if not words:
        return None
    # Token stream plus, for each token, the index of the source word it came
    # from — so a matched token range can be converted back into a slice of the
    # original, punctuated text.
    haystack: list[str] = []
    owner: list[int] = []
    for index, word in enumerate(words):
        for token in normalize(word):
            haystack.append(token)
            owner.append(index)
    if not haystack:
        return None

    matcher = SequenceMatcher(a=needle, b=haystack, autojunk=False)
    blocks = [block for block in matcher.get_matching_blocks() if block.size]
    ratio = sum(block.size for block in blocks) / len(needle)
    if ratio < QUOTE_MATCH_RATIO or not blocks:
        return None

    first = owner[blocks[0].b]
    last = owner[min(blocks[-1].b + blocks[-1].size - 1, len(owner) - 1)]
    span = " ".join(words[first : last + 1]).strip()
    return QuoteMatch(text=span, ratio=round(ratio, 4))


# ─── Resolution ──────────────────────────────────────────────────────────────


def resolve_conflicts(
    pairs: Sequence[CandidatePair],
    votes: Sequence[AdjudicationVote],
) -> tuple[list[Conflict], dict[str, int]]:
    """Adjudicated pairs into conflicts, with the rejections counted.

    Six gates, in order, each of which drops the pair and is tallied so a run
    can say *why* it found few conflicts rather than only that it did:

    1. no repeat called it a conflict;
    2. some did, but not a strict majority — see :class:`AdjudicationVote`.
       Counted apart from gate 1 on purpose: it is the number that says how
       close to the threshold this corpus sits, and the first sweep of this
       module could not report it because it only ever drew once;
    3. it said conflict but could not name an axis or say why the two positions
       are incompatible — a conflict nobody can state is not reportable;
    4. one side's quote is not in the stored transcript (see
       :func:`verbatim_span`);
    5. one side's quote is not a *statement* (see :func:`states_a_position`).
       A rhetorical question resolves perfectly against the transcript and shows
       a reader nothing;
    6. one of the two chunks already backs a conflict. This is the exclusivity
       rule from :func:`src.evals.critique.ground_findings`: distinct evidence
       per claim, so the count cannot be inflated by restating one disagreement.

    Order matters for gate 6 — the highest-similarity pair wins the chunk — and
    ``pairs`` arrives sorted, so the result does not depend on adjudication
    completion order.
    """
    tally = {
        "not_a_conflict": 0,
        "minority_verdict": 0,
        "unstated_axis": 0,
        "quote_not_in_transcript": 0,
        "quote_is_not_a_position": 0,
        "duplicate_evidence": 0,
    }
    conflicts: list[Conflict] = []
    spent: set[str] = set()
    for pair, vote in zip(pairs, votes):
        if not vote.is_conflict:
            tally["minority_verdict" if vote.votes else "not_a_conflict"] += 1
            continue
        verdict = vote.carried
        if not verdict.axis or not verdict.why_incompatible:
            tally["unstated_axis"] += 1
            continue
        left = verbatim_span(verdict.quote_a, pair.left.chunk_text)
        right = verbatim_span(verdict.quote_b, pair.right.chunk_text)
        if left is None or right is None:
            tally["quote_not_in_transcript"] += 1
            continue
        if not states_a_position(left.text) or not states_a_position(right.text):
            tally["quote_is_not_a_position"] += 1
            continue
        if pair.left.chunk_id in spent or pair.right.chunk_id in spent:
            tally["duplicate_evidence"] += 1
            continue
        spent.add(pair.left.chunk_id)
        spent.add(pair.right.chunk_id)
        conflicts.append(
            Conflict(
                conflict_id=f"conflict:{len(conflicts)}",
                axis=verdict.axis,
                why_incompatible=verdict.why_incompatible,
                left=_side(pair.left, verdict.position_a, left),
                right=_side(pair.right, verdict.position_b, right),
                similarity=round(pair.similarity, 4),
                cross_channel=pair.cross_channel,
                kind=verdict.kind,
                votes=vote.votes,
                repeats=vote.repeats,
            )
        )
    return conflicts, tally


def _side(claim: ClaimRef, position: str, quote: QuoteMatch) -> Side:
    return Side(
        video_id=claim.video_id,
        chunk_id=claim.chunk_id,
        channel_name=claim.channel_name,
        title=claim.title,
        start_seconds=claim.start_seconds,
        end_seconds=claim.end_seconds,
        position=position,
        quote=quote.text,
        quote_ratio=quote.ratio,
        watch_url=claim.watch_url,
    )


def near_misses(
    pairs: Sequence[CandidatePair],
    votes: Sequence[AdjudicationVote],
    limit: int = 25,
) -> list[dict[str, Any]]:
    """Pairs that persuaded at least one look but not a majority.

    The interesting list for anyone auditing this layer, and the one the
    aggregate counts cannot give: a ``minority_verdict`` total says six pairs sat
    near the threshold, and says nothing about *which*. Without them a reviewer
    re-running the sweep and getting a different four has no way to tell whether
    the layer moved or the judge did.

    The axis is the minority verdict's own, recorded so the near-miss can be read
    and dismissed on its merits rather than only counted. It is explicitly not a
    conflict and never reaches :attr:`ConflictIndex.conflicts`.
    """
    near = [(pair, vote) for pair, vote in zip(pairs, votes) if vote.votes and not vote.is_conflict]
    near.sort(key=lambda item: (-item[1].votes, -item[0].similarity, item[0].key))
    return [
        {
            "chunk_ids": list(pair.key),
            "channels": sorted({pair.left.channel_name, pair.right.channel_name}),
            "votes": vote.votes,
            "repeats": vote.repeats,
            "similarity": round(pair.similarity, 4),
            "axis_claimed": vote.carried.axis,
        }
        for pair, vote in near[:limit]
    ]


#: Share of looks a pair must win before its conflict verdict is called *firm*
#: rather than a coin flip.
#:
#: Two thirds, which at nine looks means six. Chosen because it is the band that
#: separates the two populations this layer actually contains: a pair the judge
#: is sure about (per-call probability around 0.9) clears 6/9 essentially always
#: (0.99), while a pair it is undecided about (0.5) clears it a quarter of the
#: time. At three looks the same band is 2/3, which a coin-flip pair clears
#: *half* the time — which is precisely why the single-draw and three-look
#: builds could not tell the two apart, and why two cards moved 3/3 -> 0/3
#: between runs.
FIRM_VOTE_SHARE = 2 / 3

#: :data:`FIRM_VOTE_SHARE` as an exact ratio, because the band edges have to be
#: compared exactly and floats cannot do it.
#:
#: A vote share is a ratio of two small integers and so is the edge, but neither
#: ``2 / 3`` nor ``1 / 3`` is representable in binary: ``1 - 2 / 3`` rounds to
#: ``0.33333333333333337``, which is *above* ``1 / 3``. So the undecided band's
#: lower edge silently excluded every pair that drew conflict on exactly a third
#: of its looks — 1/3, 2/6, 3/9, 5/15, 7/21 — which is the most common undecided
#: pair there is and the **only** one three looks can produce. The shipped
#: three-look artifact has eleven of them and would have reported
#: ``undecided_pairs: 0`` beside its own ``minority_verdict: 11``.
#:
#: Derived from :data:`FIRM_VOTE_SHARE` rather than written out a second time, so
#: that changing the share cannot leave the two disagreeing.
_FIRM_SHARE = Fraction(FIRM_VOTE_SHARE).limit_denominator(1000)


def _is_firm(votes: int, repeats: int) -> bool:
    """Whether this tally clears :data:`FIRM_VOTE_SHARE` of its looks."""
    return bool(repeats) and Fraction(votes, repeats) >= _FIRM_SHARE


def _is_undecided(votes: int, repeats: int) -> bool:
    """Whether this tally sits between the two firm bands — see :data:`_FIRM_SHARE`.

    Firm *against* is the mirror of firm *for*, so the band is everything that is
    neither: a share of at least ``1 - FIRM_VOTE_SHARE`` and less than
    ``FIRM_VOTE_SHARE``.
    """
    return bool(repeats) and 1 - _FIRM_SHARE <= Fraction(votes, repeats) < _FIRM_SHARE


def _ships_probability(votes: int, repeats: int, at_repeats: int) -> float:
    """P(a pair with this observed rate carries a strict majority of ``at_repeats``).

    The pair's per-call conflict probability is estimated by its own vote share
    and the majority is then a binomial tail. Crude — the estimate is itself
    noisy at these repeat counts — but it is the only error bar available from a
    single run, and an approximate spread on the record beats a bare count that
    invites being read as exact.

    **How crude, measured.** The plug-in is *upward* biased, and the bias is a
    function of ``repeats`` rather than of the corpus:

    * ``q(p) = 3p² - 2p³`` is convex near zero, so by Jensen ``E[q(p̂)] > q(p)``
      when ``p̂`` is noisy. A pair whose true rate is 0.1 has a 2.8% chance of
      carrying three looks; feed the same pair's three-look draws through here
      and the average answer is 8.4%. Nearly every pair in this corpus sits in
      that regime, so the sum over them inherits the bias.
    * At ``repeats = 3`` the only rates this can see are 0, 1/3, 2/3 and 1, and
      the first and last contribute no variance at all. So
      ``stability_statistics`` collapses to ``0.4382 * sqrt(S)`` at three looks,
      where ``S`` is the number of pairs that split — and the whole *shape* of
      its 3 → 21 curve is a property of that four-point grid, not of the corpus.
      Simulate a corpus with no pair within 0.4 of even odds, whose true spread
      really does fall 0.77 → 0.05 by nine looks, and the three-look plug-in
      still reports 1.49 → 1.18 and calls the variance irreducible.
    * There is a reason three looks cannot do better. ``q(1 - q)`` is degree six
      in ``p``, and ``n`` draws per pair identify ``p¹ … pⁿ`` and no further, so
      the spread is not estimable at all below six looks. Nine — see
      :data:`DEFAULT_ADJUDICATE_REPEATS` — is the first odd count that clears it.

    A cheap check on any figure this produces: the count is a sum of independent
    indicators, so ``Var = Σ q(1 - q) ≤ Σ q = E[count]`` and the spread can never
    exceed ``sqrt`` of the count it is printed beside. The shipped three-look
    artifact (465 pairs at 0/3, 11 at 1/3, 2 at 3/3) puts this estimator at 1.45
    against a ceiling of ``sqrt(2) = 1.41``, and the model behind it expects
    ``11 * 7/27 + 2 = 4.85`` conflicts from the run that returned 2. Both are the
    same bias, showing up in the mean and in the spread.
    """
    from math import comb

    if repeats <= 0:
        return 0.0
    p = votes / repeats
    return sum(
        comb(at_repeats, k) * p**k * (1 - p) ** (at_repeats - k)
        for k in range(at_repeats // 2 + 1, at_repeats + 1)
    )


def stability_statistics(
    pairs: Sequence[CandidatePair],
    votes: Sequence[AdjudicationVote],
) -> dict[str, Any]:
    """How much of the headline count is measurement rather than corpus.

    Two facts, kept apart because the obvious link between them does not hold.
    Two builds of this layer over an identical candidate set returned 4
    conflicts and then 2, with two cards moving 3/3 to 0/3. And the nine-look
    sweep found a genuinely ambiguous band: 20 of its 710 pairs drew conflict
    on between two and seven of their nine looks, where a pair the judge
    answers at even odds carries a majority half the time however often it is
    asked. It is tempting to explain the first by the second — the closing
    paragraph below is why that explanation fails.

    An earlier version of this docstring went further still and called the
    spread *irreducible*, on a curve that barely moved between three looks and
    twenty-one. That was wrong, and wrong in a way worth leaving on the record:
    it was computed from three-look vote shares, which can only place a pair at
    0, 1/3, 2/3 or 1 — and which therefore produce nearly the same curve for
    corpora whose true spread differs by eight orders of magnitude. See
    :data:`DEFAULT_ADJUDICATE_REPEATS` for that simulation. More looks *do*
    reduce the spread within a build, slowly: the count still carries roughly
    ±1 at any repeat count worth paying for.

    So it ships with that spread, and with the two populations separated:

    * ``firm_conflicts`` — carried by at least :data:`FIRM_VOTE_SHARE` of looks.
      These are what the layer is actually asserting.
    * ``undecided_pairs`` — drew conflict on between a third and two thirds of
      looks. Named individually in :attr:`ConflictIndex.near_misses` when they
      lost, on the cards when they won, and countable here either way. A run
      that finds three firm conflicts and six undecided pairs is telling a very
      different story from one that finds three firm and none.

    ``subsample_counts_at_3`` is the direct measurement rather than the
    modelled one: when the run has repeats to spare, its draws are split into
    disjoint groups of three and the whole resolution is re-run on each. Those
    counts are what independent three-look builds of this same corpus would
    have reported, observed rather than predicted — which is the evidence that
    the 4-then-2 discrepancy was the judge and not the layer.

    ``count_sd_estimate`` is only worth reading at six looks or more. See
    :func:`_ships_probability`: below that the spread is not an estimable
    quantity, the number reduces to ``0.4382 * sqrt(split pairs)``, and on the
    three-look artifact this layer shipped before the nine-look sweep it lands
    at 1.45 — above the ``sqrt(count)`` ceiling that a sum of independent
    indicators cannot exceed.
    ``subsample_counts_at_3`` is the honest version and needs the looks to
    spare. Nothing here is gated on the repeat count, because a run that reports
    a bad error bar beside its own histogram is still better than one that
    reports a bare integer, but a reader comparing the two should trust the
    sub-runs.

    **Both numbers are lower bounds, including the honest one.** Each is
    computed from a single run's draws — the plug-in from the vote shares, the
    sub-runs from disjoint thirds of them — so both measure how far the count
    moves when the judge is asked again *inside one build*, and neither can see
    anything that differs between builds. The two builds above say something
    does. Under the iid-per-look model both estimators assume, a pair drawing
    3/3 and then 0/3 has probability ``p³(1 - p)³``, at most ``1/64`` at
    ``p = 1/2``, so two pairs doing it is about ``2e-4``. An observation that
    unlikely under a model is evidence against the model, not a run of bad luck,
    and what it points at is a between-run component — judge version, serving
    stack, or state carried across builds — that no amount of repeating within
    one run samples. So the measured curve (1.90 at three looks, 1.37 at nine,
    1.01 at twenty-one, 0.77 at forty-five) is a **floor** under the spread a
    reader comparing two artifacts actually cares about, not an estimate of it.
    Measuring that one means building twice and matching the runs pair by pair,
    which is the other thing :func:`vote_ledger` exists for.
    """
    if not pairs:
        return {}
    repeats = max((vote.repeats for vote in votes), default=0)
    histogram = {index: 0 for index in range(repeats + 1)}
    for vote in votes:
        histogram[vote.votes] = histogram.get(vote.votes, 0) + 1

    variance = sum(
        _ships_probability(vote.votes, vote.repeats, repeats)
        * (1 - _ships_probability(vote.votes, vote.repeats, repeats))
        for vote in votes
    )
    undecided = [vote for vote in votes if _is_undecided(vote.votes, vote.repeats)]

    stats: dict[str, Any] = {
        # The distribution the single number is drawn from. A run whose pairs
        # are all 0/9 or 9/9 has a trustworthy count; one with a pile at 4/9 and
        # 5/9 does not, and only this tells them apart.
        "vote_histogram": {str(key): value for key, value in sorted(histogram.items())},
        "undecided_pairs": len(undecided),
        "count_sd_estimate": round(variance**0.5, 2),
    }
    if repeats >= 6 and repeats % 3 == 0:
        groups = [
            [AdjudicationVote(verdicts=vote.verdicts[start : start + 3]) for vote in votes]
            for start in range(0, repeats, 3)
        ]
        stats["subsample_counts_at_3"] = [
            len(resolve_conflicts(pairs, group)[0]) for group in groups
        ]
    return stats


def vote_ledger(
    pairs: Sequence[CandidatePair],
    votes: Sequence[AdjudicationVote],
) -> list[dict[str, Any]]:
    """Every adjudicated pair and the rate it drew, in candidate order.

    The rate set this run measured, on the record in full rather than
    summarised. It exists because the claim that once drove this layer's design
    — that the spread on the count is irreducible, and that paying for more
    looks therefore buys little — was argued from a rate set that lived only in
    a build log, and did not survive being checked against one that did not. A
    conclusion about variance has to be recomputable from the artifact by
    someone who did not watch the run, and the histogram alone cannot support
    the other thing this is for: matching a pair to *the same pair* in another
    run, which is what turns two builds into a reproducibility measurement
    instead of two numbers. That is the only route to the between-build
    component of the spread — the one :func:`stability_statistics` can put a
    floor under from a single run but, by construction, cannot see.

    Deliberately the whole population, not the interesting tail. The pairs that
    never drew a conflict verdict are most of the corpus and most of the
    evidence that a low count is a property of the corpus rather than of a
    filter; :func:`near_misses` keeps its own ranked, truncated list for
    reading, and this one is for arithmetic.
    """
    return [
        {
            "chunk_ids": list(pair.key),
            "video_ids": sorted({pair.left.video_id, pair.right.video_id}),
            "votes": vote.votes,
            "repeats": vote.repeats,
            "similarity": round(pair.similarity, 4),
            "errors": sum(1 for verdict in vote.verdicts if verdict.verdict == "error"),
        }
        for pair, vote in zip(pairs, votes)
    ]


def conflict_statistics(
    conflicts: Sequence[Conflict],
    candidates: int,
    tally: dict[str, int],
) -> dict[str, Any]:
    """The deterministic numbers, computed once and stored with the artifact.

    ``conflict_precision`` is the anti-gaming measure: conflicts over candidates
    *adjudicated*, so proposing more candidates can only lower it. There is no
    metric here that rewards volume, and ``conflicts`` on its own is meaningless
    without the denominator beside it — which is why they ship together.
    """
    channels = sorted(
        {side.channel_name for conflict in conflicts for side in (conflict.left, conflict.right)}
    )
    videos = sorted(
        {side.video_id for conflict in conflicts for side in (conflict.left, conflict.right)}
    )
    cross = sum(1 for conflict in conflicts if conflict.cross_channel)
    return {
        "conflicts": len(conflicts),
        "cross_channel_conflicts": cross,
        "within_channel_conflicts": len(conflicts) - cross,
        # Split out because they are not the same claim. A unanimous conflict is
        # one the adjudicator reproduced on every look; a split one is a coin
        # that landed twice the same way, and a reader deciding how much to
        # trust a card needs to know which it is.
        "unanimous_conflicts": sum(1 for conflict in conflicts if conflict.unanimous),
        "split_conflicts": sum(1 for conflict in conflicts if not conflict.unanimous),
        "factual_conflicts": sum(1 for conflict in conflicts if conflict.kind == "factual"),
        "axis_conflicts": sum(1 for conflict in conflicts if conflict.kind != "factual"),
        "candidates_adjudicated": candidates,
        "conflict_precision": round(len(conflicts) / candidates, 4) if candidates else None,
        "channels_involved": len(channels),
        "videos_involved": len(videos),
        "channels": channels,
        "rejected": dict(tally),
        "quotes_resolved": 2 * len(conflicts),
        "min_quote_ratio": round(
            min(
                (
                    side.quote_ratio
                    for conflict in conflicts
                    for side in (conflict.left, conflict.right)
                ),
                default=0.0,
            ),
            4,
        ),
    }


# ─── Calibration probes ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class Probe:
    """One pair with a known answer, run through the real adjudicator.

    Half of these are the point of the whole module and half are the trap it
    must not fall into, because an adjudicator that says "conflict" to
    everything scores perfectly on the first half alone.
    """

    probe_id: str
    expect: str
    why: str
    left_creator: str
    left_text: str
    right_creator: str
    right_text: str


#: Calibration in both directions.
#:
#: ``planted-*`` are flat contradictions written to be unmissable: if these do
#: not surface, the finder is broken and its silence means nothing. They are
#: written rather than lifted from the corpus precisely so that they cannot be
#: mistaken for a finding — nothing planted here is ever reported as a corpus
#: disagreement, and they are never written to any store.
#:
#: ``complementary-*`` are the harder half, and the reason the count in this
#: artifact can be believed. ``complementary-same-subject`` and
#: ``complementary-detail`` are both *about the same subject as each other* and
#: use the same vocabulary, which is exactly what a candidate generator ranks
#: first; an adjudicator that pattern-matches on "two people talking about
#: caching" instead of applying the one-person test fails them.
#: ``complementary-exception`` is the general rule and an exception to it, the
#: shape most often mistaken for disagreement. ``complementary-hedged`` is "do
#: X" beside "X is hard", which one person says in consecutive sentences.
PROBES: tuple[Probe, ...] = (
    Probe(
        probe_id="planted-flat",
        expect="conflict",
        why="One says never, the other says always, about the same decision.",
        left_creator="Planted creator A",
        left_text=(
            "So my rule is simple. You should never put a cache in front of your "
            "database. Caching is the single worst decision you can make in a "
            "system like this, because it always ends up serving stale data to "
            "your users and you will spend the rest of the year chasing bugs. "
            "Just make the database faster instead. Do not cache. Ever."
        ),
        right_creator="Planted creator B",
        right_text=(
            "The first thing I do on every single design is put a cache in front "
            "of the database. You should always cache, without exception, because "
            "reads are what kill you at scale and nothing else you do will buy you "
            "that much headroom for that little work. If you take one thing away "
            "from this video, always put a cache in front of your database."
        ),
    ),
    Probe(
        probe_id="planted-number",
        expect="conflict",
        why="The same quantity, given two incompatible values as advice.",
        left_creator="Planted creator C",
        left_text=(
            "Keep your resume to a single page. One page, no exceptions, even if "
            "you have twenty years of experience. Anything longer than one page "
            "does not get read, and the second page is where good candidates go to "
            "be ignored. If it does not fit on one page, cut it until it does."
        ),
        right_creator="Planted creator D",
        right_text=(
            "A one page resume is a mistake and I want you to stop doing it. Two "
            "pages is the correct length for anybody with real experience, and "
            "cutting your work down to one page is how you delete the very "
            "achievements that would have got you the interview. Use two pages."
        ),
    ),
    Probe(
        probe_id="complementary-same-subject",
        expect="complementary",
        why=(
            "Same subject, same vocabulary, different aspects of it — the shape a "
            "candidate generator ranks first and a lazy adjudicator calls a fight."
        ),
        left_creator="Complementary creator A",
        left_text=(
            "A cache is only useful when your read pattern is skewed. If ninety "
            "percent of your traffic is asking for the same one percent of the "
            "keys, a cache in front of the database will take almost all of that "
            "load off it, and that is the case where I reach for one."
        ),
        right_creator="Complementary creator B",
        right_text=(
            "The thing people forget about caches is invalidation. Once you put a "
            "cache in front of the database you now own the problem of deciding "
            "when the cached copy is wrong, and you need a strategy for that, "
            "whether that is a short time to live or writing through the cache."
        ),
    ),
    Probe(
        probe_id="complementary-exception",
        expect="complementary",
        why="A general rule and an exception to it. One person holds both, routinely.",
        left_creator="Complementary creator C",
        left_text=(
            "Tailor your resume to every single job you apply for. Read the "
            "posting, work out what they actually need, and rewrite your bullets "
            "so the first thing they see is the thing they asked for. Sending the "
            "same document to everybody is why you are not hearing back."
        ),
        right_creator="Complementary creator D",
        right_text=(
            "There is one case where you do not tailor, and that is a referral. "
            "When somebody inside the company is walking your resume to the hiring "
            "manager, the generic version is fine, because a human is already "
            "vouching for you and nobody is screening that document."
        ),
    ),
    Probe(
        probe_id="complementary-hedged",
        expect="complementary",
        why=(
            "'Do X' beside 'X is hard'. One person says both of these in "
            "consecutive sentences all the time."
        ),
        left_creator="Complementary creator E",
        left_text=(
            "So when you are asked about delivery guarantees in an interview, you "
            "want to talk about exactly once delivery, because that is what the "
            "interviewer is listening for and it shows you know the three "
            "semantics and which one the business actually needs."
        ),
        right_creator="Complementary creator F",
        right_text=(
            "Exactly once delivery is genuinely hard to achieve in practice. There "
            "are so many points of failure in a distributed queue, the producer, "
            "the replication, the consumer, that most systems you will actually "
            "meet settle for at least once delivery and deduplicate downstream."
        ),
    ),
)


def probe_pair(probe: Probe) -> CandidatePair:
    """A probe as a candidate pair, so it goes through the shipped adjudicator.

    Chunk ids are namespaced ``probe:`` so that a probe can never collide with a
    real chunk id, and so that anything derived from a probe is identifiable as
    such anywhere downstream.
    """

    def side(suffix: str, creator: str, text: str) -> ClaimRef:
        return ClaimRef(
            claim_id=f"probe:{probe.probe_id}:{suffix}#0",
            text=text,
            chunk_id=f"probe:{probe.probe_id}:{suffix}",
            chunk_text=text,
            video_id=f"probe-{probe.probe_id}-{suffix}",
            channel_name=creator,
            title=f"Calibration probe {probe.probe_id} ({suffix})",
            start_seconds=0.0,
            end_seconds=0.0,
        )

    return CandidatePair(
        left=side("a", probe.left_creator, probe.left_text),
        right=side("b", probe.right_creator, probe.right_text),
        similarity=1.0,
    )


def run_probes(
    adjudicate: AdjudicateFn,
    probes: Sequence[Probe] = PROBES,
    *,
    repeats: int = 1,
) -> list[dict[str, Any]]:
    """Every probe through the real adjudicator, ``repeats`` times each.

    The adjudicator is an LLM and does not have to agree with itself, so a
    single pass proves nothing about the next one. Each probe records every
    verdict it drew and whether they were unanimous; a probe that passes 3/5 is
    a probe that is telling you the threshold is in the wrong place, and the
    artifact should say so rather than round it to a tick.
    """
    results: list[dict[str, Any]] = []
    for probe in probes:
        pair = probe_pair(probe)
        verdicts = [adjudicate(pair) for _ in range(max(1, repeats))]
        got = [verdict.verdict for verdict in verdicts]
        passes = [value == probe.expect for value in got]
        conflict_verdict = next((v for v in verdicts if v.is_conflict), None)
        results.append(
            {
                "probe_id": probe.probe_id,
                "expect": probe.expect,
                "why": probe.why,
                "verdicts": got,
                "passed": all(passes),
                "unanimous": len(set(got)) == 1,
                # The axis is only meaningful for a probe that was *supposed* to
                # be a conflict; recording it is how "surfaced as a conflict, not
                # fused into one blended statement" is checkable rather than
                # asserted — two named positions, or it did fuse them.
                "axis": conflict_verdict.axis if conflict_verdict else "",
                "position_a": conflict_verdict.position_a if conflict_verdict else "",
                "position_b": conflict_verdict.position_b if conflict_verdict else "",
            }
        )
    return results


def probes_passed(results: Iterable[dict[str, Any]]) -> bool:
    """Whether every probe drew its expected verdict on every repeat."""
    rows = list(results)
    return bool(rows) and all(row.get("passed") for row in rows)


# ─── Store ───────────────────────────────────────────────────────────────────


class ConflictStore:
    """The conflict layer on disk: one JSON artifact, wholly derived.

    A file rather than a collection for the same reason as
    :class:`src.rag.themes.ThemeStore` — nothing queries conflicts by vector,
    the UI lists them and links out to the videos, and a plain file can be read
    by the API without opening the embedding stack.

    Unlike the theme artifact this one **does** store its quotes, because a
    quote here is not a pointer into a chunk but a specific span *within* one
    that the adjudicator picked out. ``chunk_id`` and the timestamps are stored
    beside it so the span can always be re-checked against the store, which is
    what :func:`verbatim_span` does at build time and what a reviewer can redo.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def exists(self) -> bool:
        return self.path.is_file()

    def load(self) -> ConflictIndex | None:
        if not self.path.is_file():
            return None
        return ConflictIndex.model_validate_json(self.path.read_text(encoding="utf-8"))

    def save(self, index: ConflictIndex) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(index.model_dump(mode="json"), indent=2, sort_keys=False) + "\n",
            encoding="utf-8",
        )
        return self.path


# ─── Build ───────────────────────────────────────────────────────────────────


def build_conflicts(
    claims: Sequence[ClaimRef],
    embed: EmbedFn,
    adjudicate: AdjudicateFn,
    *,
    config: ConflictConfig | None = None,
    embedding_model: str = "",
    adjudicator_model: str = "",
    chunk_collection: str = "",
    corpus: dict[str, Any] | None = None,
    probe_repeats: int = 1,
    max_workers: int = 6,
    cache: AdjudicationCache | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> ConflictIndex:
    """Candidates, then adjudication, then resolution — the whole layer.

    The probes run **first**. If the adjudicator is calling everything a
    conflict, the conflicts it then reports are worthless, and finding that out
    after spending two hundred calls on the corpus is finding it out too late.
    Their results ship inside the artifact rather than being printed and
    forgotten, so the file carries the evidence that its own adjudicator was
    calibrated when it produced these conflicts.

    Corpus pairs get the **same repeat-and-vote discipline as the probes**. They
    did not, once, and that was this module's worst bug: the probes proved the
    adjudicator was stable on five written pairs while every card that shipped
    rested on a single draw. The calibration strip was certifying a property the
    cards had never been tested for.

    The vote is the *only* gate a pair passes into
    :attr:`ConflictIndex.conflicts`, and every consumer — the view, ``stats``,
    ``conflict_precision`` and the ``contested_coverage`` denominator in
    :func:`src.evals.critique_run.contested_pairs` — reads that one list. A pair
    that loses its vote is therefore absent from all of them by construction
    rather than by four separate filters agreeing with each other, which is how
    the first version of this managed to have them disagree.
    """
    settings = config or ConflictConfig()
    progress = on_progress or (lambda _message: None)
    repeats = max(1, settings.adjudicate_repeats)

    probes = run_probes(adjudicate, repeats=probe_repeats)
    progress(
        f"probes: {sum(1 for row in probes if row['passed'])}/{len(probes)} passed "
        f"({probe_repeats} repeat(s) each)"
    )

    pairs = candidate_pairs(claims, embed, settings)
    progress(
        f"{len(claims)} claims -> {len(pairs)} candidate chunk pairs "
        f"x {repeats} repeat(s) = {len(pairs) * repeats} adjudications"
    )

    votes = _adjudicate_all(pairs, adjudicate, repeats, max_workers, progress, cache)
    conflicts, tally = resolve_conflicts(pairs, votes)
    flapping = sum(1 for vote in votes if not vote.unanimous)
    progress(
        f"{len(conflicts)} conflicts kept; {flapping} of {len(pairs)} pairs drew "
        f"more than one verdict; rejected {tally}"
    )

    return ConflictIndex(
        generated_at=datetime.now(timezone.utc).isoformat(),
        embedding_model=embedding_model,
        adjudicator_model=adjudicator_model,
        chunk_collection=chunk_collection,
        corpus={
            **(corpus or {}),
            # Chunks that actually reached the candidate stage. A chunk with no
            # cached extraction contributes no claims and is invisible here
            # however much it disagrees with anything, so the gap between this
            # and ``chunks`` is the layer's blind spot, stated rather than left
            # for a reader to infer from a claim count.
            "chunks_with_claims": len({claim.chunk_id for claim in claims}),
            "videos_with_claims": len({claim.video_id for claim in claims}),
        },
        config=asdict(settings),
        stats={
            **conflict_statistics(conflicts, len(pairs), tally),
            **stability_statistics(pairs, votes),
            "firm_conflicts": sum(
                1 for conflict in conflicts if _is_firm(conflict.votes, conflict.repeats)
            ),
            "claims": len(claims),
            "probes_passed": probes_passed(probes),
            "probe_repeats": probe_repeats,
            "adjudicate_repeats": repeats,
            "adjudications": len(pairs) * repeats,
            # How often the adjudicator disagreed with itself about the same
            # pair. Published because it is the honest error bar on every other
            # number in this artifact.
            "pairs_with_split_verdicts": flapping,
            "verdict_agreement": round(1 - flapping / len(pairs), 4) if pairs else None,
        },
        probes=probes,
        vote_ledger=vote_ledger(pairs, votes),
        near_misses=near_misses(pairs, votes),
        conflicts=conflicts,
    )


def _adjudicate_all(
    pairs: Sequence[CandidatePair],
    adjudicate: AdjudicateFn,
    repeats: int,
    max_workers: int,
    progress: Callable[[str], None],
    cache: AdjudicationCache | None = None,
) -> list[AdjudicationVote]:
    """Every pair adjudicated ``repeats`` times, concurrently, in input order.

    Repeats are submitted as individual tasks rather than looped inside one, so
    the pool is saturated by ``pairs * repeats`` work and three passes cost
    roughly three times the wall clock of one rather than three times the wall
    clock of a serial pass.

    Order is preserved because :func:`resolve_conflicts` awards a contested
    chunk to the first pair that claims it, and "first" has to mean highest
    similarity rather than whichever call the endpoint happened to answer
    first — otherwise the artifact would differ between two runs over identical
    input.

    The optional ``cache`` is consulted here rather than inside ``adjudicate``
    because this is the only place that knows *which attempt* a call is — and
    the attempt has to be in the key or the nine looks collapse into one. See
    :class:`AdjudicationCache`.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if not pairs:
        return []
    total = len(pairs) * repeats
    results: list[list[Adjudication | None]] = [[None] * repeats for _ in pairs]
    workers = max(1, min(max_workers, total))
    done = 0

    def work(pair: CandidatePair, attempt: int) -> Adjudication:
        if cache is not None:
            cached = cache.get(pair, attempt)
            if cached is not None:
                return cached
        verdict = adjudicate(pair)
        if cache is not None:
            cache.put(pair, attempt, verdict)
        return verdict

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(work, pair, attempt): (index, attempt)
            for index, pair in enumerate(pairs)
            for attempt in range(repeats)
        }
        for future in as_completed(futures):
            index, attempt = futures[future]
            results[index][attempt] = future.result()
            done += 1
            if done % 50 == 0 or done == total:
                served = f", {cache.hits} from cache" if cache is not None else ""
                progress(f"adjudicated {done}/{total}{served}")
    return [
        AdjudicationVote(
            verdicts=tuple(
                result or Adjudication(verdict="error", error="no result") for result in row
            )
        )
        for row in results
    ]


def _json_object(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Adjudicator response was not valid JSON: {content[:200]}") from exc
    if not isinstance(value, dict):
        raise ValueError("Adjudicator response JSON must be an object")
    return value

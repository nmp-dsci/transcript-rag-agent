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
* a low count is a legitimate result and is reported as one. The statistics
  carry ``candidates_adjudicated`` beside ``conflicts`` so that a run which
  looked at 200 pairs and found 4 cannot be confused with one that looked at 6.

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
    def cross_creator(self) -> bool:
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
    cross_creator: bool


class ConflictIndex(BaseModel):
    """The artifact on disk: derived state, rebuilt by ``index-conflicts``."""

    version: int = 1
    generated_at: str
    embedding_model: str = ""
    adjudicator_model: str = ""
    chunk_collection: str = ""
    config: dict[str, Any] = Field(default_factory=dict)
    stats: dict[str, Any] = Field(default_factory=dict)
    #: Calibration results — see :data:`PROBES`. Stored with the run so the
    #: artifact carries the evidence that its adjudicator was working in both
    #: directions on the day it produced these conflicts.
    probes: list[dict[str, Any]] = Field(default_factory=list)
    conflicts: list[Conflict] = Field(default_factory=list)


# ─── Configuration ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ConflictConfig:
    """Knobs for candidate generation. All deterministic; none reach the LLM."""

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
    max_candidates: int = 240
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
    #: Only pairs from different creators are adjudicated by default. Two videos
    #: by one creator restating a position is not a corpus disagreement, and
    #: within-creator "conflicts" are overwhelmingly one person qualifying
    #: themselves ("usually X, but when Y, not X") — the textbook shape of
    #: something one person can hold entirely.
    cross_creator_only: bool = True


#: Similarity above which the adjudicator's quote counts as present in the
#: chunk. Below 1.0 because a model retyping a quote drops a filler word or
#: normalises "gonna"; well above chance because the check exists to catch
#: invention. Matches :data:`src.evals.critique.QUOTE_MATCH_RATIO` on purpose —
#: a quote that ships from here must pass the same bar the critique scorer sets.
QUOTE_MATCH_RATIO = 0.80

#: Words a quote must have before it is worth showing. A three-word "it depends"
#: resolves perfectly and proves nothing.
MIN_QUOTE_WORDS = 8

_WORD = re.compile(r"[a-z0-9']+")


def normalize(text: str) -> list[str]:
    """Lowercase word tokens — the comparable form of transcript text."""
    return _WORD.findall(str(text).lower())


# ─── Claims, with provenance stamped from the chunk ──────────────────────────


#: ``chunk -> the claim sentences extracted from it``. The CLI wires this to the
#: cached GraphRAG extractions; a list is enough for the tests. Typed on ``Any``
#: rather than a ``TranscriptChunk`` because the only thing this layer needs
#: from a chunk is the handful of attributes read below, and importing the
#: storage models here would drag chromadb into a module that is otherwise pure.
ClaimTextsFn = Callable[[Any], Sequence[str]]


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

    Same-video and (by default) same-creator pairs are masked to ``-1`` rather
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
    if config.cross_creator_only:
        creators = np.array([claim.channel_name.strip().lower() for claim in claims])
        scores[creators[:, None] == creators[None, :]] = -1.0
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

Quotes: copy the words VERBATIM from the excerpt, including any transcription
errors. Do not tidy, do not paraphrase, do not join two separated sentences.
Each quote must be at least 8 words and must be the words that state that
side's position.

Return only JSON:
{"verdict": "conflict" | "complementary" | "unrelated",
 "axis": "the one question they answer differently, as a question",
 "why_incompatible": "why one person could not hold both",
 "position_a": "excerpt A's answer, one line",
 "position_b": "excerpt B's answer, one line",
 "quote_a": "verbatim from excerpt A",
 "quote_b": "verbatim from excerpt B"}

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
    error: str | None = None

    @property
    def is_conflict(self) -> bool:
        return self.verdict == "conflict"


#: ``(pair) -> an Adjudication``. The shipped implementation is
#: :class:`LlmAdjudicator`; the tests pass a dict-backed fake.
AdjudicateFn = Callable[[CandidatePair], Adjudication]


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
    adjudications: Sequence[Adjudication],
) -> tuple[list[Conflict], dict[str, int]]:
    """Adjudicated pairs into conflicts, with the rejections counted.

    Four gates, in order, each of which drops the pair and is tallied so a run
    can say *why* it found few conflicts rather than only that it did:

    1. the adjudicator said complementary, unrelated, or errored;
    2. it said conflict but could not name an axis or say why the two positions
       are incompatible — a conflict nobody can state is not reportable;
    3. one side's quote is not in the stored transcript (see
       :func:`verbatim_span`);
    4. one of the two chunks already backs a conflict. This is the exclusivity
       rule from :func:`src.evals.critique.ground_findings`: distinct evidence
       per claim, so the count cannot be inflated by restating one disagreement.

    Order matters for gate 4 — the highest-similarity pair wins the chunk — and
    ``pairs`` arrives sorted, so the result does not depend on adjudication
    completion order.
    """
    tally = {
        "not_a_conflict": 0,
        "unstated_axis": 0,
        "quote_not_in_transcript": 0,
        "duplicate_evidence": 0,
    }
    conflicts: list[Conflict] = []
    spent: set[str] = set()
    for pair, verdict in zip(pairs, adjudications):
        if not verdict.is_conflict:
            tally["not_a_conflict"] += 1
            continue
        if not verdict.axis or not verdict.why_incompatible:
            tally["unstated_axis"] += 1
            continue
        left = verbatim_span(verdict.quote_a, pair.left.chunk_text)
        right = verbatim_span(verdict.quote_b, pair.right.chunk_text)
        if left is None or right is None:
            tally["quote_not_in_transcript"] += 1
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
                cross_creator=pair.cross_creator,
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
    creators = sorted(
        {side.channel_name for conflict in conflicts for side in (conflict.left, conflict.right)}
    )
    videos = sorted(
        {side.video_id for conflict in conflicts for side in (conflict.left, conflict.right)}
    )
    cross = sum(1 for conflict in conflicts if conflict.cross_creator)
    return {
        "conflicts": len(conflicts),
        "cross_creator_conflicts": cross,
        "within_creator_conflicts": len(conflicts) - cross,
        "candidates_adjudicated": candidates,
        "conflict_precision": round(len(conflicts) / candidates, 4) if candidates else None,
        "creators_involved": len(creators),
        "videos_involved": len(videos),
        "creators": creators,
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
    probe_repeats: int = 1,
    max_workers: int = 6,
    on_progress: Callable[[str], None] | None = None,
) -> ConflictIndex:
    """Candidates, then adjudication, then resolution — the whole layer.

    The probes run **first**. If the adjudicator is calling everything a
    conflict, the conflicts it then reports are worthless, and finding that out
    after spending two hundred calls on the corpus is finding it out too late.
    Their results ship inside the artifact rather than being printed and
    forgotten, so the file carries the evidence that its own adjudicator was
    calibrated when it produced these conflicts.
    """
    settings = config or ConflictConfig()
    progress = on_progress or (lambda _message: None)

    probes = run_probes(adjudicate, repeats=probe_repeats)
    progress(
        f"probes: {sum(1 for row in probes if row['passed'])}/{len(probes)} passed "
        f"({probe_repeats} repeat(s) each)"
    )

    pairs = candidate_pairs(claims, embed, settings)
    progress(f"{len(claims)} claims -> {len(pairs)} candidate chunk pairs to adjudicate")

    adjudications = _adjudicate_all(pairs, adjudicate, max_workers, progress)
    conflicts, tally = resolve_conflicts(pairs, adjudications)
    progress(f"{len(conflicts)} conflicts kept; rejected {tally}")

    return ConflictIndex(
        generated_at=datetime.now(timezone.utc).isoformat(),
        embedding_model=embedding_model,
        adjudicator_model=adjudicator_model,
        chunk_collection=chunk_collection,
        config=asdict(settings),
        stats={
            **conflict_statistics(conflicts, len(pairs), tally),
            "claims": len(claims),
            "probes_passed": probes_passed(probes),
            "probe_repeats": probe_repeats,
        },
        probes=probes,
        conflicts=conflicts,
    )


def _adjudicate_all(
    pairs: Sequence[CandidatePair],
    adjudicate: AdjudicateFn,
    max_workers: int,
    progress: Callable[[str], None],
) -> list[Adjudication]:
    """Every pair adjudicated, concurrently, preserving input order.

    Order is preserved because :func:`resolve_conflicts` awards a contested
    chunk to the first pair that claims it, and "first" has to mean highest
    similarity rather than whichever call the endpoint happened to answer
    first — otherwise the artifact would differ between two runs over identical
    input.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if not pairs:
        return []
    results: list[Adjudication | None] = [None] * len(pairs)
    workers = max(1, min(max_workers, len(pairs)))
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(adjudicate, pair): index for index, pair in enumerate(pairs)}
        for future in as_completed(futures):
            index = futures[future]
            results[index] = future.result()
            done += 1
            if done % 20 == 0 or done == len(pairs):
                progress(f"adjudicated {done}/{len(pairs)}")
    return [result or Adjudication(verdict="error", error="no result") for result in results]


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

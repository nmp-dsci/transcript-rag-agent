"""Scoring a critique against an expert the system was never allowed to read.

Every other harness in ``src/evals`` measures retrieval or answer quality against
labels somebody wrote *for* this corpus. This one measures something the product
claim actually rests on: **can the distilled corpus reach a named expert's
conclusions without having seen that expert.**

The experiment
--------------
One video is held out — a resume teardown in which the reviewer walks two
documents top to bottom. The *criteria* that reviewer applies are extracted by
hand into :data:`DEFAULT_DATASET_PATH`, each with the timestamp it was said at:
not the specific criticisms ("this summary is three lines too long") but the
general, checkable rules behind them ("a professional summary should be two to
three lines, because the work experience proves the same thing"). The system then
reviews a *different* artifact — the user's portfolio — retrieving only from the
other 52 videos, and the findings it produces independently are matched against
that held-out list.

Note what this is **not**. It is not "did the system find the same faults in the
same document": those resumes exist only as pixels in a video and are not in any
corpus. It is "did the system arrive at the same *rules*", which is the
transferable thing an expert actually has and the only part of the teardown that
could be reached from other videos at all.

The four measures
-----------------
``criteria_recall``
    Held-out criteria the system applied **in a finding the corpus actually paid
    for** — see the grounding gate below, which is what stops this being a test
    of what the answering model already knew. Matching is semantic, not id
    overlap (two people phrase the same rule differently), and it is the one
    thing here that could not be made deterministic: calibration showed the
    shipped bi-encoder cannot separate a restatement of a rule from a *different*
    rule about the same subject (see :data:`MATCH_THRESHOLD`). So it is an LLM
    call, repeated :data:`DEFAULT_MATCH_REPEATS` times and resolved by per-criterion
    majority vote, and the run reports the **spread across those repeats** beside
    the score. One call's pairing moved this number by ±20% relative on a fixed
    matrix; a scorer whose noise is the size of the effect later slices are
    trying to show is not a scorer.
``evidence_precision``
    Findings resting on **exclusive** evidence: at least one resolving citation
    that no other finding also cites. Not merely "has a citation that resolves" —
    see :func:`ground_findings`.
``provenance``
    Citations whose quoted words are actually present in the transcript at the
    timestamp cited. Pure string work against the chunk store: no LLM, no judge.
``contested_rate``
    Findings that surfaced a disagreement between videos instead of averaging it
    away, counted only when at least two *distinct* videos are cited and their
    citations resolve. A reporting-behaviour measure, not an accuracy one.

The grounding gate, and the attack that forced it
-------------------------------------------------
An independent reviewer ran this scorer over 18 findings that simply restated the
18 applicable criteria, **each stapled to the same one real quote** lifted from a
committed run, and scored ``criteria_recall 1.000 · evidence_precision 1.000 ·
provenance 1.000``. A perfect sweep from a system that recited generic advice it
already knew. Two holes made that work, and both are now closed:

* ``evidence_precision`` was a plain rate, so *extra findings were free*. It now
  requires **exclusive** evidence, so N findings need N distinct chunks; the
  attack's 18 findings share one chunk and none of them is grounded.
* ``criteria_recall`` counted a match to any finding at all. It now counts only
  matches to **grounded** findings, because the whole claim being measured is
  that the *corpus* reached the expert's conclusions — a finding with no corpus
  evidence behind it came from the model's prior, and crediting it measures the
  opposite of what this harness is for.

Together these also restore the trade-off that an earlier version of this
docstring claimed but did not have: recall is now bounded by how many distinct
pieces of corpus evidence the system actually retrieved and used, so padding the
output can no longer buy it. Note what that costs: a single chunk that genuinely
supports two different rules is penalised. That is the conservative direction on
purpose — this is the baseline every later slice must beat, and a baseline scorer
should under-credit rather than over-credit.

**There is deliberately no composite.** A weighted blend of these four is exactly
the shape that lets a run look better while getting worse — the matrix has
already produced a cell whose composite rose 0.368 because context_precision hit
1.000 on a narrowed context while answer_correctness collapsed to 0.137. These
four measure different things (coverage, grounding, honesty, calibration) and
trade against each other by design. Averaging them would hide the trade, so they
are reported side by side and a later slice has to beat the baseline on the
metric it claims to improve.

Everything here is pure: dataset in, findings in, scores out. The retrieval, the
answering call and the chunk-store lookups live in :mod:`src.evals.critique_run`,
behind the callables this module declares.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

#: Metrics reported per setup, in table order — coverage, then grounding, then
#: the two behavioural ones. See the module docstring for why no composite.
CRITIQUE_METRICS: list[str] = [
    "criteria_recall",
    "evidence_precision",
    "provenance",
    "contested_rate",
]

DEFAULT_DATASET_PATH = Path(__file__).resolve().parent / "critique_dataset.json"

#: Cosine above which a finding counts as having applied a criterion, for the
#: *offline* matcher only.
#:
#: This threshold does not exist. Calibrating it on this dataset with the
#: shipped MiniLM bi-encoder found no separating value: hand-written
#: restatements of a criterion in someone else's words score 0.25-0.63 against
#: their own criterion, while the highest cosine between two criteria that are
#: genuinely *different* rules is 0.599 (c07 "build a flagship project in the
#: target domain" vs c19 "reframe the earlier career as its foundation"). The
#: bands overlap almost completely, because every criterion in this list is a
#: sentence about resumes and careers and a bi-encoder trained for retrieval
#: compresses them into one neighbourhood.
#:
#: So the shipped default is :func:`~src.evals.critique_run.llm_matcher`, and
#: this constant is what :func:`embedding_matcher` uses when a run wants a
#: model-free matcher for a smoke test. 0.60 is kept as the value that at least
#: sits above every distinct pair — it buys precision at the cost of recall, and
#: a run using it should be read as a lower bound, not as a measurement.
MATCH_THRESHOLD = 0.60

#: How far a cited timestamp may sit from a chunk's span and still be taken to
#: name that chunk. The minimum gap between consecutive chunk starts anywhere in
#: this corpus is 39.7s, so ±20s cannot make two chunks both plausible — the
#: window is wide enough to absorb a model rounding 179.2 to 180, and far too
#: narrow to let a citation drift onto its neighbour.
TIMESTAMP_TOLERANCE_SECONDS = 20.0

#: Similarity above which a quote counts as present in a chunk. Below 1.0
#: because a model transcribing a quote out of context reliably drops a filler
#: word or normalises "gonna"; well above chance because the check exists to
#: catch invention, and an invented quote scores nothing like 0.8 against real
#: transcript text.
QUOTE_MATCH_RATIO = 0.80

#: How many times the matcher is run before its pairings are resolved by vote.
#: Odd, so a per-criterion majority always exists. Five was chosen by measuring
#: the thing it fixes: 13 single-call matcher runs over one byte-identical
#: 24x9 matrix produced applicable-match counts of 6,6,5,5,5,5,5,5 and (at
#: temperature 0) 5,6,5,5,6 — the score moved ±20% relative with the system's
#: output held completely fixed, and pinning temperature did not help. The vote
#: is over the *pairing*, not the count, because the pairing drifted (c02 vs c22
#: swapping) even in runs whose totals agreed.
DEFAULT_MATCH_REPEATS = 5

#: Version of the exclusion mechanism the run was produced under. Part of the
#: cache fingerprint (see :func:`src.evals.matrix_cache.cell_fingerprint`), so a
#: change to *how* a video is held out invalidates cells scored under the old
#: way rather than silently reusing them.
EXCLUSION_VERSION = "v1"

_WORD = re.compile(r"[a-z0-9']+")


def normalize(text: str) -> list[str]:
    """Lowercase word tokens — the comparable form of transcript text.

    Transcripts carry no markup and inconsistent punctuation, so comparing
    quotes on characters would fail on a comma the model did not reproduce.
    """
    return _WORD.findall(str(text).lower())


# ── the held-out dataset ──────────────────────────────────────────────────


@dataclass(frozen=True)
class Criterion:
    """One general rule the held-out expert applies, and where they said it.

    ``criterion`` is the checkable rule, stated so it could be applied to a
    document nobody has seen. ``quote`` and ``start_seconds`` are the evidence
    that the rule is really in the video rather than something the dataset's
    author believes — both are verified against the transcript by
    :func:`verify_dataset`, so a criterion cannot drift away from its source.

    ``applies_to`` is load-bearing. The held-out video reviews *resumes*, and
    some of what it says is about resume mechanics ("keep it to one page", "write
    it in LaTeX so an agent can check it") that a portfolio website has no
    equivalent of. Scoring a portfolio review against those would report a
    ceiling the system could not reach for reasons that have nothing to do with
    retrieval, so the headline recall is over the applicable subset and the
    whole-list recall is reported beside it.
    """

    id: str
    criterion: str
    quote: str
    video_id: str
    start_seconds: float
    applies_to: tuple[str, ...] = ()
    note: str | None = None
    #: Criteria that are the same rule said twice. Defaults to the criterion's
    #: own id, so an ungrouped rule is a group of one and the grouped recall
    #: equals the per-criterion one when nothing is grouped.
    group: str = ""

    def applies(self, kind: str) -> bool:
        return kind in self.applies_to

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "criterion": self.criterion,
            "quote": self.quote,
            "video_id": self.video_id,
            "start_seconds": self.start_seconds,
            "applies_to": list(self.applies_to),
            "note": self.note,
            "group": self.group,
        }


@dataclass(frozen=True)
class CritiqueDataset:
    """The held-out expert, the artifact under review, and the criteria list."""

    held_out_video_id: str
    held_out_title: str
    artifact_id: str
    artifact_url: str
    artifact_kind: str
    criteria: tuple[Criterion, ...]

    def applicable(self) -> list[Criterion]:
        return [c for c in self.criteria if c.applies(self.artifact_kind)]


def load_critique_dataset(path: Path | None = None) -> CritiqueDataset:
    data = json.loads((path or DEFAULT_DATASET_PATH).read_text(encoding="utf-8"))
    return CritiqueDataset(
        held_out_video_id=str(data["held_out_video_id"]),
        held_out_title=str(data.get("held_out_title", "")),
        artifact_id=str(data["artifact_id"]),
        artifact_url=str(data["artifact_url"]),
        artifact_kind=str(data.get("artifact_kind", "document")),
        criteria=tuple(
            Criterion(
                id=str(row["id"]),
                criterion=str(row["criterion"]),
                quote=str(row["quote"]),
                video_id=str(row.get("video_id", data["held_out_video_id"])),
                start_seconds=float(row["start_seconds"]),
                applies_to=tuple(str(k) for k in row.get("applies_to", [])),
                note=row.get("note"),
                group=str(row.get("group") or row["id"]),
            )
            for row in data["criteria"]
        ),
    )


# ── what the system produced ──────────────────────────────────────────────


@dataclass(frozen=True)
class Citation:
    """One (video, timestamp, quote) triple a finding rests on."""

    video_id: str
    start_seconds: float
    quote: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "start_seconds": self.start_seconds,
            "quote": self.quote,
        }


@dataclass(frozen=True)
class Finding:
    """One point the system made about the artifact.

    ``criterion`` is the general rule it applied — the field matched against the
    held-out list. ``detail`` is what it said about *this* artifact, which is
    what a reader wants and what nothing is scored on: two systems can apply the
    same rule and phrase the specific criticism completely differently.
    """

    id: str
    criterion: str
    detail: str
    citations: tuple[Citation, ...] = ()
    contested: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "criterion": self.criterion,
            "detail": self.detail,
            "citations": [c.to_dict() for c in self.citations],
            "contested": self.contested,
        }


def parse_findings(payload: Any) -> list[Finding]:
    """Findings out of the answering call's JSON, skipping anything unusable.

    A malformed finding is dropped rather than repaired: an entry with no
    criterion text cannot be matched against anything, and inventing a criterion
    for it would put a score on the parser instead of on the system.
    """
    rows = payload.get("findings") if isinstance(payload, dict) else payload
    findings: list[Finding] = []
    for index, row in enumerate(rows or []):
        if not isinstance(row, dict):
            continue
        criterion = str(row.get("criterion") or "").strip()
        if not criterion:
            continue
        citations: list[Citation] = []
        for cite in row.get("citations") or []:
            if not isinstance(cite, dict):
                continue
            video_id = str(cite.get("video_id") or "").strip()
            quote = str(cite.get("quote") or "").strip()
            start = cite.get("start_seconds")
            if not video_id or not quote or not isinstance(start, (int, float)):
                continue
            citations.append(Citation(video_id=video_id, start_seconds=float(start), quote=quote))
        findings.append(
            Finding(
                id=str(row.get("id") or f"f{index + 1:02d}"),
                criterion=criterion,
                detail=str(row.get("detail") or "").strip(),
                citations=tuple(citations),
                contested=bool(row.get("contested")),
            )
        )
    return findings


# ── provenance: does the quote really appear where it was cited? ──────────

#: ``(video_id, start_seconds) -> candidate (chunk_id, text) pairs``. Supplied by
#: :mod:`src.evals.critique_run` over the real chunk store; a dict-backed fake
#: is enough for the tests.
#:
#: The chunk *id* is carried, not just the text, because two citations a few
#: seconds apart are the same piece of evidence and :func:`ground_findings` has
#: to be able to say so. Timestamps alone cannot: chunks overlap, so 140.6 and
#: 145.0 are one chunk while looking like two citations.
ChunkTextFn = Callable[[str, float], list[tuple[str, str]]]


@dataclass
class CitationCheck:
    """Whether one citation resolved, and why not when it did not.

    Not frozen: ``shared`` is decided by :func:`ground_findings`, which can only
    know it once every finding's citations have been resolved.
    """

    citation: Citation
    resolved: bool
    reason: str
    ratio: float = 0.0
    #: Which chunk the quote was found in. The unit of evidence, so two
    #: citations that land here together are one piece of evidence.
    chunk_id: str | None = None
    #: Set when another finding also rests on this chunk (see
    #: :func:`ground_findings`). Recorded rather than silently discounted,
    #: because "your evidence is somebody else's" is the diagnosis.
    shared: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.citation.to_dict(),
            "resolved": self.resolved,
            "reason": self.reason,
            "ratio": round(self.ratio, 3),
            "chunk_id": self.chunk_id,
            "shared": self.shared,
        }


def quote_ratio(quote: str, text: str) -> float:
    """How completely ``quote``'s words appear, in order, inside ``text``.

    Token-level rather than character-level, and one-sided: the score is the
    share of the *quote* that the chunk contains, so quoting eight words out of a
    seventy-word chunk is a perfect match rather than a poor one. That is the
    right asymmetry — the question is whether the speaker said this, not whether
    this is all they said.
    """
    needle = normalize(quote)
    haystack = normalize(text)
    if not needle or not haystack:
        return 0.0
    matcher = SequenceMatcher(a=needle, b=haystack, autojunk=False)
    matched = sum(block.size for block in matcher.get_matching_blocks())
    return matched / len(needle)


def check_citation(citation: Citation, chunk_text: ChunkTextFn) -> CitationCheck:
    """Resolve a citation to transcript text and confirm the quote is in it."""
    candidates = chunk_text(citation.video_id, citation.start_seconds)
    if not candidates:
        return CitationCheck(
            citation=citation,
            resolved=False,
            reason="no chunk at this video and timestamp",
        )
    ratio, chunk_id = max(
        ((quote_ratio(citation.quote, text), chunk_id) for chunk_id, text in candidates),
        default=(0.0, None),
    )
    if ratio >= QUOTE_MATCH_RATIO:
        return CitationCheck(
            citation=citation, resolved=True, reason="ok", ratio=ratio, chunk_id=chunk_id
        )
    return CitationCheck(
        citation=citation,
        resolved=False,
        reason="quote not found in the transcript at this timestamp",
        ratio=ratio,
        chunk_id=chunk_id,
    )


# ── criteria recall: fuzzy, semantic, one-to-one ──────────────────────────

#: ``texts -> unit-normalised vectors``. The shipped MiniLM model in practice.
EmbedFn = Callable[[Sequence[str]], list[list[float]]]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


@dataclass
class CriterionMatch:
    """One held-out criterion and the finding (if any) that reached it."""

    criterion: Criterion
    finding_id: str | None = None
    finding_criterion: str | None = None
    score: float = 0.0
    #: Why the matcher paired these, in its own words — recorded for unmatched
    #: rows too, since "not raised" is the reason a reader most wants. A tick
    #: with no reason is not reviewable, and this score is the one a reader is
    #: most likely to dispute.
    why: str | None = None
    #: Share of matcher repeats that backed this verdict (see
    #: :func:`consensus`). 1.0 is unanimous; 0.6 is three of five.
    agreement: float = 1.0
    #: Set when the finding this criterion was paired with turned out to rest on
    #: no exclusive corpus evidence, so the pairing does not count towards
    #: recall. Kept visible rather than silently unmatched — "you made this
    #: point but the corpus did not pay for it" is a different failure from
    #: "you never made this point".
    ungrounded: bool = False

    @property
    def matched(self) -> bool:
        """Paired with a finding, whether or not that finding was grounded."""
        return self.finding_id is not None

    @property
    def counted(self) -> bool:
        """Paired with a finding the corpus paid for — what recall counts."""
        return self.finding_id is not None and not self.ungrounded

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.criterion.to_dict(),
            "matched": self.matched,
            "finding_id": self.finding_id,
            "finding_criterion": self.finding_criterion,
            "score": round(self.score, 3),
            "why": self.why,
            "agreement": round(self.agreement, 3),
            "ungrounded": self.ungrounded,
            "counted": self.counted,
        }


def match_criteria(
    criteria: Sequence[Criterion],
    findings: Sequence[Finding],
    embed: EmbedFn,
    *,
    threshold: float = MATCH_THRESHOLD,
) -> list[CriterionMatch]:
    """Pair each criterion with the finding that best restates it.

    Greedy over the whole (criterion, finding) similarity matrix, best pair
    first, and one-to-one: a system that emits the same point five times must not
    thereby cover five criteria, and one finding that happens to sit between two
    criteria must be spent on the closer one. Ties break on id so the pairing is
    reproducible.
    """
    matches = [CriterionMatch(criterion=c) for c in criteria]
    if not criteria or not findings:
        return matches

    vectors = embed([c.criterion for c in criteria] + [f.criterion for f in findings])
    criterion_vectors = vectors[: len(criteria)]
    finding_vectors = vectors[len(criteria) :]

    pairs = [
        (cosine(criterion_vectors[i], finding_vectors[j]), criteria[i].id, findings[j].id, i, j)
        for i in range(len(criteria))
        for j in range(len(findings))
    ]
    pairs.sort(key=lambda pair: (-pair[0], pair[1], pair[2]))

    used_criteria: set[int] = set()
    used_findings: set[int] = set()
    for score, _, _, i, j in pairs:
        if score < threshold:
            break
        if i in used_criteria or j in used_findings:
            continue
        used_criteria.add(i)
        used_findings.add(j)
        matches[i].finding_id = findings[j].id
        matches[i].finding_criterion = findings[j].criterion
        matches[i].score = score

    # Record the best near-miss for an unmatched criterion, so a reviewer can
    # see whether the system missed the point entirely or merely fell short of
    # the threshold. Without this an unmatched row is unfalsifiable.
    #
    # This is the *embedding* matcher's answer to that; the shipped LLM matcher
    # has no similarity to report and instead carries the reason it gave for
    # leaving the criterion alone (``CriterionMatch.why``), which is why that
    # field is now recorded for unmatched rows too.
    for i, match in enumerate(matches):
        if match.matched:
            continue
        best = max(
            ((cosine(criterion_vectors[i], finding_vectors[j]), j) for j in range(len(findings))),
            default=(0.0, -1),
        )
        if best[1] >= 0:
            match.score = best[0]
            match.finding_criterion = findings[best[1]].criterion
    return matches


@dataclass
class MatchResult:
    """The consensus pairing, plus every repeat that voted on it.

    ``runs`` is what makes the score falsifiable: one entry per matcher repeat,
    mapping criterion id to the finding it chose (or ``None``). The run file
    reports the spread across them, so a reader can see whether a recall of
    0.278 was 0.278 every time or 0.222-0.333 depending on the call.
    """

    matches: list["CriterionMatch"] = field(default_factory=list)
    runs: list[dict[str, str | None]] = field(default_factory=list)

    @property
    def repeats(self) -> int:
        return len(self.runs) or 1


#: ``(criteria, findings) -> a MatchResult``. The shipped implementation is
#: :func:`~src.evals.critique_run.llm_matcher`; the tests pass a fake, and
#: :func:`embedding_matcher` is the model-free fallback.
MatchFn = Callable[[Sequence[Criterion], Sequence[Finding]], "MatchResult"]


def consensus(
    criteria: Sequence[Criterion],
    findings: Sequence[Finding],
    runs: list[dict[str, str | None]],
) -> list[CriterionMatch]:
    """Resolve several matcher runs into one pairing by per-criterion vote.

    Majority over *pairings*, not over counts. A criterion that four of five runs
    left unmatched is unmatched; one that three of five paired with ``f03`` is
    paired with ``f03`` even if the other two chose different findings. Ties break
    on finding id so the result never depends on run order, and ``None`` is a
    candidate like any other — "most runs did not raise this" is a verdict.

    ``agreement`` records how many runs backed the winner, so a 3/5 pairing is
    visibly weaker than a 5/5 one instead of both rendering as a tick.
    """
    by_finding = {finding.id: finding for finding in findings}
    matches: list[CriterionMatch] = []
    for criterion in criteria:
        votes: dict[str | None, int] = {}
        for run in runs:
            choice = run.get(criterion.id)
            votes[choice] = votes.get(choice, 0) + 1
        winner, count = max(votes.items(), key=lambda item: (item[1], item[0] or ""))
        match = CriterionMatch(criterion=criterion)
        match.agreement = count / (len(runs) or 1)
        if winner is not None and winner in by_finding:
            match.finding_id = winner
            match.finding_criterion = by_finding[winner].criterion
            match.score = match.agreement
        matches.append(match)
    return matches


def repeated_matcher(inner: MatchFn, repeats: int = DEFAULT_MATCH_REPEATS) -> MatchFn:
    """Run a matcher several times and resolve its pairings by vote.

    Wraps any :data:`MatchFn`, so the offline and LLM matchers get the same
    treatment. A matcher that is already deterministic votes unanimously with
    itself and is unchanged apart from the recorded ``agreement`` of 1.0.
    """

    def match(criteria: Sequence[Criterion], findings: Sequence[Finding]) -> MatchResult:
        runs: list[dict[str, str | None]] = []
        for _ in range(max(1, repeats)):
            result = inner(criteria, findings)
            paired = enforce_one_to_one(list(result.matches))
            runs.append({row.criterion.id: row.finding_id for row in paired})
        merged = enforce_one_to_one(consensus(criteria, findings, runs))
        return MatchResult(matches=merged, runs=runs)

    return match


def ground_findings(
    findings: Sequence[Finding],
    checks: dict[str, list[CitationCheck]],
) -> dict[str, list[str]]:
    """The chunks each finding can claim as **its own** evidence.

    A finding is grounded when it cites at least one resolving chunk that no
    other finding cites. Sharing is symmetric — if two findings rest on the same
    single chunk then *neither* is grounded, rather than the first one listed
    winning it — because any tie-break would decide the score by output order,
    and this scorer already has one source of run-to-run drift too many.

    This is the lever that kills the padding attack in the module docstring: an
    output of N findings all pointing at one quote yields no exclusive evidence
    at all, so nothing is grounded and both ``evidence_precision`` and
    ``criteria_recall`` collapse. The cost is that a chunk genuinely supporting
    two rules grounds neither — deliberately conservative, see the docstring.
    """
    owners: dict[str, set[str]] = {}
    for finding in findings:
        for check in checks.get(finding.id, []):
            if check.resolved and check.chunk_id is not None:
                owners.setdefault(check.chunk_id, set()).add(finding.id)
    exclusive: dict[str, list[str]] = {}
    for finding in findings:
        mine = [
            check.chunk_id
            for check in checks.get(finding.id, [])
            if check.resolved
            and check.chunk_id is not None
            and owners.get(check.chunk_id) == {finding.id}
        ]
        exclusive[finding.id] = sorted(dict.fromkeys(mine))
        for check in checks.get(finding.id, []):
            if check.resolved and check.chunk_id is not None:
                check.shared = len(owners.get(check.chunk_id, ())) > 1
    return exclusive


def embedding_matcher(embed: EmbedFn, *, threshold: float = MATCH_THRESHOLD) -> MatchFn:
    """:func:`match_criteria` as a :data:`MatchFn`. See :data:`MATCH_THRESHOLD`
    for why this is a lower bound rather than the default."""

    def match(criteria: Sequence[Criterion], findings: Sequence[Finding]) -> MatchResult:
        matches = match_criteria(criteria, findings, embed, threshold=threshold)
        return MatchResult(
            matches=matches,
            runs=[{row.criterion.id: row.finding_id for row in matches}],
        )

    return match


def enforce_one_to_one(matches: list[CriterionMatch]) -> list[CriterionMatch]:
    """Drop repeat uses of one finding, keeping its best-scoring criterion.

    Applies to any matcher, including an LLM one — a model asked to pair 18
    criteria against 10 findings will happily spend the same finding on three of
    them, and that is precisely how a critique that made one broad point would
    score as having covered three rules.
    """
    best_for_finding: dict[str, CriterionMatch] = {}
    for match in matches:
        if not match.matched or match.finding_id is None:
            continue
        current = best_for_finding.get(match.finding_id)
        if current is None or match.score > current.score:
            best_for_finding[match.finding_id] = match
    keep = {id(match) for match in best_for_finding.values()}
    for match in matches:
        if match.matched and id(match) not in keep:
            match.finding_id = None
    return matches


# ── the run ───────────────────────────────────────────────────────────────


@dataclass
class SetupCritique:
    """One setup's critique of the artifact, before it is scored."""

    setup: str
    findings: list[Finding] = field(default_factory=list)
    retrieved_chunk_ids: list[str] = field(default_factory=list)
    retrieved_video_ids: list[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    token_estimate: int = 0
    answer: str = ""
    error: str | None = None
    trace: list[dict[str, Any]] = field(default_factory=list)


def held_out_leaks(
    held_out_video_id: str,
    *,
    chunk_ids: Iterable[str] = (),
    video_ids: Iterable[str] = (),
    citations: Iterable[Citation] = (),
) -> list[str]:
    """Every trace of the held-out video anywhere in what the run produced.

    Deliberately post-hoc and independent of the filters that are supposed to
    make it empty: chunk ids are ``chunk:{video_id}:{index}``, so this is a
    prefix test that would still catch a leak if every ``$nin`` in the stack
    were deleted. The gate reads this list out of the committed run file rather
    than trusting that exclusion was wired up — which is the only version of the
    check worth having, since a filter that silently stopped applying looks
    exactly like a filter that is working.
    """
    prefix = f"chunk:{held_out_video_id}:"
    leaks: list[str] = []
    leaks += [cid for cid in chunk_ids if cid.startswith(prefix) or cid == held_out_video_id]
    leaks += [f"video:{vid}" for vid in video_ids if vid == held_out_video_id]
    leaks += [
        f"citation:{c.video_id}@{c.start_seconds}"
        for c in citations
        if c.video_id == held_out_video_id
    ]
    return sorted(dict.fromkeys(leaks))


def score_critique(
    critique: SetupCritique,
    dataset: CritiqueDataset,
    match: MatchFn,
    chunk_text: ChunkTextFn,
) -> dict[str, Any]:
    """One setup's cell: four scores, the matched/unmatched split, the leaks."""
    checks: dict[str, list[CitationCheck]] = {
        finding.id: [check_citation(c, chunk_text) for c in finding.citations]
        for finding in critique.findings
    }
    exclusive = ground_findings(critique.findings, checks)
    grounded_ids = {fid for fid, chunks in exclusive.items() if chunks}
    grounded = [f for f in critique.findings if f.id in grounded_ids]

    all_checks = [check for finding in critique.findings for check in checks[finding.id]]
    resolved_checks = [check for check in all_checks if check.resolved]

    contested_findings = [
        finding
        for finding in critique.findings
        if finding.contested
        and len({check.citation.video_id for check in checks[finding.id] if check.resolved}) >= 2
    ]

    # Matched once over the whole list, then read twice. Matching the applicable
    # subset separately would let a finding that lost a resume-only criterion to
    # the one-to-one rule reappear against an applicable one, so the headline
    # recall would depend on which criteria happened to be in the same call.
    result = match(dataset.criteria, critique.findings)
    matches_all = enforce_one_to_one(list(result.matches))
    for row in matches_all:
        row.ungrounded = row.matched and row.finding_id not in grounded_ids

    applicable = dataset.applicable()
    applicable_ids = {c.id for c in applicable}
    matches_applicable = [m for m in matches_all if m.criterion.id in applicable_ids]
    counted = sum(m.counted for m in matches_applicable)

    # The same score computed from each matcher repeat on its own, so the run
    # carries the range its own scorer produced rather than one draw from it.
    spread = _recall_spread(dataset, result.runs, applicable_ids, grounded_ids)

    groups = _group_recall(applicable, matches_applicable)
    total = len(critique.findings)
    scores = {
        "criteria_recall": _ratio(counted, len(applicable)),
        "evidence_precision": _ratio(len(grounded), total),
        "provenance": _ratio(len(resolved_checks), len(all_checks)),
        "contested_rate": _ratio(len(contested_findings), total),
    }

    leaks = held_out_leaks(
        dataset.held_out_video_id,
        chunk_ids=critique.retrieved_chunk_ids,
        video_ids=critique.retrieved_video_ids,
        citations=[c for finding in critique.findings for c in finding.citations],
    )

    return {
        "setup": critique.setup,
        "scores": scores,
        "score_spread": spread,
        "match_repeats": result.repeats,
        "criteria_recall_all": _ratio(sum(m.counted for m in matches_all), len(dataset.criteria)),
        "criteria_recall_grouped": _ratio(groups[0], groups[1]),
        "criteria_groups": groups[1],
        "findings_total": total,
        "findings_grounded": len(grounded),
        "findings_sharing_evidence": total - len(grounded),
        "citations_total": len(all_checks),
        "citations_resolved": len(resolved_checks),
        "criteria_applicable": len(applicable),
        "criteria_matched": counted,
        "criteria_matched_ungrounded": sum(
            1 for m in matches_applicable if m.matched and m.ungrounded
        ),
        "contested_findings": len(contested_findings),
        "held_out_leaks": len(leaks),
        "held_out_leak_ids": leaks,
        "retrieved_chunk_ids": list(critique.retrieved_chunk_ids),
        "retrieved_video_ids": list(critique.retrieved_video_ids),
        "elapsed_seconds": round(critique.elapsed_seconds, 2),
        "token_estimate": critique.token_estimate,
        "error": critique.error,
        "matches": [m.to_dict() for m in matches_all],
        "match_runs": result.runs,
        "findings": [
            {
                **finding.to_dict(),
                "citation_checks": [check.to_dict() for check in checks[finding.id]],
                "grounded": finding.id in grounded_ids,
                "exclusive_chunk_ids": exclusive.get(finding.id, []),
            }
            for finding in critique.findings
        ],
        "answer": critique.answer,
        "trace": list(critique.trace),
    }


def _recall_spread(
    dataset: CritiqueDataset,
    runs: list[dict[str, str | None]],
    applicable_ids: set[str],
    grounded_ids: set[str],
) -> dict[str, float | None]:
    """Min / median / max applicable recall across the matcher's own repeats.

    The consensus score is a point estimate from a matcher that does not agree
    with itself; this is the interval it came out of. A later slice that moves
    recall by less than ``max - min`` has not shown anything, and the UI renders
    the range next to the number so that comparison is possible without opening
    the run file.
    """
    if not runs or not applicable_ids:
        return {
            "criteria_recall_min": None,
            "criteria_recall_median": None,
            "criteria_recall_max": None,
        }
    per_run = sorted(
        len(
            [
                cid
                for cid, fid in run.items()
                if cid in applicable_ids and fid is not None and fid in grounded_ids
            ]
        )
        / len(applicable_ids)
        for run in runs
    )
    middle = per_run[len(per_run) // 2]
    return {
        "criteria_recall_min": round(per_run[0], 4),
        "criteria_recall_median": round(middle, 4),
        "criteria_recall_max": round(per_run[-1], 4),
    }


def _group_recall(
    applicable: Sequence[Criterion], matches: Sequence[CriterionMatch]
) -> tuple[int, int]:
    """Recall counted once per *rule*, not once per phrasing of it.

    Four pairs in the held-out list are near-duplicates of each other (see
    ``group`` in the dataset). One-to-one matching means a system that makes one
    correct point about linking to your code can only ever satisfy one of the
    two link criteria — a recall ceiling that has nothing to do with retrieval.
    Reported alongside the per-criterion score rather than instead of it: the
    per-criterion number stays the conservative headline.
    """
    reached: set[str] = set()
    groups: set[str] = set()
    counted = {m.criterion.id for m in matches if m.counted}
    for criterion in applicable:
        groups.add(criterion.group)
        if criterion.id in counted:
            reached.add(criterion.group)
    return len(reached), len(groups)


def _ratio(numerator: int, denominator: int) -> float | None:
    """A rate, or ``None`` when there was nothing to take a rate over.

    Zero findings is not zero precision — it is an unmeasured cell, and
    reporting 0.0 would let a setup that produced nothing look like a setup that
    produced only ungrounded claims.
    """
    if denominator <= 0:
        return None
    return round(numerator / denominator, 4)


def build_run(
    dataset: CritiqueDataset,
    cells: list[dict[str, Any]],
    *,
    config: dict[str, Any],
    baseline: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    """The committed snapshot, in the shape ``load_experiments`` reads.

    The run id carries the held-out video, because ``save_run`` overwrites by id
    and a timestamp alone would let two runs holding out *different* videos
    collide — which is the one collision that would make the whole experiment
    meaningless while looking entirely healthy.
    """
    moment = now or datetime.now(timezone.utc)
    return {
        "run_id": (f"critique-{dataset.held_out_video_id}-{moment.strftime('%Y%m%d-%H%M%S')}"),
        "created_at": moment.isoformat(),
        "kind": "critique-eval",
        "held_out_video_id": dataset.held_out_video_id,
        "held_out_title": dataset.held_out_title,
        "artifact_id": dataset.artifact_id,
        "artifact_url": dataset.artifact_url,
        "artifact_kind": dataset.artifact_kind,
        "criteria_total": len(dataset.criteria),
        "criteria_applicable": len(dataset.applicable()),
        "metrics": CRITIQUE_METRICS,
        "baseline": baseline,
        "match_repeats": config.get("match_repeats", DEFAULT_MATCH_REPEATS),
        "exclusion_version": EXCLUSION_VERSION,
        "criteria_groups": len({c.group for c in dataset.applicable()}),
        "held_out_leaks": sum(int(cell.get("held_out_leaks") or 0) for cell in cells),
        "config": config,
        "cells": cells,
    }


def verify_dataset(dataset: CritiqueDataset, chunk_text: ChunkTextFn) -> list[CitationCheck]:
    """Check every criterion's own quote resolves at its own timestamp.

    The ground truth has to pass the bar it sets. A criterion whose quote cannot
    be found where it says it was is not evidence of anything, and would let the
    dataset's author put words in the expert's mouth — so this runs in the tests
    and again before every run.
    """
    return [
        check_citation(
            Citation(
                video_id=criterion.video_id,
                start_seconds=criterion.start_seconds,
                quote=criterion.quote,
            ),
            chunk_text,
        )
        for criterion in dataset.criteria
    ]


def format_table(run: dict[str, Any]) -> str:
    """A fixed-width comparison table for the terminal, baseline row first."""
    metrics = run["metrics"]
    header = "  ".join([f"{'setup':<20}"] + [f"{m:>18}" for m in metrics])
    lines = [header, "-" * len(header)]
    for cell in run["cells"]:
        scores = cell.get("scores", {})
        values = [
            f"{scores[m]:>18.3f}" if isinstance(scores.get(m), (int, float)) else f"{'—':>18}"
            for m in metrics
        ]
        lines.append("  ".join([f"{cell['setup']:<20}"] + values))
    lines.append("")
    for cell in run["cells"]:
        spread = cell.get("score_spread") or {}
        low, high = spread.get("criteria_recall_min"), spread.get("criteria_recall_max")
        if isinstance(low, (int, float)) and isinstance(high, (int, float)) and low != high:
            lines.append(
                f"  {cell['setup']}: criteria_recall ranged {low:.3f}-{high:.3f} across "
                f"{cell.get('match_repeats', '?')} matcher runs — a later slice has to "
                "beat that range, not the point estimate"
            )
    lines.append(
        f"held-out {run['held_out_video_id']} · leaks {run['held_out_leaks']} · "
        f"{run['criteria_applicable']}/{run['criteria_total']} criteria applicable "
        f"({run.get('criteria_groups', '?')} groups)"
    )
    return "\n".join(lines)

"""Judging a document one rubric at a time, instead of against a bag of chunks.

The chunk-dump review (``DOC_REVIEW_SYSTEM_PROMPT``) hands the model ten
retrieved chunks and asks for feedback. What comes back is whatever the model
decided to notice — a good essay, but nobody can tell what it *did not* check,
because there was never a list. That is the thing this module changes: the
criteria exist before the review does. They are the shipped expert packs under
``experts/``, and every one of them is a checkable rule with a creator's own
words and the second they said it behind it.

So a review here is not prose with citations attached; it is **one verdict per
rubric**, and the rubric it belongs to is not something the model chose.

Three rules run through all of it.

**Nothing citable comes from the model.** The model is given rubric ids and
returns rubric ids, a verdict, a severity and which document sections it is
talking about — and that is the entire surface. The criterion text, the check,
the creator, the video, the timestamp and the deep link are all read out of the
pack afterwards, keyed by the rubric id. This is not caution for its own sake:
the earlier review path let the model supply ``chunk_index`` and it was wrong on
35 of 39 references. A field the model cannot get wrong is a field the model is
never asked for.

**A verdict that names no section of the document is not a review of it.** A
``fail`` has to point at ``[§N]``, and one that does not is recorded as
``unjudged`` with the reason rather than counted. Without that rule the cheapest
way to score well is to recite the whole pack back as failures — which is
exactly the hole ``src/evals/KNOWN_GAP_attack2.md`` documents, and exactly what
the recall comparison in the Critique panel would otherwise reward.

**``n-a`` is a real answer.** Most of what a résumé pack knows does not apply to
a portfolio page, and V3 measured that: 19 of 24 held-out criteria applied. A
reviewer that forces a pass or a fail onto "put your dates on the right-hand
side" is not being thorough, it is being wrong quietly. So ``n-a`` is a
first-class verdict, it is rendered, and its share is reported.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence


#: Every shipped pack, in the order a reviewer reads them: the two career packs
#: first because the artifact is usually a career document, then the two
#: engineering packs whose rubrics judge the *work* a portfolio describes.
#:
#: All four run on every review, deliberately. Routing a pack away up front
#: would look tidier and would hide the honest answer — "fifteen system-design
#: rules were checked against this page and none of them applied" is a result,
#: and a pack that was never asked cannot report it. It also keeps pack
#: selection deterministic: no router call sits between a document and the
#: criteria it was judged against.
PACK_TOPICS: tuple[str, ...] = (
    "resume-design",
    "job-search",
    "system-design",
    "app-architecture",
)

#: ``experts/`` at the repo root, independent of the server's working directory
#: — the same resolution :mod:`src.api.packs` uses.
DEFAULT_PACKS_DIR = Path(__file__).resolve().parents[2] / "experts"

#: The three verdicts a rubric can reach, plus the one the *reviewer* can reach
#: about itself. ``unjudged`` is never returned by the model: it is what a row
#: becomes when it broke a rule (no section on a fail, an unknown rubric id, a
#: rubric the model skipped), and it is rendered rather than dropped so the
#: count of rubrics nobody actually decided is visible instead of inferred.
VERDICTS: tuple[str, ...] = ("pass", "fail", "n-a", "unjudged")

#: How badly a failing document fails. The one field on a verdict the model
#: supplies that is not an identifier — validated against this tuple, and forced
#: to ``none`` on anything that is not a ``fail``, because a severity on a pass
#: is a number with nothing behind it.
SEVERITIES: tuple[str, ...] = ("blocker", "major", "minor", "none")

_VERDICT_ALIASES = {
    "pass": "pass",
    "passes": "pass",
    "ok": "pass",
    "fail": "fail",
    "fails": "fail",
    "failed": "fail",
    "n-a": "n-a",
    "na": "n-a",
    "n/a": "n-a",
    "not applicable": "n-a",
    "notapplicable": "n-a",
    "n a": "n-a",
}

_SEVERITY_ALIASES = {
    "blocker": "blocker",
    "critical": "blocker",
    "high": "blocker",
    "major": "major",
    "medium": "major",
    "moderate": "major",
    "minor": "minor",
    "low": "minor",
    "nit": "minor",
}


@dataclass(frozen=True)
class VerdictEvidence:
    """One creator quote behind a rubric, as the browser needs it.

    Every field is read off :class:`src.rag.packs.RubricEvidence`. ``url`` in
    particular is built from ``video_id`` and never from ``source_url`` — eight
    of the corpus's stored urls already carry a ``t=`` parameter, and appending
    a second one gives a link YouTube opens about twenty seconds early.
    """

    video_id: str
    chunk_id: str
    quote: str
    channel_name: str | None
    title: str | None
    start_seconds: float
    url: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "chunk_id": self.chunk_id,
            "quote": self.quote,
            "channel_name": self.channel_name,
            "title": self.title,
            "start_seconds": self.start_seconds,
            "url": self.url,
        }


@dataclass
class RubricVerdict:
    """One rubric, the verdict it reached, and where both halves came from.

    "Both halves" is the point of the shape. ``criterion``/``check``/``evidence``
    say what was checked and who said it should be — all copied from the pack.
    ``verdict``/``severity``/``finding``/``sections`` say what this document did
    about it. A reader can follow either direction: from a finding back to the
    creator's own sentence, or from a rubric forward to the section it lands on.
    """

    rubric_id: str
    topic: str
    pack_name: str
    criterion: str
    check: str
    why: str
    unit_title: str
    creators: list[str]
    verdict: str
    severity: str = "none"
    #: What this document does about the rubric, in the reviewer's words. Empty
    #: on a ``pass`` is fine; empty on a ``fail`` is not, and is caught below.
    finding: str = ""
    #: Zero-based section indices, the same numbering ``DocumentSection.index``
    #: uses. Rendered as ``§N+1`` to match the ``[§N]`` markers in the context.
    sections: list[int] = field(default_factory=list)
    evidence: list[VerdictEvidence] = field(default_factory=list)
    #: Set when the reviewer's row was rejected and the verdict became
    #: ``unjudged`` — the rule it broke, said out loud rather than dropped.
    note: str = ""

    @property
    def timestamps(self) -> list[float]:
        return [item.start_seconds for item in self.evidence]

    def to_dict(self) -> dict[str, Any]:
        return {
            "rubric_id": self.rubric_id,
            "topic": self.topic,
            "pack_name": self.pack_name,
            "criterion": self.criterion,
            "check": self.check,
            "why": self.why,
            "unit_title": self.unit_title,
            "creators": list(self.creators),
            "verdict": self.verdict,
            "severity": self.severity,
            "finding": self.finding,
            "sections": list(self.sections),
            "evidence": [item.to_dict() for item in self.evidence],
            "note": self.note,
        }


@dataclass
class PackOutcome:
    """What one pack's review call produced, including when it produced nothing.

    A pack that errored is a row in the UI, not a silence. Fifteen rubrics that
    were never asked and fifteen that all came back ``n-a`` are different
    results, and a review that cannot tell them apart is claiming coverage it
    does not have.
    """

    topic: str
    name: str
    artifact: str
    rubrics: int
    elapsed_seconds: float = 0.0
    error: str | None = None
    #: Rows the model returned that named a rubric this pack does not contain.
    unknown_rubric_ids: list[str] = field(default_factory=list)
    #: Rubrics the model returned twice; only the first row is kept.
    duplicate_rubric_ids: list[str] = field(default_factory=list)
    #: Rubrics the model never mentioned.
    missing_rubric_ids: list[str] = field(default_factory=list)
    #: Failing rows that pointed at no section of the document.
    unanchored_failures: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "topic": self.topic,
            "name": self.name,
            "artifact": self.artifact,
            "rubrics": self.rubrics,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "error": self.error,
            "unknown_rubric_ids": list(self.unknown_rubric_ids),
            "duplicate_rubric_ids": list(self.duplicate_rubric_ids),
            "missing_rubric_ids": list(self.missing_rubric_ids),
            "unanchored_failures": list(self.unanchored_failures),
        }


@dataclass
class RubricReview:
    """Every rubric that was applied to one document, and what it decided."""

    document_id: str
    document_url: str
    document_kind: str
    verdicts: list[RubricVerdict] = field(default_factory=list)
    packs: list[PackOutcome] = field(default_factory=list)

    @property
    def stats(self) -> dict[str, Any]:
        """The deterministic numbers the panel states as claims.

        Recomputed from the rows rather than accumulated while building them, so
        the header cannot drift from the list underneath it.
        """
        total = len(self.verdicts)
        counts = {value: 0 for value in VERDICTS}
        severities = {value: 0 for value in SEVERITIES}
        for verdict in self.verdicts:
            counts[verdict.verdict] = counts.get(verdict.verdict, 0) + 1
            if verdict.verdict == "fail":
                severities[verdict.severity] = severities.get(verdict.severity, 0) + 1
        with_id = sum(1 for v in self.verdicts if v.rubric_id)
        with_timestamp = sum(1 for v in self.verdicts if v.evidence)
        with_both = sum(1 for v in self.verdicts if v.rubric_id and v.evidence)
        links = [item.url for verdict in self.verdicts for item in verdict.evidence]
        return {
            "rubrics_total": total,
            "verdicts": counts,
            "severities": severities,
            "packs_used": len(self.packs),
            "packs_failed": sum(1 for pack in self.packs if pack.error),
            "with_rubric_id": with_id,
            "with_timestamp": with_timestamp,
            "with_id_and_timestamp": with_both,
            "id_and_timestamp_share": round(with_both / total, 4) if total else 0.0,
            "evidence_links": len(links),
            "sections_cited": sorted(
                {index for verdict in self.verdicts for index in verdict.sections}
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "document_url": self.document_url,
            "document_kind": self.document_kind,
            "verdicts": [verdict.to_dict() for verdict in self.verdicts],
            "packs": [pack.to_dict() for pack in self.packs],
            "stats": self.stats,
        }


# ─── Loading the criteria ────────────────────────────────────────────────────


def load_review_packs(packs_dir: Path | str | None = None) -> list[Any]:
    """Every shipped pack, in :data:`PACK_TOPICS` order, skipping the unbuilt.

    A pack that is absent or unreadable reads as "not built" rather than taking
    the review down — the same rule :mod:`src.api.packs` follows. A review over
    three packs is still a review; a 500 is not.
    """
    from src.rag.packs import PackStore

    store = PackStore(Path(packs_dir) if packs_dir else DEFAULT_PACKS_DIR)
    packs: list[Any] = []
    for topic in PACK_TOPICS:
        try:
            pack = store.load_pack(topic)
        except (OSError, ValueError):
            continue
        if pack is not None and pack.rubrics:
            packs.append(pack)
    return packs


def format_rubrics(pack: Any) -> str:
    """One pack's rubrics as the review call sees them.

    The id leads every line because the id is the entire answer surface: the
    model reads it here and writes it back, and nothing else it says is used to
    identify anything. ``why`` is deliberately left out — it is the argument for
    the rule, and a reviewer that is told why a rule matters starts arguing with
    the rule instead of applying it.
    """
    blocks = []
    for rubric in pack.rubrics:
        check = f"\n    CHECK: {rubric.check}" if rubric.check else ""
        blocks.append(f"{rubric.rubric_id}: {rubric.criterion}{check}")
    return "\n".join(blocks)


def _evidence_rows(rubric: Any) -> list[VerdictEvidence]:
    return [
        VerdictEvidence(
            video_id=item.video_id,
            chunk_id=item.chunk_id,
            quote=item.quote,
            channel_name=item.channel_name,
            title=item.title,
            # The chunk's own start, which is what a provenance check resolves
            # against; the deep link uses the interpolated quote offset.
            start_seconds=item.start_seconds,
            url=item.youtube_url(),
        )
        for item in rubric.evidence
    ]


# ─── Reading the model back ──────────────────────────────────────────────────


def _normalise_verdict(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    return _VERDICT_ALIASES.get(text)


def _normalise_severity(value: Any, verdict: str) -> str:
    if verdict != "fail":
        return "none"
    text = str(value or "").strip().lower()
    # An unrecognised severity on a real failure is downgraded rather than
    # dropped: the failure still happened, and defaulting it to the loudest
    # bucket would let a typo shout.
    return _SEVERITY_ALIASES.get(text, "minor")


def _section_indices(value: Any, section_indices: Sequence[int]) -> list[int]:
    """``[§3]`` in the prompt means section index 2 here.

    The prompt numbers sections from one because that is what the document
    context is labelled with; everything downstream — ``DocumentSection.index``,
    ``document_sections_selected`` — is zero-based, and converting at the only
    boundary where the two meet is what stops the off-by-one from spreading.
    """
    allowed = set(section_indices)
    found: list[int] = []
    for raw in value if isinstance(value, (list, tuple)) else []:
        try:
            number = int(str(raw).strip().lstrip("§#"))
        except (TypeError, ValueError):
            continue
        index = number - 1
        if index in allowed and index not in found:
            found.append(index)
    return sorted(found)


def parse_pack_verdicts(
    payload: Any,
    pack: Any,
    section_indices: Sequence[int],
) -> tuple[list[RubricVerdict], PackOutcome]:
    """One pack's model reply, turned into a verdict for every rubric it holds.

    The output length is the pack's rubric count, always — not the number of
    rows the model happened to return. A rubric the model skipped is present and
    says so, because "the reviewer did not answer for this rule" is information
    and an absent row looks identical to a rule that was never in the pack.
    """
    outcome = PackOutcome(
        topic=pack.topic,
        name=getattr(pack, "name", pack.topic),
        artifact=getattr(pack, "artifact", "document"),
        rubrics=len(pack.rubrics),
    )
    by_id = {rubric.rubric_id: rubric for rubric in pack.rubrics}
    rows = payload.get("verdicts") if isinstance(payload, dict) else None
    decided: dict[str, tuple[str, str, str, list[int]]] = {}
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            continue
        rubric_id = str(row.get("rubric_id") or "").strip()
        if rubric_id not in by_id:
            outcome.unknown_rubric_ids.append(rubric_id or "(blank)")
            continue
        if rubric_id in decided:
            outcome.duplicate_rubric_ids.append(rubric_id)
            continue
        verdict = _normalise_verdict(row.get("verdict"))
        if verdict is None or verdict == "unjudged":
            continue
        sections = _section_indices(row.get("sections"), section_indices)
        finding = str(row.get("finding") or "").strip()
        if verdict == "fail" and not sections:
            # The anti-recitation rule. A failure that points at no section is a
            # statement about documents in general — true, perhaps, but it is
            # not a finding about this one, and counting it would make "emit
            # every rubric as a failure" the highest-scoring strategy.
            outcome.unanchored_failures.append(rubric_id)
            decided[rubric_id] = (
                "unjudged",
                "none",
                finding,
                [],
            )
            continue
        decided[rubric_id] = (
            verdict,
            _normalise_severity(row.get("severity"), verdict),
            finding,
            sections,
        )

    verdicts: list[RubricVerdict] = []
    for rubric in pack.rubrics:
        verdict, severity, finding, sections = decided.get(
            rubric.rubric_id, ("unjudged", "none", "", [])
        )
        note = ""
        if rubric.rubric_id not in decided:
            outcome.missing_rubric_ids.append(rubric.rubric_id)
            note = "the reviewer returned no verdict for this rubric"
        elif rubric.rubric_id in outcome.unanchored_failures:
            note = "failed, but named no section of the document — not counted as a finding"
        verdicts.append(
            RubricVerdict(
                rubric_id=rubric.rubric_id,
                topic=pack.topic,
                pack_name=outcome.name,
                criterion=rubric.criterion,
                check=rubric.check,
                why=rubric.why,
                unit_title=rubric.unit_title,
                creators=list(rubric.creators),
                verdict=verdict,
                severity=severity,
                finding=finding,
                sections=sections,
                evidence=_evidence_rows(rubric),
                note=note,
            )
        )
    return verdicts, outcome


# ─── The answer a chat bubble shows ──────────────────────────────────────────

#: Failures are read worst-first, and this is the order.
SEVERITY_ORDER = {"blocker": 0, "major": 1, "minor": 2, "none": 3}


def sort_verdicts(verdicts: Iterable[RubricVerdict]) -> list[RubricVerdict]:
    """Failures worst-first, then everything else, stable within a group.

    Pack order is the tiebreak rather than rubric id, so a reader scanning the
    list sees one pack's rules together instead of interleaved by an id whose
    numbering is an artefact of the build.
    """
    order = {"fail": 0, "unjudged": 1, "pass": 2, "n-a": 3}
    topics = {topic: index for index, topic in enumerate(PACK_TOPICS)}
    return sorted(
        verdicts,
        key=lambda v: (
            order.get(v.verdict, 4),
            SEVERITY_ORDER.get(v.severity, 3),
            topics.get(v.topic, len(topics)),
            v.rubric_id,
        ),
    )


def build_references(verdicts: Sequence[RubricVerdict]) -> list[dict[str, Any]]:
    """The corpus sources behind the failing rubrics, numbered for ``[N]``.

    Only the failures. A source list that included every quote behind every
    ``n-a`` would be a bibliography of the whole pack, and the citations in the
    answer would point into it at random — the numbers have to line up with the
    claims the answer actually makes.
    """
    references: list[dict[str, Any]] = []
    seen: dict[str, int] = {}
    for verdict in verdicts:
        if verdict.verdict != "fail":
            continue
        for item in verdict.evidence:
            if item.chunk_id in seen:
                continue
            seen[item.chunk_id] = len(references) + 1
            references.append(
                {
                    "label": f"[{len(references) + 1}]",
                    "video_id": item.video_id,
                    "source_url": item.url,
                    "timestamp_url": item.url,
                    "start_seconds": item.start_seconds,
                    "chunk_id": item.chunk_id,
                    "quote": item.quote,
                    "channel_name": item.channel_name,
                }
            )
    return references


def _citation_numbers(verdict: RubricVerdict, references: Sequence[dict[str, Any]]) -> list[int]:
    by_chunk = {ref["chunk_id"]: index + 1 for index, ref in enumerate(references)}
    return [by_chunk[item.chunk_id] for item in verdict.evidence if item.chunk_id in by_chunk]


def verdicts_as_findings(review: RubricReview) -> list[Any]:
    """The review as :class:`src.evals.critique.Finding` rows, for scoring.

    **Only the failures.** This is the whole anti-inflation design, and it is
    worth being explicit about why, because the obvious alternative is wrong in
    a way that would not show up in the number.

    The held-out harness measures recall: how many of an unseen expert's
    criteria a system reached. A pack reviewer could trivially "reach" all of
    them by emitting one finding per rubric — sixty-one general rules, each with
    a real corpus citation, one of which is bound to pair with whatever the
    held-out expert said. That is the attack ``KNOWN_GAP_attack2.md`` documents,
    and it beats the honest baseline 3.3x.

    Three things stop it here, none of them a threshold anyone tuned:

    * A rubric only becomes a finding when the reviewer judged this document to
      **fail** it. Passes and ``n-a`` are the majority of the review and score
      nothing, so the reviewer cannot spend rubrics it did not apply.
    * A failure must name a section of the document (:func:`parse_pack_verdicts`),
      so a rule recited without being applied is already ``unjudged`` by the time
      it gets here.
    * The scorer's own one-to-one pairing and exclusive-evidence rule mean two
      findings resting on the same quote cannot both count.

    ``detail`` carries what the reviewer actually said about *this* document,
    because that is what distinguishes an applied criterion from a recited one,
    and the matcher reads it.
    """
    from src.evals.critique import Citation, Finding

    findings: list[Any] = []
    for verdict in sort_verdicts(review.verdicts):
        if verdict.verdict != "fail":
            continue
        findings.append(
            Finding(
                # Qualified by pack. Rubric ids are numbered per pack, so
                # ``r0101`` exists in all four of them — an unqualified id would
                # hand the scorer two different rules under one name and let a
                # one-to-one pairing silently drop one of them.
                id=f"{verdict.topic}:{verdict.rubric_id}",
                criterion=verdict.criterion,
                detail=verdict.finding,
                citations=tuple(
                    Citation(
                        video_id=item.video_id,
                        start_seconds=item.start_seconds,
                        quote=item.quote,
                    )
                    for item in verdict.evidence
                ),
                contested=False,
            )
        )
    return findings


def build_answer(review: RubricReview) -> str:
    """The review as markdown, assembled from the verdicts rather than written.

    No second model call produces this. Every ``[§N]`` and every ``[N]`` in it is
    placed from a field that was already validated — so the prose in the bubble
    and the rows in the panel cannot disagree, which is the failure mode a
    separately-generated summary has by construction.
    """
    verdicts = sort_verdicts(review.verdicts)
    references = build_references(verdicts)
    stats = review.stats
    counts = stats["verdicts"]
    lines: list[str] = []
    packs = ", ".join(f"{pack.name} ({pack.rubrics})" for pack in review.packs)
    lines.append(
        f"Checked {stats['rubrics_total']} rubrics from {stats['packs_used']} expert "
        f"packs — {packs}. "
        f"{counts.get('fail', 0)} fail, {counts.get('pass', 0)} pass, "
        f"{counts.get('n-a', 0)} not applicable to a {review.document_kind}"
        + (f", {counts.get('unjudged', 0)} unjudged." if counts.get("unjudged", 0) else ".")
    )
    failed = [verdict for verdict in verdicts if verdict.verdict == "fail"]
    if not failed:
        lines.append("")
        lines.append("No rubric failed against the sections that were reviewed.")
    for group in ("blocker", "major", "minor"):
        rows = [verdict for verdict in failed if verdict.severity == group]
        if not rows:
            continue
        lines.append("")
        lines.append(f"## {group.capitalize()} ({len(rows)})")
        for verdict in rows:
            sections = " ".join(f"[§{index + 1}]" for index in verdict.sections)
            cites = " ".join(f"[{number}]" for number in _citation_numbers(verdict, references))
            body = verdict.finding or verdict.criterion
            lines.append(
                f"- **{verdict.rubric_id}** {verdict.criterion} — {body} {sections} {cites}".rstrip()
            )
    unjudged = [verdict for verdict in verdicts if verdict.verdict == "unjudged"]
    if unjudged:
        lines.append("")
        lines.append(f"## Not decided ({len(unjudged)})")
        for verdict in unjudged:
            lines.append(f"- **{verdict.rubric_id}** {verdict.criterion} — {verdict.note}")
    return "\n".join(lines).strip()

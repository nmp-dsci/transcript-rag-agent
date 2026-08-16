"""The deep-research build loop: plan, execute, criticise, republish.

:mod:`src.rag.pack_build` builds a pack in one pass — the corpus's own clusters
are the units, one LLM call each, and whatever comes out is the pack. This
module builds one the slow way instead: a **planner** decomposes the pack's
topic into sub-questions, an **executor** answers each one out of the corpus, a
**gap critic** reads the resulting pack and names what a reviewer still could
not check, and a **publisher** folds the answers to those gaps into a second
version.

This is the offline path. Nothing here runs while somebody waits for an answer;
it costs a plan, a dozen retrievals and a critique, and it produces a file.

What the loop has to prove
--------------------------
That a second round *caused by the critic* beats a first round is not the
interesting claim — the second round has spent more. The claim worth making is
that iteration beats the same spend without iteration, so this module builds
**three** packs from one planner call:

``deep-r1``
    The plan's first ``P`` probes. One executor call each.
``deep-oneshot``
    The same ``P`` probes plus the plan's own next ``Q``. The planner was asked
    for ``P + Q`` in a single call and told to order them most-important-first,
    so this arm is a strict superset of ``deep-r1`` and costs exactly what the
    loop's second round costs.
``deep-r2``
    The same ``P`` probes plus ``Q`` probes the **gap critic** wrote after
    reading ``deep-r1``.

``deep-oneshot`` and ``deep-r2`` spend the identical number of executor calls on
the identical first six probes. They differ in one thing: where the last four
questions came from. That is the whole experiment, and it is why a one-shot that
happens to score well is a failure of the hypothesis rather than a result.

Why nothing here is a second instrument
---------------------------------------
Every rubric is written by :func:`src.rag.pack_build.rubrics_for_unit` — the
same prompt, the same excerpt budget, the same ``max_rubrics``, the same
server-side citation reconciliation and the same on-disk call cache. A probe
becomes a :class:`~src.rag.packs.SourceUnit` and goes through the existing
machinery untouched, so the arms differ in *which passages the model saw* and in
nothing else. A pack this module writes is an ordinary
:class:`~src.rag.packs.ExpertPack` and is scored by the same held-out harness
:mod:`src.evals.pack_ablation` already runs.

The padding failure this is built to refuse
-------------------------------------------
``criteria_recall`` rewards having more findings — ``src/evals/KNOWN_GAP_attack2.md``
records an attack that beat the honest baseline 3.3× by reciting known advice
one citation at a time. A loop that appends a round is *shaped* like that attack,
so two guards are structural rather than reported:

* Round two's candidates are deduped against round one's surviving rules by
  :func:`~src.rag.packs.dedupe_rubrics`, at the same cosine threshold the arms
  use. A gap answered with a restatement of a rule the pack already had adds
  nothing and is counted as a drop, which is the measurement of whether the
  critic found new ground.
* Every citation still has to snap onto the stored transcript
  (:func:`~src.rag.packs.reconcile_evidence`), so a rule cannot be admitted with
  a quote nobody said.

The report carries finding count and citation count beside every score for the
same reason: an arm that gained recall by growing is visible in those two
columns, and a reader should not have to take this module's word for it.
"""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

from pydantic import BaseModel, Field

from src.config import Settings
from src.rag.pack_build import (
    DEFAULT_EXCERPTS_PER_UNIT,
    DEFAULT_MAX_RUBRICS_PER_UNIT,
    PackWorkspace,
    open_workspace,
    route_members,
    rubrics_for_unit,
)
from src.rag.packs import (
    DEDUPE_THRESHOLD,
    ChatModel,
    ExpertPack,
    PackDeclaration,
    PackGap,
    PackMember,
    PackStore,
    Rubric,
    SourceUnit,
    cosine,
    creator_profile,
    dedupe_rubrics,
    included_video_ids,
    pack_statistics,
)

ProgressFn = Callable[[str], None]

#: Where planner and critic calls are remembered. Derived state beside the
#: rubric cache: deleting it costs the calls again and nothing else. Keyed on
#: the exact prompt, so a rerun of an unchanged loop reproduces the identical
#: plan and the identical gaps rather than re-rolling them — which is the only
#: way "round two added the rubric the critic asked for" is a claim a reader can
#: re-check rather than a story about one lucky run.
DEFAULT_RESEARCH_CACHE_DIR = Path(".yt-agent/research_cache")

#: Bumped whenever :data:`PLAN_SYSTEM_PROMPT` or :data:`GAP_SYSTEM_PROMPT`
#: changes, so a cached plan written under the old wording is never served under
#: the new one.
RESEARCH_PROMPT_VERSION = "v1"

#: Executor calls in round one. Six because that is the unit budget the
#: hand-built arms actually spent on this topic (``experts/ablation.json``'s
#: ``unit_budget``), and a loop that beat the hand build on more calls would be
#: reporting its budget.
DEFAULT_ROUND_ONE_PROBES = 6

#: Executor calls the second round gets, and equally the number of extra probes
#: the one-shot control gets. Both arms therefore end on ten executor calls.
DEFAULT_GAP_PROBES = 4

#: Chunks a probe retrieves before :func:`~src.rag.packs.unit_excerpts` spreads
#: them across videos down to :data:`~src.rag.pack_build.DEFAULT_EXCERPTS_PER_UNIT`.
#: Larger than the excerpt budget on purpose — the round-robin needs slack to
#: reach a second creator, and a probe whose top twelve are all one channel is
#: the single-voice failure the pack build already refuses.
DEFAULT_RETRIEVAL_POOL = 18

#: The arms this module writes, in report order.
#:
#: The first three are the original experiment (see the module docstring). The
#: last two are V8's attempt at a loop that beats the *hand-built* pack rather
#: than its own first round, and they are additions rather than replacements —
#: an arm that fared badly is not allowed to leave the table.
RESEARCH_ARMS: tuple[str, ...] = (
    "deep-r1",
    "deep-oneshot",
    "deep-r2",
    "deep-r2-admit",
    "deep-frontier",
)

#: V8's arm: the critic sees where its own probes have not read, round two is
#: forbidden to retrieve into ground round one already had in front of it, and a
#: rule is admitted only if it rests on a passage no admitted rule rests on.
FRONTIER_ARM = "deep-frontier"

#: The ablation that isolates the admission rule: ``deep-r2`` exactly as built,
#: with nothing changed but the evidence-novelty filter. No LLM call is spent on
#: it, so the difference between it and ``deep-r2`` is the rule and nothing else.
ADMISSION_ARM = "deep-r2-admit"

#: Filename of the build report inside ``experts/<topic>/``.
REPORT_FILENAME = "research.json"


# ─── Records ─────────────────────────────────────────────────────────────────


class ResearchProbe(BaseModel):
    """One sub-question, and the reason it exists.

    ``origin`` is the load-bearing field. ``plan`` means the planner wrote it
    before anything had been built; ``gap:<id>`` names the critic finding that
    caused it. A round-two rubric's provenance is read back through this: the
    rubric names its unit, the unit is a probe, and the probe names the gap.
    Without that chain "a second round happened" and "the critic caused a second
    round" are indistinguishable in the artifact.
    """

    probe_id: str
    facet: str
    question: str
    why: str = ""
    origin: str = "plan"
    #: Position in the planner's own ordering, 1-based. The planner is asked for
    #: an importance order so a shorter plan is a prefix of a longer one, which
    #: is what makes ``deep-r1`` and ``deep-oneshot`` nested rather than two
    #: unrelated draws.
    rank: int = 0


class ResearchGap(BaseModel):
    """One thing the critic says a reviewer still cannot check.

    ``probe`` is the critic's own retrieval query, used verbatim as the next
    round's executor input. The critic does not get to write a rubric — it names
    a hole and a place to look, and whether anything fills it is decided by the
    corpus and by the same extraction call every other rubric goes through.
    """

    gap_id: str
    missing: str
    why: str = ""
    probe: str = ""


class ProbeOutcome(BaseModel):
    """What one executor call produced, including when it produced nothing."""

    probe: ResearchProbe
    unit_id: str
    chunks: int = 0
    videos: int = 0
    creator_count: int = 0
    top_creator: str = ""
    top_creator_share: float = 0.0
    eligible: bool = True
    reject_reason: str = ""
    rubric_ids: list[str] = Field(default_factory=list)
    #: Set when the probe was skipped or came back empty — the same three-way
    #: distinction the pack build's gap log makes, kept here so an unspent call
    #: is visible as an unspent call rather than as a quiet zero.
    reason: str = ""
    #: ``True`` only when an LLM extraction call was actually paid for.
    spent_call: bool = False


class RoundReport(BaseModel):
    """One round of the loop as the Build report renders it."""

    round: int
    arm: str
    label: str
    caused_by: str
    probes: list[ProbeOutcome] = Field(default_factory=list)
    executor_calls: int = 0
    planner_calls: int = 0
    critic_calls: int = 0
    rubrics: int = 0
    citations: int = 0
    multi_creator_share: float = 0.0
    #: Candidates this round produced that restated a rule an earlier round
    #: already had. The number that says whether the critic found new ground.
    deduped_against_previous: int = 0
    added_rubric_ids: list[str] = Field(default_factory=list)


# ─── Planner ─────────────────────────────────────────────────────────────────

PLAN_SYSTEM_PROMPT = """You plan a research pass over a corpus of expert video transcripts.

The goal is a rubric pack for one {artifact}: a list of checkable review
criteria somebody could hold a single real {artifact} against and reach a
verdict on.

You are given the pack's topic description and the titles of the videos that
routed into it. Decompose the topic into exactly {count} SUB-QUESTIONS, each
naming one facet a reviewer of a {artifact} would check.

Each sub-question is used as a retrieval query against the transcripts, so write
it the way a creator would say it out loud, not the way a section heading reads.

Rules:
- The facets must not overlap. Two sub-questions that would pull back the same
  passages waste half the budget.
- Only facets this corpus could plausibly cover, judging by the video titles.
- Order them most important first. A reader may take only the first few, so the
  first {count} must degrade gracefully to the first three.
- Never name a specific video, creator or example.
- A facet is a thing about the artifact, not a thing about the videos. "How
  long should it be" is a facet; "what the speakers agree on" is not.

Return only JSON in this shape:
{{"probes":[{{"facet":"...","question":"...","why":"..."}}]}}"""


class ResearchPlanner:
    """One LLM call: a pack declaration in, an ordered list of probes out."""

    def __init__(self, llm: ChatModel, model_name: str = "") -> None:
        self.llm = llm
        self.model_name = model_name

    def plan(
        self,
        declaration: PackDeclaration,
        video_titles: Sequence[str],
        *,
        count: int,
    ) -> list[dict[str, Any]]:
        from langchain_core.messages import HumanMessage, SystemMessage

        prompt = PLAN_SYSTEM_PROMPT.format(artifact=declaration.artifact, count=count)
        body = (
            f"PACK: {declaration.name}\n"
            f"ARTIFACT: {declaration.artifact}\n"
            f"TOPIC: {declaration.routing_text}\n\n"
            "VIDEOS IN THIS PACK\n" + "\n".join(f"- {title}" for title in video_titles)
        )
        response = self.llm.invoke([SystemMessage(content=prompt), HumanMessage(content=body)])
        payload = _json_object(str(getattr(response, "content", response) or ""))
        rows = payload.get("probes")
        return [row for row in (rows or []) if isinstance(row, dict)]


def parse_probes(rows: Sequence[dict[str, Any]], limit: int) -> list[ResearchProbe]:
    """Planner rows turned into probes, in the planner's own order.

    A row with no question is dropped rather than repaired: the question *is*
    the executor's input, and a probe with an empty one would spend a retrieval
    on the empty string and look like a facet the corpus could not cover.
    """
    probes: list[ResearchProbe] = []
    for row in rows:
        question = str(row.get("question") or "").strip()
        if not question:
            continue
        rank = len(probes) + 1
        probes.append(
            ResearchProbe(
                probe_id=f"p{rank:02d}",
                facet=str(row.get("facet") or "").strip() or question[:60],
                question=question,
                why=str(row.get("why") or "").strip(),
                origin="plan",
                rank=rank,
            )
        )
        if len(probes) >= limit:
            break
    return probes


# ─── Gap critic ──────────────────────────────────────────────────────────────

GAP_SYSTEM_PROMPT = """You review a draft rubric pack and name what it cannot check.

The pack is meant to let a reviewer judge one real {artifact}. Below are the
criteria it currently contains and the facets its planner set out to cover.

Name at most {count} things a reviewer of a {artifact} would want to check that
these criteria give them no way to check. For each one write a retrieval query
that would find the transcript passages covering it.

Rules:
- A gap is something MISSING. If a criterion already covers the ground, however
  clumsily worded, it is not a gap — say nothing about it.
- Each gap must be checkable on one artifact. "The tone could be better" is not
  a gap; "nothing here says what the first section has to contain" is.
- Say what is missing, not what to write. You are not drafting the rule; you are
  naming the hole and where to look for the evidence that would fill it.
- Order them by how much a reviewer would miss it.
- Do not name a video, a creator or an example.

Return only JSON in this shape:
{{"gaps":[{{"missing":"...","why":"...","probe":"..."}}]}}"""


class GapCritic:
    """One LLM call: a draft pack in, the holes in it out.

    It reads the criteria and the plan and nothing else — no evidence, no
    transcript, and above all nothing from the held-out expert. A critic that
    could see the answer key would be writing the second round's rubrics rather
    than finding the first round's gaps, and the score would then be measuring
    the leak.
    """

    def __init__(self, llm: ChatModel, model_name: str = "") -> None:
        self.llm = llm
        self.model_name = model_name

    def critique(
        self,
        declaration: PackDeclaration,
        rubrics: Sequence[Rubric],
        probes: Sequence[ResearchProbe],
        *,
        count: int,
    ) -> list[dict[str, Any]]:
        from langchain_core.messages import HumanMessage, SystemMessage

        prompt = GAP_SYSTEM_PROMPT.format(artifact=declaration.artifact, count=count)
        body = (
            f"PACK: {declaration.name}\nARTIFACT: {declaration.artifact}\n"
            f"TOPIC: {declaration.routing_text}\n\n"
            "FACETS THE PLAN COVERED\n"
            + "\n".join(f"- {probe.facet}" for probe in probes)
            + "\n\nCRITERIA IN THE DRAFT PACK\n"
            + "\n".join(
                f"{rubric.rubric_id}: {rubric.criterion}"
                + (f" — check: {rubric.check}" if rubric.check else "")
                for rubric in rubrics
            )
        )
        response = self.llm.invoke([SystemMessage(content=prompt), HumanMessage(content=body)])
        payload = _json_object(str(getattr(response, "content", response) or ""))
        rows = payload.get("gaps")
        return [row for row in (rows or []) if isinstance(row, dict)]


def parse_gaps(rows: Sequence[dict[str, Any]], limit: int) -> list[ResearchGap]:
    gaps: list[ResearchGap] = []
    for row in rows:
        missing = str(row.get("missing") or "").strip()
        probe = str(row.get("probe") or "").strip()
        if not missing or not probe:
            continue
        gaps.append(
            ResearchGap(
                gap_id=f"g{len(gaps) + 1:02d}",
                missing=missing,
                why=str(row.get("why") or "").strip(),
                probe=probe,
            )
        )
        if len(gaps) >= limit:
            break
    return gaps


def gap_probes(gaps: Sequence[ResearchGap], *, start: int = 50) -> list[ResearchProbe]:
    """The critic's gaps as executor inputs, each stamped with its cause.

    Numbered from 50 so a round-two rubric id can never collide with a
    round-one one, and so the origin of a rule is legible from its id alone.

    ``start`` moves the block for a second critic answering the same round one.
    V8's frontier round uses 60, so ``r6101`` and ``r5101`` are never two
    different rules wearing one id in two packs a reader has side by side.
    """
    return [
        ResearchProbe(
            probe_id=f"p{start + index:02d}",
            facet=gap.missing[:80],
            question=gap.probe,
            why=gap.why,
            origin=f"gap:{gap.gap_id}",
            rank=start + index,
        )
        for index, gap in enumerate(gaps, start=1)
    ]


# ─── Executor ────────────────────────────────────────────────────────────────


def probe_unit(
    probe: ResearchProbe,
    vector: Sequence[float],
    vectors: dict[str, Sequence[float]],
    records: dict[str, dict[str, Any]],
    allowed_video_ids: Sequence[str],
    *,
    pool: int = DEFAULT_RETRIEVAL_POOL,
    min_videos: int = 2,
    max_creator_share: float = 0.80,
    exclude_chunk_ids: Sequence[str] = (),
) -> SourceUnit:
    """One probe's retrieval, shaped as a source unit the pack build can use.

    Dense similarity against the chunk vectors this corpus was indexed with,
    restricted to the pack's routed members — which is how the blocked and
    held-out videos stay out, since :func:`~src.rag.pack_build.route_members`
    has already dropped them from the membership this reads.

    ``exclude_chunk_ids`` is the frontier lever (:func:`run_frontier_round`):
    passages an earlier round already had in front of it are struck from the
    candidate set before the top-``pool`` is taken, so a later probe cannot
    spend its call re-reading ground that has already had its chance. It is
    empty for every arm the original experiment builds, so those arms retrieve
    exactly as they did before.

    The same three eligibility floors the cluster arms apply are applied here,
    for the same reason and with the same numbers: a probe whose passages are
    one creator talking produces that creator's advice wearing the pack's name.
    An ineligible probe is marked and returned rather than dropped, so the round
    report can show a call that was deliberately not spent.
    """
    allowed = set(allowed_video_ids)
    blocked = set(exclude_chunk_ids)
    scored: list[tuple[float, str]] = []
    for chunk_id, record in records.items():
        if str(record.get("video_id")) not in allowed or chunk_id in blocked:
            continue
        stored = vectors.get(chunk_id)
        if stored is None:
            continue
        scored.append((cosine(vector, stored), chunk_id))
    # Ties broken on the chunk id so the same probe over the same corpus always
    # retrieves the same passages in the same order.
    scored.sort(key=lambda row: (-row[0], row[1]))
    chunk_ids = [chunk_id for _score, chunk_id in scored[:pool]]
    videos = sorted({str(records[chunk_id]["video_id"]) for chunk_id in chunk_ids})
    creators, top, share = creator_profile(chunk_ids, records)
    reason = ""
    if not chunk_ids:
        reason = "no chunk of this pack's videos could be retrieved for the probe" + (
            " outside the ground an earlier round already read" if blocked else ""
        )
    elif len(videos) < min_videos:
        reason = "every passage the probe retrieved comes from one video"
    elif creators < 2:
        reason = f"every passage the probe retrieved comes from one creator ({top})"
    elif share > max_creator_share:
        reason = f"{share:.0%} of the probe's passages are {top} — one voice with visitors"
    return SourceUnit(
        unit_id=f"probe:{probe.probe_id}",
        kind="probe",
        title=probe.facet,
        summary=probe.question,
        chunk_ids=chunk_ids,
        video_ids=videos,
        retained=len(chunk_ids),
        creator_count=creators,
        top_creator=top,
        top_creator_share=share,
        eligible=not reason,
        reject_reason=reason,
    )


def run_probes(
    workspace: PackWorkspace,
    declaration: PackDeclaration,
    probes: Sequence[ResearchProbe],
    vectors: dict[str, Sequence[float]],
    allowed_video_ids: Sequence[str],
    *,
    pool: int = DEFAULT_RETRIEVAL_POOL,
    max_workers: int = 6,
    exclude_chunk_ids: Sequence[str] = (),
    on_progress: ProgressFn | None = None,
) -> tuple[list[SourceUnit], list[Rubric], list[ProbeOutcome]]:
    """Retrieve for every probe, then extract rubrics from each one's excerpts.

    The extraction is :func:`~src.rag.pack_build.rubrics_for_unit` verbatim —
    same prompt, same excerpt budget, same cache, same citation reconciliation —
    so an executor call and a cluster-arm call cost and constrain the model
    identically. That is what makes the arms comparable at all.
    """
    progress = on_progress or (lambda _message: None)
    questions = [probe.question for probe in probes]
    query_vectors = workspace.embed(questions) if questions else []

    units = [
        probe_unit(
            probe,
            vector,
            vectors,
            workspace.by_chunk,
            allowed_video_ids,
            pool=pool,
            exclude_chunk_ids=exclude_chunk_ids,
        )
        for probe, vector in zip(probes, query_vectors)
    ]

    def work(item: tuple[ResearchProbe, SourceUnit]) -> tuple[str, tuple[list[Rubric], str]]:
        probe, unit = item
        if not unit.eligible:
            return unit.unit_id, ([], unit.reject_reason)
        return unit.unit_id, rubrics_for_unit(workspace, unit, declaration, index=probe.rank)

    results: dict[str, tuple[list[Rubric], str]] = {}
    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
        for unit_id, outcome in executor.map(work, list(zip(probes, units))):
            results[unit_id] = outcome

    rubrics: list[Rubric] = []
    outcomes: list[ProbeOutcome] = []
    for probe, unit in zip(probes, units):
        found, reason = results.get(unit.unit_id, ([], "probe was not executed"))
        rubrics.extend(found)
        outcomes.append(
            ProbeOutcome(
                probe=probe,
                unit_id=unit.unit_id,
                chunks=len(unit.chunk_ids),
                videos=len(unit.video_ids),
                creator_count=unit.creator_count,
                top_creator=unit.top_creator,
                top_creator_share=unit.top_creator_share,
                eligible=unit.eligible,
                reject_reason=unit.reject_reason,
                rubric_ids=[rubric.rubric_id for rubric in found],
                reason=reason,
                spent_call=unit.eligible,
            )
        )
        progress(
            f"  {probe.probe_id} [{probe.origin}] {len(found)} rubric(s) "
            f"from {len(unit.chunk_ids)} chunks / {unit.creator_count} creators"
            + (f" — {reason}" if reason else "")
        )
    return units, rubrics, outcomes


# ─── Publisher ───────────────────────────────────────────────────────────────


def publish(
    workspace: PackWorkspace,
    declaration: PackDeclaration,
    arm: str,
    *,
    members: Sequence[PackMember],
    units: Sequence[SourceUnit],
    candidates: Sequence[Rubric],
    outcomes: Sequence[ProbeOutcome],
    notes: Sequence[str] = (),
    reason_by_unit: dict[str, str] | None = None,
) -> tuple[ExpertPack, list[Rubric]]:
    """Candidates deduped into a pack. Returns ``(pack, dropped)``.

    Order is the argument order, and it matters: round one's surviving rules are
    passed first, so a round-two candidate that restates one of them is the
    thing that gets dropped. Round two cannot bank the pack it inherited as new
    coverage.

    ``reason_by_unit`` lets a caller that dropped a unit's rules for its own
    stated reason say so in the gap log, rather than have the default sentence
    about cosine dedupe stand in for a refusal that was not a cosine dedupe.
    """
    kept, dropped = dedupe_rubrics(candidates, workspace.embed)
    kept_units = {rubric.unit_id for rubric in kept}
    gaps: list[PackGap] = []
    for unit, outcome in zip(units, outcomes):
        if unit.unit_id in kept_units:
            continue
        reason = (
            (reason_by_unit or {}).get(unit.unit_id)
            or outcome.reason
            or "every rubric it produced restated a rule an earlier probe already covered"
        )
        gaps.append(
            PackGap(
                unit_id=unit.unit_id,
                unit_kind=unit.kind,
                unit_title=unit.title,
                videos=len(unit.video_ids),
                chunks=len(unit.chunk_ids),
                creator_count=unit.creator_count,
                top_creator=unit.top_creator,
                top_creator_share=unit.top_creator_share,
                reason=reason,
            )
        )
    stats = pack_statistics(kept, members, units, gaps, workspace.catalog.blocked())
    stats["rubrics_deduped"] = len(dropped)
    stats["rubric_candidates"] = len(candidates)
    stats["unit_budget"] = len(units)
    stats["executor_calls"] = sum(1 for outcome in outcomes if outcome.spent_call)
    provenance = workspace.provenance(notes)
    provenance.units_budget = len(units)
    return (
        ExpertPack(
            topic=declaration.topic,
            name=declaration.name,
            arm=arm,
            artifact=declaration.artifact,
            routing_text=declaration.routing_text,
            blurb=declaration.blurb,
            generated_at=datetime.now(timezone.utc).isoformat(),
            provenance=provenance,
            members=list(members),
            units=list(units),
            rubrics=kept,
            gaps=gaps,
            stats=stats,
        ),
        dropped,
    )


def rubric_diff(before: ExpertPack, after: ExpertPack) -> dict[str, Any]:
    """The v1 → v2 rubric diff the pack browser renders.

    Keyed on ``rubric_id`` rather than on the criterion text, because a rule
    that survived is literally the same object — the publisher carries round
    one's rubrics forward untouched — and a text diff would invite a reader to
    read paraphrase drift into rows where none exists.
    """
    old = {rubric.rubric_id: rubric for rubric in before.rubrics}
    new = {rubric.rubric_id: rubric for rubric in after.rubrics}
    return {
        "before_arm": before.arm,
        "after_arm": after.arm,
        "kept": [_diff_row(new[key]) for key in new if key in old],
        "added": [_diff_row(new[key]) for key in new if key not in old],
        "removed": [_diff_row(old[key]) for key in old if key not in new],
    }


def restatement_audit(
    before: Sequence[Rubric],
    added: Sequence[Rubric],
    embed: Callable[[Sequence[str]], list[list[float]]],
) -> list[dict[str, Any]]:
    """For each new rule, its nearest surviving rule and how near it sat.

    The dedupe threshold is a cliff, and a cliff has a shadow: two statements of
    the same rule that land at 0.83 both survive. That is not hypothetical here
    — round two produced "Keep the resume formatting ATS-safe: one column,
    standard font, no images, tables…" against a round-one rule reading "Format
    the resume for ATS parsing: use a single column, no tables/images/graphics…",
    and the pack kept both.

    So the nearest prior rule and its cosine travel with every added rule. This
    deliberately does **not** filter: a threshold tuned in this module would
    make the loop's own numbers a function of a knob this module owns, and the
    honest instrument is the one that shows a reader the near-misses and lets
    them judge. It is also what stops the score being read naively — a round
    whose additions all sit at 0.8 has grown, not learned.
    """
    if not added:
        return []
    prior = list(before)
    vectors = embed([rubric.criterion for rubric in [*prior, *added]])
    prior_vectors = vectors[: len(prior)]
    rows: list[dict[str, Any]] = []
    for rubric, vector in zip(added, vectors[len(prior) :]):
        ranked = sorted(
            ((cosine(vector, other), item) for other, item in zip(prior_vectors, prior)),
            key=lambda row: (-row[0], row[1].rubric_id),
        )
        best = ranked[0] if ranked else None
        rows.append(
            {
                "rubric_id": rubric.rubric_id,
                "criterion": rubric.criterion,
                "unit_id": rubric.unit_id,
                "nearest_prior_id": best[1].rubric_id if best else None,
                "nearest_prior_criterion": best[1].criterion if best else None,
                "nearest_prior_cosine": round(float(best[0]), 4) if best else None,
                "dedupe_threshold": DEDUPE_THRESHOLD,
            }
        )
    return rows


def gap_closure(
    gaps: Sequence[ResearchGap],
    before: Sequence[Rubric],
    added: Sequence[Rubric],
    origin_by_unit: dict[str, str],
    embed: Callable[[Sequence[str]], list[list[float]]],
) -> list[dict[str, Any]]:
    """Per gap: did anything the probe returned get nearer to it than round one?

    A gap is *asked about* by construction — its probe is the critic's own
    words. Whether it was *closed* is a different question, and this corpus
    answers it badly for some of them: a probe for a facet the transcripts
    barely discuss retrieves the nearest thing they do discuss, and the
    extractor dutifully writes a rule about that instead.

    The test is deliberately weak and deterministic: a gap counts as closed only
    when a rule admitted from its own probe sits closer to the critic's
    statement of the gap than every rule round one already had. It cannot prove
    a rule is *about* the gap — nothing short of a second judgement could — but
    it does separate "the loop found new ground here" from "the loop retrieved
    the same ground again under a new question", and the second of those is the
    outcome a reader is most likely to be sold as the first.

    The anchor is ``gap.missing``, which is phrased as a negation ("No criterion
    requires…"), and a negation is a fair worry for a bi-encoder. It was checked
    rather than argued: re-running the same comparison anchored on ``gap.probe``
    — the critic's positive question — moves every similarity by a few
    hundredths and changes no verdict on the resume-design run. The result is
    not an artefact of how the critic phrases a hole.
    """
    if not gaps:
        return []
    prior = list(before)
    texts = [
        *[gap.missing for gap in gaps],
        *[rubric.criterion for rubric in prior],
        *[rubric.criterion for rubric in added],
    ]
    vectors = embed(texts)
    gap_vectors = vectors[: len(gaps)]
    prior_vectors = vectors[len(gaps) : len(gaps) + len(prior)]
    added_vectors = vectors[len(gaps) + len(prior) :]

    rows: list[dict[str, Any]] = []
    for gap, gap_vector in zip(gaps, gap_vectors):
        baseline = max((cosine(gap_vector, v) for v in prior_vectors), default=0.0)
        mine = [
            (cosine(gap_vector, vector), rubric)
            for rubric, vector in zip(added, added_vectors)
            if origin_by_unit.get(rubric.unit_id) == f"gap:{gap.gap_id}"
        ]
        ranked = sorted(mine, key=lambda row: (-row[0], row[1].rubric_id))
        best = ranked[0] if ranked else None
        rows.append(
            {
                "gap_id": gap.gap_id,
                "missing": gap.missing,
                "probe": gap.probe,
                "rules_from_this_probe": len(mine),
                "best_new_rubric_id": best[1].rubric_id if best else None,
                "best_new_criterion": best[1].criterion if best else None,
                "best_new_cosine": round(float(best[0]), 4) if best else None,
                "round_one_best_cosine": round(float(baseline), 4),
                "closed": bool(best and best[0] > baseline),
            }
        )
    return rows


def _diff_row(rubric: Rubric) -> dict[str, Any]:
    return {
        "rubric_id": rubric.rubric_id,
        "criterion": rubric.criterion,
        "check": rubric.check,
        "unit_id": rubric.unit_id,
        "creators": rubric.creators,
        "citations": len(rubric.evidence),
    }


# ─── Cached calls ────────────────────────────────────────────────────────────


def _cached_json(
    cache_dir: Path,
    kind: str,
    material: str,
    produce: Callable[[], list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], bool]:
    """Run ``produce`` unless an identical call is already on disk.

    Returns ``(rows, was_cached)``. The flag is carried into the report's call
    budget so "eleven calls" always means eleven calls somebody paid for once,
    rather than eleven calls this particular rerun made.
    """
    key = hashlib.sha256(
        json.dumps({"version": RESEARCH_PROMPT_VERSION, "kind": kind, "material": material}).encode(
            "utf-8"
        )
    ).hexdigest()[:24]
    path = cache_dir / f"{kind}-{key}.json"
    if path.is_file():
        try:
            stored = json.loads(path.read_text(encoding="utf-8"))
            rows = [row for row in (stored.get("rows") or []) if isinstance(row, dict)]
            return rows, True
        except (json.JSONDecodeError, OSError):
            pass
    rows = produce()
    cache_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"rows": rows}, indent=2) + "\n", encoding="utf-8")
    return rows, False


def _json_object(content: str) -> dict[str, Any]:
    """The same tolerant JSON reader the rubric extractor uses."""
    from src.rag.packs import _json_object as read

    return read(content)


# ─── The loop ────────────────────────────────────────────────────────────────


class ResearchLoop:
    """Planner, executors, gap critic and publisher over one pack declaration."""

    def __init__(
        self,
        workspace: PackWorkspace,
        declaration: PackDeclaration,
        *,
        planner: ResearchPlanner | None,
        critic: GapCritic | None,
        vectors: dict[str, Sequence[float]],
        cache_dir: Path = DEFAULT_RESEARCH_CACHE_DIR,
    ) -> None:
        self.workspace = workspace
        self.declaration = declaration
        self.planner = planner
        self.critic = critic
        self.vectors = vectors
        self.cache_dir = cache_dir
        self.planner_calls = 0
        self.critic_calls = 0

    def plan(self, video_titles: Sequence[str], count: int) -> list[ResearchProbe]:
        planner = self.planner
        if planner is None:
            raise ValueError("No planner model configured for this run")
        material = json.dumps(
            {
                "topic": self.declaration.topic,
                "artifact": self.declaration.artifact,
                "routing_text": self.declaration.routing_text,
                "titles": list(video_titles),
                "count": count,
                "model": planner.model_name,
            },
            sort_keys=True,
        )
        rows, cached = _cached_json(
            self.cache_dir,
            "plan",
            material,
            lambda: planner.plan(self.declaration, video_titles, count=count),
        )
        if not cached:
            self.planner_calls += 1
        return parse_probes(rows, count)

    def find_gaps(
        self,
        pack: ExpertPack,
        probes: Sequence[ResearchProbe],
        count: int,
    ) -> list[ResearchGap]:
        critic = self.critic
        if critic is None:
            raise ValueError("No critic model configured for this run")
        material = json.dumps(
            {
                "topic": self.declaration.topic,
                "artifact": self.declaration.artifact,
                "routing_text": self.declaration.routing_text,
                "facets": [probe.facet for probe in probes],
                "criteria": [[r.rubric_id, r.criterion, r.check] for r in pack.rubrics],
                "count": count,
                "model": critic.model_name,
            },
            sort_keys=True,
        )
        rows, cached = _cached_json(
            self.cache_dir,
            "gaps",
            material,
            lambda: critic.critique(self.declaration, pack.rubrics, probes, count=count),
        )
        if not cached:
            self.critic_calls += 1
        return parse_gaps(rows, count)


def _round_report(
    *,
    index: int,
    arm: str,
    label: str,
    caused_by: str,
    pack: ExpertPack,
    outcomes: Sequence[ProbeOutcome],
    dropped: Sequence[Rubric],
    previous: ExpertPack | None,
    planner_calls: int = 0,
    critic_calls: int = 0,
) -> RoundReport:
    inherited = {rubric.rubric_id for rubric in previous.rubrics} if previous else set()
    # Only this round's own probes can be deduped away: the publisher is handed
    # the previous round's surviving rules first, and dedupe keeps the earlier
    # statement, so an inherited rule is never the one dropped.
    mine = {outcome.unit_id for outcome in outcomes}
    return RoundReport(
        round=index,
        arm=arm,
        label=label,
        caused_by=caused_by,
        probes=list(outcomes),
        executor_calls=sum(1 for outcome in outcomes if outcome.spent_call),
        planner_calls=planner_calls,
        critic_calls=critic_calls,
        rubrics=len(pack.rubrics),
        citations=int(pack.stats.get("citations") or 0),
        multi_creator_share=float(pack.stats.get("multi_creator_share") or 0.0),
        deduped_against_previous=sum(1 for rubric in dropped if rubric.unit_id in mine),
        added_rubric_ids=[r.rubric_id for r in pack.rubrics if r.rubric_id not in inherited],
    )


def run_deep_research(
    settings: Settings,
    *,
    topic: str,
    packs_dir: Path = Path("experts"),
    round_one_probes: int = DEFAULT_ROUND_ONE_PROBES,
    gap_probes_count: int = DEFAULT_GAP_PROBES,
    pool: int = DEFAULT_RETRIEVAL_POOL,
    cache_dir: Path = DEFAULT_RESEARCH_CACHE_DIR,
    records: Sequence[dict[str, Any]] | None = None,
    on_progress: ProgressFn | None = None,
) -> dict[str, Any]:
    """Build all three research arms for one topic and write the build report.

    One planner call serves every arm. ``deep-r1`` is the plan's first ``P``
    probes, ``deep-oneshot`` is the plan's first ``P + Q``, and ``deep-r2`` is
    the first ``P`` plus ``Q`` probes the gap critic wrote after reading
    ``deep-r1``. The one-shot and the second round therefore spend the same
    executor budget on the same opening probes and differ only in where the last
    ``Q`` questions came from — which is the comparison the loop has to win to
    be worth its extra round.
    """
    from langchain_openai import ChatOpenAI

    from src.agents.llm import chat_model_kwargs

    progress = on_progress or (lambda _message: None)
    store = PackStore(packs_dir)
    catalog = store.catalog()
    declaration = catalog.declaration(topic)
    if declaration is None:
        raise ValueError(f"No declared pack for topic {topic!r}")

    if records is None:
        from src.api.corpus import load_chunk_embeddings

        records = load_chunk_embeddings(settings.chroma_path, settings.chunk_collection)
    vectors: dict[str, Sequence[float]] = {
        str(record["chunk_id"]): list(record["embedding"])
        for record in records
        if record.get("embedding") is not None
    }
    progress(f"{len(vectors)} chunk vectors loaded for retrieval")

    workspace = open_workspace(settings, catalog, with_model=True)
    provenance = workspace.provenance()
    progress(
        f"corpus {provenance.corpus_digest} — {provenance.chunk_count} chunks, "
        f"{provenance.video_count} videos"
    )

    members = route_members(workspace, declaration, store.overrides(topic))
    allowed = included_video_ids(members)
    titles = [
        str((workspace.video_catalogue.get(video_id) or {}).get("title") or video_id)
        for video_id in allowed
    ]
    progress(f"[{topic}] {len(allowed)} member videos")

    llm = ChatOpenAI(**chat_model_kwargs(settings))
    loop = ResearchLoop(
        workspace,
        declaration,
        planner=ResearchPlanner(llm, settings.deepseek_model),
        critic=GapCritic(llm, settings.deepseek_model),
        vectors=vectors,
        cache_dir=cache_dir,
    )

    total = round_one_probes + gap_probes_count
    plan = loop.plan(titles, total)
    progress(f"[{topic}] planner returned {len(plan)} probes")
    for probe in plan:
        progress(f"  {probe.probe_id} {probe.facet} — {probe.question}")
    first = plan[:round_one_probes]
    extra = plan[round_one_probes:]

    # Round 1 — the plan's opening probes.
    progress(f"[{topic}] round 1 — {len(first)} probes")
    units_r1, cands_r1, outcomes_r1 = run_probes(
        workspace, declaration, first, vectors, allowed, pool=pool, on_progress=progress
    )
    pack_r1, dropped_r1 = publish(
        workspace,
        declaration,
        "deep-r1",
        members=members,
        units=units_r1,
        candidates=cands_r1,
        outcomes=outcomes_r1,
        notes=[f"deep-research round 1: the plan's first {len(first)} probes"],
    )
    store.save_pack(pack_r1)
    progress(f"[{topic}] deep-r1 {pack_r1.stats['rubrics']} rubrics")

    # The one-shot control — the same plan, continued, with no critic.
    progress(f"[{topic}] one-shot control — {len(extra)} further plan probes")
    units_os, cands_os, outcomes_os = run_probes(
        workspace, declaration, extra, vectors, allowed, pool=pool, on_progress=progress
    )
    pack_os, dropped_os = publish(
        workspace,
        declaration,
        "deep-oneshot",
        members=members,
        units=[*units_r1, *units_os],
        candidates=[*pack_r1.rubrics, *cands_os],
        outcomes=[*outcomes_r1, *outcomes_os],
        notes=[
            "deep-research one-shot control: the same plan taken to "
            f"{len(plan)} probes in a single round, no critic"
        ],
    )
    store.save_pack(pack_os)
    progress(f"[{topic}] deep-oneshot {pack_os.stats['rubrics']} rubrics")

    # The critic reads round 1 and writes round 2's probes.
    gaps = loop.find_gaps(pack_r1, first, gap_probes_count)
    progress(f"[{topic}] gap critic named {len(gaps)} gaps")
    for gap in gaps:
        progress(f"  {gap.gap_id} MISSING: {gap.missing}")
        progress(f"       probe: {gap.probe}")
    probes_r2 = gap_probes(gaps)
    units_r2, cands_r2, outcomes_r2 = run_probes(
        workspace, declaration, probes_r2, vectors, allowed, pool=pool, on_progress=progress
    )
    pack_r2, dropped_r2 = publish(
        workspace,
        declaration,
        "deep-r2",
        members=members,
        units=[*units_r1, *units_r2],
        candidates=[*pack_r1.rubrics, *cands_r2],
        outcomes=[*outcomes_r1, *outcomes_r2],
        notes=[
            f"deep-research round 2: round 1 plus {len(probes_r2)} probes the gap critic asked for"
        ],
    )
    store.save_pack(pack_r2)
    progress(f"[{topic}] deep-r2 {pack_r2.stats['rubrics']} rubrics")
    inherited = {rubric.rubric_id for rubric in pack_r1.rubrics}
    added_r2 = [rubric for rubric in pack_r2.rubrics if rubric.rubric_id not in inherited]

    rounds = [
        _round_report(
            index=1,
            arm="deep-r1",
            label="round 1 — the plan's opening probes",
            caused_by="planner",
            pack=pack_r1,
            outcomes=outcomes_r1,
            dropped=dropped_r1,
            previous=None,
            planner_calls=1,
        ),
        _round_report(
            index=2,
            arm="deep-r2",
            label="round 2 — the gap critic's probes",
            caused_by="gap critic reading deep-r1",
            pack=pack_r2,
            outcomes=outcomes_r2,
            dropped=dropped_r2,
            previous=pack_r1,
            critic_calls=1,
        ),
    ]
    control = _round_report(
        index=2,
        arm="deep-oneshot",
        label="control — the same plan continued, no critic",
        caused_by="planner",
        pack=pack_os,
        outcomes=outcomes_os,
        dropped=dropped_os,
        previous=pack_r1,
    )

    report: dict[str, Any] = {
        "kind": "deep-research",
        "topic": topic,
        "name": declaration.name,
        "artifact": declaration.artifact,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "corpus_digest": provenance.corpus_digest,
        "chunk_count": provenance.chunk_count,
        "model": settings.deepseek_model,
        "members": len(allowed),
        "excluded_video_ids": list(catalog.excluded_video_ids),
        "held_out_video_ids": list(catalog.held_out_video_ids),
        "settings": {
            "round_one_probes": round_one_probes,
            "gap_probes": gap_probes_count,
            "retrieval_pool": pool,
            "excerpts_per_unit": DEFAULT_EXCERPTS_PER_UNIT,
            "max_rubrics_per_unit": DEFAULT_MAX_RUBRICS_PER_UNIT,
        },
        "plan": [probe.model_dump(mode="json") for probe in plan],
        "gaps": [gap.model_dump(mode="json") for gap in gaps],
        # The two diagnostics that stop the round-2 delta being read as a win by
        # default: which gaps the second round actually got nearer to, and how
        # close each new rule sits to a rule the pack already had.
        "gap_closure": gap_closure(
            gaps,
            pack_r1.rubrics,
            added_r2,
            {outcome.unit_id: outcome.probe.origin for outcome in outcomes_r2},
            workspace.embed,
        ),
        "restatements": restatement_audit(pack_r1.rubrics, added_r2, workspace.embed),
        "rounds": [row.model_dump(mode="json") for row in rounds],
        "control": control.model_dump(mode="json"),
        "diff": rubric_diff(pack_r1, pack_r2),
        "control_diff": rubric_diff(pack_r1, pack_os),
        "budget": budget_table(rounds, control, pack_r1, pack_os, pack_r2),
        "arms": {
            pack.arm: {
                "rubrics": pack.stats.get("rubrics"),
                "citations": pack.stats.get("citations"),
                "multi_creator_share": pack.stats.get("multi_creator_share"),
                "executor_calls": pack.stats.get("executor_calls"),
                "rubrics_deduped": pack.stats.get("rubrics_deduped"),
                "gaps": pack.stats.get("gaps"),
            }
            for pack in (pack_r1, pack_os, pack_r2)
        },
        "scores": None,
    }
    save_report(store, topic, report)
    return report


def budget_table(
    rounds: Sequence[RoundReport],
    control: RoundReport,
    pack_r1: ExpertPack,
    pack_os: ExpertPack,
    pack_r2: ExpertPack,
) -> list[dict[str, Any]]:
    """Calls per arm, stated rather than implied.

    A loop that beat a one-shot on more calls has reported its budget, not its
    architecture. These rows are what a reader checks that against: the planner
    call is shared by every arm, the one-shot and the second round spend the
    same executor budget, and the second round's only extra spend is the single
    critic call.
    """
    rows: list[dict[str, Any]] = []
    for report, pack in ((rounds[0], pack_r1), (control, pack_os), (rounds[1], pack_r2)):
        critic_calls = 1 if report.arm == "deep-r2" else 0
        executor_calls = int(pack.stats.get("executor_calls") or 0)
        rows.append(
            {
                "arm": report.arm,
                "label": report.label,
                "planner_calls": 1,
                "critic_calls": critic_calls,
                "executor_calls": executor_calls,
                "probes_budgeted": int(pack.stats.get("unit_budget") or 0),
                "total_llm_calls": 1 + critic_calls + executor_calls,
                "rubrics": len(pack.rubrics),
                "citations": int(pack.stats.get("citations") or 0),
            }
        )
    return rows


# ─── Report storage ──────────────────────────────────────────────────────────


def report_path(store: PackStore, topic: str) -> Path:
    return store.topic_dir(topic) / REPORT_FILENAME


def save_report(store: PackStore, topic: str, report: dict[str, Any]) -> Path:
    path = report_path(store, topic)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return path


def load_report(store: PackStore, topic: str) -> dict[str, Any] | None:
    """The build report, or ``None`` if it is absent *or* unreadable.

    A half-written report must read as "no loop has been run", the same rule the
    pack reader follows — it must never take the tab down with a 500.
    """
    path = report_path(store, topic)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return data if isinstance(data, dict) else None


# ─── Scoring ─────────────────────────────────────────────────────────────────


def score_research(
    settings: Settings,
    *,
    topic: str,
    packs_dir: Path = Path("experts"),
    arms: Sequence[str] = RESEARCH_ARMS,
    baseline_arm: str = "merged",
    repeats: int = 5,
    records: Sequence[dict[str, Any]] | None = None,
    max_workers: int = 4,
    on_progress: ProgressFn | None = None,
) -> dict[str, Any]:
    """Score the research arms and the hand-built pack on the held-out harness.

    The hand-built arm is scored in the same call rather than quoted from
    ``experts/ablation.json``, so the comparison is one table produced by one
    resolver — and because the matcher cache is keyed on the findings text, an
    unchanged hand pack costs nothing to include.

    Cells are scored concurrently. Each is five serial matcher calls over its own
    matrix and they share nothing but the cache directory, where every write is
    to a filename derived from that cell's own fingerprint.
    """
    from src.evals.critique import (
        CRITIQUE_METRICS,
        build_run,
        load_critique_dataset,
        repeated_matcher,
        score_critique,
    )
    from src.evals.critique_run import MATCHER_VERSION, cached_matcher, llm_matcher
    from src.evals.pack_ablation import (
        baseline_table,
        chunk_text_from_records,
        pack_as_critique,
        winner,
    )
    from src.rag.packs import corpus_digest

    progress = on_progress or (lambda _message: None)
    store = PackStore(packs_dir)
    wanted = [baseline_arm, *arms] if baseline_arm else list(arms)
    loaded = [pack for pack in (store.load_pack(topic, arm) for arm in wanted) if pack is not None]
    if not loaded:
        raise ValueError(f"No built arms for pack {topic!r}. Run deep-research first.")

    if records is None:
        from src.api.corpus import load_chunk_embeddings

        records = load_chunk_embeddings(settings.chroma_path, settings.chunk_collection)
    chunk_text = chunk_text_from_records(records)
    dataset = load_critique_dataset()
    match = cached_matcher(
        repeated_matcher(llm_matcher(settings), repeats=repeats), repeats=repeats
    )

    def score(pack: ExpertPack) -> dict[str, Any]:
        progress(f"[{topic}/{pack.arm}] scoring {len(pack.rubrics)} rubrics")
        cell = score_critique(pack_as_critique(pack), dataset, match, chunk_text)
        cell["rubrics"] = len(pack.rubrics)
        cell["multi_creator_share"] = pack.stats.get("multi_creator_share")
        cell["unit_budget"] = pack.stats.get("unit_budget")
        cell["executor_calls"] = pack.stats.get("executor_calls")
        cell["units_by_kind"] = pack.stats.get("units_by_kind")
        progress(f"[{topic}/{pack.arm}] done")
        return cell

    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
        cells = list(executor.map(score, loaded))

    run = build_run(
        dataset,
        cells,
        config={
            "harness": "deep-research",
            "arms": [pack.arm for pack in loaded],
            "rubric_model": loaded[0].provenance.rubric_model,
            "matcher": "llm",
            "matcher_model": settings.deepseek_model,
            "matcher_version": MATCHER_VERSION,
            "match_repeats": repeats,
            "corpus_digest": loaded[0].provenance.corpus_digest,
            "held_out_video_ids": loaded[0].provenance.held_out_video_ids,
            "resolver": "chunk_text_from_records",
            # Two digests, never one. The arms were built from the corpus as it
            # stood; the citations are resolved against the corpus as it stands
            # now, and ingestion does not stop for an experiment. When these
            # differ the arms are still comparable to *each other* — they were
            # all built at the build digest — but a reader must be told which
            # corpus the provenance column was measured on rather than being
            # left to assume the two are the same. ``provenance`` is the number
            # that would show the drift biting: a stored quote whose chunk
            # boundary moved stops resolving, because the resolver matches on
            # (video_id, timestamp span) and then on the quoted words, never on
            # a chunk index.
            **_scoring_corpus(records, loaded[0].provenance.corpus_digest, corpus_digest),
        },
        baseline=loaded[0].arm,
    )
    # Deliberately left as ``critique-eval``, the kind
    # :func:`src.api.experiments.load_experiments` surfaces to the Critique
    # panel. This *is* a held-out critique run — same dataset, same matcher,
    # same four metrics — and stamping it with a private kind would have hidden
    # the loop's rows from the one table a reviewer compares runs in. Which
    # harness produced it is recorded in ``config.harness`` and in ``harness``,
    # so the two are still told apart by anything that cares.
    run["harness"] = "deep-research"
    run["topic"] = topic
    run["verdicts"] = {metric: winner(run, metric) for metric in CRITIQUE_METRICS}
    # Beside the verdicts, never instead of them. ``winner`` ranks the arms and
    # reports the leader against the *runner-up*; the gate this harness exists
    # to settle asks whether a loop arm reaches the *hand-built* pack, and when
    # two loop arms tie for the lead ``winner`` says "tied" and answers nothing.
    # Committed into the run rather than computed by the panel, because a
    # finding that lives only at render time is missing from the file.
    run["against_baseline"] = baseline_table(run)
    run["generated_at"] = datetime.now(timezone.utc).isoformat()
    return run


def _scoring_corpus(
    records: Sequence[dict[str, Any]],
    build_digest: str,
    digest_of_records: Callable[[Sequence[dict[str, Any]]], str],
) -> dict[str, Any]:
    """The corpus the scoring actually ran against, next to the one built from.

    Separated out so the comparison is a value in the run file rather than a
    thing a reader has to notice. ``scored_on_build_corpus`` is the one-bit
    answer; the counts and digests are there to be checked.
    """
    scoring_digest = digest_of_records(records)
    return {
        "scoring_corpus_digest": scoring_digest,
        "scoring_chunk_count": len(records),
        "scoring_video_count": len({str(record.get("video_id")) for record in records}),
        "scored_on_build_corpus": scoring_digest == build_digest,
    }


def score_rows(run: dict[str, Any]) -> list[dict[str, Any]]:
    """The score table the Build report renders, one row per arm.

    Finding count and citation count sit beside every score on purpose. The
    attack ``criteria_recall`` is open to is padding — more findings, one
    citation each — so a reader has to be able to see the shape of an arm that
    gained, and not be handed the number on its own.
    """
    return [
        {
            "arm": cell.get("setup"),
            "scores": cell.get("scores") or {},
            "score_spread": cell.get("score_spread") or {},
            "findings_total": cell.get("findings_total"),
            "findings_grounded": cell.get("findings_grounded"),
            "citations_total": cell.get("citations_total"),
            "citations_resolved": cell.get("citations_resolved"),
            "criteria_matched": cell.get("criteria_matched"),
            "criteria_applicable": cell.get("criteria_applicable"),
            "criteria_recall_grouped": cell.get("criteria_recall_grouped"),
            "executor_calls": cell.get("executor_calls"),
            "unit_budget": cell.get("unit_budget"),
            "held_out_leaks": cell.get("held_out_leaks"),
        }
        for cell in run.get("cells") or []
    ]


def attach_scores(
    store: PackStore,
    topic: str,
    run: dict[str, Any],
) -> dict[str, Any] | None:
    """Fold a scoring run into the stored build report and save it."""
    from src.evals.critique import CRITIQUE_METRICS
    from src.evals.pack_ablation import baseline_table, winner

    report = load_report(store, topic)
    if report is None:
        return None
    config = run.get("config") or {}
    report["scores"] = {
        "generated_at": run.get("generated_at"),
        "run_id": run.get("run_id"),
        "metrics": run.get("metrics") or [],
        "baseline": run.get("baseline"),
        "held_out_title": run.get("held_out_title"),
        "held_out_video_id": run.get("held_out_video_id"),
        "criteria_total": run.get("criteria_total"),
        "criteria_applicable": run.get("criteria_applicable"),
        "match_repeats": config.get("match_repeats"),
        # Carried into the report, not left in the run file, because the Build
        # report is where somebody reads these scores and "which corpus was
        # this measured on" is part of the score.
        "build_corpus_digest": config.get("corpus_digest"),
        "scoring_corpus_digest": config.get("scoring_corpus_digest"),
        "scoring_chunk_count": config.get("scoring_chunk_count"),
        "scored_on_build_corpus": config.get("scored_on_build_corpus"),
        "rows": score_rows(run),
        # Recomputed rather than carried over, for the same reason the pack
        # reader recomputes: ``winner()`` is pure, and a report written before a
        # verdict bug was found would otherwise keep serving the old reading.
        "verdicts": {metric: winner(run, metric) for metric in CRITIQUE_METRICS},
        "against_baseline": run.get("against_baseline") or baseline_table(run),
    }
    save_report(store, topic, report)
    return report


def research_report(topic: str, packs_dir: Path | None = None) -> dict[str, Any] | None:
    """The build report as the browser reads it, with the verdicts re-derived.

    The report is otherwise served exactly as written. The one thing recomputed
    is ``verdicts``: ``winner()`` is pure, and a report saved before a verdict
    bug was found would keep serving the old reading forever. Every stored
    report predates the three-way ``basis``, so without this the Build report
    still announces "leader clears the runner-up's own spread" on metrics that
    were only ever scored once per arm.

    ``score_rows`` names the arm ``arm`` while ``winner`` reads ``setup``; the
    scores and spreads are otherwise the same fields, so the rows are relabelled
    rather than re-derived from a run file the reader does not have.
    """
    from src.api.packs import DEFAULT_PACKS_DIR
    from src.evals.critique import CRITIQUE_METRICS
    from src.evals.pack_ablation import baseline_table, winner

    report = load_report(PackStore(packs_dir or DEFAULT_PACKS_DIR), topic)
    if not report:
        return report
    scores = report.get("scores")
    rows = (scores or {}).get("rows") or []
    if scores and rows:
        as_run = {
            "cells": [{**row, "setup": row.get("arm")} for row in rows],
            "baseline": scores.get("baseline"),
        }
        scores["verdicts"] = {metric: winner(as_run, metric) for metric in CRITIQUE_METRICS}
        # Same rule as the verdicts, for the same reason and one slice later: a
        # report written before ``against_baseline`` existed has the numbers it
        # needs but not the comparison, and re-deriving it is pure arithmetic
        # over rows the file already carries.
        scores["against_baseline"] = baseline_table(as_run)
    store = PackStore(packs_dir or DEFAULT_PACKS_DIR)
    _mark_rediscoveries(store, topic, report)
    _fill_frontier_round(store, topic, report)
    return report


def _mark_rediscoveries(store: PackStore, topic: str, report: dict[str, Any]) -> None:
    """Flag each added rule that the shipped pack already covered from the same chunk.

    V8's gate asks whether the loop beats the hand-built pack, and it does not.
    Its evaluator found the mechanical reason and noted the page was silent on
    it: the loop's one genuinely novel rule, r5403, cites the same chunk as the
    hand build's r0403. The rule is new *against round one*, which is all the
    gap critic can see — so the novelty clause passes honestly — but it is not
    new against the pack it is being measured against. Without this a reader
    sees ten additions and no way to tell rediscovery from discovery.

    Overlap is measured on cited chunk ids, not on wording. Two rules phrased
    differently off the same transcript second are the same finding, and a text
    similarity score would be a judgement where an identity check will do.
    """
    shipped = _load_arm(store, topic, None)
    if shipped is None:
        return
    # chunk id -> the shipped rules citing it.
    shipped_by_chunk: dict[str, list[str]] = {}
    for rubric in shipped.rubrics:
        for item in rubric.evidence:
            shipped_by_chunk.setdefault(item.chunk_id, []).append(rubric.rubric_id)

    # Every diff on the page, or none of them. The control's additions render
    # directly beneath the loop's, so marking only the loop would leave every
    # control row blank — which reads as "all new" and flatters the arm the loop
    # is supposed to beat. The same argument covers V8's frontier arm, whose
    # whole claim is that it rediscovers less.
    for diff in _diffs_in(report):
        added = diff.get("added") or []
        arm = _load_arm(store, topic, diff.get("after_arm")) if added else None
        if arm is None:
            continue
        by_id = {rubric.rubric_id: rubric for rubric in arm.rubrics}
        for row in added:
            rubric = by_id.get(row.get("rubric_id"))
            if rubric is None:
                continue
            row["already_in_shipped"] = sorted(
                {
                    shipped_id
                    for item in rubric.evidence
                    for shipped_id in shipped_by_chunk.get(item.chunk_id, ())
                }
            )


def _fill_frontier_round(store: PackStore, topic: str, report: dict[str, Any]) -> None:
    """Give the frontier arm a row in the Rounds table, like every other arm.

    The Rounds table is where a reader compares what each round *spent* against
    what it *added*, and the arm with the strongest claim was the one missing
    from it — which reads as a claim made somewhere the comparison is not.

    Derived rather than stored so a report written before this existed gains the
    row without the packs being rebuilt: every field is already in the frontier
    block, and none of it is a judgement. Skipped when the block already carries
    a round, so a future build's own record is never overwritten by a
    reconstruction of it.
    """
    frontier = report.get("frontier")
    if not isinstance(frontier, dict) or isinstance(frontier.get("round"), dict):
        return
    outcomes = [row for row in (frontier.get("probes") or []) if isinstance(row, dict)]
    added = [
        row for row in ((frontier.get("diff") or {}).get("added") or []) if isinstance(row, dict)
    ]
    budget = next(
        (row for row in (frontier.get("budget") or []) if row.get("arm") == frontier.get("arm")),
        {},
    )
    pack = _load_arm(store, topic, str(frontier.get("arm") or ""))
    frontier["round"] = {
        "round": 2,
        "arm": str(frontier.get("arm") or ""),
        "label": "frontier round — a coverage-aware critic, off round 1's ground",
        "caused_by": "coverage-aware gap critic reading deep-r1",
        "probes": outcomes,
        "executor_calls": sum(1 for row in outcomes if row.get("spent_call")),
        "planner_calls": 0,
        "critic_calls": 1,
        "rubrics": int(budget.get("rubrics") or 0),
        "citations": int(budget.get("citations") or 0),
        "multi_creator_share": float((pack.stats.get("multi_creator_share") if pack else 0) or 0.0),
        # Both refusals, not just the cosine one. The wording dedupe and the
        # evidence-novelty rule are two ways of saying "this round added
        # nothing here", and a column that counted one of them would understate
        # what the round threw away.
        "deduped_against_previous": len(frontier.get("deduped_against_previous") or [])
        + len(frontier.get("refused") or []),
        "added_rubric_ids": [str(row.get("rubric_id")) for row in added],
    }


def _diffs_in(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Every rubric diff the Build report renders, wherever it is stored.

    The frontier arm's diff lives inside its own block rather than at the top
    level, and a marker that reached only the top-level pair would leave the one
    arm whose claim is "it rediscovers less" as the only unmarked table.
    """
    found = [report.get("diff"), report.get("control_diff")]
    frontier = report.get("frontier")
    if isinstance(frontier, dict):
        found.append(frontier.get("diff"))
    return [row for row in found if isinstance(row, dict)]


def _load_arm(store: PackStore, topic: str, arm: str | None) -> Any | None:
    try:
        return store.load_pack(topic, arm)
    except (OSError, ValueError):
        return None


# ─── V8: the frontier round ──────────────────────────────────────────────────
#
# The loop above beats its own first round and loses to the hand-built pack. The
# mechanism was measured rather than guessed: of ``deep-r2``'s ten added rules,
# four cite chunks the shipped pack already cites, and the one-shot control
# rediscovers at exactly the same rate — four of ten. Round two was not searching
# better than no loop at all, it was searching *more*, and the extra search
# landed back on ground somebody had already worked.
#
# Two facts make that measurable here rather than arguable. The member corpus is
# 182 chunks; round one's six probes put 72 of them in front of the model, and
# every round-two probe drew between five and twelve of its eighteen passages out
# of those same 72. Meanwhile the shipped pack's units span 174 of the 182 — it
# has *seen* nearly everything and distilled 23 chunks of it. So the ground a
# second round could add is real and large, and nothing in the loop was pointing
# at it.
#
# Three changes, each deterministic in what it forbids:
#
# **The critic is told where its own probes have not read.** It still sees the
# round-one criteria and the plan's facets and nothing else of the *content* —
# no evidence, no transcript, nothing from the held-out expert. What it gains is
# a coverage table: the member videos, their titles, and how many of each one's
# passages round one's probes actually retrieved. Video titles are not a new
# information class in this loop — :data:`PLAN_SYSTEM_PROMPT` already hands the
# planner exactly that list — so the critic is being told what the loop already
# knew about itself, which is the one thing it could not previously tell:
# whether a hole is missing from the pack or missing from the corpus.
#
# **A round-two probe may not retrieve into ground round one already read.**
# Round one's retrieved chunks are struck from the candidate set. This is free,
# deterministic and computed from the loop's own output; it is not a threshold
# and there is nothing in it to tune.
#
# **A rule is admitted only if it rests on a passage no admitted rule rests on.**
# The existing dedupe compares *wordings* at a cosine cliff, and the shadow under
# that cliff is documented: six of round two's ten additions sat between 0.744
# and 0.859 against a 0.86 threshold and were kept. Evidence identity has no
# cliff — either a rule brings a transcript second of its own or it does not.
#
# What the third change is *not* allowed to be sold as: it mirrors the shape of
# the scorer's exclusivity rule, so any ``evidence_precision`` it buys is partly
# bought by construction. That is why :data:`ADMISSION_ARM` exists — the same
# rule applied to ``deep-r2`` with nothing else changed and no call spent — so a
# reader can subtract it.
#
# This comment used to say that ``criteria_recall`` *cannot* be bought that way,
# because dropping rules can only ever remove chances to match. **That was
# false, and V8's evaluator caught it.** ``deep-r2-admit`` is a strict subset of
# ``deep-r2`` — the same rules minus ``r5201`` and ``r5202`` — and it scores
# *higher*, 0.3684 against 0.4211. The mechanism is not exclusivity: ``r0401``
# is grounded on the same chunk in all four deep arms. It is
# :func:`~src.evals.critique.enforce_one_to_one`. A criterion may be paired with
# at most one finding, so thinning the pool removes competitors, and ``c22``
# — which the matcher had been giving to an *ungrounded* rule in ``deep-r1`` and
# to nothing at all in ``deep-r2`` — lands on the grounded ``r0401`` once two
# rules stop contending for it.
#
# So the honest claim is weaker: dropping rules is **not a reliable way to buy
# recall**. It moved one criterion in nineteen here, well inside the matcher's
# own spread, and it moved it by changing a pairing rather than by adding
# coverage. What the admission rule cannot do is manufacture *new ground* — the
# criteria a pack reaches are still bounded by the passages it distilled.

#: Bumped when :data:`COVERAGE_GAP_SYSTEM_PROMPT` changes. Separate from
#: :data:`RESEARCH_PROMPT_VERSION` on purpose — bumping that one would invalidate
#: the cached plan the committed arms were built from, and the arms would stop
#: being nested inside a plan nobody could reproduce.
FRONTIER_PROMPT_VERSION = "v1"

#: Where the frontier round's probe numbering starts. Ten past the original
#: loop's block, so ``deep-r2``'s ``r5201`` and this arm's ``r6201`` are never
#: the same id on two different rules in two packs a reader compares.
FRONTIER_PROBE_OFFSET = 60


class CoverageProfile(BaseModel):
    """One member video, and how much of it a round's probes actually read."""

    video_id: str
    title: str = ""
    channel_name: str = ""
    chunks: int = 0
    read: int = 0

    @property
    def unread(self) -> int:
        return max(0, self.chunks - self.read)


def coverage_profile(
    records: Sequence[dict[str, Any]],
    allowed_video_ids: Sequence[str],
    retrieved_chunk_ids: Sequence[str],
) -> list[CoverageProfile]:
    """Per member video: its passages, and how many a round has read.

    Ordered by unread passages, most first — which is the order the critic is
    asked to read it in, and the only ordering decision in the whole signal. No
    cut-off is applied: every member video is listed whether it was read or not,
    so a critic that ignores the table cannot be said to have been steered by a
    filter this module chose.
    """
    read = set(retrieved_chunk_ids)
    allowed = set(allowed_video_ids)
    rows: dict[str, CoverageProfile] = {}
    for record in records:
        video_id = str(record.get("video_id"))
        if video_id not in allowed:
            continue
        row = rows.setdefault(
            video_id,
            CoverageProfile(
                video_id=video_id,
                title=str(record.get("title") or ""),
                channel_name=str(record.get("channel_name") or ""),
            ),
        )
        row.chunks += 1
        if str(record.get("chunk_id")) in read:
            row.read += 1
    return sorted(rows.values(), key=lambda row: (-row.unread, row.video_id))


COVERAGE_GAP_SYSTEM_PROMPT = """You review a draft rubric pack and name what it cannot check.

The pack is meant to let a reviewer judge one real {artifact}. Below are the
criteria it currently contains, the facets its planner set out to cover, and a
coverage table saying how much of each source video the draft's probes actually
read.

Name at most {count} things a reviewer of a {artifact} would want to check that
these criteria give them no way to check. For each one write a retrieval query
that would find the transcript passages covering it.

Rules:
- A gap is something MISSING. If a criterion already covers the ground, however
  clumsily worded or from whatever angle, it is not a gap — say nothing about it.
- Prefer gaps the UNREAD part of this corpus could plausibly answer. The coverage
  table is the only thing telling you where the draft has not looked; a gap in
  material that was read and produced nothing is a gap the corpus cannot fill,
  and spending a probe on it wastes the round.
- Each gap must be checkable on one artifact. "The tone could be better" is not
  a gap; "nothing here says what the first section has to contain" is.
- Say what is missing, not what to write. You are not drafting the rule; you are
  naming the hole and where to look for the evidence that would fill it.
- Write each retrieval query the way a creator would say it out loud, and make it
  specific enough that it would not pull back the same passages as the facets
  already listed.
- Order them by how much a reviewer would miss it.
- Do not name a video, a creator or an example.

Return only JSON in this shape:
{{"gaps":[{{"missing":"...","why":"...","probe":"..."}}]}}"""


class CoverageGapCritic:
    """The gap critic, plus a map of where its own probes have not read.

    It reads the criteria, the plan's facets and the coverage table, and nothing
    else. Still no evidence, still no transcript, and above all still nothing
    from the held-out expert — the coverage table is counts and the video titles
    the planner was already given, so the critic learns where the loop has been,
    never what is there.
    """

    def __init__(self, llm: ChatModel, model_name: str = "") -> None:
        self.llm = llm
        self.model_name = model_name

    def critique(
        self,
        declaration: PackDeclaration,
        rubrics: Sequence[Rubric],
        probes: Sequence[ResearchProbe],
        coverage: Sequence[CoverageProfile],
        *,
        count: int,
    ) -> list[dict[str, Any]]:
        from langchain_core.messages import HumanMessage, SystemMessage

        prompt = COVERAGE_GAP_SYSTEM_PROMPT.format(artifact=declaration.artifact, count=count)
        body = (
            f"PACK: {declaration.name}\nARTIFACT: {declaration.artifact}\n"
            f"TOPIC: {declaration.routing_text}\n\n"
            "FACETS THE PLAN COVERED\n"
            + "\n".join(f"- {probe.facet}" for probe in probes)
            + "\n\nCRITERIA IN THE DRAFT PACK\n"
            + "\n".join(
                f"{rubric.rubric_id}: {rubric.criterion}"
                + (f" — check: {rubric.check}" if rubric.check else "")
                for rubric in rubrics
            )
            + "\n\nHOW MUCH OF EACH SOURCE THE DRAFT READ\n"
            + "\n".join(
                f"- {row.unread} of {row.chunks} passages still unread — {row.title}"
                for row in coverage
            )
        )
        response = self.llm.invoke([SystemMessage(content=prompt), HumanMessage(content=body)])
        payload = _json_object(str(getattr(response, "content", response) or ""))
        rows = payload.get("gaps")
        return [row for row in (rows or []) if isinstance(row, dict)]


def admit_novel_evidence(
    prior: Sequence[Rubric],
    candidates: Sequence[Rubric],
) -> tuple[list[Rubric], list[dict[str, Any]]]:
    """Admit a candidate only if it rests on a passage no admitted rule rests on.

    Returns ``(admitted, refused)``. A refusal row carries the chunks the rule
    cited and the rules that were already resting on them, so a reader can see
    the refusal was an identity check and not a judgement about wording.

    The rule is the complement of :func:`~src.rag.packs.dedupe_rubrics`, which
    compares criterion text at a cosine threshold and therefore has a cliff and
    a shadow under it. This has neither: a rule either quotes a transcript
    second nothing admitted quotes, or it does not. It is applied to the new
    round's candidates only, exactly as the cosine dedupe effectively is — the
    inherited round is passed first and is never the thing dropped.

    Note what it does **not** do: it does not simulate the scorer. A rule that
    brings one new chunk is admitted even if its other citations collide with an
    existing rule's only evidence, which the scorer's exclusivity gate would
    punish. Mirroring the gate any more closely would make this a scoring
    strategy rather than a build rule.
    """
    used: dict[str, list[str]] = {}
    for rubric in prior:
        for item in rubric.evidence:
            used.setdefault(item.chunk_id, []).append(rubric.rubric_id)
    admitted: list[Rubric] = []
    refused: list[dict[str, Any]] = []
    for rubric in candidates:
        chunk_ids = [item.chunk_id for item in rubric.evidence]
        fresh = [chunk_id for chunk_id in chunk_ids if chunk_id not in used]
        if fresh:
            admitted.append(rubric)
        else:
            refused.append(
                {
                    "rubric_id": rubric.rubric_id,
                    "criterion": rubric.criterion,
                    "unit_id": rubric.unit_id,
                    "chunk_ids": sorted(dict.fromkeys(chunk_ids)),
                    "already_rested_on_by": sorted(
                        {name for chunk_id in chunk_ids for name in used.get(chunk_id, ())}
                    ),
                }
            )
            continue
        for chunk_id in chunk_ids:
            used.setdefault(chunk_id, []).append(rubric.rubric_id)
    return admitted, refused


def rediscovery_rate(
    shipped: ExpertPack | None,
    added: Sequence[Rubric],
) -> dict[str, Any]:
    """How many of these rules cite a chunk the shipped pack already cites.

    The same identity check :func:`_mark_rediscoveries` renders per row, reduced
    to the one number the V8 gate turns on. Four of ten is what ``deep-r2`` and
    the one-shot control both scored; an arm that does not move it has not moved
    the finding, whatever its scores do.
    """
    if shipped is None:
        return {"added": len(added), "rediscovered": None, "rate": None, "rows": []}
    by_chunk: dict[str, list[str]] = {}
    for rubric in shipped.rubrics:
        for item in rubric.evidence:
            by_chunk.setdefault(item.chunk_id, []).append(rubric.rubric_id)
    rows: list[dict[str, Any]] = []
    for rubric in added:
        overlap = sorted(
            {name for item in rubric.evidence for name in by_chunk.get(item.chunk_id, ())}
        )
        rows.append(
            {
                "rubric_id": rubric.rubric_id,
                "criterion": rubric.criterion,
                "already_in_shipped": overlap,
            }
        )
    hit = sum(1 for row in rows if row["already_in_shipped"])
    return {
        "shipped_arm": shipped.arm,
        "added": len(added),
        "rediscovered": hit,
        "rate": round(hit / len(added), 4) if added else None,
        "rows": rows,
    }


def corpus_as_built(
    records: Sequence[dict[str, Any]],
    digest: str,
) -> list[dict[str, Any]]:
    """The subset of the live corpus that reproduces a pack's build digest.

    Ingestion does not stop for an experiment: the corpus these arms were built
    from is 1372 chunks over 56 videos, and the store now holds more. A new arm
    built on the larger corpus would not be comparable with the hand-built pack
    it has to beat, and the comparison is the whole point — so the new arm is
    built against the old corpus, and *proved* to be, rather than assumed.

    Videos are added in the order the store returns them, which is ingestion
    order, and each prefix is fingerprinted with the same
    :func:`~src.rag.packs.corpus_digest` a pack records. The answer is either a
    subset whose digest matches exactly — a cryptographic check, not a
    heuristic — or a ``ValueError``. There is no "close enough" branch: a
    silently wrong corpus is the failure mode this exists to make impossible.
    """
    from src.rag.packs import corpus_digest

    order: list[str] = []
    seen: set[str] = set()
    for record in records:
        video_id = str(record.get("video_id"))
        if video_id not in seen:
            seen.add(video_id)
            order.append(video_id)
    for size in range(1, len(order) + 1):
        allowed = set(order[:size])
        subset = [record for record in records if str(record.get("video_id")) in allowed]
        if corpus_digest(subset) == digest:
            return subset
    raise ValueError(
        f"no prefix of the live corpus reproduces build digest {digest!r} — "
        "the corpus has changed underneath these arms in a way that is not "
        "pure growth, and nothing built now would be comparable with them"
    )


def restrict_workspace(
    workspace: PackWorkspace, records: Sequence[dict[str, Any]]
) -> PackWorkspace:
    """The same workspace over a narrower corpus.

    Everything expensive — the embedding model, the LLM client, the theme index,
    the graph extractions and the on-disk call cache — is shared by reference, so
    this costs a dictionary rebuild. What changes is the only thing that must:
    the chunk records, and therefore ``by_chunk``, the video catalogue and the
    provenance digest.
    """
    narrowed = PackWorkspace(
        workspace.settings,
        workspace.catalog,
        embed=workspace.embed,
        extractor=workspace.extractor,
        records=records,
        themes=workspace.themes,
        theme_generated_at=workspace.theme_generated_at,
        extractions=workspace.extractions,
        route=workspace.route,
        cache_dir=workspace.cache_dir,
    )
    return narrowed


def build_admission_arm(
    workspace: PackWorkspace,
    declaration: PackDeclaration,
    store: PackStore,
    pack_r1: ExpertPack,
    pack_r2: ExpertPack,
) -> tuple[ExpertPack, list[dict[str, Any]]]:
    """``deep-r2`` with the evidence-novelty rule and nothing else changed.

    No probe is re-run and no call is spent: the rules are the ones ``deep-r2``
    already published, filtered. It exists so the frontier arm's numbers can be
    read against the rule on its own — if the whole of a gain is here, then the
    critic and the frontier retrieval bought nothing and the honest report says
    so.
    """
    inherited = {rubric.rubric_id for rubric in pack_r1.rubrics}
    added = [rubric for rubric in pack_r2.rubrics if rubric.rubric_id not in inherited]
    admitted, refused = admit_novel_evidence(pack_r1.rubrics, added)
    outcomes = [
        ProbeOutcome(
            probe=ResearchProbe(probe_id=unit.unit_id, facet=unit.title, question=unit.summary),
            unit_id=unit.unit_id,
            chunks=len(unit.chunk_ids),
            videos=len(unit.video_ids),
            creator_count=unit.creator_count,
            top_creator=unit.top_creator,
            top_creator_share=unit.top_creator_share,
            spent_call=False,
        )
        for unit in pack_r2.units
    ]
    pack, _dropped = publish(
        workspace,
        declaration,
        ADMISSION_ARM,
        members=pack_r2.members,
        units=pack_r2.units,
        candidates=[*pack_r1.rubrics, *admitted],
        outcomes=outcomes,
        notes=[
            "deep-r2 with one change: a round-two rule is admitted only if it rests on a "
            "passage no admitted rule rests on. No probe was re-run and no call was spent."
        ],
        reason_by_unit={
            str(row["unit_id"]): "every rule it produced rested only on passages an admitted "
            "rule already rested on"
            for row in refused
        },
    )
    # The published pack still spent deep-r2's calls; only the admission changed.
    pack.stats["executor_calls"] = pack_r2.stats.get("executor_calls")
    pack.stats["rubrics_refused_no_new_evidence"] = len(refused)
    store.save_pack(pack)
    return pack, refused


def run_frontier_round(
    settings: Settings,
    *,
    topic: str,
    packs_dir: Path = Path("experts"),
    gap_probes_count: int = DEFAULT_GAP_PROBES,
    pool: int = DEFAULT_RETRIEVAL_POOL,
    cache_dir: Path = DEFAULT_RESEARCH_CACHE_DIR,
    records: Sequence[dict[str, Any]] | None = None,
    on_progress: ProgressFn | None = None,
) -> dict[str, Any]:
    """Build ``deep-frontier`` and ``deep-r2-admit`` beside the existing arms.

    Round one is not rebuilt. It is read back off the committed ``deep-r1``
    pack, which is what makes this arm nested inside the same plan as
    ``deep-r2`` and the one-shot control rather than a fourth independent draw:
    the six opening probes, their retrievals, their rules and the pack's
    membership are literally the same objects. The only spend is one critic call
    and this round's executor calls, which is exactly what ``deep-r2`` spent.
    """
    from langchain_openai import ChatOpenAI

    from src.agents.llm import chat_model_kwargs

    progress = on_progress or (lambda _message: None)
    store = PackStore(packs_dir)
    catalog = store.catalog()
    declaration = catalog.declaration(topic)
    if declaration is None:
        raise ValueError(f"No declared pack for topic {topic!r}")

    pack_r1 = store.load_pack(topic, "deep-r1")
    pack_r2 = store.load_pack(topic, "deep-r2")
    if pack_r1 is None or pack_r2 is None:
        raise ValueError(f"{topic!r} has no deep-r1/deep-r2 on disk. Run deep-research first.")

    if records is None:
        from src.api.corpus import load_chunk_embeddings

        records = load_chunk_embeddings(settings.chroma_path, settings.chunk_collection)
    build_digest = pack_r1.provenance.corpus_digest
    as_built = corpus_as_built(records, build_digest)
    vectors: dict[str, Sequence[float]] = {
        str(record["chunk_id"]): list(record["embedding"])
        for record in as_built
        if record.get("embedding") is not None
    }
    plain = [{key: value for key, value in row.items() if key != "embedding"} for row in as_built]
    progress(
        f"[{topic}] built-corpus digest {build_digest} reproduced — "
        f"{len(plain)} chunks, {len({str(r['video_id']) for r in plain})} videos, "
        f"{len(vectors)} vectors"
    )

    workspace = restrict_workspace(open_workspace(settings, catalog, with_model=True), plain)
    provenance = workspace.provenance()
    if provenance.corpus_digest != build_digest:
        raise ValueError(
            f"workspace corpus {provenance.corpus_digest} is not the build corpus {build_digest}"
        )

    members = list(pack_r1.members)
    allowed = included_video_ids(members)
    round_one_retrieved = sorted({cid for unit in pack_r1.units for cid in unit.chunk_ids})
    round_one_probes = [
        ResearchProbe(
            probe_id=unit.unit_id.split(":")[-1],
            facet=unit.title,
            question=unit.summary,
            rank=index,
        )
        for index, unit in enumerate(pack_r1.units, start=1)
    ]
    coverage = coverage_profile(plain, allowed, round_one_retrieved)
    progress(
        f"[{topic}] round 1 read {len(round_one_retrieved)} of "
        f"{sum(row.chunks for row in coverage)} member passages"
    )
    for row in coverage:
        progress(f"  {row.read}/{row.chunks} read — {row.title}")

    llm = ChatOpenAI(**chat_model_kwargs(settings))
    critic = CoverageGapCritic(llm, settings.deepseek_model)
    material = json.dumps(
        {
            "version": FRONTIER_PROMPT_VERSION,
            "topic": topic,
            "artifact": declaration.artifact,
            "routing_text": declaration.routing_text,
            "facets": [probe.facet for probe in round_one_probes],
            "criteria": [[r.rubric_id, r.criterion, r.check] for r in pack_r1.rubrics],
            "coverage": [row.model_dump(mode="json") for row in coverage],
            "count": gap_probes_count,
            "model": critic.model_name,
        },
        sort_keys=True,
    )
    rows, cached = _cached_json(
        cache_dir,
        "gaps-coverage",
        material,
        lambda: critic.critique(
            declaration, pack_r1.rubrics, round_one_probes, coverage, count=gap_probes_count
        ),
    )
    gaps = parse_gaps(rows, gap_probes_count)
    progress(
        f"[{topic}] coverage-aware critic named {len(gaps)} gaps" + (" (cached)" if cached else "")
    )
    for gap in gaps:
        progress(f"  {gap.gap_id} MISSING: {gap.missing}")
        progress(f"       probe: {gap.probe}")

    probes = gap_probes(gaps, start=FRONTIER_PROBE_OFFSET)
    units, candidates, outcomes = run_probes(
        workspace,
        declaration,
        probes,
        vectors,
        allowed,
        pool=pool,
        exclude_chunk_ids=round_one_retrieved,
        on_progress=progress,
    )
    admitted, refused = admit_novel_evidence(pack_r1.rubrics, candidates)
    for row in refused:
        progress(f"  refused {row['rubric_id']} — no passage of its own ({row['chunk_ids']})")
    pack, dropped = publish(
        workspace,
        declaration,
        FRONTIER_ARM,
        members=members,
        units=[*pack_r1.units, *units],
        candidates=[*pack_r1.rubrics, *admitted],
        outcomes=[
            *[_inherited_outcome(unit) for unit in pack_r1.units],
            *outcomes,
        ],
        notes=[
            "deep-research frontier round: round 1 plus "
            f"{len(probes)} probes a coverage-aware gap critic asked for, retrieved outside "
            "the ground round 1 already read, and admitted only when they rested on a "
            "passage no admitted rule rested on"
        ],
        reason_by_unit={
            str(row["unit_id"]): "every rule it produced rested only on passages an admitted "
            "rule already rested on"
            for row in refused
        },
    )
    pack.stats["executor_calls"] = int(pack_r1.stats.get("executor_calls") or 0) + sum(
        1 for outcome in outcomes if outcome.spent_call
    )
    pack.stats["rubrics_refused_no_new_evidence"] = len(refused)
    store.save_pack(pack)
    progress(f"[{topic}] {FRONTIER_ARM} {pack.stats['rubrics']} rubrics")

    pack_admit, refused_admit = build_admission_arm(workspace, declaration, store, pack_r1, pack_r2)
    progress(
        f"[{topic}] {ADMISSION_ARM} {pack_admit.stats['rubrics']} rubrics "
        f"({len(refused_admit)} refused)"
    )

    shipped = _load_arm(store, topic, None)
    pack_os = store.load_pack(topic, "deep-oneshot")
    inherited = {rubric.rubric_id for rubric in pack_r1.rubrics}
    added = [rubric for rubric in pack.rubrics if rubric.rubric_id not in inherited]
    added_admit = [rubric for rubric in pack_admit.rubrics if rubric.rubric_id not in inherited]
    added_r2 = [rubric for rubric in pack_r2.rubrics if rubric.rubric_id not in inherited]
    added_os = [
        rubric
        for rubric in (pack_os.rubrics if pack_os else [])
        if rubric.rubric_id not in inherited
    ]

    block: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "arm": FRONTIER_ARM,
        "admission_arm": ADMISSION_ARM,
        "model": settings.deepseek_model,
        "corpus_digest": provenance.corpus_digest,
        "chunk_count": provenance.chunk_count,
        "video_count": provenance.video_count,
        "member_videos": len(allowed),
        "member_chunks": sum(row.chunks for row in coverage),
        "round_one_read": len(round_one_retrieved),
        "settings": {
            "gap_probes": gap_probes_count,
            "retrieval_pool": pool,
            "excluded_chunks": len(round_one_retrieved),
            "prompt_version": FRONTIER_PROMPT_VERSION,
        },
        "coverage": [row.model_dump(mode="json") for row in coverage],
        "gaps": [gap.model_dump(mode="json") for gap in gaps],
        "probes": [outcome.model_dump(mode="json") for outcome in outcomes],
        "probe_overlap": [
            {
                "unit_id": unit.unit_id,
                "chunks": len(unit.chunk_ids),
                "from_round_one_ground": len(
                    [cid for cid in unit.chunk_ids if cid in set(round_one_retrieved)]
                ),
            }
            for unit in units
        ],
        "refused": refused,
        # The fairness check the admission rule needs, and the one a reader will
        # reach for: the new arms get a rule the hand-built pack was never put
        # through, so does the hand-built pack have anything to lose to it? On
        # resume-design the answer is no — every one of merged's rules already
        # rests on a passage of its own, so the rule refuses nothing there and
        # the precision gap between the arms is not an artefact of applying it
        # to one side. (It is not free in general: the same rule would refuse
        # three of raptor's eighteen and one of communities' eighteen.)
        "admission_on_shipped": _admission_counterfactual(shipped),
        "deduped_against_previous": [rubric.rubric_id for rubric in dropped],
        "diff": rubric_diff(pack_r1, pack),
        # All four arms that add to round one, not just the new one. The number
        # to beat is four of ten, and the one-shot control's identical four of
        # ten is what says the loop was not searching better than no loop — an
        # arm reporting its own rate without those two beside it is unreadable.
        "rediscovery": {
            FRONTIER_ARM: rediscovery_rate(shipped, added),
            ADMISSION_ARM: rediscovery_rate(shipped, added_admit),
            "deep-r2": rediscovery_rate(shipped, added_r2),
            "deep-oneshot": rediscovery_rate(shipped, added_os),
        },
        "gap_closure": gap_closure(
            gaps,
            pack_r1.rubrics,
            added,
            {outcome.unit_id: outcome.probe.origin for outcome in outcomes},
            workspace.embed,
        ),
        "restatements": restatement_audit(pack_r1.rubrics, added, workspace.embed),
        "budget": [
            {
                "arm": FRONTIER_ARM,
                "label": "frontier round — a coverage-aware critic, off round 1's ground",
                "planner_calls": 1,
                "critic_calls": 1,
                "executor_calls": int(pack.stats.get("executor_calls") or 0),
                "probes_budgeted": len(pack.units),
                "total_llm_calls": 2 + int(pack.stats.get("executor_calls") or 0),
                "rubrics": len(pack.rubrics),
                "citations": int(pack.stats.get("citations") or 0),
            },
            {
                "arm": ADMISSION_ARM,
                "label": "ablation — deep-r2, evidence-novelty admission only, no new call",
                "planner_calls": 1,
                "critic_calls": 1,
                "executor_calls": int(pack_admit.stats.get("executor_calls") or 0),
                "probes_budgeted": len(pack_admit.units),
                "total_llm_calls": 2 + int(pack_admit.stats.get("executor_calls") or 0),
                "rubrics": len(pack_admit.rubrics),
                "citations": int(pack_admit.stats.get("citations") or 0),
            },
        ],
    }

    report = load_report(store, topic) or {}
    report["frontier"] = block
    # The two new rows join the budget table the panel already renders rather
    # than starting a second one: an arm that is not beside the arms it claims
    # to beat is an arm a reader has to go and find.
    existing = [
        row
        for row in (report.get("budget") or [])
        if row.get("arm") not in {FRONTIER_ARM, ADMISSION_ARM}
    ]
    report["budget"] = [*existing, *block["budget"]]
    save_report(store, topic, report)
    return report


def _admission_counterfactual(shipped: ExpertPack | None) -> dict[str, Any]:
    """What the evidence-novelty rule would refuse in the pack being beaten.

    Applied from that pack's first rule onward rather than against an inherited
    round, because the hand build has no rounds — the question is whether every
    rule it ships rests on a passage of its own, which is the property the new
    arms are being given by construction.
    """
    if shipped is None:
        return {"arm": None, "rubrics": 0, "would_refuse": None, "refused_ids": []}
    _kept, refused = admit_novel_evidence([], shipped.rubrics)
    return {
        "arm": shipped.arm,
        "rubrics": len(shipped.rubrics),
        "would_refuse": len(refused),
        "refused_ids": [str(row["rubric_id"]) for row in refused],
    }


def _inherited_outcome(unit: SourceUnit) -> ProbeOutcome:
    """Round one's probe as an outcome row, with no call charged to this round.

    ``publish`` walks units and outcomes in step to build the gap log, so the
    inherited units need a row each. ``spent_call`` is false because this round
    did not spend them — the executor budget is restored from ``deep-r1``'s own
    statistics afterwards, so the arm is charged for round one exactly once.
    """
    return ProbeOutcome(
        probe=ResearchProbe(
            probe_id=unit.unit_id.split(":")[-1], facet=unit.title, question=unit.summary
        ),
        unit_id=unit.unit_id,
        chunks=len(unit.chunk_ids),
        videos=len(unit.video_ids),
        creator_count=unit.creator_count,
        top_creator=unit.top_creator,
        top_creator_share=unit.top_creator_share,
        spent_call=False,
    )

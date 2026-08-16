"""D2: scoring a rubric pack's three arms on the held-out critique harness.

:mod:`src.evals.critique` measures whether a system reached a named expert's
conclusions without having been allowed to read that expert. A rubric pack is
exactly a list of conclusions with citations, so it goes through that scorer
unchanged — the pack's rubrics *are* the findings, and no second instrument has
to be built or defended.

What this module adds is only the plumbing: turn each arm's pack into findings,
score all three against the same held-out criteria with the same matcher, and
write one comparison run. Nothing here re-implements a metric.

Two properties worth stating, because they are what make the comparison mean
anything:

* **The pack never saw the held-out video.** It is in
  ``experts/packs.json``'s ``held_out_video_ids``, so it is stripped from the
  summary index every pack's membership is routed through — the same mechanism
  the D5 property exclusion uses. The proof is not the mechanism:
  :func:`~src.evals.critique.held_out_leaks` re-scans every chunk id the build
  fed the model and every citation the pack shipped, by prefix, and the count
  goes in the run.
* **A bigger arm cannot win by being bigger.** ``criteria_recall`` counts only
  matches to findings holding *exclusive* resolving evidence, and
  ``evidence_precision`` is grounded-over-total, so an arm that pads its pack
  with restatements of one chunk loses on both. That property belongs to the
  V3 harness; this module inherits it by not going around it.

The scorer's own noise is reported beside the score. Its matcher does not agree
with itself across repeats, so each arm carries the ``score_spread`` its five
votes produced, and an arm that leads by less than that range has not won.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

from src.config import Settings
from src.evals.critique import (
    CRITIQUE_METRICS,
    ChunkTextFn,
    CritiqueDataset,
    MatchFn,
    SetupCritique,
    build_run,
    load_critique_dataset,
    score_critique,
)
from src.rag.packs import PACK_ARMS, ExpertPack, PackStore, rubrics_as_findings

ProgressFn = Callable[[str], None]


def chunk_text_from_records(
    records: Sequence[dict[str, Any]],
    *,
    tolerance: float = 20.0,
) -> ChunkTextFn:
    """A provenance resolver over already-loaded chunk records.

    Behaviourally the same lookup as
    :func:`src.evals.critique_run.chunk_text_lookup` — every chunk whose span
    covers the timestamp, within the same tolerance — but built from rows this
    process already has rather than from a store that instantiates the
    sentence-transformers model. That matters on a host where loading the model
    is what makes a run take forty minutes; it changes nothing about the answer,
    and the numbers this module reports are re-derived against the store-backed
    resolver before they are believed.
    """
    spans: dict[str, list[tuple[float, float, str, str]]] = {}
    for record in records:
        start = float(record.get("start_seconds") or 0.0)
        end = float(record.get("end_seconds") or start)
        spans.setdefault(str(record["video_id"]), []).append(
            (start, end, str(record["chunk_id"]), str(record.get("text") or ""))
        )

    def lookup(video_id: str, seconds: float) -> list[tuple[str, str]]:
        return [
            (chunk_id, text)
            for start, end, chunk_id, text in spans.get(video_id, [])
            if start - tolerance <= seconds <= end + tolerance
        ]

    return lookup


def pack_as_critique(pack: ExpertPack) -> SetupCritique:
    """One arm's pack as the harness's unit of work.

    ``retrieved_chunk_ids`` is every chunk the build actually showed the model,
    not just the ones that ended up cited — the leak check has to see what the
    system was *exposed* to, since a held-out video that reached the prompt and
    was merely not quoted has still contaminated the experiment.

    Each finding also carries the narrower thing the grounding gate needs: the
    chunks of the one unit *its own* rubric was distilled from
    (:func:`~src.evals.critique_run.pack_finding_provenance`). That is what lets
    these arms be graded under
    :data:`~src.evals.critique.GATE_PROVENANCE` at all — a rubric quoting a
    chunk from somebody else's unit is not evidence it retrieved.
    """
    from src.evals.critique import attach_provenance
    from src.evals.critique_run import pack_finding_provenance

    exposed = sorted({chunk_id for unit in pack.units for chunk_id in unit.chunk_ids})
    return SetupCritique(
        setup=pack.arm,
        findings=attach_provenance(rubrics_as_findings(pack), pack_finding_provenance([pack])),
        retrieved_chunk_ids=exposed,
        retrieved_video_ids=sorted(
            {unit_video for unit in pack.units for unit_video in unit.video_ids}
        ),
        answer="",
    )


def score_arms(
    packs: Sequence[ExpertPack],
    dataset: CritiqueDataset,
    match: MatchFn,
    chunk_text: ChunkTextFn,
    *,
    config: dict[str, Any] | None = None,
    on_progress: ProgressFn | None = None,
) -> dict[str, Any]:
    """Score each arm against the held-out criteria and assemble one run."""
    progress = on_progress or (lambda _message: None)
    cells: list[dict[str, Any]] = []
    for pack in packs:
        progress(f"[{pack.topic}/{pack.arm}] scoring {len(pack.rubrics)} rubrics")
        cell = score_critique(pack_as_critique(pack), dataset, match, chunk_text)
        cell["rubrics"] = len(pack.rubrics)
        cell["multi_creator_share"] = pack.stats.get("multi_creator_share")
        cell["unit_budget"] = pack.stats.get("unit_budget")
        cell["units_by_kind"] = pack.stats.get("units_by_kind")
        cells.append(cell)
    run = build_run(
        dataset,
        cells,
        config={
            "harness": "pack-d2",
            "arms": [pack.arm for pack in packs],
            **(config or {}),
        },
        baseline=packs[0].arm if packs else PACK_ARMS[0],
    )
    run["kind"] = "pack-ablation"
    run["topic"] = packs[0].topic if packs else ""
    return run


def winner(run: dict[str, Any], metric: str = "criteria_recall") -> dict[str, Any]:
    """Which arm leads on ``metric``, and whether the lead survives the noise.

    The scorer's matcher disagrees with itself; every cell therefore carries the
    range its own repeats produced. ``decisive`` is true only when the leader's
    score clears the runner-up's *maximum*, which is the bar the V3 fix set for
    any later slice. When it is false the honest report is "these are the same",
    and the decided fallback applies: ship the winner and say so.

    Three outcomes, not two. Only ``criteria_recall`` is scored repeatedly, so
    only it has a ``*_max`` to clear; every other metric is a single draw. This
    used to read the missing key as ``None`` and fall through to ``decisive``,
    which meant ``evidence_precision`` and ``provenance`` always announced
    "leader clears the runner-up's own spread" — a comparison that had not been
    performed. A lead with no repeats behind it is not decisive and not inside
    the noise either; both of those claim knowledge of a range nobody measured.
    So ``basis`` names which of the three situations produced the verdict, and
    ``decisive`` is now reserved for the one case that was actually checked.
    """
    scored = [
        (cell["setup"], cell.get("scores", {}).get(metric), cell.get("score_spread") or {})
        for cell in run.get("cells", [])
    ]
    ranked = sorted(
        [row for row in scored if isinstance(row[1], (int, float))],
        key=lambda row: (-float(row[1]), row[0]),
    )
    if not ranked:
        return {
            "metric": metric,
            "leader": None,
            "decisive": False,
            "basis": "no-scored-arm",
            "reason": "no scored arm",
        }
    leader, best, _ = ranked[0]
    if len(ranked) == 1:
        return {
            "metric": metric,
            "leader": leader,
            "value": best,
            "decisive": False,
            "basis": "single-arm",
            "reason": "only one arm scored — nothing to compare against",
        }
    tied = [name for name, value, _ in ranked if value == best]
    runner_max = max(
        (
            row[2].get(f"{metric}_max")
            for row in ranked[1:]
            if isinstance(row[2].get(f"{metric}_max"), (int, float))
        ),
        default=None,
    )
    if len(tied) > 1:
        basis = "tied"
        reason = "two or more arms scored the same"
    elif runner_max is None:
        # No repeats were run for this metric, so there is no range to clear and
        # no range to sit inside. Say that, rather than borrowing either verdict.
        basis = "unrepeated"
        reason = "scored once per arm — no repeat spread, so the gap's reliability is unmeasured"
    elif float(best) > float(runner_max):
        basis = "cleared-spread"
        reason = "leader clears the runner-up's own spread"
    else:
        basis = "inside-spread"
        reason = "the lead sits inside the scorer's own range across repeats"
    return {
        "metric": metric,
        "leader": leader,
        "value": best,
        "tied": tied,
        "runner_up_max": runner_max,
        "decisive": basis == "cleared-spread",
        "basis": basis,
        "reason": reason,
    }


#: Metrics scored once per arm because nothing in their path can disagree.
#:
#: ``evidence_precision`` is quote resolution, set exclusivity and the unit each
#: rubric was distilled from; ``provenance`` is quote resolution alone. Neither
#: has a model in it, so there is no spread to clear and no noise to sit inside.
#: Filing them under "unrepeated" would put a deterministic number under the
#: same doubt as a judged one, which is why :func:`against_baseline` names them
#: rather than letting the missing ``*_max`` decide.
DETERMINISTIC_METRICS: tuple[str, ...] = ("evidence_precision", "provenance")


def against_baseline(run: dict[str, Any], metric: str = "criteria_recall") -> list[dict[str, Any]]:
    """Every arm against the run's **baseline**, not against the runner-up.

    :func:`winner` answers "which arm leads, and does the lead survive the
    scorer's noise". That is the right question for an ablation and the wrong
    one for a gate phrased as *"does the loop-built pack reach the hand-built
    one"*: when two loop arms tie for the lead, ``winner`` reports ``tied`` and
    says nothing whatever about the pack they were built to beat.

    So this states that pair directly, one row per non-baseline arm, under the
    same rule the V3 fix set — a lead smaller than the ranges under it is not a
    lead. The form is the most conservative one available: the arm's *minimum*
    across repeats against the baseline's *maximum*. Two ranges that touch count
    as overlapping.

    It lives here, beside ``winner``, rather than in the panel that renders it,
    because a finding that exists only at render time is absent from the run
    file, and the run file is what a reader outside the browser has.
    """
    cells = {str(cell.get("setup") or ""): cell for cell in run.get("cells") or []}
    base_name = str(run.get("baseline") or "")
    base = cells.get(base_name)
    if base is None:
        return []
    base_value = (base.get("scores") or {}).get(metric)
    base_max = (base.get("score_spread") or {}).get(f"{metric}_max")
    rows: list[dict[str, Any]] = []
    for name, cell in cells.items():
        if name == base_name:
            continue
        value = (cell.get("scores") or {}).get(metric)
        low = (cell.get("score_spread") or {}).get(f"{metric}_min")
        if not isinstance(value, (int, float)) or not isinstance(base_value, (int, float)):
            rows.append(
                {
                    "arm": name,
                    "baseline": base_name,
                    "metric": metric,
                    "value": value,
                    "baseline_value": base_value,
                    "delta": None,
                    "beats_baseline": None,
                    "basis": "ungraded",
                    "reason": "one of the two cells is not graded under this run's gate",
                }
            )
            continue
        delta = round(float(value) - float(base_value), 4)
        if metric in DETERMINISTIC_METRICS:
            basis = "deterministic"
            beats = delta > 0
            reason = (
                "scored once per arm because there is no model in this metric's path — "
                "the gap is the whole of the comparison"
            )
        elif isinstance(low, (int, float)) and isinstance(base_max, (int, float)):
            beats = float(low) > float(base_max)
            basis = "ranges-disjoint" if beats else "ranges-overlap"
            reason = (
                f"this arm's worst of the run's repeats ({low}) sits above the baseline's "
                f"best ({base_max})"
                if beats
                else "the two repeat ranges cross, so neither arm is credited with a lead"
            )
        else:
            beats = None
            basis = "unrepeated"
            reason = "no repeat spread on one side, so the gap's reliability is unmeasured"
        rows.append(
            {
                "arm": name,
                "baseline": base_name,
                "metric": metric,
                "value": value,
                "baseline_value": base_value,
                "arm_min": low,
                "baseline_max": base_max,
                "delta": delta,
                "beats_baseline": beats,
                "basis": basis,
                "reason": reason,
            }
        )
    return sorted(rows, key=lambda row: (-(row["delta"] or 0.0), row["arm"]))


def baseline_table(run: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """:func:`against_baseline` for every metric the harness scores."""
    return {metric: against_baseline(run, metric) for metric in CRITIQUE_METRICS}


def run_pack_ablation(
    settings: Settings,
    *,
    topic: str,
    packs_dir: Path = Path("experts"),
    arms: Sequence[str] = PACK_ARMS,
    repeats: int = 5,
    records: Sequence[dict[str, Any]] | None = None,
    on_progress: ProgressFn | None = None,
) -> dict[str, Any]:
    """Load the three arms of one pack and score them on the held-out harness."""
    from src.evals.critique import repeated_matcher
    from src.evals.critique_run import MATCHER_VERSION, cached_matcher, llm_matcher

    store = PackStore(packs_dir)
    loaded = [pack for pack in (store.load_pack(topic, arm) for arm in arms) if pack is not None]
    if not loaded:
        raise ValueError(f"No built arms for pack {topic!r}. Run build-packs first.")

    if records is None:
        from src.api.corpus import load_chunk_embeddings

        records = load_chunk_embeddings(settings.chroma_path, settings.chunk_collection)
    chunk_text = chunk_text_from_records(records)
    dataset = load_critique_dataset()
    match = cached_matcher(
        repeated_matcher(llm_matcher(settings), repeats=repeats), repeats=repeats
    )
    run = score_arms(
        loaded,
        dataset,
        match,
        chunk_text,
        config={
            "rubric_model": loaded[0].provenance.rubric_model,
            "matcher": "llm",
            "matcher_model": settings.deepseek_model,
            "matcher_version": MATCHER_VERSION,
            "match_repeats": repeats,
            "corpus_digest": loaded[0].provenance.corpus_digest,
            "excluded_video_ids": loaded[0].provenance.excluded_video_ids,
            "held_out_video_ids": loaded[0].provenance.held_out_video_ids,
            "unit_budget": loaded[0].stats.get("unit_budget"),
            "resolver": "chunk_text_from_records",
        },
        on_progress=on_progress,
    )
    run["verdicts"] = {metric: winner(run, metric) for metric in CRITIQUE_METRICS}
    run["against_baseline"] = baseline_table(run)
    run["generated_at"] = datetime.now(timezone.utc).isoformat()
    return run

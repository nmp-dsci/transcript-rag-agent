"""Read the built rubric packs under ``experts/`` for the Experiments tab.

Everything this module serves is already on disk. A pack is a JSON file
:mod:`src.rag.pack_build` wrote, and the point of that choice is that the app
renders exactly what a reviewer can open in the repo and diff — so this module
loads, reshapes for the browser, and adds nothing a rebuild would not reproduce.

Two things are computed here rather than read:

* **The deep link for every quote.** :meth:`~src.rag.packs.RubricEvidence.youtube_url`
  builds it from the video id and the interpolated quote offset. It is not built
  from ``source_url``: eight of the corpus's stored urls already carry a ``t=``
  parameter, and appending a second one gives a url with two, of which YouTube
  honours the first — the link then lands about twenty seconds before the
  sentence it is supposed to prove.
* **The gate numbers, per pack.** ``multi_creator_share`` counts distinct
  *creators*, not distinct video ids, because the corpus audit showed a theme
  can span four videos and still be one podcast talking. The pack build already
  stores that count; this module re-derives the quote-resolution rate beside it
  so the header carries both halves of the claim the panel makes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.evals.pack_ablation import winner
from src.rag.packs import ExpertPack, PackStore

#: ``experts/`` at the repo root, independent of the server's working directory.
DEFAULT_PACKS_DIR = Path(__file__).resolve().parents[2] / "experts"

#: A quote counts as resolved when it snapped onto the transcript essentially
#: intact. ``reconcile_evidence`` already drops anything below its own floor, so
#: this is a display threshold and a second, visible line of defence rather than
#: the only check.
RESOLVED_RATIO = 0.9


def _store(packs_dir: Path | None) -> PackStore:
    return PackStore(packs_dir or DEFAULT_PACKS_DIR)


def _load(store: PackStore, topic: str, arm: str | None = None) -> ExpertPack | None:
    """One pack, or ``None`` if it is absent *or* unreadable.

    A half-written or hand-edited pack file must not take the whole tab down
    with a 500 — the same rule the committed-run reader follows. It reads as
    "not built", which is what a file the app cannot parse effectively is.
    """
    try:
        return store.load_pack(topic, arm)
    except (OSError, ValueError):
        return None


def _evidence_row(evidence: Any) -> dict[str, Any]:
    """One quote as the browser needs it: text, creator, timestamp, deep link."""
    return {
        "video_id": evidence.video_id,
        "chunk_id": evidence.chunk_id,
        "quote": evidence.quote,
        "model_quote": evidence.model_quote,
        "channel_name": evidence.channel_name,
        "title": evidence.title,
        "start_seconds": evidence.start_seconds,
        "quote_start_seconds": evidence.quote_start_seconds,
        "ratio": evidence.ratio,
        "resolved": evidence.ratio >= RESOLVED_RATIO,
        "url": evidence.youtube_url(),
    }


def _rubric_row(rubric: Any) -> dict[str, Any]:
    return {
        "rubric_id": rubric.rubric_id,
        "criterion": rubric.criterion,
        "check": rubric.check,
        "why": rubric.why,
        "contested": rubric.contested,
        "unit_id": rubric.unit_id,
        "unit_kind": rubric.unit_kind,
        "unit_title": rubric.unit_title,
        "creators": rubric.creators,
        "videos": rubric.videos,
        "evidence": [_evidence_row(item) for item in rubric.evidence],
    }


def pack_checks(pack: ExpertPack) -> dict[str, Any]:
    """The deterministic numbers the panel header states as claims.

    Recomputed from the rubrics rather than copied out of ``stats`` so the
    header cannot drift from the rows underneath it — if a rubric is dropped by
    hand from the JSON, these move and the stored stats do not.
    """
    evidence = [item for rubric in pack.rubrics for item in rubric.evidence]
    resolved = [item for item in evidence if item.ratio >= RESOLVED_RATIO]
    multi = [rubric for rubric in pack.rubrics if len(rubric.creators) >= 2]
    return {
        "rubrics": len(pack.rubrics),
        "evidence_total": len(evidence),
        "evidence_resolved": len(resolved),
        "quote_resolution": round(len(resolved) / len(evidence), 4) if evidence else 0.0,
        "multi_creator_rubrics": len(multi),
        "multi_creator_share": (round(len(multi) / len(pack.rubrics), 4) if pack.rubrics else 0.0),
        "contested_rubrics": sum(1 for rubric in pack.rubrics if rubric.contested),
        "creators": len({item.channel_name or item.video_id for item in evidence}),
        "excluded_video_citations": sum(
            1 for item in evidence if item.video_id in set(pack.provenance.excluded_video_ids)
        ),
    }


def _member_rows(pack: ExpertPack) -> list[dict[str, Any]]:
    """Membership, with what each video actually contributed marked on it.

    Being routed into a pack and having a say in its rules are two different
    things, and the membership table is the one place a reader would otherwise
    conflate them. A video whose chunks reached no source unit contributed
    nothing — the honest reason in this corpus is that the theme layer and the
    entity graph are derived state built at an earlier chunk count, so three
    videos ingested after the last ``index-themes`` route into a pack and can
    never be cited by it. Rendering them as ordinary members would let the
    membership count stand in for coverage, which is exactly the overstatement
    the corpus audit found.
    """
    in_units = {video for unit in pack.units for video in unit.video_ids}
    cited = {item.video_id for rubric in pack.rubrics for item in rubric.evidence}
    rows: list[dict[str, Any]] = []
    for member in pack.members:
        row = member.model_dump(mode="json")
        row["in_units"] = member.video_id in in_units
        row["cited"] = member.video_id in cited
        rows.append(row)
    return rows


def _ablation_rows(run: dict[str, Any] | None, topic: str) -> dict[str, Any] | None:
    """The D2 three-way comparison, if one has been run for this topic.

    Returned whole rather than summarised: it is three rows of numbers, and the
    panel's job is to put them side by side with the spread beside the point
    estimate so nobody reads a lead smaller than the scorer's own noise as a
    result.

    Verdicts are **recomputed** from the cells rather than read from the run.
    ``winner()`` is pure — it reads the scores already in the file and calls no
    model — so recomputing costs nothing and is the only way a run committed
    before a verdict bug was found gets the corrected reading. Trusting the
    stored verdicts would have left every existing ablation still announcing
    "leader clears the runner-up's own spread" on metrics that were scored once,
    which is the exact claim that fix removed.
    """
    if not run or run.get("topic") != topic:
        return None
    cells = [
        {
            "arm": cell.get("setup"),
            "scores": cell.get("scores") or {},
            "score_spread": cell.get("score_spread") or {},
            "rubrics": cell.get("rubrics"),
            "criteria_recall_all": cell.get("criteria_recall_all"),
            "criteria_recall_grouped": cell.get("criteria_recall_grouped"),
            "findings_total": cell.get("findings_total"),
            "findings_grounded": cell.get("findings_grounded"),
            "citations_total": cell.get("citations_total"),
            "citations_resolved": cell.get("citations_resolved"),
            "held_out_leaks": cell.get("held_out_leaks"),
            "units_by_kind": cell.get("units_by_kind"),
        }
        for cell in run.get("cells") or []
    ]
    return {
        "generated_at": run.get("generated_at"),
        "metrics": run.get("metrics") or [],
        "baseline": run.get("baseline"),
        "held_out_title": run.get("held_out_title"),
        "artifact_url": run.get("artifact_url"),
        "criteria_total": run.get("criteria_total"),
        "criteria_applicable": run.get("criteria_applicable"),
        "match_repeats": (run.get("config") or {}).get("match_repeats"),
        "cells": cells,
        "verdicts": {metric: winner(run, metric) for metric in (run.get("metrics") or [])},
    }


def list_packs(packs_dir: Path | None = None) -> dict[str, Any]:
    """Every declared pack with its shipped arm's headline numbers.

    A declared pack that has not been built yet still appears, carrying
    ``built: false`` and the command that builds it — the catalog is the claim
    about what packs exist, and hiding the unbuilt ones would let the tab look
    complete while three quarters of it was missing.
    """
    store = _store(packs_dir)
    try:
        catalog = store.catalog()
    except (OSError, ValueError):
        return {"packs": [], "build_command": "uv run python -m src.cli build-packs"}

    rows: list[dict[str, Any]] = []
    for declaration in catalog.packs:
        pack = _load(store, declaration.topic)
        row: dict[str, Any] = {
            "topic": declaration.topic,
            "name": declaration.name,
            "blurb": declaration.blurb,
            "artifact": declaration.artifact,
            "built": pack is not None,
        }
        if pack is not None:
            row["arm"] = pack.arm
            row["generated_at"] = pack.generated_at
            row["corpus_digest"] = pack.provenance.corpus_digest
            row["checks"] = pack_checks(pack)
            row["members"] = pack.stats.get("members_included")
        rows.append(row)
    return {
        "packs": rows,
        "excluded_video_ids": list(catalog.excluded_video_ids),
        "exclusion_reason": catalog.exclusion_reason,
        "held_out_video_ids": list(catalog.held_out_video_ids),
        "build_command": "uv run python -m src.cli build-packs",
    }


def corpus_staleness(pack: ExpertPack, live_videos: list[dict[str, Any]]) -> dict[str, Any]:
    """How far the pack's build corpus is behind the one being browsed.

    A pack is a snapshot: its rules were distilled from the corpus as it stood
    at build time, and the corpus keeps growing. The panel used to print the
    build's own digest and counts and stop there, which reads as a statement of
    fact about the present — V5's acceptance gate failed on exactly that, with
    the pack claiming 56 videos while the header above it said 67.

    So this compares the two and says the difference out loud. It deliberately
    reports *counts*, not a bare "stale" badge: "behind by 11 videos" tells a
    reader how much to discount what they are looking at, and a warning triangle
    does not. The digest is recomputed the same way
    :func:`src.evals.matrix_cache.corpus_digest` does — sorted unique video ids
    plus the total chunk count — so a re-chunk of the same videos still counts
    as a change.
    """
    from src.evals.matrix_cache import corpus_digest

    live_ids = [str(video.get("video_id")) for video in live_videos if video.get("video_id")]
    live_chunks = sum(int(video.get("chunk_count") or 0) for video in live_videos)
    live_digest = corpus_digest(live_ids, live_chunks)
    build_videos = int(pack.provenance.video_count or 0)
    build_chunks = int(pack.provenance.chunk_count or 0)
    return {
        "build_digest": pack.provenance.corpus_digest,
        "build_videos": build_videos,
        "build_chunks": build_chunks,
        "live_digest": live_digest,
        "live_videos": len(set(live_ids)),
        "live_chunks": live_chunks,
        # Signed: a re-chunk can shrink the corpus, and "behind by -4 chunks"
        # would be a lie in the other direction.
        "behind_videos": len(set(live_ids)) - build_videos,
        "behind_chunks": live_chunks - build_chunks,
        "current": live_digest == pack.provenance.corpus_digest,
    }


def pack_detail(
    topic: str,
    packs_dir: Path | None = None,
    live_videos: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """One pack: its rubrics, its membership, its gaps, and the D2 rows.

    The three arms are loaded alongside the shipped one so the panel can say
    what the ablation compared without a second request, and so a reader can see
    that the pack on screen *is* one of the three arms rather than a fourth
    thing assembled for display.

    ``live_videos`` is the corpus listing as it stands now. Passing it adds a
    ``staleness`` block; omitting it leaves that key absent rather than
    guessing, because "no comparison available" and "up to date" must not render
    the same.
    """
    store = _store(packs_dir)
    try:
        catalog = store.catalog()
    except (OSError, ValueError):
        return None
    declaration = next((item for item in catalog.packs if item.topic == topic), None)
    if declaration is None:
        return None
    pack = _load(store, topic)
    manifest = store.load_manifest(topic)
    arms: dict[str, Any] = {}
    for arm in ("raptor", "communities", "merged"):
        loaded = _load(store, topic, arm)
        if loaded is not None:
            arms[arm] = {
                "arm": arm,
                "checks": pack_checks(loaded),
                "units_by_kind": loaded.stats.get("units_by_kind") or {},
                "unit_budget": loaded.stats.get("unit_budget"),
                "gaps": loaded.stats.get("gaps"),
            }

    detail: dict[str, Any] = {
        "topic": declaration.topic,
        "name": declaration.name,
        "blurb": declaration.blurb,
        "artifact": declaration.artifact,
        "routing_text": declaration.routing_text,
        "built": pack is not None,
        "overrides": {
            str(key): bool(value)
            for key, value in (manifest.get("overrides") or {}).items()
            if isinstance(value, bool)
        },
        "arms": arms,
        "ablation": _ablation_rows(store.load_ablation(), topic),
        "excluded_video_ids": list(catalog.excluded_video_ids),
        "held_out_video_ids": list(catalog.held_out_video_ids),
        "build_command": "uv run python -m src.cli build-packs",
    }
    if pack is None:
        return detail

    if live_videos is not None:
        detail["staleness"] = corpus_staleness(pack, live_videos)

    detail.update(
        {
            "arm": pack.arm,
            "version": pack.version,
            "generated_at": pack.generated_at,
            "provenance": pack.provenance.model_dump(mode="json"),
            "stats": pack.stats,
            "checks": pack_checks(pack),
            "rubrics": [_rubric_row(rubric) for rubric in pack.rubrics],
            "members": _member_rows(pack),
            "units": [
                {
                    "unit_id": unit.unit_id,
                    "kind": unit.kind,
                    "title": unit.title,
                    "chunks": len(unit.chunk_ids),
                    "videos": len(unit.video_ids),
                    "creator_count": unit.creator_count,
                    "top_creator": unit.top_creator,
                    "top_creator_share": unit.top_creator_share,
                }
                for unit in pack.units
            ],
            "gaps": [gap.model_dump(mode="json") for gap in pack.gaps],
        }
    )
    return detail


def set_member_override(
    topic: str,
    video_id: str,
    included: bool | None,
    packs_dir: Path | None = None,
) -> dict[str, Any]:
    """Pin a video into or out of a pack, or hand it back to the router.

    Written to the manifest, not to the pack. Membership is *routed* — a hand
    decision that lived in the pack file would be erased by the next build, and
    one that silently rewrote the rendered pack would make the panel disagree
    with the artifact on disk. So this records the decision and says plainly
    that it applies at the next build; nothing on screen changes until then.

    An excluded video cannot be re-admitted here. ``route_members`` drops
    blocked ids after overrides are applied, so accepting one would only store a
    decision the build will refuse — better to refuse it at the point a human
    makes it.
    """
    store = _store(packs_dir)
    catalog = store.catalog()
    if video_id in set(catalog.blocked()):
        raise ValueError(f"{video_id} is excluded from every pack and cannot be overridden in")
    overrides = store.set_override(topic, video_id, included)
    return {
        "topic": topic,
        "overrides": overrides,
        "applies": "at the next build",
        "build_command": "uv run python -m src.cli build-packs",
    }

"""The live half of the rubric packs — stores, routing, one call per unit.

:mod:`src.rag.packs` is arithmetic and records. This module is what opens the
chunk store, routes each declared pack's topic description through the summary
index, assembles source units from both abstraction layers, and spends the LLM
calls that turn a bundle of excerpts into criteria.

Two things here are load-bearing.

**Every call is cached by its own inputs.** The key is the prompt, the model and
the exact excerpt text — so the merged arm, which by construction re-visits units
the single arms already covered, costs almost nothing after the first two arms,
and a rebuild of an unchanged corpus reproduces the identical pack rather than
re-rolling it. That is the difference between a pack a reviewer can diff and a
pack that changes every time somebody looks at it.

**Exclusion happens at the router, not afterwards.** The five Australian
property videos are removed from the summary store the membership query runs
against, so a property video is never a candidate for a pack in the first place;
the held-out video is removed the same way. The check that this worked is
independent of the mechanism — :func:`src.rag.packs.pack_statistics` counts
citations and members belonging to blocked videos by id, which would still catch
a leak if the exclusion argument were deleted.
"""

from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

from src.config import Settings
from src.rag.packs import (
    PACK_ARMS,
    EmbedFn,
    ExpertPack,
    PackCatalog,
    PackDeclaration,
    PackGap,
    PackMember,
    PackProvenance,
    PackStore,
    Rubric,
    RubricExtractor,
    SourceUnit,
    apply_overrides,
    community_units,
    corpus_digest,
    dedupe_rubrics,
    digest_of,
    format_excerpts,
    included_video_ids,
    pack_statistics,
    raptor_units,
    reconcile_evidence,
    select_units,
    unit_excerpts,
)

#: Where per-unit rubric calls are remembered. Derived state beside the other
#: caches: deleting it only costs the calls to regenerate it.
DEFAULT_RUBRIC_CACHE_DIR = Path(".yt-agent/pack_cache")

#: Bumped whenever :data:`~src.rag.packs.RUBRIC_SYSTEM_PROMPT` changes, so a
#: cached rubric written under the old wording is never served under the new.
#: A pack and the prompt that produced it must not be able to drift apart.
PROMPT_VERSION = "v1"

#: Source units per arm. The same number for all three, because the merged arm
#: is a claim about *provenance*, not about budget — see
#: :func:`src.rag.packs.merge_units`.
DEFAULT_UNIT_BUDGET = 8
DEFAULT_EXCERPTS_PER_UNIT = 12
DEFAULT_MAX_RUBRICS_PER_UNIT = 3

#: Videos a pack's topic description is allowed to route to. Wider than the
#: retrieval default (5) because a pack is a standing artifact rather than one
#: answer's context window: the cost of an extra video is a slightly broader
#: unit, and the cost of missing one is a creator the pack can never cite.
DEFAULT_ROUTING_TOP_K = 14
DEFAULT_ROUTING_MIN_SCORE = 0.25

ProgressFn = Callable[[str], None]


class PackWorkspace:
    """Everything four packs and three arms share, loaded once.

    The chunk records, the theme layer and the graph extractions are each read
    from disk exactly once; routing, unit assembly and every LLM call are then
    pure functions of them. Loading per pack would also mean the arms could
    silently be built from different corpora, which is the reproducibility bug
    this project has already paid for once.
    """

    def __init__(
        self,
        settings: Settings,
        catalog: PackCatalog,
        *,
        embed: EmbedFn,
        extractor: RubricExtractor | None,
        records: Sequence[dict[str, Any]],
        themes: Sequence[Any],
        theme_generated_at: str = "",
        extractions: Sequence[Any] = (),
        route: Callable[[str, int, float], list[Any]] | None = None,
        cache_dir: Path = DEFAULT_RUBRIC_CACHE_DIR,
    ) -> None:
        self.settings = settings
        self.catalog = catalog
        self.embed = embed
        self.extractor = extractor
        self.records = list(records)
        self.by_chunk = {str(record["chunk_id"]): record for record in self.records}
        self.themes = list(themes)
        self.theme_generated_at = theme_generated_at
        self.extractions = list(extractions)
        self.route = route
        self.cache_dir = cache_dir
        self.calls = 0
        self.cache_hits = 0
        self.video_catalogue = _video_catalogue(self.records)

    def provenance(self, notes: Sequence[str] = ()) -> PackProvenance:
        return PackProvenance(
            corpus_digest=corpus_digest(self.records),
            chunk_count=len(self.records),
            video_count=len(self.video_catalogue),
            theme_generated_at=self.theme_generated_at,
            theme_count=len(self.themes),
            graph_cache_digest=digest_of(str(item.chunk_id) for item in self.extractions),
            graph_extractions=len(self.extractions),
            excluded_video_ids=list(self.catalog.excluded_video_ids),
            held_out_video_ids=list(self.catalog.held_out_video_ids),
            embedding_model=self.settings.embedding_model,
            rubric_model=self.extractor.model_name if self.extractor else "",
            units_budget=DEFAULT_UNIT_BUDGET,
            excerpts_per_unit=DEFAULT_EXCERPTS_PER_UNIT,
            max_rubrics_per_unit=DEFAULT_MAX_RUBRICS_PER_UNIT,
            routing_top_k=DEFAULT_ROUTING_TOP_K,
            routing_min_score=DEFAULT_ROUTING_MIN_SCORE,
            community_min_entities=5,
            notes=list(notes),
        )


def _video_catalogue(records: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    catalogue: dict[str, dict[str, Any]] = {}
    for record in records:
        video_id = str(record["video_id"])
        row = catalogue.setdefault(
            video_id,
            {
                "title": record.get("title"),
                "channel_name": record.get("channel_name"),
                "chunk_count": 0,
            },
        )
        row["chunk_count"] = int(row["chunk_count"]) + 1
    return catalogue


# ─── Membership ──────────────────────────────────────────────────────────────


def route_members(
    workspace: PackWorkspace,
    declaration: PackDeclaration,
    overrides: dict[str, bool],
    *,
    top_k: int = DEFAULT_ROUTING_TOP_K,
    min_score: float = DEFAULT_ROUTING_MIN_SCORE,
) -> list[PackMember]:
    """Membership computed by routing the topic description like a question.

    This is the decision the design turns on: a pack exists because somebody
    declared it, and *what is in it* is whatever the corpus's own summary index
    says is about that description. The routing text is embedded and matched
    against per-video summaries by exactly the call a live question makes, so a
    pack's membership is reproducible from the corpus and reviewable next to a
    retrieval trace, rather than being a hand-kept list that rots.
    """
    routed: list[PackMember] = []
    if workspace.route is not None:
        for summary in workspace.route(declaration.routing_text, top_k, min_score):
            video_id = str(summary.video_id)
            meta = workspace.video_catalogue.get(video_id) or {}
            routed.append(
                PackMember(
                    video_id=video_id,
                    title=summary.title or meta.get("title"),
                    channel_name=meta.get("channel_name"),
                    score=round(float(summary.score), 4),
                    routed=True,
                    chunk_count=int(meta.get("chunk_count") or 0),
                )
            )
    members = apply_overrides(routed, overrides, workspace.video_catalogue)
    blocked = set(workspace.catalog.blocked())
    # Belt and braces: the router never saw these, but a hand override could
    # name one, and a property rubric is the single most visible failure this
    # slice has. An override cannot re-admit an excluded video.
    return [member for member in members if member.video_id not in blocked]


# ─── Rubric calls ────────────────────────────────────────────────────────────


def _cache_key(excerpts: Sequence[dict[str, Any]], artifact: str, model: str) -> str:
    material = json.dumps(
        {
            "prompt_version": PROMPT_VERSION,
            "artifact": artifact,
            "model": model,
            "max_rubrics": DEFAULT_MAX_RUBRICS_PER_UNIT,
            "excerpts": format_excerpts(excerpts),
        },
        sort_keys=True,
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]


def rubrics_for_unit(
    workspace: PackWorkspace,
    unit: SourceUnit,
    declaration: PackDeclaration,
    *,
    index: int,
) -> tuple[list[Rubric], str]:
    """One unit's rubrics, cached, with citation metadata derived server-side.

    Returns the surviving rubrics and, when there are none, the reason — which
    the caller records as a gap rather than dropping on the floor.
    """
    excerpts = unit_excerpts(unit, workspace.by_chunk, DEFAULT_EXCERPTS_PER_UNIT)
    if not excerpts:
        return [], "no excerpts survived the restriction to this pack's videos"
    model = workspace.extractor.model_name if workspace.extractor else ""
    key = _cache_key(excerpts, declaration.artifact, model)
    path = workspace.cache_dir / f"{key}.json"
    rows: list[dict[str, Any]] | None = None
    if path.is_file():
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
            rows = [row for row in (cached.get("rubrics") or []) if isinstance(row, dict)]
            workspace.cache_hits += 1
        except (json.JSONDecodeError, OSError):
            rows = None
    if rows is None:
        if workspace.extractor is None:
            return [], "no rubric model configured for this build"
        try:
            rows = workspace.extractor.extract(
                excerpts,
                artifact=declaration.artifact,
                max_rubrics=DEFAULT_MAX_RUBRICS_PER_UNIT,
            )
        except (ValueError, KeyError) as exc:
            return [], f"rubric call failed: {type(exc).__name__}: {exc}"
        workspace.calls += 1
        workspace.cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"rubrics": rows}, indent=2) + "\n", encoding="utf-8")

    rubrics: list[Rubric] = []
    unverifiable = 0
    for position, row in enumerate(rows, start=1):
        criterion = str(row.get("criterion") or "").strip()
        if not criterion:
            continue
        evidence = reconcile_evidence(row.get("evidence") or [], excerpts)
        if not evidence:
            unverifiable += 1
            continue
        rubrics.append(
            Rubric(
                rubric_id=f"r{index:02d}{position:02d}",
                criterion=criterion,
                check=str(row.get("check") or "").strip(),
                why=str(row.get("why") or "").strip(),
                # A contested rubric has to cite two creators or it is just an
                # assertion that people disagree; the scorer applies the same
                # rule, so applying it here keeps the pack and its score aligned.
                contested=bool(row.get("contested"))
                and len({e.video_id for e in evidence}) >= 2,
                unit_id=unit.unit_id,
                unit_kind=unit.kind,
                unit_title=unit.title,
                evidence=evidence,
            )
        )
    if rubrics:
        return rubrics, ""
    if unverifiable:
        return [], f"{unverifiable} rubric(s) returned, none with a quote that resolved"
    return [], "the model returned no rubric for this unit"


# ─── Build ───────────────────────────────────────────────────────────────────


def build_pack(
    workspace: PackWorkspace,
    declaration: PackDeclaration,
    arm: str,
    *,
    members: Sequence[PackMember],
    raptor_pool: Sequence[SourceUnit],
    community_pool: Sequence[SourceUnit],
    budget: int = DEFAULT_UNIT_BUDGET,
    on_progress: ProgressFn | None = None,
    max_workers: int = 6,
) -> ExpertPack:
    """One topic, one arm: units in, deduped rubrics and a gap log out."""
    progress = on_progress or (lambda _message: None)
    units = select_units(arm, raptor_pool, community_pool, budget)
    selected = {unit.unit_id for unit in units}

    results: dict[str, tuple[list[Rubric], str]] = {}

    def work(item: tuple[int, SourceUnit]) -> tuple[str, tuple[list[Rubric], str]]:
        position, unit = item
        return unit.unit_id, rubrics_for_unit(workspace, unit, declaration, index=position)

    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
        for unit_id, outcome in pool.map(work, list(enumerate(units, start=1))):
            results[unit_id] = outcome

    candidates: list[Rubric] = []
    gaps: list[PackGap] = []
    for unit in units:
        rubrics, reason = results.get(unit.unit_id, ([], "unit was not processed"))
        candidates.extend(rubrics)
        if not rubrics:
            gaps.append(
                PackGap(
                    unit_id=unit.unit_id,
                    unit_kind=unit.kind,
                    unit_title=unit.title,
                    videos=len(unit.video_ids),
                    chunks=len(unit.chunk_ids),
                    reason=reason,
                )
            )
        progress(f"[{declaration.topic}/{arm}] {unit.unit_id}: {len(rubrics)} rubric(s)")

    kept, dropped = dedupe_rubrics(candidates, workspace.embed)
    kept_units = {rubric.unit_id for rubric in kept}
    for unit in units:
        if unit.unit_id in kept_units or any(gap.unit_id == unit.unit_id for gap in gaps):
            continue
        gaps.append(
            PackGap(
                unit_id=unit.unit_id,
                unit_kind=unit.kind,
                unit_title=unit.title,
                videos=len(unit.video_ids),
                chunks=len(unit.chunk_ids),
                reason="every rubric it produced restated a rule an earlier unit already covered",
            )
        )
    # Units the topic reached that no rubric came from. Logged rather than
    # dropped: "we never looked at it", "it is one creator talking" and "we
    # looked and found nothing" are three different admissions, and a coverage
    # claim that cannot tell them apart is not a coverage claim. The
    # single-creator rows are the ones to read — that is where the corpus's
    # dissenting voices sit, and they cannot honestly become shared rules.
    for pool_units in (raptor_pool, community_pool):
        for unit in pool_units:
            if unit.unit_id in selected or any(gap.unit_id == unit.unit_id for gap in gaps):
                continue
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
                    reason=unit.reject_reason
                    or f"relevant but outside the {budget}-unit budget for this arm",
                )
            )

    stats = pack_statistics(kept, members, units, gaps, workspace.catalog.blocked())
    stats["rubrics_deduped"] = len(dropped)
    stats["rubric_candidates"] = len(candidates)
    stats["raptor_pool"] = len(raptor_pool)
    stats["community_pool"] = len(community_pool)
    stats["unit_budget"] = budget
    provenance = workspace.provenance()
    provenance.units_budget = budget
    return ExpertPack(
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
    )


def shared_budget(
    budget: int,
    raptor_pool: Sequence[SourceUnit],
    community_pool: Sequence[SourceUnit],
) -> int:
    """The largest unit count all three arms can actually spend.

    A pack whose raptor pool holds six units and whose community pool holds
    twelve would, at a budget of eight, hand the communities arm two extra LLM
    calls — and the comparison would then be reporting that more calls beat
    fewer. Capping every arm at the smaller pool is the only version of this
    comparison that is about *where the abstraction came from*. Both pools being
    empty is a pack with nothing to say, and the budget is then irrelevant.
    """
    available = [len(pool) for pool in (raptor_pool, community_pool) if pool]
    return min([budget, *available]) if available else budget


def build_topic(
    workspace: PackWorkspace,
    declaration: PackDeclaration,
    store: PackStore,
    *,
    arms: Sequence[str] = PACK_ARMS,
    budget: int = DEFAULT_UNIT_BUDGET,
    on_progress: ProgressFn | None = None,
) -> dict[str, ExpertPack]:
    """Every arm of one pack, sharing one routing pass and one unit assembly."""
    progress = on_progress or (lambda _message: None)
    overrides = store.overrides(declaration.topic)
    members = route_members(workspace, declaration, overrides)
    videos = included_video_ids(members)
    progress(
        f"[{declaration.topic}] routed {len(videos)} videos "
        f"({len(overrides)} override(s)): {', '.join(videos)}"
    )
    raptor_pool = raptor_units(workspace.themes, videos)
    community_pool = community_units(workspace.extractions, videos)
    effective = shared_budget(budget, raptor_pool, community_pool)
    progress(
        f"[{declaration.topic}] units available — raptor {len(raptor_pool)}, "
        f"communities {len(community_pool)}; budget {effective}"
    )

    packs: dict[str, ExpertPack] = {}
    for arm in arms:
        pack = build_pack(
            workspace,
            declaration,
            arm,
            members=members,
            raptor_pool=raptor_pool,
            community_pool=community_pool,
            budget=effective,
            on_progress=on_progress,
        )
        store.save_pack(pack)
        packs[arm] = pack
        progress(
            f"[{declaration.topic}/{arm}] {pack.stats['rubrics']} rubrics, "
            f"{pack.stats['multi_creator_share']:.0%} multi-creator, "
            f"{pack.stats['gaps']} gaps"
        )

    store.save_manifest(
        declaration.topic,
        {
            "topic": declaration.topic,
            "name": declaration.name,
            "routing_text": declaration.routing_text,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "overrides": dict(sorted(overrides.items())),
            "provenance": workspace.provenance().model_dump(mode="json"),
            "members": [member.model_dump(mode="json") for member in members],
            "arms": {
                arm: {
                    "rubrics": pack.stats["rubrics"],
                    "gaps": pack.stats["gaps"],
                    "multi_creator_share": pack.stats["multi_creator_share"],
                }
                for arm, pack in packs.items()
            },
        },
    )
    return packs


# ─── Live wiring ─────────────────────────────────────────────────────────────


def open_workspace(
    settings: Settings,
    catalog: PackCatalog,
    *,
    with_model: bool = True,
) -> PackWorkspace:
    """The workspace over the real stores, with every blocked video removed.

    ``with_model=False`` builds everything except the LLM client, which is what
    a dry run wants: routing, unit assembly and the deterministic counts cost
    nothing and are worth checking before any call is paid for.
    """
    from src.api.corpus import load_chunk_embeddings
    from src.rag.embeddings import HuggingFaceEmbeddingModel
    from src.rag.packs import load_extractions
    from src.rag.summaries import TranscriptSummaryStore
    from src.rag.themes import ThemeStore

    records = load_chunk_embeddings(settings.chroma_path, settings.chunk_collection)
    for record in records:
        record.pop("embedding", None)
    theme_index = ThemeStore(settings.theme_path).load()
    extractions = (
        load_extractions(settings.graph_cache_dir) if settings.graph_cache_dir else []
    )

    embedding_model = HuggingFaceEmbeddingModel(settings.embedding_model)
    summary_store = TranscriptSummaryStore(
        settings.chroma_path,
        embedding_model=embedding_model,
        embedding_model_name=settings.embedding_model,
        collection_name=settings.transcript_summary_collection,
        exclude_video_ids=catalog.blocked(),
    )

    def route(text: str, top_k: int, min_score: float) -> list[Any]:
        return list(summary_store.query_relevant_transcripts(text, top_k, min_score))

    def embed(texts: Sequence[str]) -> list[list[float]]:
        return embedding_model.embed_documents(list(texts))

    extractor: RubricExtractor | None = None
    if with_model:
        from langchain_openai import ChatOpenAI

        from src.agents.llm import chat_model_kwargs

        extractor = RubricExtractor(
            ChatOpenAI(**chat_model_kwargs(settings)), settings.deepseek_model
        )

    return PackWorkspace(
        settings,
        catalog,
        embed=embed,
        extractor=extractor,
        records=records,
        themes=theme_index.themes if theme_index is not None else [],
        theme_generated_at=theme_index.generated_at if theme_index is not None else "",
        extractions=extractions,
        route=route,
    )


def build_all(
    settings: Settings,
    *,
    packs_dir: Path = Path("experts"),
    topics: Sequence[str] | None = None,
    arms: Sequence[str] = PACK_ARMS,
    budget: int = DEFAULT_UNIT_BUDGET,
    with_model: bool = True,
    on_progress: ProgressFn | None = None,
) -> dict[str, dict[str, ExpertPack]]:
    """Build every declared pack, three ways, from one shared workspace."""
    progress = on_progress or (lambda _message: None)
    store = PackStore(packs_dir)
    catalog = store.catalog()
    started = time.monotonic()
    workspace = open_workspace(settings, catalog, with_model=with_model)
    progress(
        f"corpus {workspace.provenance().corpus_digest} — {len(workspace.records)} chunks, "
        f"{len(workspace.themes)} themes, {len(workspace.extractions)} extractions, "
        f"{len(catalog.blocked())} videos blocked"
    )
    chosen = [
        declaration
        for declaration in catalog.packs
        if topics is None or declaration.topic in topics
    ]
    built: dict[str, dict[str, ExpertPack]] = {}
    for declaration in chosen:
        built[declaration.topic] = build_topic(
            workspace, declaration, store, arms=arms, budget=budget, on_progress=progress
        )
    progress(
        f"done in {time.monotonic() - started:.1f}s — {workspace.calls} LLM call(s), "
        f"{workspace.cache_hits} cache hit(s)"
    )
    return built

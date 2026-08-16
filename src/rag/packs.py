"""Rubric packs: the corpus distilled into criteria a reviewer can apply.

Every other layer in this project answers a question. This one produces the
thing the project exists for — a **distilled expert reviewer**: a list of
checkable rules, each carrying the transcript words that put it there, that
somebody could hold a real resume (or a real service boundary) against.

The distinction this module is built to defend is **criterion, not summary**.
"Creators stress the importance of quantifying your impact" is a summary: it
tells a reader what a video was about and cannot be applied to anything. "Every
experience bullet carries a number — a percentage, a count, a currency figure or
a duration" is a criterion: one artifact, one look, pass or fail. Each rubric
therefore carries a separate ``check`` field naming the observable thing a
reviewer counts, and a rubric whose check reads "assess whether the summary is
good" is one this module has failed to produce.

Three decisions worth reading
-----------------------------
**Packs are declared; membership is computed.** A pack exists because
``experts/packs.json`` says so — a name and a topic description *written as
routing text*. Which videos are in it is then decided by routing that
description through the per-video summary index exactly as if it were a user
question (:meth:`~src.rag.summaries.TranscriptSummaryStore.query_relevant_transcripts`).
Clustering never gets to decide which packs exist, only what goes in them,
because a pack list that reshuffles itself every time the corpus grows is not
something a reader can be asked to trust. Routing is also reviewable: the
computed membership is written to the manifest with each video's score, and a
human can pin a video in or out through the UI, which persists an override.

**The distillate has two possible sources, and they are compared.** A rubric is
written from a *source unit* — a bundle of excerpts the corpus itself grouped.
Two layers can supply those bundles: the RAPTOR theme layer
(:mod:`src.rag.themes`, clusters of chunk embeddings) and GraphRAG's Leiden
communities over the entity graph (:mod:`src.rag.communities`). Both arms and
their merge are built with **the same unit budget and the same prompt**, so the
comparison is about where the abstraction came from and not about who was
allowed more LLM calls.

**Citation metadata is derived, never accepted.** The model is shown numbered
excerpts and may only answer with an excerpt *number* and a quote; the video id,
timestamp, chunk id and channel are read off the excerpt this project retrieved.
An earlier slice in this repo shipped 39 citations of which 35 carried a
model-invented ``chunk_index``. The quote is then snapped to the exact
contiguous span of the stored transcript that it matches (:func:`snap_quote`),
so what renders in the UI is transcript text rather than the model's
recollection of it, and anything that does not resolve is dropped rather than
displayed.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol, Sequence

from pydantic import BaseModel, Field

#: The three ways a pack can be built, in the order the comparison renders them.
#: ``merged`` is not a third algorithm — it is the two pools interleaved under
#: the same unit budget, so it trades half of each arm's units for the other's.
PACK_ARMS: tuple[str, ...] = ("raptor", "communities", "merged")

#: Where packs live. Files are the source of truth: the API renders them and the
#: only thing that writes them is the CLI build (plus membership overrides).
DEFAULT_PACKS_DIR = Path("experts")

#: The pack declaration file — hand-written, and the only place a pack is born.
CATALOG_FILENAME = "packs.json"

#: Minimum token overlap for a quote to count as present in a chunk. The same
#: value :mod:`src.evals.critique` uses, on purpose: the shipped pack's
#: provenance is checked by that module's own resolver, so a quote this module
#: accepts and that one rejects would be a gate scored against itself.
QUOTE_MATCH_RATIO = 0.80

#: How far a snapped span may exceed the quote before it is treated as a
#: stitch-up rather than a quotation. A model that joined two sentences from
#: opposite ends of a chunk produces a technically-high token overlap over a
#: span nothing like the quote's length; without this the UI would render 400
#: characters of transcript as an eight-word quote.
MAX_SNAP_EXPANSION = 3.0

#: Cosine above which two rubrics are the same rule said twice. Applied
#: identically to all three arms — without it the merged arm would score its
#: overlap with itself as extra coverage, which is exactly the padding the
#: critique harness exists to refuse.
DEDUPE_THRESHOLD = 0.86

_WORD = re.compile(r"[a-z0-9']+")
_TOKEN = re.compile(r"[A-Za-z0-9']+")


class ChatModel(Protocol):
    def invoke(self, messages: list) -> object: ...


# ─── Declaration ─────────────────────────────────────────────────────────────


class PackDeclaration(BaseModel):
    """One declared pack. ``routing_text`` is a query, not a description.

    It is embedded and run against the per-video summary index, so it should
    read like something a user would ask — the vocabulary a relevant video's
    summary would contain — rather than like a section heading.
    """

    topic: str
    name: str
    routing_text: str
    blurb: str = ""
    #: What kind of artifact this pack's rubrics judge. Carried so a reader can
    #: see the rubrics were written to be applied to something.
    artifact: str = "document"


class PackCatalog(BaseModel):
    """The declared packs plus the exclusions every build must honour."""

    version: int = 1
    #: The D5 exclusion: videos that stay indexed for Q&A but may never reach a
    #: pack. Property/tax material in a job-search rubric is instantly visible
    #: leakage, so it is excluded at the router rather than filtered afterwards.
    excluded_video_ids: list[str] = Field(default_factory=list)
    exclusion_reason: str = ""
    #: Videos held out so the packs can be scored against an expert the build
    #: was never allowed to read (:mod:`src.evals.critique`). Excluded from
    #: *every* pack, uniformly, so the shipped pack and the scored pack are the
    #: same artifact rather than two builds that might differ.
    held_out_video_ids: list[str] = Field(default_factory=list)
    packs: list[PackDeclaration] = Field(default_factory=list)

    def blocked(self) -> list[str]:
        return sorted(dict.fromkeys([*self.excluded_video_ids, *self.held_out_video_ids]))

    def declaration(self, topic: str) -> PackDeclaration | None:
        return next((pack for pack in self.packs if pack.topic == topic), None)


def load_catalog(packs_dir: Path | str = DEFAULT_PACKS_DIR) -> PackCatalog:
    path = Path(packs_dir) / CATALOG_FILENAME
    return PackCatalog.model_validate_json(path.read_text(encoding="utf-8"))


# ─── Membership ──────────────────────────────────────────────────────────────


class PackMember(BaseModel):
    """One video's standing in a pack, and how it got there.

    ``routed``/``score`` record what the summary router decided; ``override``
    records what a human decided afterwards. Both are kept, so the UI can show
    "the router ranked this 0.41 and you pinned it out" rather than silently
    presenting the human's answer as the machine's.
    """

    video_id: str
    title: str | None = None
    channel_name: str | None = None
    score: float = 0.0
    routed: bool = True
    #: ``True`` pinned in, ``False`` pinned out, ``None`` left to the router.
    override: bool | None = None
    chunk_count: int = 0

    @property
    def included(self) -> bool:
        return self.routed if self.override is None else self.override


def apply_overrides(
    routed: Sequence[PackMember],
    overrides: dict[str, bool],
    catalogue: dict[str, dict[str, Any]] | None = None,
) -> list[PackMember]:
    """Routed membership with human pins applied, blocked videos never added.

    A video pinned *in* that the router never returned has no score and no
    metadata of its own, so ``catalogue`` (video_id → title/channel/chunks) is
    consulted to build a row for it. Ordering is by inclusion then score, which
    is the order the UI reviews them in: what is in the pack first, loudest
    first, and the near misses underneath.
    """
    rows = {member.video_id: member.model_copy() for member in routed}
    for video_id, decision in overrides.items():
        existing = rows.get(video_id)
        if existing is not None:
            existing.override = decision
            continue
        meta = (catalogue or {}).get(video_id) or {}
        rows[video_id] = PackMember(
            video_id=video_id,
            title=meta.get("title"),
            channel_name=meta.get("channel_name"),
            score=0.0,
            routed=False,
            override=decision,
            chunk_count=int(meta.get("chunk_count") or 0),
        )
    return sorted(rows.values(), key=lambda row: (not row.included, -row.score, row.video_id))


def included_video_ids(members: Sequence[PackMember]) -> list[str]:
    return [member.video_id for member in members if member.included]


# ─── Source units ────────────────────────────────────────────────────────────


class SourceUnit(BaseModel):
    """One bundle of excerpts a rubric can be written from.

    ``kind`` is the arm it came from. ``chunk_ids`` is what the LLM will
    actually be shown, already restricted to the pack's videos — a theme that
    spans the whole corpus contributes only its in-pack part, so a pack never
    quietly cites a video it does not contain.

    ``top_creator_share`` is here because an independent audit of the theme
    layer found that a theme spanning four videos can draw 145 of its 146 chunks
    from one podcast. Video count is therefore not a diversity measure — a large
    bucket collects stray videos for free — so a unit carries the share of its
    chunks belonging to its loudest channel, and a reader can see when "eight
    videos" means "one lecture with seven visitors".
    """

    unit_id: str
    kind: str
    title: str
    summary: str = ""
    chunk_ids: list[str] = Field(default_factory=list)
    video_ids: list[str] = Field(default_factory=list)
    #: How much of the unit survived the restriction to this pack — the ranking
    #: key, and the number that says whether a unit is really about this topic.
    retained: int = 0
    creator_count: int = 0
    top_creator: str = ""
    top_creator_share: float = 0.0
    #: Units below the diversity floor are kept rather than filtered away, so
    #: the gap log can say "this exists and we chose not to use it".
    eligible: bool = True
    reject_reason: str = ""


def creator_profile(
    chunk_ids: Sequence[str],
    records: dict[str, dict[str, Any]],
) -> tuple[int, str, float]:
    """``(distinct creators, loudest creator, its share of the chunks)``.

    Counted over chunks rather than videos on purpose: one channel with three
    videos in a unit is still one voice, and the failure this guards against is
    a rubric that looks corroborated because a cluster happened to include a
    second creator's single stray chunk.
    """
    tally: dict[str, int] = {}
    for chunk_id in chunk_ids:
        record = records.get(chunk_id)
        if record is None:
            continue
        creator = str(record.get("channel_name") or record.get("video_id") or "")
        tally[creator] = tally.get(creator, 0) + 1
    total = sum(tally.values())
    if not total:
        return 0, "", 0.0
    top = max(tally, key=lambda name: (tally[name], name))
    return len(tally), top, round(tally[top] / total, 4)


def raptor_units(
    themes: Iterable[Any],
    video_ids: Sequence[str],
    records: dict[str, dict[str, Any]],
    *,
    min_chunks: int = 5,
    min_videos: int = 2,
    max_creator_share: float = 0.80,
) -> list[SourceUnit]:
    """Theme clusters restricted to a pack's videos, biggest first.

    Two floors, and the second one is the interesting one. ``min_videos`` is why
    this layer is preferred to the per-video summaries at all: a unit drawn from
    one video can only produce that video's advice. ``max_creator_share`` is the
    audit's finding turned into a rule — a theme whose chunks are 80% one
    channel is that channel's lecture with corroborating visitors, and a rubric
    written from it would satisfy a "spans two videos" test while resting on one
    voice.

    Rejected units are returned marked rather than dropped, because a pack that
    silently discards a creator is exactly the failure a coverage claim hides.
    The corpus's most contrarian résumé voice sits in a single-video theme; it
    cannot honestly become a shared rule, and it must not vanish either.

    ``records`` is required rather than optional. It was optional once, and the
    three packs that were never built failed exactly there: with no lookup every
    unit profiles as zero creators, trips ``creators < 2``, and the pack comes
    out empty with a gap log blaming the corpus. A missing argument has to be a
    ``TypeError``, not a plausible-looking empty pack.
    """
    units: list[SourceUnit] = []
    allowed = set(video_ids)
    lookup = records
    for theme in themes:
        members = [m for m in theme.members if m.video_id in allowed]
        videos = sorted({m.video_id for m in members})
        if len(members) < min_chunks:
            continue
        chunk_ids = [m.chunk_id for m in members]
        creators, top, share = creator_profile(chunk_ids, lookup)
        reason = ""
        if len(videos) < min_videos:
            reason = "only one of this pack's videos is in the theme — one creator's position"
        elif creators < 2:
            reason = f"every chunk comes from one creator ({top})"
        elif share > max_creator_share:
            reason = f"{share:.0%} of the theme's chunks are {top} — one voice with visitors"
        units.append(
            SourceUnit(
                unit_id=f"raptor:{theme.theme_id}",
                kind="raptor",
                title=theme.title,
                summary=theme.summary,
                chunk_ids=chunk_ids,
                video_ids=videos,
                retained=len(members),
                creator_count=creators,
                top_creator=top,
                top_creator_share=share,
                eligible=not reason,
                reject_reason=reason,
            )
        )
    return sorted(units, key=lambda unit: (-unit.retained, unit.unit_id))


@dataclass(frozen=True)
class _Claim:
    claim_id: str
    text: str
    chunk_id: str
    video_id: str
    entity_ids: tuple[str, ...]


def load_extractions(cache_dir: Path | str) -> list[Any]:
    """Every cached chunk extraction, in a fixed order.

    Read from :mod:`src.rag.graph_extract`'s disk cache rather than from Neo4j.
    The cache is what Neo4j was loaded *from* — one file per chunk, keyed on the
    chunk's id and text — so the entity graph it yields is the same graph, and
    it can be rebuilt on a machine with no database running. Sorted by chunk id
    because Leiden is seeded but the node *order* still decides the labelling.
    """
    from src.rag.graph_models import ChunkExtraction

    extractions: list[Any] = []
    for path in sorted(Path(cache_dir).glob("*.json")):
        try:
            extraction = ChunkExtraction.model_validate_json(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
        if extraction.error is None and extraction.chunk_id:
            extractions.append(extraction)
    return sorted(extractions, key=lambda item: str(item.chunk_id))


def entity_graph(
    extractions: Sequence[Any],
) -> tuple[list[str], list[tuple[str, str, float]], dict[str, str], list[_Claim]]:
    """The weighted entity graph, rebuilt exactly as the Neo4j export builds it.

    Mirrors :meth:`src.rag.graph_store.GraphStore.entity_edges`: explicit
    ``RELATES`` weights (max per relation type, summed over types) plus one unit
    per claim that co-mentions both entities. Relations and claim anchors are
    filtered to entities named in the *same* extraction, which is what the
    upsert does, so a relation to an entity the chunk never introduced is
    dropped here for the same reason it is dropped there.
    """
    from src.rag.graph_models import claim_id_for, entity_id_for

    names: dict[str, str] = {}
    relates: dict[tuple[str, str, str], float] = {}
    claims: dict[str, _Claim] = {}
    for extraction in extractions:
        known = {entity_id_for(entity.name): entity.name for entity in extraction.entities}
        for entity_id, name in known.items():
            names.setdefault(entity_id, name)
        for relation in extraction.relations:
            source, target = entity_id_for(relation.source), entity_id_for(relation.target)
            if source not in known or target not in known or source == target:
                continue
            key = (*sorted((source, target)), str(relation.type))
            weight = float(relation.weight)
            relates[key] = max(relates.get(key, weight), weight)
        for claim in extraction.claims:
            anchors = tuple(
                sorted(
                    {entity_id_for(name) for name in claim.entities if entity_id_for(name) in known}
                )
            )
            claim_id = claim_id_for(str(extraction.chunk_id), claim.text)
            claims[claim_id] = _Claim(
                claim_id=claim_id,
                text=claim.text,
                chunk_id=str(extraction.chunk_id),
                video_id=str(extraction.video_id),
                entity_ids=anchors,
            )

    edges: dict[tuple[str, str], float] = {}
    for (source, target, _type), weight in relates.items():
        edges[(source, target)] = edges.get((source, target), 0.0) + weight
    for claim in claims.values():
        anchors = claim.entity_ids
        for index, source in enumerate(anchors):
            for target in anchors[index + 1 :]:
                key = (source, target)
                edges[key] = edges.get(key, 0.0) + 1.0

    nodes = sorted(names)
    edge_list = [(source, target, weight) for (source, target), weight in sorted(edges.items())]
    return nodes, edge_list, names, sorted(claims.values(), key=lambda claim: claim.claim_id)


def community_units(
    extractions: Sequence[Any],
    video_ids: Sequence[str],
    records: dict[str, dict[str, Any]],
    *,
    min_entities: int = 5,
    min_chunks: int = 5,
    min_videos: int = 2,
    max_creator_share: float = 0.80,
) -> list[SourceUnit]:
    """Leiden communities over the entity graph, restricted to a pack's videos.

    ``min_entities`` is not a detail. This corpus's entity graph is badly
    fragmented — 841 communities, most of them one or two entities that were
    extracted once and related to nothing — so an unfiltered communities arm
    would lose a comparison against RAPTOR because its inputs were thin, not
    because entity co-mention is a worse abstraction than embedding proximity.
    Filtering to communities with a real entity count is what makes the
    comparison about the thing it claims to be about.
    """
    from src.rag.communities import detect_communities

    nodes, edges, names, claims = entity_graph(extractions)
    assignments = detect_communities(nodes, edges)
    allowed = set(video_ids)

    members: dict[int, list[str]] = {}
    for entity_id, community_id in assignments.items():
        members.setdefault(community_id, []).append(entity_id)

    lookup = records
    units: list[SourceUnit] = []
    for community_id, entity_ids in sorted(members.items()):
        if len(entity_ids) < min_entities:
            continue
        entity_set = set(entity_ids)
        anchored = [
            claim
            for claim in claims
            if claim.video_id in allowed and entity_set.intersection(claim.entity_ids)
        ]
        chunk_ids = sorted({claim.chunk_id for claim in anchored})
        videos = sorted({claim.video_id for claim in anchored})
        if len(chunk_ids) < min_chunks:
            continue
        top = sorted(entity_ids, key=lambda eid: (-_mentions(eid, claims), eid))[:6]
        creators, loudest, share = creator_profile(chunk_ids, lookup)
        reason = ""
        if len(videos) < min_videos:
            reason = "only one of this pack's videos mentions these entities"
        elif creators < 2:
            reason = f"every chunk comes from one creator ({loudest})"
        elif share > max_creator_share:
            reason = (
                f"{share:.0%} of the community's chunks are {loudest} — one voice with visitors"
            )
        units.append(
            SourceUnit(
                unit_id=f"community:{community_id}",
                kind="communities",
                title=", ".join(names.get(entity_id, entity_id) for entity_id in top),
                summary=" ".join(claim.text for claim in anchored[:4]),
                chunk_ids=chunk_ids,
                video_ids=videos,
                retained=len(anchored),
                creator_count=creators,
                top_creator=loudest,
                top_creator_share=share,
                eligible=not reason,
                reject_reason=reason,
            )
        )
    return sorted(units, key=lambda unit: (-unit.retained, unit.unit_id))


def eligible_units(units: Sequence[SourceUnit]) -> list[SourceUnit]:
    return [unit for unit in units if unit.eligible]


def _mentions(entity_id: str, claims: Sequence[_Claim]) -> int:
    return sum(1 for claim in claims if entity_id in claim.entity_ids)


def merge_units(
    raptor: Sequence[SourceUnit],
    communities: Sequence[SourceUnit],
    limit: int,
) -> list[SourceUnit]:
    """The two pools interleaved, best-first, up to the shared unit budget.

    Strict alternation rather than a re-ranking: the merged arm has to spend the
    *same* number of LLM calls as either single arm, so what it is really being
    asked is whether half of each pool beats all of one. A merge that got a
    larger budget would win the comparison by size.
    """
    raptor, communities = eligible_units(raptor), eligible_units(communities)
    merged: list[SourceUnit] = []
    for index in range(limit):
        pool = raptor if index % 2 == 0 else communities
        other = communities if index % 2 == 0 else raptor
        taken = {unit.unit_id for unit in merged}
        pick = next((unit for unit in pool if unit.unit_id not in taken), None)
        if pick is None:
            pick = next((unit for unit in other if unit.unit_id not in taken), None)
        if pick is None:
            break
        merged.append(pick)
    return merged


def select_units(
    arm: str,
    raptor: Sequence[SourceUnit],
    communities: Sequence[SourceUnit],
    limit: int,
) -> list[SourceUnit]:
    if arm == "raptor":
        return eligible_units(raptor)[:limit]
    if arm == "communities":
        return eligible_units(communities)[:limit]
    if arm == "merged":
        return merge_units(raptor, communities, limit)
    raise ValueError(f"Unknown pack arm: {arm!r}. Expected one of {', '.join(PACK_ARMS)}")


# ─── Excerpts ────────────────────────────────────────────────────────────────


def unit_excerpts(
    unit: SourceUnit,
    records: dict[str, dict[str, Any]],
    limit: int = 12,
) -> list[dict[str, Any]]:
    """Up to ``limit`` of a unit's chunks, spread across its videos.

    Round-robin by video rather than by rank. Taking the unit's chunks in order
    would hand the model twelve excerpts from whichever video dominates it, and
    the rubric written from that is one creator's advice wearing a pack's name —
    which is the failure the ``>= 2 creators`` gate exists to catch.
    """
    by_video: dict[str, list[dict[str, Any]]] = {}
    for chunk_id in unit.chunk_ids:
        record = records.get(chunk_id)
        if record is None:
            continue
        by_video.setdefault(str(record["video_id"]), []).append(record)
    for rows in by_video.values():
        rows.sort(key=lambda row: int(row.get("chunk_index") or 0))
    order = sorted(by_video, key=lambda video_id: (-len(by_video[video_id]), video_id))
    picked: list[dict[str, Any]] = []
    depth = 0
    while len(picked) < limit:
        added = False
        for video_id in order:
            if depth < len(by_video[video_id]):
                picked.append(by_video[video_id][depth])
                added = True
                if len(picked) >= limit:
                    break
        if not added:
            break
        depth += 1
    return picked


def format_excerpts(excerpts: Sequence[dict[str, Any]]) -> str:
    """Numbered excerpts labelled by creator. The number is the only handle.

    The model is never given a chunk id or asked for one — it answers with the
    number, and every other citation field is read off the record behind it.
    """
    blocks = []
    for index, excerpt in enumerate(excerpts, start=1):
        creator = excerpt.get("channel_name") or "Unknown creator"
        title = excerpt.get("title") or excerpt.get("video_id") or "untitled"
        text = str(excerpt.get("text") or "").strip().replace("\n", " ")
        blocks.append(f"[{index}] {creator} — {title}\n{text}")
    return "\n\n".join(blocks)


# ─── Rubric extraction ───────────────────────────────────────────────────────

RUBRIC_SYSTEM_PROMPT = """You turn expert video transcripts into review criteria.

You are given numbered EXCERPTS from several different creators, grouped because
they cover the same ground. Extract the checkable rules those creators apply —
rules a reviewer could hold one real {artifact} against and reach a verdict.

A rubric is not a summary. Test each one before you write it:
- Could a reviewer look at a single artifact and say pass or fail? If deciding
  needs taste and nothing else, drop it.
- Does it say what to do or what to look for, rather than what the creators
  talked about? "Creators emphasise quantifying your impact" is a summary and is
  wrong. "Every experience bullet carries a number - a percentage, a count, a
  currency figure or a duration" is a rubric.
- Would it still mean something applied to an artifact none of these creators
  has seen? If it only makes sense about one video's own example, drop it.

Fields per rubric:
- criterion: the rule. Imperative, one sentence, specific enough that an
  artifact can fail it.
- check: how a reviewer decides - the observable thing they count, look for or
  measure, with the threshold if the creators gave one. Never "assess whether".
- why: the consequence the creators say follows from getting it wrong.
- evidence: one or two entries. "excerpt" is the excerpt's NUMBER. "quote" is 6
  to 25 words copied EXACTLY from that excerpt, character for character, from
  one continuous run of its text. Do not tidy grammar, do not paraphrase, do not
  join words from two places.
- contested: true only when excerpts from two DIFFERENT creators disagree about
  the rule; then cite both sides.

Prefer rubrics whose evidence spans two creators.
Write at most {max_rubrics}. Three sharp rubrics beat eight vague ones.
Use only the excerpts. Never invent a number, a name or a threshold.

Return only JSON in this shape:
{{"rubrics":[{{"criterion":"...","check":"...","why":"...","contested":false,\
"evidence":[{{"excerpt":3,"quote":"..."}}]}}]}}"""


class RubricEvidence(BaseModel):
    """One quote, with every field but the quote derived server-side."""

    video_id: str
    chunk_id: str
    chunk_index: int
    start_seconds: float
    end_seconds: float
    #: The exact transcript span the model's quote matched — what the UI shows.
    quote: str
    #: What the model actually wrote, kept so a reviewer can see the drift.
    model_quote: str
    #: Where in the chunk the quote sits, interpolated by character offset. Used
    #: for the deep link only; ``start_seconds`` stays the chunk's own start,
    #: because that is the timestamp the provenance check resolves against.
    quote_start_seconds: float
    channel_name: str | None = None
    title: str | None = None
    source_url: str | None = None
    ratio: float = 0.0

    def youtube_url(self) -> str:
        start = max(0, int(self.quote_start_seconds) - 2)
        return f"https://www.youtube.com/watch?v={self.video_id}&t={start}s"


class Rubric(BaseModel):
    """One checkable criterion and the transcript words behind it."""

    rubric_id: str
    criterion: str
    check: str = ""
    why: str = ""
    contested: bool = False
    unit_id: str = ""
    unit_kind: str = ""
    unit_title: str = ""
    evidence: list[RubricEvidence] = Field(default_factory=list)

    @property
    def creators(self) -> list[str]:
        return sorted({e.channel_name or e.video_id for e in self.evidence})

    @property
    def videos(self) -> list[str]:
        return sorted({e.video_id for e in self.evidence})


class PackGap(BaseModel):
    """A source unit that produced no rubric, and why.

    The gate is that every global theme for a topic reaches a rubric **or a
    logged gap**: silence is the failure mode a coverage claim hides behind, so
    a unit that was selected and yielded nothing, and a unit that was relevant
    but lost to the budget, both leave a row here.

    The creator columns are the ones to read on a rejected unit. ``videos`` is
    the number a coverage claim overstates with — a theme can span four videos
    and still be 145 chunks of one podcast — so the count of distinct creators
    and the loudest one's share are carried beside it rather than left to be
    parsed out of ``reason``. They were passed by the builder and silently
    dropped for want of these three fields, which is why every gap row shipped
    so far says "one creator" in prose and nothing in data.
    """

    unit_id: str
    unit_kind: str
    unit_title: str
    videos: int = 0
    chunks: int = 0
    creator_count: int = 0
    top_creator: str = ""
    top_creator_share: float = 0.0
    reason: str = ""


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
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match is None:
            raise ValueError(f"Rubric response was not JSON: {content[:200]}")
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("Rubric response JSON must be an object")
    return value


class RubricExtractor:
    """One LLM call per source unit: excerpts in, candidate rubrics out."""

    def __init__(self, llm: ChatModel, model_name: str = "") -> None:
        self.llm = llm
        self.model_name = model_name

    def extract(
        self,
        excerpts: Sequence[dict[str, Any]],
        *,
        artifact: str,
        max_rubrics: int = 3,
    ) -> list[dict[str, Any]]:
        from langchain_core.messages import HumanMessage, SystemMessage

        prompt = RUBRIC_SYSTEM_PROMPT.format(artifact=artifact, max_rubrics=max_rubrics)
        response = self.llm.invoke(
            [SystemMessage(content=prompt), HumanMessage(content=format_excerpts(excerpts))]
        )
        payload = _json_object(str(getattr(response, "content", response) or ""))
        rows = payload.get("rubrics")
        return [row for row in (rows or []) if isinstance(row, dict)]


# ─── Quote snapping and citation reconciliation ──────────────────────────────


def normalize(text: str) -> list[str]:
    return _WORD.findall(str(text).lower())


def snap_quote(quote: str, text: str) -> tuple[str, float, int]:
    """The contiguous span of ``text`` a model's quote matched, verbatim.

    Returns ``(span, ratio, char_offset)``. The ratio is the share of the
    *quote*'s tokens found in order inside the chunk — one-sided on purpose,
    the same asymmetry :func:`src.evals.critique.quote_ratio` uses, because the
    question is whether the speaker said this and not whether it is all they
    said.

    Returning the span rather than the quote is what makes "verbatim" true
    instead of asserted. A model reliably drops a filler word or normalises
    "gonna"; rendering its version puts words in a creator's mouth, and this
    repo has already shipped one panel of transcript text that was not the
    transcript. A span more than :data:`MAX_SNAP_EXPANSION` times the quote's
    length is rejected instead, because that is a quote stitched from two places
    rather than a quotation of one.
    """
    needle = normalize(quote)
    tokens = [(match.group(0), match.start(), match.end()) for match in _TOKEN.finditer(text)]
    haystack = [token.lower().strip("'") or token.lower() for token, _, _ in tokens]
    haystack = [_WORD.findall(token)[0] if _WORD.findall(token) else "" for token in haystack]
    if not needle or not haystack:
        return "", 0.0, 0
    matcher = SequenceMatcher(a=needle, b=haystack, autojunk=False)
    blocks = [block for block in matcher.get_matching_blocks() if block.size]
    matched = sum(block.size for block in blocks)
    ratio = matched / len(needle)
    if not blocks:
        return "", ratio, 0
    first = blocks[0].b
    last = blocks[-1].b + blocks[-1].size - 1
    if (last - first + 1) > MAX_SNAP_EXPANSION * len(needle) + 4:
        return "", 0.0, 0
    start, end = tokens[first][1], tokens[last][2]
    return text[start:end], ratio, start


def reconcile_evidence(
    rows: Sequence[Any],
    excerpts: Sequence[dict[str, Any]],
) -> list[RubricEvidence]:
    """Model evidence rows turned into citations this project can stand behind.

    The model supplies an excerpt number and a quote and nothing else. Every
    other field — video, timestamps, chunk id, channel — comes off the excerpt
    record, which came out of the chunk store. A row naming an excerpt that was
    not shown, or a quote that does not appear in the excerpt it names, is
    dropped: an unverifiable citation is worse than no citation, because it
    renders identically to a real one.
    """
    found: list[RubricEvidence] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        number = row.get("excerpt")
        quote = str(row.get("quote") or "").strip()
        if not isinstance(number, (int, float)) or not quote:
            continue
        index = int(number) - 1
        if index < 0 or index >= len(excerpts):
            continue
        record = excerpts[index]
        text = str(record.get("text") or "")
        span, ratio, offset = snap_quote(quote, text)
        if ratio < QUOTE_MATCH_RATIO or not span:
            continue
        chunk_id = str(record["chunk_id"])
        key = (chunk_id, span)
        if key in seen:
            continue
        seen.add(key)
        start = float(record.get("start_seconds") or 0.0)
        end = float(record.get("end_seconds") or start)
        share = offset / len(text) if text else 0.0
        found.append(
            RubricEvidence(
                video_id=str(record["video_id"]),
                chunk_id=chunk_id,
                chunk_index=int(record.get("chunk_index") or 0),
                start_seconds=start,
                end_seconds=end,
                quote=span,
                model_quote=quote,
                quote_start_seconds=round(start + share * max(0.0, end - start), 2),
                channel_name=record.get("channel_name"),
                title=record.get("title"),
                source_url=record.get("source_url"),
                ratio=round(ratio, 4),
            )
        )
    return found


# ─── Dedupe ──────────────────────────────────────────────────────────────────

#: ``texts -> vectors``. The shipped MiniLM model in practice; the tests pass a
#: deterministic fake.
EmbedFn = Callable[[Sequence[str]], list[list[float]]]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return 0.0 if na == 0.0 or nb == 0.0 else dot / (na * nb)


def dedupe_rubrics(
    rubrics: Sequence[Rubric],
    embed: EmbedFn,
    *,
    threshold: float = DEDUPE_THRESHOLD,
) -> tuple[list[Rubric], list[Rubric]]:
    """Keep the first statement of each rule; return ``(kept, dropped)``.

    Order matters and is deliberate: units are visited best-first, so the
    surviving phrasing is the one written from the strongest unit. Two rubrics
    that are the same rule are not extra coverage — this is the arm-fairness
    lever, because the merged arm sees both pools and would otherwise bank its
    own overlap as a larger pack.
    """
    if not rubrics:
        return [], []
    vectors = embed([rubric.criterion for rubric in rubrics])
    kept: list[Rubric] = []
    kept_vectors: list[Sequence[float]] = []
    dropped: list[Rubric] = []
    for rubric, vector in zip(rubrics, vectors):
        if any(cosine(vector, other) >= threshold for other in kept_vectors):
            dropped.append(rubric)
            continue
        kept.append(rubric)
        kept_vectors.append(vector)
    return kept, dropped


# ─── The pack ────────────────────────────────────────────────────────────────


class PackProvenance(BaseModel):
    """What this pack was built from, recorded in the pack itself.

    Eval cells in this repo have already been compared across different
    corpora, and the local cross-encoder ranks differently on MPS than on CPU.
    A pack that does not carry its own inputs is a number nobody can reproduce,
    so the corpus digest, the theme artifact's timestamp, the graph cache's
    digest and both exclusion lists travel with the rubrics.
    """

    corpus_digest: str = ""
    chunk_count: int = 0
    video_count: int = 0
    theme_generated_at: str = ""
    theme_count: int = 0
    graph_cache_digest: str = ""
    graph_extractions: int = 0
    excluded_video_ids: list[str] = Field(default_factory=list)
    held_out_video_ids: list[str] = Field(default_factory=list)
    embedding_model: str = ""
    rubric_model: str = ""
    units_budget: int = 0
    excerpts_per_unit: int = 0
    max_rubrics_per_unit: int = 0
    routing_top_k: int = 0
    routing_min_score: float = 0.0
    community_min_entities: int = 0
    notes: list[str] = Field(default_factory=list)


class ExpertPack(BaseModel):
    """One topic, one arm: the shipped artifact the app renders."""

    version: int = 1
    topic: str
    name: str
    arm: str
    artifact: str = "document"
    routing_text: str = ""
    blurb: str = ""
    generated_at: str = ""
    provenance: PackProvenance = Field(default_factory=PackProvenance)
    members: list[PackMember] = Field(default_factory=list)
    units: list[SourceUnit] = Field(default_factory=list)
    rubrics: list[Rubric] = Field(default_factory=list)
    gaps: list[PackGap] = Field(default_factory=list)
    stats: dict[str, Any] = Field(default_factory=dict)


def corpus_digest(records: Sequence[dict[str, Any]]) -> str:
    """A stable fingerprint of the chunk corpus a pack was built from."""
    hasher = hashlib.sha256()
    for record in sorted(records, key=lambda row: str(row.get("chunk_id"))):
        hasher.update(str(record.get("chunk_id")).encode("utf-8"))
        hasher.update(b"\x00")
        hasher.update(str(record.get("text") or "").encode("utf-8"))
        hasher.update(b"\x00")
    return hasher.hexdigest()[:16]


def digest_of(values: Iterable[str]) -> str:
    hasher = hashlib.sha256()
    for value in sorted(values):
        hasher.update(value.encode("utf-8"))
        hasher.update(b"\x00")
    return hasher.hexdigest()[:16]


def pack_statistics(
    rubrics: Sequence[Rubric],
    members: Sequence[PackMember],
    units: Sequence[SourceUnit],
    gaps: Sequence[PackGap],
    excluded: Sequence[str] = (),
) -> dict[str, Any]:
    """The deterministic gate numbers, computed once and stored with the pack.

    ``multi_creator_share`` is the one to read first: a rubric backed by one
    creator is that creator's opinion, and a pack of them is a channel summary
    with extra steps.
    """
    citations = [evidence for rubric in rubrics for evidence in rubric.evidence]
    blocked = set(excluded)
    multi = [rubric for rubric in rubrics if len(rubric.creators) >= 2]
    checked = [rubric for rubric in rubrics if rubric.check.strip()]
    return {
        "rubrics": len(rubrics),
        "citations": len(citations),
        "rubrics_with_check": len(checked),
        "multi_creator_rubrics": len(multi),
        "multi_creator_share": round(len(multi) / len(rubrics), 4) if rubrics else 0.0,
        "contested_rubrics": sum(1 for rubric in rubrics if rubric.contested),
        "creators": len({e.channel_name or e.video_id for e in citations}),
        "videos_cited": len({e.video_id for e in citations}),
        "members_included": sum(1 for member in members if member.included),
        "members_routed": sum(1 for member in members if member.routed),
        "members_overridden": sum(1 for member in members if member.override is not None),
        "units_used": len(units),
        "units_by_kind": {
            kind: sum(1 for unit in units if unit.kind == kind)
            for kind in sorted({unit.kind for unit in units})
        },
        "gaps": len(gaps),
        "excluded_video_citations": sum(1 for e in citations if e.video_id in blocked),
        "excluded_video_members": sum(
            1 for member in members if member.included and member.video_id in blocked
        ),
    }


# ─── Store ───────────────────────────────────────────────────────────────────


class PackStore:
    """``experts/`` on disk. Files are the source of truth; the app renders them.

    A directory of JSON rather than a database because a rubric pack is
    something a reviewer should be able to read in a diff — "which rules changed
    when the corpus grew" is the question this artifact exists to answer, and it
    is a question ``git log`` answers for free on files.
    """

    def __init__(self, root: Path | str = DEFAULT_PACKS_DIR) -> None:
        self.root = Path(root)

    def catalog(self) -> PackCatalog:
        return load_catalog(self.root)

    def topic_dir(self, topic: str) -> Path:
        return self.root / topic

    def arm_path(self, topic: str, arm: str) -> Path:
        return self.topic_dir(topic) / f"{arm}.json"

    def pack_path(self, topic: str) -> Path:
        return self.topic_dir(topic) / "pack.json"

    def manifest_path(self, topic: str) -> Path:
        return self.topic_dir(topic) / "manifest.json"

    def ablation_path(self) -> Path:
        return self.root / "ablation.json"

    def load_pack(self, topic: str, arm: str | None = None) -> ExpertPack | None:
        path = self.pack_path(topic) if arm is None else self.arm_path(topic, arm)
        if not path.is_file():
            return None
        return ExpertPack.model_validate_json(path.read_text(encoding="utf-8"))

    def save_pack(self, pack: ExpertPack, *, shipped: bool = False) -> Path:
        path = self.pack_path(pack.topic) if shipped else self.arm_path(pack.topic, pack.arm)
        return _write_json(path, pack.model_dump(mode="json"))

    def load_manifest(self, topic: str) -> dict[str, Any]:
        path = self.manifest_path(topic)
        if not path.is_file():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
        return data if isinstance(data, dict) else {}

    def save_manifest(self, topic: str, manifest: dict[str, Any]) -> Path:
        return _write_json(self.manifest_path(topic), manifest)

    def overrides(self, topic: str) -> dict[str, bool]:
        stored = self.load_manifest(topic).get("overrides") or {}
        return {
            str(video_id): bool(value)
            for video_id, value in stored.items()
            if isinstance(value, bool)
        }

    def set_override(self, topic: str, video_id: str, included: bool | None) -> dict[str, bool]:
        """Pin a video in or out of a pack, or hand it back to the router.

        Persisted to the manifest rather than to the pack, because a pack is
        rebuilt from the corpus and a human's decision must survive that. The
        rebuild reads these back and reapplies them.
        """
        manifest = self.load_manifest(topic)
        overrides = {
            str(key): bool(value)
            for key, value in (manifest.get("overrides") or {}).items()
            if isinstance(value, bool)
        }
        if included is None:
            overrides.pop(video_id, None)
        else:
            overrides[video_id] = included
        manifest["topic"] = topic
        manifest["overrides"] = dict(sorted(overrides.items()))
        manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
        self.save_manifest(topic, manifest)
        return manifest["overrides"]

    def load_ablation(self) -> dict[str, Any] | None:
        path = self.ablation_path()
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        return data if isinstance(data, dict) else None

    def save_ablation(self, run: dict[str, Any]) -> Path:
        return _write_json(self.ablation_path(), run)


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return path


# ─── Scoring bridge ──────────────────────────────────────────────────────────


def rubrics_as_findings(pack: ExpertPack) -> list[Any]:
    """A pack rendered as :class:`src.evals.critique.Finding` rows.

    The held-out harness scores *criteria a system applied*, and a rubric is
    exactly that — so a pack can be put through the same scorer as a live
    critique without a second instrument. ``detail`` carries the check and the
    rationale because the matcher reads it; nothing is scored on it.
    """
    from src.evals.critique import Citation, Finding

    findings: list[Any] = []
    for rubric in pack.rubrics:
        findings.append(
            Finding(
                id=rubric.rubric_id,
                criterion=rubric.criterion,
                detail=" ".join(part for part in (rubric.check, rubric.why) if part).strip(),
                citations=tuple(
                    Citation(
                        video_id=evidence.video_id,
                        start_seconds=evidence.start_seconds,
                        quote=evidence.quote,
                    )
                    for evidence in rubric.evidence
                ),
                contested=rubric.contested,
            )
        )
    return findings

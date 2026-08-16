"""Cross-video themes: RAPTOR level 2 over the stored chunk embeddings.

The corpus already has two levels of a RAPTOR-style hierarchy:

* **level 0** — chunks (:mod:`src.rag.storage`),
* **level 1** — one summary per video (:mod:`src.rag.summaries`).

Level 1 can never say anything a single video does not, because its unit *is*
the video. This module adds **level 2**: clusters of chunks drawn from the whole
corpus at once, so a cluster can gather the same argument from creators who have
never heard of each other, and one LLM call per cluster turns it into a theme.

Why not reuse the Leiden pass in :mod:`src.rag.communities`? That clusters
*entities* in Neo4j by co-mention, not chunk embeddings — a different graph, a
different unit, and one that needs the whole GraphRAG extraction to have run
first. It is also fragmented in practice (hundreds of 1-2 entity communities),
so it cannot be the raptor arm of a raptor-vs-communities comparison. Working at
level 0 has a second, concrete advantage here: three property videos have chunks
but no per-video summary, so they are invisible to level 1 and still reachable
by this layer.

Clustering is free — the vectors are already stored — and follows RAPTOR:
reduce with PCA, fit a Gaussian mixture, pick the component count by BIC, and
assign softly so one chunk can belong to two themes. It is done twice: once
globally, then again inside each global cluster, which is what turns a handful
of continent-sized blobs into themes at a readable grain.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Protocol, Sequence

from pydantic import BaseModel, Field

THEME_SYSTEM_PROMPT = """You name themes that run across a corpus of YouTube transcripts.

You are given excerpts drawn from SEVERAL DIFFERENT videos by different
creators that an embedding clustering put together. Your job is to state what
these creators are collectively saying, not to summarise any one video.

Rules:
- The title is a claim or a topic several of these speakers share, 3-9 words,
  no video title, no channel name, no "and more".
- The summary is 2-4 sentences: the shared position, the concrete specifics
  (tools, numbers, scheme names, tactics) that recur, and any disagreement
  between the speakers. Name a speaker's stance only when they differ.
- Never write "this video", "the speaker", or "the transcript" — there are
  many. Write about what the creators say.
- Use only the excerpts. Do not invent numbers or names.

Return only JSON in this shape:
{"title":"short claim","summary":"2-4 sentences"}"""


class ChatModel(Protocol):
    def invoke(self, messages: list) -> object: ...


# ─── Clustering ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ThemeClusterConfig:
    """Knobs for the two-pass RAPTOR clustering.

    The defaults were chosen against this corpus (1329 chunks / 53 videos) by
    sweeping covariance type and dimensionality and reading the resulting
    clusters: ``full`` covariance globally finds the eight broad subject areas,
    and ``diag`` locally splits them further, because a diagonal mixture spends
    far fewer parameters per component and so BIC lets it keep more of them.
    """

    #: PCA width for the global pass. RAPTOR reduces before fitting because a
    #: full-covariance mixture over 384 raw dimensions is mostly parameters.
    global_dims: int = 10
    #: PCA width inside a global cluster, where there is less to separate.
    local_dims: int = 8
    global_covariance: str = "full"
    local_covariance: str = "diag"
    max_global_clusters: int = 30
    #: Chunks per theme the local pass aims for; also the split threshold —
    #: a global cluster at or below twice this is already theme-sized.
    #:
    #: 50, not 30. At 30 this corpus split harder than it warranted: six
    #: single-video themes, three separate "tailor your résumé" themes, and
    #: **16 of 30 themes where one channel supplied 80%+ of the members** —
    #: cross-video by chunk count, one voice in substance. Raising it to 50
    #: gives 20 themes, 2 single-video, 7 above that 80% line, and drops the
    #: median top-channel share from 0.85 to 0.61, while still covering all 53
    #: videos. Going further (80) reaches 13 themes and a 0.46 median but the
    #: layer stops being browsable and starts being a table of contents.
    target_cluster_size: int = 50
    #: Soft-assignment floor. A chunk joins every component that gives it at
    #: least this posterior, so one chunk can sit in two themes (RAPTOR's own
    #: threshold is 0.1).
    membership_threshold: float = 0.10
    #: Below this a cluster is noise, not a theme worth an LLM call.
    min_cluster_size: int = 4
    random_state: int = 0


def _reduce(matrix: Any, dims: int, random_state: int) -> Any:
    from sklearn.decomposition import PCA

    width = min(dims, matrix.shape[0] - 1, matrix.shape[1])
    if width < 1:
        return matrix
    # svd_solver="full" is exact and deterministic; the randomized solver would
    # make the same corpus cluster differently between runs.
    pca = PCA(n_components=width, random_state=random_state, svd_solver="full")
    return pca.fit_transform(matrix)


def _fit_by_bic(matrix: Any, low: int, high: int, covariance: str, random_state: int) -> Any:
    """The mixture with the lowest BIC over ``low..high`` components."""
    from sklearn.mixture import GaussianMixture

    best = None
    best_bic = math.inf
    for count in range(low, high + 1):
        if count >= matrix.shape[0]:
            break
        model = GaussianMixture(
            n_components=count,
            covariance_type=covariance,
            random_state=random_state,
            reg_covar=1e-4,
            max_iter=200,
        )
        model.fit(matrix)
        bic = model.bic(matrix)
        if bic < best_bic:
            best, best_bic = model, bic
    return best


def _soft_columns(probabilities: Any, threshold: float) -> list[list[int]]:
    """Row indices assigned to each mixture component, one list per component."""
    return [
        [row for row in range(probabilities.shape[0]) if probabilities[row][column] >= threshold]
        for column in range(probabilities.shape[1])
    ]


def cluster_embeddings(
    vectors: Sequence[Sequence[float]],
    config: ThemeClusterConfig | None = None,
) -> list[list[tuple[int, float]]]:
    """Two-pass soft clustering over unit-normalised chunk vectors.

    Returns one list per cluster of ``(row index, membership probability)``,
    ordered by descending probability. Clusters overlap by design.
    """
    import numpy as np

    settings = config or ThemeClusterConfig()
    if not vectors:
        return []
    matrix = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    # Cosine is the metric the retrieval side uses; normalising first makes the
    # Euclidean mixture agree with it.
    matrix = matrix / np.where(norms == 0, 1.0, norms)

    reduced = _reduce(matrix, settings.global_dims, settings.random_state)
    top = _fit_by_bic(
        reduced,
        2,
        settings.max_global_clusters,
        settings.global_covariance,
        settings.random_state,
    )
    if top is None:
        return []
    global_probabilities = top.predict_proba(reduced)

    clusters: list[list[tuple[int, float]]] = []
    for column, rows in enumerate(
        _soft_columns(global_probabilities, settings.membership_threshold)
    ):
        if len(rows) <= settings.target_cluster_size * 2:
            if len(rows) >= settings.min_cluster_size:
                clusters.append([(row, float(global_probabilities[row][column])) for row in rows])
            continue
        # Second pass, RAPTOR's "local" step: the global mixture separates
        # subject areas, and re-fitting inside one of them is what separates
        # "sharding" from "caching" within distributed systems.
        local_reduced = _reduce(matrix[rows], settings.local_dims, settings.random_state)
        local = _fit_by_bic(
            local_reduced,
            2,
            max(2, len(rows) // settings.target_cluster_size),
            settings.local_covariance,
            settings.random_state,
        )
        if local is None:
            clusters.append([(row, float(global_probabilities[row][column])) for row in rows])
            continue
        local_probabilities = local.predict_proba(local_reduced)
        for local_column, offsets in enumerate(
            _soft_columns(local_probabilities, settings.membership_threshold)
        ):
            if len(offsets) < settings.min_cluster_size:
                continue
            clusters.append(
                [
                    (rows[offset], float(local_probabilities[offset][local_column]))
                    for offset in offsets
                ]
            )
    return [sorted(cluster, key=lambda item: (-item[1], item[0])) for cluster in clusters]


# ─── Topic tagging (deterministic, for the purity check) ─────────────────────

#: Declaration order is only the tie-break; the winner is whichever domain
#: matches the most distinct terms. Single-keyword precedence does not survive
#: this corpus — "System Design Interview - Distributed Message Queue" contains
#: "interview" and "Machine Learning Interviews" is a résumé video — but
#: counting terms separates them (three architecture words beat one, one
#: ML word loses to résumé plus interview).
DOMAIN_TERMS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "property",
        (
            "propert",
            "real estate",
            "housing market",
            "negative gearing",
            "gold coast",
            "suburb",
            "rentvest",
            "land tax",
            "budget",
        ),
    ),
    (
        "system_design",
        (
            "system design",
            "architecture",
            "monolith",
            "microservice",
            "shard",
            "caching",
            "cache",
            "distributed",
            "database",
            "transaction",
            r"\brest\b",
            "queue",
            "coupling",
            "hexagonal",
            "domain.centric",
            "domain.driven",
            "ports & adapters",
            "ports and adapters",
            "fastapi",
        ),
    ),
    (
        "job_search",
        (
            "resume",
            "r[eé]sum[eé]",
            r"\bcv\b",
            "linkedin",
            "recruiter",
            "interview",
            "hiring",
            "hired",
            "job search",
            "get a job",
            "career",
            r"\bats\b",
            "cover letter",
        ),
    ),
    (
        "ai_engineering",
        (
            "agentic",
            "ai engineer",
            "mlops",
            "llmops",
            "llm",
            r"\brag\b",
            "prompt",
            "automation stack",
            "data scientist",
            "machine learning",
            "ai coding",
            "claude",
            "cursor",
        ),
    ),
    ("ui_design", (r"\bui\b", r"\bux\b", "design", "premium", "illustration")),
)

DOMAIN_OTHER = "other"


def domain_scores(title: str | None, channel_name: str | None = None) -> dict[str, int]:
    """How many distinct terms of each domain a video's title and channel hit."""
    haystack = f"{title or ''} {channel_name or ''}".lower()
    return {
        domain: sum(1 for term in terms if re.search(term, haystack))
        for domain, terms in DOMAIN_TERMS
    }


def video_domain(title: str | None, channel_name: str | None = None) -> str:
    """A coarse subject tag for one video, from its title and channel.

    Deliberately rule-based rather than model-based: the purity check it feeds
    ("does a job-search theme quietly contain Australian property tax?") is a
    gate, and a gate scored by an LLM is a gate that moves.

    A known limit: the unit is the *video*, so every chunk of a video carries
    its video's tag. A résumé video that spends four minutes on Australian tax
    would not register as contamination here. It is a per-video signal used to
    audit clusters, not a per-chunk classifier.
    """
    scores = domain_scores(title, channel_name)
    order = {domain: index for index, (domain, _) in enumerate(DOMAIN_TERMS)}
    best = max(scores, key=lambda domain: (scores[domain], -order[domain]))
    return best if scores[best] else DOMAIN_OTHER


# ─── Records ─────────────────────────────────────────────────────────────────


class ThemeMember(BaseModel):
    chunk_id: str
    video_id: str
    chunk_index: int
    probability: float


class ThemeVideo(BaseModel):
    video_id: str
    title: str | None = None
    channel_name: str | None = None
    member_count: int
    domain: str = DOMAIN_OTHER


class Theme(BaseModel):
    theme_id: str
    title: str
    summary: str
    member_count: int
    video_count: int
    channel_count: int
    #: False when every member came from one video — the case where this layer
    #: has added nothing over the per-video summaries that already exist.
    cross_video: bool
    #: The creator who supplied the most members, and their share of them.
    #:
    #: ``cross_video`` counts videos; this counts *voices*, and the two come
    #: apart hard. A theme can span four videos and still be 99% one podcast
    #: with three visitors contributing a chunk each — cross-video by the
    #: letter, one creator in substance. Published on the theme row so the
    #: reader sees the imbalance without opening the member list, because the
    #: summary itself will not show it: ``representative_excerpts`` samples
    #: round-robin across videos, so a dominant creator holding 55% of the
    #: members supplies only ~17% of the excerpts the summariser reads.
    top_channel: str | None = None
    top_channel_share: float = 0.0
    #: The domain the majority of members come from, and the share of members
    #: from property videos. Together these answer "is this job-search theme
    #: actually clean?" without anyone eyeballing the members.
    domain: str
    domain_mix: dict[str, float] = Field(default_factory=dict)
    property_share: float = 0.0
    #: The largest share held by a domain *other* than this theme's own — the
    #: general form of the property check, so contamination is visible whatever
    #: the theme is about.
    off_domain_share: float = 0.0
    videos: list[ThemeVideo] = Field(default_factory=list)
    members: list[ThemeMember] = Field(default_factory=list)


class ThemeIndex(BaseModel):
    version: int = 1
    generated_at: str
    embedding_model: str = ""
    summary_model: str = ""
    chunk_collection: str = ""
    config: dict[str, Any] = Field(default_factory=dict)
    stats: dict[str, Any] = Field(default_factory=dict)
    themes: list[Theme] = Field(default_factory=list)


class ThemeStore:
    """The theme layer on disk: one JSON artifact, wholly derived.

    A file rather than a Chroma collection because nothing queries themes by
    vector — the UI lists them and drills into members — and because a plain
    file can be read by the API without opening the embedding stack at all.
    Members are stored as ids only; their text and timestamps are hydrated from
    the chunk collection on read, so re-chunking can never leave stale
    transcript text sitting in this file.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def exists(self) -> bool:
        return self.path.is_file()

    def load(self) -> ThemeIndex | None:
        if not self.path.is_file():
            return None
        return ThemeIndex.model_validate_json(self.path.read_text(encoding="utf-8"))

    def save(self, index: ThemeIndex) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(index.model_dump(mode="json"), indent=2, sort_keys=False) + "\n",
            encoding="utf-8",
        )
        return self.path


# ─── Summarisation ───────────────────────────────────────────────────────────


class ThemeSummarizer:
    """One LLM call per cluster: excerpts in, title + summary out."""

    def __init__(self, llm: ChatModel, model_name: str = "") -> None:
        self.llm = llm
        self.model_name = model_name

    def summarize(self, excerpts: list[dict[str, Any]]) -> tuple[str, str]:
        from langchain_core.messages import HumanMessage, SystemMessage

        response = self.llm.invoke(
            [
                SystemMessage(content=THEME_SYSTEM_PROMPT),
                HumanMessage(content=format_excerpts(excerpts)),
            ]
        )
        content = str(getattr(response, "content", response) or "")
        data = _json_object(content)
        title = str(data.get("title", "")).strip()
        summary = str(data.get("summary", "")).strip()
        if not title or not summary:
            raise ValueError("Theme summary LLM returned an empty title or summary")
        return title, summary


def format_excerpts(excerpts: list[dict[str, Any]], max_chars: int = 900) -> str:
    """Excerpts labelled by creator, so the model can see it is many voices."""
    lines = []
    for excerpt in excerpts:
        creator = excerpt.get("channel_name") or "Unknown creator"
        title = excerpt.get("title") or excerpt.get("video_id") or "untitled"
        text = str(excerpt.get("text") or "").strip().replace("\n", " ")
        if len(text) > max_chars:
            text = text[:max_chars].rstrip() + "…"
        lines.append(f"[{creator} — {title}]\n{text}")
    return "\n\n".join(lines)


def representative_excerpts(
    members: list[tuple[int, float]],
    records: Sequence[dict[str, Any]],
    limit: int = 12,
) -> list[dict[str, Any]]:
    """Up to ``limit`` members chosen to spread across videos, not to rank.

    Taking the highest-probability chunks outright would hand the summariser
    twelve excerpts from whichever video dominates the cluster, and it would
    then write that video's outline — exactly the failure this layer exists to
    avoid. So it takes each video's best chunk in turn before anyone's second.
    """
    by_video: dict[str, list[tuple[int, float]]] = {}
    for row, probability in members:
        by_video.setdefault(str(records[row]["video_id"]), []).append((row, probability))
    order = sorted(
        by_video,
        key=lambda video_id: (-len(by_video[video_id]), video_id),
    )
    picked: list[int] = []
    depth = 0
    while len(picked) < limit:
        added = False
        for video_id in order:
            if depth < len(by_video[video_id]):
                picked.append(by_video[video_id][depth][0])
                added = True
                if len(picked) >= limit:
                    break
        if not added:
            break
        depth += 1
    return [dict(records[row]) for row in picked]


# ─── Build ───────────────────────────────────────────────────────────────────


def sort_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Chunk records in a fixed order, so clustering is reproducible.

    Chroma returns rows in its own storage order, which is not a promise; PCA
    and a Gaussian mixture are both order-sensitive enough that an unsorted
    input would give a different theme layer on a re-run of the same corpus.
    """
    return sorted(
        records,
        key=lambda record: (str(record.get("video_id", "")), int(record.get("chunk_index", 0))),
    )


def build_themes(
    records: Sequence[dict[str, Any]],
    summarizer: ThemeSummarizer | None = None,
    *,
    config: ThemeClusterConfig | None = None,
    embedding_model: str = "",
    chunk_collection: str = "",
    excerpts_per_theme: int = 12,
    on_progress: Any = None,
) -> ThemeIndex:
    """Cluster every stored chunk embedding, then summarise each cluster.

    ``records`` are :func:`src.api.corpus.load_chunk_embeddings` rows. Without a
    ``summarizer`` the clusters are still built and measured — the gate numbers
    are deterministic and cost nothing, so they can be checked before spending
    one LLM call per theme.
    """
    settings = config or ThemeClusterConfig()
    progress = on_progress or (lambda _message: None)
    ordered = sort_records(records)
    clusters = cluster_embeddings([record["embedding"] for record in ordered], settings)
    progress(f"clustered {len(ordered)} chunks into {len(clusters)} candidate themes")

    themes: list[Theme] = []
    for number, members in enumerate(_ordered_clusters(clusters, ordered)):
        videos = _theme_videos(members, ordered)
        channels = {
            str(ordered[row].get("channel_name") or ordered[row].get("channel_id") or "")
            for row, _ in members
        }
        mix = _domain_mix(videos)
        title, summary = f"Theme {number + 1}", ""
        if summarizer is not None:
            excerpts = representative_excerpts(members, ordered, excerpts_per_theme)
            title, summary = summarizer.summarize(excerpts)
            progress(
                f"theme {number + 1}/{len(clusters)}: {len(members)} chunks, "
                f"{len(videos)} videos → {title}"
            )
        themes.append(
            Theme(
                theme_id=f"theme:{number}",
                title=title,
                summary=summary,
                member_count=len(members),
                video_count=len(videos),
                channel_count=len(channels),
                cross_video=len(videos) >= 2,
                domain=_dominant_domain(mix),
                domain_mix=mix,
                property_share=round(mix.get("property", 0.0), 4),
                videos=videos,
                members=[
                    ThemeMember(
                        chunk_id=str(ordered[row]["chunk_id"]),
                        video_id=str(ordered[row]["video_id"]),
                        chunk_index=int(ordered[row]["chunk_index"]),
                        probability=round(probability, 4),
                    )
                    for row, probability in members
                ],
            )
        )

    return ThemeIndex(
        generated_at=datetime.now(timezone.utc).isoformat(),
        embedding_model=embedding_model,
        summary_model=summarizer.model_name if summarizer is not None else "",
        chunk_collection=chunk_collection,
        config=asdict(settings),
        stats=theme_statistics(themes, len(ordered)),
        themes=themes,
    )


def _ordered_clusters(
    clusters: list[list[tuple[int, float]]],
    records: Sequence[dict[str, Any]],
) -> list[list[tuple[int, float]]]:
    """Cross-video clusters first, then by size — the UI's default reading order.

    Themes that span videos are the whole point of the layer, so a
    single-video cluster never occupies the top of the list even when it is
    large.
    """

    def key(members: list[tuple[int, float]]) -> tuple[int, int, str]:
        videos = {str(records[row]["video_id"]) for row, _ in members}
        return (
            0 if len(videos) >= 2 else 1,
            -len(members),
            str(records[members[0][0]]["chunk_id"]),
        )

    return sorted(clusters, key=key)


def _theme_videos(
    members: list[tuple[int, float]],
    records: Sequence[dict[str, Any]],
) -> list[ThemeVideo]:
    grouped: dict[str, ThemeVideo] = {}
    for row, _ in members:
        record = records[row]
        video_id = str(record["video_id"])
        existing = grouped.get(video_id)
        if existing is None:
            grouped[video_id] = ThemeVideo(
                video_id=video_id,
                title=record.get("title") or None,
                channel_name=record.get("channel_name") or None,
                member_count=1,
                domain=video_domain(record.get("title"), record.get("channel_name")),
            )
        else:
            existing.member_count += 1
    return sorted(grouped.values(), key=lambda video: (-video.member_count, video.video_id))


def _domain_mix(videos: list[ThemeVideo]) -> dict[str, float]:
    total = sum(video.member_count for video in videos) or 1
    tally: dict[str, int] = {}
    for video in videos:
        tally[video.domain] = tally.get(video.domain, 0) + video.member_count
    return {
        domain: round(count / total, 4)
        for domain, count in sorted(tally.items(), key=lambda item: (-item[1], item[0]))
    }


def _dominant_domain(mix: dict[str, float]) -> str:
    if not mix:
        return DOMAIN_OTHER
    return max(mix, key=lambda domain: (mix[domain], domain))


def theme_statistics(themes: list[Theme], chunk_count: int = 0) -> dict[str, Any]:
    """The deterministic gate numbers, computed once and stored with the index."""
    distribution: dict[str, int] = {}
    for theme in themes:
        key = str(theme.video_count)
        distribution[key] = distribution.get(key, 0) + 1
    covered = {video.video_id for theme in themes for video in theme.videos}
    impure = [
        theme.theme_id
        for theme in themes
        if theme.domain == "job_search" and theme.property_share > 0.20
    ]
    return {
        "themes": len(themes),
        "cross_video_themes": sum(1 for theme in themes if theme.cross_video),
        "single_video_themes": sum(1 for theme in themes if not theme.cross_video),
        "video_count_distribution": dict(
            sorted(distribution.items(), key=lambda item: int(item[0]))
        ),
        "max_videos_in_a_theme": max((theme.video_count for theme in themes), default=0),
        "chunks_clustered": chunk_count,
        "videos_covered": len(covered),
        "job_search_themes": sum(1 for theme in themes if theme.domain == "job_search"),
        "impure_job_search_themes": impure,
        "max_property_share_in_job_search_theme": round(
            max(
                (theme.property_share for theme in themes if theme.domain == "job_search"),
                default=0.0,
            ),
            4,
        ),
    }


# ─── Intruder probe ──────────────────────────────────────────────────────────


def build_intruder_probe(
    themes: list[Theme],
    theme_index: int,
    *,
    options: int = 5,
    seed: int = 0,
) -> dict[str, Any]:
    """One coherence trial: this theme's members plus one chunk from another.

    The topic-model intrusion test, applied to themes. A theme that is a real
    shared claim makes the outsider obvious; a theme that is just "assorted
    talking" hides it. Returns chunk ids only — the caller hydrates the text,
    so the probe itself never depends on the chunk store.
    """
    import random

    rng = random.Random(f"{seed}:{theme_index}")
    theme = themes[theme_index]
    own = [member.chunk_id for member in theme.members]
    mine = set(own)
    pool = [
        member.chunk_id
        for index, other in enumerate(themes)
        if index != theme_index
        for member in other.members
        if member.chunk_id not in mine
    ]
    if not pool or len(own) < options - 1:
        raise ValueError("Not enough material for an intruder probe")
    intruder = rng.choice(sorted(set(pool)))
    kept = rng.sample(own[: max(options * 3, options)], options - 1)
    shuffled = [*kept, intruder]
    rng.shuffle(shuffled)
    return {
        "theme_id": theme.theme_id,
        "title": theme.title,
        "chunk_ids": shuffled,
        "intruder_chunk_id": intruder,
    }


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
        raise ValueError(f"LLM response was not valid JSON: {content}") from exc
    if not isinstance(value, dict):
        raise ValueError("LLM response JSON must be an object")
    return value

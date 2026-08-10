"""Persistent, config-aware cache for one (engine, golden question) eval cell.

Judging an answer is the expensive part of a matrix run — several RAGAS
sub-calls per metric, times every engine, times every question. Without this,
adding one new engine variant to ``eval-matrix`` re-scores every *existing*
engine from scratch too, burning time and tokens on cells that have not
changed. This cache remembers a cell once it is scored, so only genuinely new
or genuinely changed cells cost anything on the next run.

Cache identity is a fingerprint of everything that could change the score:
the question and its reference answer (editing the golden set invalidates a
cell), **the corpus that was retrieved from** (see :func:`corpus_digest` — a
RAG score is meaningless without it, and an ingestion mid-sweep is what makes
that concrete), and the answering + judging configuration — swap the model, the
retrieval mode, ``top_k``, the judge itself, or a setting the answering engine
reads (its recursion budget, its iteration cap), and the fingerprint changes,
so the old score is never mistaken for still being valid. Settings that
cannot change what got scored (Neo4j's password, cache directories) are
deliberately excluded from the fingerprint, and the engine-specific ones are
scoped to the engines that read them — see :func:`_engine_material`. This
mirrors
:mod:`src.rag.graph_extract`'s chunk-hash cache for the same reason: cache
identity should track only what the cached thing actually depends on.

One JSON file per cell, keyed by the fingerprint, not one growing log: a kill
mid-write corrupts at most the cell being written, never the whole cache, and
every cell is independently inspectable and deletable. A cached cell that
recorded an error is never reused — exactly like the chunk-extraction
cache, a failure should retry on the next run rather than being pinned.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

from src.config import Settings
from src.evals.critique import EXCLUSION_VERSION
from src.evals.golden import GoldenEntry

DEFAULT_CACHE_DIR = Path(".yt-agent/eval_cache")

#: Settings outside the shared material above that change an answer, mapped to
#: the setups whose engine actually reads them: the recursion budget only
#: shapes ``rag_llm_recursive``'s follow-up fan-out, the iteration cap only
#: bounds ``rag_agent``'s ReAct loop, and the summary pre-filter's thresholds are
#: threaded through the request by every vector engine (``graph_rag`` calls
#: ``get_context`` without them) — ``rag_llm_filtered`` is the one that always
#: switches it on, so its cells are the ones those thresholds actually move.
#: Retrieval breadth is read by every engine that retrieves through the shared
#: context provider.
_VECTOR_SETUPS = (
    "rag_llm",
    "rag_llm_recursive",
    "rag_agent",
    "rag_llm_hyde",
    "rag_llm_contextual",
    "rag_llm_filtered",
)

_ENGINE_SETTINGS: dict[str, tuple[str, ...]] = {
    "retrieval_candidates": (*_VECTOR_SETUPS, "graph_rag"),
    "transcript_filter_top_k": _VECTOR_SETUPS,
    "transcript_filter_min_score": _VECTOR_SETUPS,
    "rag_max_depth": ("rag_llm_recursive",),
    "rag_max_followups": ("rag_llm_recursive",),
    "rag_followup_top_k": ("rag_llm_recursive",),
    "rag_novelty_min_chunks": ("rag_llm_recursive",),
    "rag_max_total_followups": ("rag_llm_recursive",),
    "rag_agent_max_iterations": ("rag_agent",),
    # Which parallel index the contextual variant reads. Note the boundary this
    # shares with graph_rag's ``neo4j_uri``: it identifies the *store*, not its
    # contents, so re-running index-contextual with a changed situating prompt
    # needs ``eval-matrix --refresh`` to be rescored.
    "contextual_chunk_collection": ("rag_llm_contextual",),
}

_SETTINGS_DEFAULTS: dict[str, Any] = {
    field.name: field.default
    for field in dataclasses.fields(Settings)
    if field.default is not dataclasses.MISSING
}


def corpus_digest(video_ids: Iterable[str], chunk_count: int) -> str:
    """A short digest of the corpus an engine retrieved from.

    The video-id *set* plus the chunk count: the first catches ingestion and
    deletion, the second catches a re-chunk of the same videos, and neither
    needs the chunk text. Cheap enough to compute once per run from one bulk
    metadata read, which is why :func:`cell_fingerprint` takes the digest rather
    than the store — a fingerprint is computed per cell and must not walk the
    corpus 140 times.
    """
    ids = ",".join(sorted(dict.fromkeys(video_ids)))
    payload = f"{ids}|{chunk_count}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def corpus_digest_for(chunk_store: Any) -> str:
    """:func:`corpus_digest` for a live chunk store, in one metadata read."""
    result = chunk_store.collection.get(include=["metadatas"])
    metadatas = result.get("metadatas") or []
    return corpus_digest((str(meta.get("video_id")) for meta in metadatas if meta), len(metadatas))


def _engine_material(setup: str, settings: Settings) -> dict[str, Any]:
    """The engine-specific half of the fingerprint material.

    Two rules, both load-bearing. A field is included only for the setups that
    read it, so tuning ``rag_agent``'s loop does not throw away every
    ``rag_llm`` cell. And a field is included only when it *deviates* from its
    declared default, so cells scored before a field was tracked stay valid at
    the shipped configuration while any deviation from it — the case the cache
    would otherwise answer with a stale score — changes the fingerprint. That
    is a fingerprint, not a config record: the run's own ``config`` block is
    what documents the settings a run used.
    """
    material: dict[str, Any] = {}
    for name, setups in _ENGINE_SETTINGS.items():
        if setup not in setups:
            continue
        default = _SETTINGS_DEFAULTS[name]
        value = getattr(settings, name, default)
        if value != default:
            material[name] = value
    return material


def cell_fingerprint(
    setup: str,
    entry: GoldenEntry,
    settings: Settings,
    *,
    top_k: int | None = None,
    judge_model: str | None = None,
    judge_samples: int | None = None,
    ragas_version: str | None = None,
    reference_scored: bool = False,
    exclude_video_ids: Sequence[str] | None = None,
    corpus: str | None = None,
) -> str:
    """A hash identifying everything that determines this cell's score.

    Two calls that produce the same fingerprint asked the same question of
    the same engine configuration **over the same corpus**, and scored it with
    the same judge configuration — so a cell cached under one is always safe to
    reuse for the other, with no separate invalidation logic required.

    ``corpus`` is that third clause, and it is not optional in practice. A RAG
    cell's score is a function of what the corpus *contained*: ingest fifteen
    videos and every engine retrieves from a different store, yet nothing else
    in this material moves. Without it a run started before an ingestion and
    finished after it silently compares a stale arm against a fresh one — two
    corpora, reported as two engines. That is not a hypothetical: it is how a
    ``rag_llm`` baseline scored over 38 videos came to be averaged against a
    ``rag_llm_filtered`` arm that routed over 53.

    It deliberately does **not** follow ``_engine_material``'s "omit at the
    default" rule. That rule exists so cells scored before a field was tracked
    stay valid; here those cells are exactly the ones that are *not* valid,
    because nothing recorded which corpus produced them. Passing ``None`` is
    for unit tests that construct no store; every real caller passes a digest,
    and doing so invalidates the untracked history on purpose.

    ``exclude_video_ids`` is the held-out set the engine retrieved under. It is
    the sharpest collision this cache could have: a full-corpus cell and a
    held-out cell ask the identical question of the identical configuration and
    differ only in what the corpus contained, so without it the held-out run is
    silently answered with the answer from a run that *had* seen the video it is
    supposed to be blind to — a clean-looking, completely meaningless number.
    The exclusion *mechanism*'s version rides along for the same reason a judge
    upgrade rescores an unchanged answer: a change in how a video is held out
    can change what was retrieved even when the id list is identical.
    """
    material = {
        "setup": setup,
        "question": entry.question,
        "reference_answer": entry.reference_answer,
        "expected_chunk_ids": sorted(entry.expected_chunk_ids),
        "expected_video_ids": sorted(entry.expected_video_ids),
        # The "model_id" of this RAG variant: everything about its current
        # configuration that could change the answer it gives.
        "answer_model": settings.deepseek_model,
        "embedding_model": settings.embedding_model,
        "retrieval_mode": settings.retrieval_mode,
        "top_k": top_k or settings.rag_top_k,
        "rerank_enabled": settings.rerank_enabled,
        "rerank_model": settings.rerank_model if settings.rerank_enabled else None,
        "neighbor_span": settings.neighbor_span,
        "neo4j_uri": settings.neo4j_uri if setup == "graph_rag" else None,
        # The judge's identity: a judge upgrade should rescore even an
        # unchanged answer, since the score itself may no longer agree.
        "judge_model": judge_model,
        "judge_samples": judge_samples,
        "ragas_version": ragas_version,
        "reference_scored": reference_scored,
    }
    # Absent from the material entirely when nothing is held out, so every cell
    # scored before held-out runs existed keeps its fingerprint — the same rule
    # ``_engine_material`` follows for a setting sitting at its default.
    if corpus is not None:
        material["corpus"] = corpus
    if exclude_video_ids:
        material["exclude_video_ids"] = sorted(dict.fromkeys(exclude_video_ids))
        material["exclusion_version"] = EXCLUSION_VERSION
    material.update(_engine_material(setup, settings))
    encoded = json.dumps(material, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24]


def load_cell(fingerprint: str, cache_dir: Path = DEFAULT_CACHE_DIR) -> dict[str, Any] | None:
    """The cached entry-result dict for this fingerprint, or ``None`` on a
    miss — including a corrupt file or a previously-recorded error, both of
    which are treated as "not cached" so the cell is simply recomputed."""
    path = cache_dir / f"{fingerprint}.json"
    if not path.exists():
        return None
    try:
        cached = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if cached.get("error"):
        return None
    return cached


def save_cell(
    fingerprint: str,
    entry_result: dict[str, Any],
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> None:
    """Persist one scored cell. Errors are not written — see :func:`load_cell`."""
    if entry_result.get("error"):
        return
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{fingerprint}.json"
    path.write_text(json.dumps(entry_result, indent=2) + "\n", encoding="utf-8")

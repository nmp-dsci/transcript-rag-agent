"""Reciprocal Rank Fusion for combining semantic and BM25 rankings.

Semantic retrieval finds paraphrases but misses exact terms; BM25 finds exact
terms but misses paraphrases. RRF merges the two by *rank* rather than by
score, which is the point: the two systems produce scores on incomparable
scales (cosine distance vs. Okapi BM25), so no normalization of raw scores is
trustworthy. Ranks are comparable by construction.

The fused score for a document is::

    score(d) = sum_i  weight_i / (k + rank_i(d))

with ranks 1-based and lists that omit the document contributing nothing. The
constant ``k`` (60 by convention, from Cormack et al. 2009) damps the influence
of the very top positions so a single system cannot dominate the fusion.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Sequence, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from src.rag.models import RetrievedChunk

DEFAULT_K = 60

# key -> RetrievedChunk, for keys that have no chunk object in the pool.
Resolver = Callable[[str], "RetrievedChunk | None"]


def chunk_key(video_id: Any, chunk_index: Any) -> str:
    """The identity shared by semantic chunks and BM25 records."""
    return f"{video_id}:{chunk_index}"


def key_of(item: Any) -> str:
    """Key a chunk object (``.video_id``/``.chunk_index``) or a BM25 dict."""
    if isinstance(item, dict):
        return chunk_key(item.get("video_id", ""), item.get("chunk_index", 0))
    return chunk_key(getattr(item, "video_id", ""), getattr(item, "chunk_index", 0))


def reciprocal_rank_fusion(
    rankings: list[list[str]],
    k: int = DEFAULT_K,
    weights: list[float] | None = None,
) -> list[tuple[str, float]]:
    """Fuse ranked lists of keys into one ``(key, score)`` list, best first.

    ``rankings`` are ordered best-first. A key missing from a list contributes
    nothing from that list. A key repeated within one list is counted once, at
    its best (first) rank.

    Ties are broken deterministically by first appearance — scanning the lists
    in order, then positions within each list — so equal-scoring keys come back
    in a stable, reproducible order rather than an arbitrary one.
    """
    if weights is not None and len(weights) != len(rankings):
        raise ValueError(
            f"weights has {len(weights)} entries but there are {len(rankings)} rankings"
        )
    if k < 0:
        raise ValueError("k must be non-negative")

    scores: dict[str, float] = {}
    first_seen: dict[str, int] = {}
    order = 0

    for list_index, ranking in enumerate(rankings):
        weight = 1.0 if weights is None else float(weights[list_index])
        seen_here: set[str] = set()
        for rank, key in enumerate(ranking, start=1):
            if key in seen_here:
                # A duplicate within one list is the same document; its best
                # rank already scored, and counting it twice would let one
                # system inflate a document by repeating it.
                continue
            seen_here.add(key)
            scores[key] = scores.get(key, 0.0) + weight / (k + rank)
            if key not in first_seen:
                first_seen[key] = order
                order += 1

    return sorted(scores.items(), key=lambda item: (-item[1], first_seen[item[0]]))


def fuse_chunks(
    semantic: Sequence[Any],
    bm25: Sequence[Any],
    top_k: int,
    k: int = DEFAULT_K,
    weights: list[float] | None = None,
    *,
    resolver: Resolver | None = None,
    pool: Iterable[Any] = (),
) -> list["RetrievedChunk"]:
    """Fuse semantic chunks with BM25 records, returning ``RetrievedChunk``s.

    ``semantic`` holds objects with ``.video_id``/``.chunk_index`` (normally
    ``RetrievedChunk``); ``bm25`` holds the dict records produced by
    ``src.api.corpus.load_chunk_corpus``. Both are keyed as
    ``f"{video_id}:{chunk_index}"``.

    **Output is restricted to chunks that resolve to a real ``RetrievedChunk``**
    — by default, those present in ``semantic``. This is a deliberate choice
    forced by the data: BM25 records carry no ``transcript_id`` and may carry a
    null ``source_url``, both of which ``RetrievedChunk`` requires, so a
    BM25-only hit cannot be faithfully reconstructed. Fabricating placeholder
    values would put invented identity into answer citations.

    BM25-only keys still take part in the scoring; they are dropped only at the
    output step. That costs nothing in ordering, because an RRF score depends
    solely on the document's own ranks — removing a document never reorders the
    ones that remain. What BM25 contributes is a *boost* to chunks both systems
    found, which is the effect worth having.

    To include BM25-only chunks, supply either ``pool`` (extra chunk objects to
    resolve against, e.g. a wider retrieval) or ``resolver`` (a callable mapping
    a key to a ``RetrievedChunk``, e.g. a store lookup).

    Each returned chunk keeps its original ``.score`` from semantic retrieval,
    untouched. The fused score is *not* attached to the chunk: ``RetrievedChunk``
    is a pydantic model without such a field, and ``model_copy(update=...)``
    would set an attribute that silently vanishes from ``model_dump()``. Use
    :func:`fuse_chunks_with_scores` when the fused score is needed.
    """
    scored = fuse_chunks_with_scores(
        semantic, bm25, top_k, k, weights, resolver=resolver, pool=pool
    )
    return [chunk for chunk, _score in scored]


def fuse_chunks_with_scores(
    semantic: Sequence[Any],
    bm25: Sequence[Any],
    top_k: int,
    k: int = DEFAULT_K,
    weights: list[float] | None = None,
    *,
    resolver: Resolver | None = None,
    pool: Iterable[Any] = (),
) -> list[tuple["RetrievedChunk", float]]:
    """:func:`fuse_chunks`, paired with each chunk's fused RRF score.

    Same rules and same ordering; use this when the fused score must be shown
    or logged. Scores are RRF sums, not probabilities — compare them only
    within one fusion.
    """
    if top_k <= 0:
        return []

    # Semantic chunks resolve first so their objects win over anything in the
    # pool, keeping the retrieval-time `.score` that the caller expects.
    resolvable: dict[str, Any] = {key_of(item): item for item in pool}
    resolvable.update({key_of(chunk): chunk for chunk in semantic})

    fused = reciprocal_rank_fusion(
        [[key_of(item) for item in semantic], [key_of(item) for item in bm25]],
        k=k,
        weights=weights,
    )

    results: list[tuple[Any, float]] = []
    for key, score in fused:
        chunk = resolvable.get(key)
        if chunk is None and resolver is not None:
            chunk = resolver(key)
        if chunk is None:
            continue
        results.append((chunk, score))
        if len(results) >= top_k:
            break
    return results

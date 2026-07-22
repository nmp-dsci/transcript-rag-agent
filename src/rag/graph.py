"""k-nearest-neighbour similarity graph over transcript chunk embeddings.

The Retrieval Lab shows chunks as points in a PCA projection, which answers
"where does this chunk sit?" but not "what is it actually close to?". This
module answers the second question by linking every chunk to its ``k`` most
similar neighbours, producing a graph the frontend renders force-directed.

The corpus is small (hundreds of chunks), so similarity is computed exactly as
a dense n x n cosine matrix rather than through an approximate index. numpy
does the work when it is importable; a pure-Python path mirrors it otherwise,
matching the conventions in ``src.dashboard.chunk_space``.

Two properties the visualisation depends on:

* **Undirected, de-duplicated edges.** A listing B in its top-k and B listing A
  is one edge, not two, and ``degree`` counts distinct neighbours after that
  merge. Force layouts double-weight duplicated links, so this matters.
* **Determinism.** Starting coordinates come from PCA over the same embeddings,
  not from a random seed, so a reload reproduces the same picture. Ties are
  broken by index everywhere for the same reason.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

try:  # pragma: no cover - exercised by whichever path is installed
    import numpy as np

    _HAS_NUMPY = True
except ImportError:  # pragma: no cover - numpy ships with scikit-learn
    np = None  # type: ignore[assignment]
    _HAS_NUMPY = False

try:  # pragma: no cover - scikit-learn is a declared dependency
    from src.dashboard.chunk_space import fit_chunk_projection, nearest_chunks_for_question
except ImportError:  # pragma: no cover
    fit_chunk_projection = None  # type: ignore[assignment]
    nearest_chunks_for_question = None  # type: ignore[assignment]

__all__ = [
    "build_chunk_graph",
    "build_chunk_graph_cached",
    "clear_cache",
    "nearest_chunks",
]

PREVIEW_CHARS = 120
_LAYOUT_MODEL_LABEL = "chunk-graph-layout"
_ROUND_TO = 6
MAX_GRAPH_CHUNKS = 5000

_CACHE: dict[tuple[int, int, float], dict[str, Any]] = {}


def build_chunk_graph(
    records: Sequence[Mapping[str, Any]],
    k: int = 5,
    min_similarity: float = 0.0,
    layout: bool = True,
) -> dict[str, Any]:
    """Build an undirected k-NN similarity graph over chunk embeddings.

    ``records`` is the output of ``TranscriptChunkStore.all_embeddings()``.
    Records without a usable embedding are dropped rather than raising, so a
    partially indexed corpus still renders.

    Returns a JSON-serialisable dict of ``nodes``, ``edges`` and ``stats``.
    """
    k = max(int(k), 0)
    min_similarity = float(min_similarity)

    usable = _usable_records(records)
    if not usable:
        return {
            "nodes": [],
            "edges": [],
            "stats": _stats(0, [], k, min_similarity, 0, 0),
        }

    vectors = [list(record["embedding"]) for record in usable]
    unit_vectors = _unit_vectors(vectors)
    similarity = _similarity_matrix(unit_vectors)
    index_edges, degrees = _build_edges(similarity, len(usable), k, min_similarity)

    coords = _layout_coordinates(unit_vectors, usable) if layout else [(0.0, 0.0)] * len(usable)

    chunk_ids = [_chunk_id(record, index) for index, record in enumerate(usable)]
    nodes = [
        _node(record, chunk_ids[index], degrees[index], coords[index])
        for index, record in enumerate(usable)
    ]
    edges = [
        {
            "source": chunk_ids[left],
            "target": chunk_ids[right],
            "similarity": round(score, _ROUND_TO),
        }
        for left, right, score in index_edges
    ]

    channels = {
        channel
        for record in usable
        if (channel := record.get("channel_id") or record.get("channel_name"))
    }
    isolated = sum(1 for degree in degrees if degree == 0)

    return {
        "nodes": nodes,
        "edges": edges,
        "stats": _stats(
            len(nodes),
            [score for _left, _right, score in index_edges],
            k,
            min_similarity,
            len(channels),
            isolated,
        ),
    }


def build_chunk_graph_cached(
    records: Sequence[Mapping[str, Any]],
    k: int = 5,
    min_similarity: float = 0.0,
    layout: bool = True,
    max_chunks: int = MAX_GRAPH_CHUNKS,
) -> dict[str, Any]:
    """Cached ``build_chunk_graph``, capped so the corpus can't grow unbounded.

    The graph is O(n^2) in the number of chunks, so a request over a corpus
    past ``max_chunks`` fails fast instead of getting slower and slower.
    Results are cached per ``(chunk_count, k, min_similarity)``, keyed the same
    way ``rag.bm25`` caches its index, so a growing/shrinking corpus
    invalidates automatically. Every call returns a fresh shallow copy so a
    caller mutating the top-level dict (e.g. adding a ``"query"`` key) can
    never corrupt what is stored for the next cache hit.
    """
    count = len(records)
    if count > max_chunks:
        raise ValueError(
            f"corpus has {count} chunks, exceeding the {max_chunks}-chunk "
            "graph limit; narrow the query or increase the limit"
        )
    key = (count, int(k), float(min_similarity))
    cached = _CACHE.get(key)
    if cached is None:
        cached = build_chunk_graph(records, k=k, min_similarity=min_similarity, layout=layout)
        _CACHE[key] = cached
    return {**cached}


def clear_cache() -> None:
    _CACHE.clear()


def nearest_chunks(
    records: Sequence[Mapping[str, Any]],
    query_embedding: Sequence[float],
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Rank chunks by cosine similarity to ``query_embedding``.

    Scored in the original embedding space, never in the 2-D layout: the
    projection is a viewing aid and distances in it are lossy. The frontend
    uses this to highlight a question's retrieval neighbourhood on the graph,
    so the highlighted set has to match what retrieval would actually return.
    """
    usable = _usable_records(records)
    top_k = max(int(top_k), 0)
    if not usable or top_k == 0:
        return []

    query = [float(value) for value in query_embedding]
    dimension = len(usable[0]["embedding"])
    if len(query) != dimension:
        raise ValueError(
            f"query_embedding has {len(query)} dimensions, chunks have {dimension}"
        )

    chunk_ids = [_chunk_id(record, index) for index, record in enumerate(usable)]
    vectors = [list(record["embedding"]) for record in usable]

    if _HAS_NUMPY and nearest_chunks_for_question is not None:
        ranked = nearest_chunks_for_question(
            np.asarray(query, dtype=float),
            np.asarray(vectors, dtype=float),
            chunk_ids,
            top_k,
        )
        return [
            {"chunk_id": item.chunk_id, "similarity": round(_clamp(item.score), _ROUND_TO)}
            for item in ranked
        ]

    scores = [(_cosine(query, vector), index) for index, vector in enumerate(vectors)]
    scores.sort(key=lambda item: (-item[0], item[1]))
    return [
        {"chunk_id": chunk_ids[index], "similarity": round(_clamp(score), _ROUND_TO)}
        for score, index in scores[:top_k]
    ]


def _usable_records(
    records: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    """Keep records carrying a finite, non-empty embedding of consistent width."""
    usable: list[Mapping[str, Any]] = []
    dimension: int | None = None
    for record in records:
        embedding = record.get("embedding")
        if not isinstance(embedding, (list, tuple)) or not embedding:
            continue
        try:
            values = [float(value) for value in embedding]
        except (TypeError, ValueError):
            continue
        if not all(math.isfinite(value) for value in values):
            continue
        if dimension is None:
            dimension = len(values)
        elif len(values) != dimension:
            continue
        usable.append({**record, "embedding": values})
    return usable


def _chunk_id(record: Mapping[str, Any], index: int) -> str:
    chunk_id = record.get("chunk_id")
    if chunk_id:
        return str(chunk_id)
    return f"chunk:{record.get('video_id', '')}:{record.get('chunk_index', index)}"


def _node(
    record: Mapping[str, Any],
    chunk_id: str,
    degree: int,
    coords: tuple[float, float],
) -> dict[str, Any]:
    return {
        "id": chunk_id,
        "video_id": record.get("video_id"),
        "chunk_index": record.get("chunk_index"),
        "channel_id": record.get("channel_id"),
        "channel_name": record.get("channel_name"),
        "title": record.get("title"),
        "preview": _preview(record.get("text")),
        "start_seconds": record.get("start_seconds"),
        "end_seconds": record.get("end_seconds"),
        "source_url": record.get("source_url"),
        "degree": degree,
        "x": coords[0],
        "y": coords[1],
    }


def _preview(text: Any) -> str:
    """First ~120 characters of chunk text, whitespace collapsed for tooltips."""
    collapsed = " ".join(str(text or "").split())
    if len(collapsed) <= PREVIEW_CHARS:
        return collapsed
    return collapsed[:PREVIEW_CHARS].rstrip() + "..."


def _stats(
    node_count: int,
    scores: Sequence[float],
    k: int,
    min_similarity: float,
    channels: int,
    isolated: int,
) -> dict[str, Any]:
    mean = sum(scores) / len(scores) if scores else 0.0
    return {
        "nodes": node_count,
        "edges": len(scores),
        "k": k,
        "min_similarity": min_similarity,
        "channels": channels,
        "mean_similarity": round(mean, _ROUND_TO),
        "isolated_nodes": isolated,
    }


def _unit_vectors(vectors: Sequence[Sequence[float]]) -> Any:
    """L2-normalise so a dot product is the cosine similarity.

    Zero vectors stay zero and therefore score 0.0 against everything, which is
    the same convention ``chunk_space.nearest_chunks_for_question`` uses for a
    zero denominator.
    """
    if _HAS_NUMPY:
        matrix = np.asarray(vectors, dtype=float)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms != 0)
    unit: list[list[float]] = []
    for vector in vectors:
        norm = math.sqrt(sum(value * value for value in vector))
        unit.append([0.0] * len(vector) if norm == 0 else [value / norm for value in vector])
    return unit


def _similarity_matrix(unit_vectors: Any) -> Any:
    if _HAS_NUMPY:
        return unit_vectors @ unit_vectors.T
    size = len(unit_vectors)
    return [
        [sum(a * b for a, b in zip(unit_vectors[i], unit_vectors[j])) for j in range(size)]
        for i in range(size)
    ]


def _neighbour_order(row: Any, size: int) -> Sequence[int]:
    """Indices of ``row`` sorted by descending similarity, ties by index."""
    if _HAS_NUMPY:
        return np.argsort(-row, kind="stable")
    return sorted(range(size), key=lambda index: (-row[index], index))


def _build_edges(
    similarity: Any,
    size: int,
    k: int,
    min_similarity: float,
) -> tuple[list[tuple[int, int, float]], list[int]]:
    """Collect each node's top-k neighbours, then merge reciprocal links.

    An edge is keyed by its sorted endpoints, so A->B and B->A collapse into a
    single entry. Self-edges are skipped explicitly rather than by masking the
    diagonal, which keeps the filter correct for negative ``min_similarity``.
    """
    merged: dict[tuple[int, int], float] = {}
    neighbours: list[set[int]] = [set() for _ in range(size)]

    if k > 0:
        for i in range(size):
            row = similarity[i]
            taken = 0
            for j in _neighbour_order(row, size):
                index = int(j)
                if index == i:
                    continue
                score = _clamp(float(row[index]))
                if score < min_similarity:
                    break  # descending order: nothing further can qualify
                key = (i, index) if i < index else (index, i)
                merged.setdefault(key, score)
                neighbours[i].add(index)
                neighbours[index].add(i)
                taken += 1
                if taken >= k:
                    break

    edges = [(left, right, score) for (left, right), score in merged.items()]
    edges.sort(key=lambda edge: (edge[0], edge[1]))
    degrees = [len(neighbour_set) for neighbour_set in neighbours]
    return edges, degrees


def _layout_coordinates(
    unit_vectors: Any,
    records: Sequence[Mapping[str, Any]],
) -> list[tuple[float, float]]:
    """Deterministic 2-D starting positions, PCA-projected and normalised.

    Reuses the dashboard's fitted projection so the graph and the scatter plot
    place the same chunk in the same relative position. PCA is fit on the
    normalised vectors, matching the metric the edges were built with.
    """
    size = len(records)
    if size == 0:
        return []
    if size == 1:
        return [(0.0, 0.0)]

    coords: list[tuple[float, float]] | None = None
    if _HAS_NUMPY and fit_chunk_projection is not None:
        chunk_ids = [_chunk_id(record, index) for index, record in enumerate(records)]
        try:
            # Identical embeddings give PCA zero total variance, and its
            # explained-variance ratio warns on the 0/0. Only the coordinates
            # are read here, so the warning is noise.
            with np.errstate(invalid="ignore", divide="ignore"):
                projection = fit_chunk_projection(
                    np.asarray(unit_vectors, dtype=float),
                    chunk_ids,
                    _LAYOUT_MODEL_LABEL,
                )
            coords = [(x, y) for _chunk_id_, x, y in projection.chunk_coords]
        except ValueError:
            coords = None

    if coords is None:
        coords = _circle_layout(size)
    return _normalise_coordinates(coords)


def _circle_layout(size: int) -> list[tuple[float, float]]:
    """Fallback ring layout, used when PCA is unavailable or degenerate."""
    step = 2 * math.pi / size
    return [(math.cos(step * index), math.sin(step * index)) for index in range(size)]


def _normalise_coordinates(
    coords: Sequence[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Centre and scale into [-1, 1], preserving the projection's aspect ratio.

    Both axes share one divisor. Scaling them independently would stretch a
    narrow second component to full width and imply structure PCA did not find.
    """
    safe = [
        (x, y) if math.isfinite(x) and math.isfinite(y) else (0.0, 0.0) for x, y in coords
    ]
    xs = [x for x, _ in safe]
    ys = [y for _, y in safe]
    x_mid = (min(xs) + max(xs)) / 2
    y_mid = (min(ys) + max(ys)) / 2
    half_span = max((max(xs) - min(xs)) / 2, (max(ys) - min(ys)) / 2)
    if half_span <= 0:  # every chunk identical: no spread to normalise
        return [(0.0, 0.0)] * len(safe)
    return [
        (round((x - x_mid) / half_span, _ROUND_TO), round((y - y_mid) / half_span, _ROUND_TO))
        for x, y in safe
    ]


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    denominator = left_norm * right_norm
    if denominator == 0:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / denominator


def _clamp(score: float) -> float:
    """Keep floating-point drift from reporting a similarity outside [-1, 1]."""
    return max(-1.0, min(1.0, float(score)))

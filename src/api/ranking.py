"""Side-by-side semantic, BM25, and graph rankings for the Retrieval Lab.

All modes rank the same chunk corpus for one query, and the results are
aligned by chunk id so the UI can show where they disagree: each row carries
its rank in the *other* modes (``other_rank``, the best rank among the modes
it also appears in), which is ``None`` when only one selected mode found that
chunk at all.

Graph mode depends on a live Neo4j; losing it must not take the other columns
down, but it must not look like a legitimately empty result either. A failing
graph ranking is therefore reported as ``errors["graph"]`` and left out of the
alignment and the overlap count, so "the graph is unavailable" stays
distinguishable from "the graph matched nothing" and the semantic-vs-BM25
agreement number survives the outage.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Literal, Sequence

from src.rag import bm25

logger = logging.getLogger(__name__)

RankMode = Literal["semantic", "bm25", "graph"]

#: Every mode identifier the ranking pipeline understands.
MODES: tuple[RankMode, ...] = ("semantic", "bm25", "graph")
#: Selected by default when a caller does not name modes explicitly. Graph
#: mode needs a live Neo4j connection, so it stays opt-in rather than
#: silently added to every request.
DEFAULT_MODES: tuple[RankMode, ...] = ("semantic", "bm25")
PREVIEW_CHARS = 220

# (query, top_k) -> ranked chunk records, best first.
SemanticFn = Callable[[str, int], Sequence[dict[str, Any]]]
RecordsFn = Callable[[], Sequence[dict[str, Any]]]
#: (query, top_k) -> ranked chunk records, each optionally carrying
#: ``matched_entities`` — the graph counterpart of ``SemanticFn``.
GraphFn = Callable[[str, int], Sequence[dict[str, Any]]]


def chunk_id(record: dict[str, Any]) -> str:
    return f"{record.get('video_id', '')}:{record.get('chunk_index', 0)}"


def _row(record: dict[str, Any], rank: int, score: float | None) -> dict[str, Any]:
    text = str(record.get("text") or "")
    preview = text[:PREVIEW_CHARS] + ("…" if len(text) > PREVIEW_CHARS else "")
    row = {
        "chunk_id": chunk_id(record),
        "video_id": record.get("video_id"),
        "chunk_index": record.get("chunk_index"),
        "rank": rank,
        "score": None if score is None else round(float(score), 4),
        "preview": preview,
        "start_seconds": record.get("start_seconds"),
        "end_seconds": record.get("end_seconds"),
        "source_url": record.get("source_url"),
        "other_rank": None,
    }
    if "matched_entities" in record:
        row["matched_entities"] = record["matched_entities"]
    return row


def _align(rankings: dict[str, list[dict[str, Any]]]) -> None:
    """Fill ``other_rank`` on every row, in place — the best rank the same
    chunk holds in any *other* selected mode."""
    if len(rankings) < 2:
        return
    ranks = {
        mode: {row["chunk_id"]: row["rank"] for row in rows} for mode, rows in rankings.items()
    }
    for mode, rows in rankings.items():
        others = [ranks[other] for other in ranks if other != mode]
        for row in rows:
            best = min(
                (other[row["chunk_id"]] for other in others if row["chunk_id"] in other),
                default=None,
            )
            row["other_rank"] = best


def build_rankings(
    query: str,
    *,
    modes: Sequence[str],
    top_k: int,
    semantic_fn: SemanticFn,
    records_fn: RecordsFn,
    graph_fn: GraphFn | None = None,
    video_id: str | None = None,
    cache_key: str | None = None,
) -> dict[str, Any]:
    selected = [mode for mode in MODES if mode in modes]
    rankings: dict[str, list[dict[str, Any]]] = {}
    errors: dict[str, str] = {}

    if "semantic" in selected:
        rankings["semantic"] = [
            _row(record, rank, record.get("score"))
            for rank, record in enumerate(semantic_fn(query, top_k), start=1)
        ]
    if "bm25" in selected:
        results = bm25.search(list(records_fn()), query, top_k, cache_key=cache_key or video_id)
        rankings["bm25"] = [_row(record, record["rank"], record["score"]) for record in results]
    if "graph" in selected and graph_fn is not None:
        try:
            rankings["graph"] = [
                _row(record, rank, record.get("score"))
                for rank, record in enumerate(graph_fn(query, top_k), start=1)
            ]
        except Exception as exc:
            # The column still exists, empty and labelled — the caller asked
            # for it, and silently dropping it would read as "no matches".
            logger.warning("graph ranking unavailable", exc_info=True)
            rankings["graph"] = []
            errors["graph"] = str(exc)

    answered = {mode: rows for mode, rows in rankings.items() if mode not in errors}
    _align(answered)

    # Chunks every *answering* mode agreed on, not just a pairwise intersection
    # — with three modes this naturally tightens to a 3-way agreement, and a
    # mode that errored contributes no disagreement, so it is left out rather
    # than collapsing the count to zero.
    overlap: list[str] = []
    if len(answered) >= 2:
        id_sets = [set(row["chunk_id"] for row in rows) for rows in answered.values()]
        overlap = sorted(set.intersection(*id_sets))

    return {
        "query": query,
        "video_id": video_id,
        "top_k": top_k,
        "modes": rankings,
        "errors": errors,
        "overlap": {
            "count": len(overlap),
            "of": min([len(rows) for rows in answered.values()], default=0),
            "chunk_ids": overlap,
        },
    }

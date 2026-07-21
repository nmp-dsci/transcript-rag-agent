"""Side-by-side semantic and BM25 rankings for the Retrieval Lab.

Both modes rank the same chunk corpus for one query, and the results are
aligned by chunk id so the UI can show where keyword and embedding retrieval
disagree: each row carries its rank in the *other* mode (``other_rank``), which
is ``None`` when only one mode found that chunk at all.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

from src.rag import bm25

MODES = ("semantic", "bm25")
PREVIEW_CHARS = 220

# (query, top_k) -> ranked chunk records, best first.
SemanticFn = Callable[[str, int], Sequence[dict[str, Any]]]
RecordsFn = Callable[[], Sequence[dict[str, Any]]]


def chunk_id(record: dict[str, Any]) -> str:
    return f"{record.get('video_id', '')}:{record.get('chunk_index', 0)}"


def _row(record: dict[str, Any], rank: int, score: float | None) -> dict[str, Any]:
    text = str(record.get("text") or "")
    preview = text[:PREVIEW_CHARS] + ("…" if len(text) > PREVIEW_CHARS else "")
    return {
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


def _align(rankings: dict[str, list[dict[str, Any]]]) -> None:
    """Fill ``other_rank`` on every row, in place."""
    if len(rankings) < 2:
        return
    ranks = {
        mode: {row["chunk_id"]: row["rank"] for row in rows}
        for mode, rows in rankings.items()
    }
    for mode, rows in rankings.items():
        others = [ranks[other] for other in ranks if other != mode]
        for row in rows:
            for other in others:
                if row["chunk_id"] in other:
                    row["other_rank"] = other[row["chunk_id"]]
                    break


def build_rankings(
    query: str,
    *,
    modes: Sequence[str],
    top_k: int,
    semantic_fn: SemanticFn,
    records_fn: RecordsFn,
    video_id: str | None = None,
    cache_key: str | None = None,
) -> dict[str, Any]:
    selected = [mode for mode in MODES if mode in modes]
    rankings: dict[str, list[dict[str, Any]]] = {}

    if "semantic" in selected:
        rankings["semantic"] = [
            _row(record, rank, record.get("score"))
            for rank, record in enumerate(semantic_fn(query, top_k), start=1)
        ]
    if "bm25" in selected:
        results = bm25.search(
            list(records_fn()), query, top_k, cache_key=cache_key or video_id
        )
        rankings["bm25"] = [
            _row(record, record["rank"], record["score"]) for record in results
        ]

    _align(rankings)

    overlap: list[str] = []
    if len(rankings) == 2:
        first, second = (set(r["chunk_id"] for r in rows) for rows in rankings.values())
        overlap = sorted(first & second)

    return {
        "query": query,
        "video_id": video_id,
        "top_k": top_k,
        "modes": rankings,
        "overlap": {
            "count": len(overlap),
            "of": min([len(rows) for rows in rankings.values()], default=0),
            "chunk_ids": overlap,
        },
    }

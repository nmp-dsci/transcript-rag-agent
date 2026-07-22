"""Lightweight corpus listing for the Library view.

Reads raw-transcript and chunk metadata straight from Chroma with a plain
client — deliberately avoiding ``RagSetupRunner`` so listing the corpus never
loads the embedding model or the retrieval stack.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def list_corpus(
    chroma_path: Path,
    raw_collection: str = "raw_transcripts",
    chunk_collection: str = "transcript_chunks",
) -> dict[str, Any]:
    import chromadb

    try:
        client = chromadb.PersistentClient(path=str(chroma_path))
        raw = client.get_collection(raw_collection)
        metadatas = raw.get(include=["metadatas"]).get("metadatas") or []
    except Exception:
        return {
            "videos": [],
            "channels": [],
            "totals": {"videos": 0, "chunks": 0, "channels": 0},
            "insights": [],
        }

    chunk_counts: dict[str, int] = {}
    total_chunks = 0
    try:
        chunk_metas = (
            client.get_collection(chunk_collection)
            .get(include=["metadatas"])
            .get("metadatas")
            or []
        )
        for meta in chunk_metas:
            video_id = str((meta or {}).get("video_id", ""))
            if video_id:
                chunk_counts[video_id] = chunk_counts.get(video_id, 0) + 1
        total_chunks = len(chunk_metas)
    except Exception:
        pass

    videos = []
    for meta in metadatas:
        meta = meta or {}
        video_id = str(meta.get("video_id", ""))
        videos.append(
            {
                "video_id": video_id,
                "title": meta.get("title") or None,
                "channel_name": meta.get("channel_name") or None,
                # Must match the channel_id stamped on chunks, or a channel
                # filter built from this list would select nothing.
                "channel_id": meta.get("channel_id") or None,
                "thumbnail_url": meta.get("thumbnail_url") or None,
                "source_url": meta.get("source_url") or None,
                "duration_seconds": meta.get("duration_seconds") or None,
                "upload_date": meta.get("upload_date") or None,
                "view_count": meta.get("view_count") or None,
                "summary": meta.get("summary") or None,
                "fetched_at": meta.get("fetched_at") or None,
                "chunk_count": chunk_counts.get(video_id, 0),
            }
        )
    videos.sort(key=lambda video: str(video.get("fetched_at") or ""), reverse=True)
    channels = build_channels(videos)
    return {
        "videos": videos,
        "channels": channels,
        "totals": {
            "videos": len(videos),
            "chunks": total_chunks,
            "channels": len(channels),
        },
        "insights": build_insights(videos, channels, total_chunks),
    }


def build_channels(videos: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One record per channel, so the UI can scope without regrouping videos.

    Channels are keyed by ``channel_id`` where present, falling back to a
    name-derived key so a channel indexed without an id is still selectable —
    the same fallback the chunk-metadata backfill uses.
    """
    grouped: dict[str, dict[str, Any]] = {}
    for video in videos:
        name = video.get("channel_name") or "Unknown channel"
        key = str(video.get("channel_id") or f"name:{name}")
        channel = grouped.setdefault(
            key,
            {
                "channel_id": key,
                "channel_name": name,
                "video_count": 0,
                "chunk_count": 0,
                "video_ids": [],
            },
        )
        channel["video_count"] += 1
        channel["chunk_count"] += int(video.get("chunk_count") or 0)
        channel["video_ids"].append(video.get("video_id"))
    return sorted(grouped.values(), key=lambda c: -c["chunk_count"])


def build_insights(
    videos: list[dict[str, Any]],
    channels: list[dict[str, Any]],
    total_chunks: int,
) -> list[dict[str, Any]]:
    """Observations about corpus shape that affect retrieval quality.

    These are the things that silently skew results — one channel dominating
    the index, or videos the summary filter can never select — surfaced as
    actionable chips rather than left for the user to infer from counts.
    """
    insights: list[dict[str, Any]] = []

    if channels and total_chunks:
        top = channels[0]
        share = top["chunk_count"] / total_chunks
        if share >= 0.4:
            insights.append(
                {
                    "kind": "channel_skew",
                    "level": "warn" if share >= 0.5 else "info",
                    "message": (
                        f"{top['channel_name']} holds {top['chunk_count']} of "
                        f"{total_chunks} chunks ({share:.0%}) — whole-corpus "
                        "retrieval will skew toward it"
                    ),
                    "channel_id": top["channel_id"],
                }
            )

    missing = [v for v in videos if not v.get("summary")]
    if missing:
        insights.append(
            {
                "kind": "missing_summaries",
                "level": "warn",
                "message": (
                    f"{len(missing)} of {len(videos)} videos have no transcript "
                    "summary, so the summary filter can never select them"
                ),
                "video_ids": [v.get("video_id") for v in missing],
            }
        )

    empty = [v for v in videos if not v.get("chunk_count")]
    if empty:
        insights.append(
            {
                "kind": "unindexed",
                "level": "bad",
                "message": (
                    f"{len(empty)} video(s) have a transcript but no chunks — "
                    "they are invisible to retrieval"
                ),
                "video_ids": [v.get("video_id") for v in empty],
            }
        )

    if videos:
        sizes = [int(v.get("chunk_count") or 0) for v in videos if v.get("chunk_count")]
        if sizes:
            insights.append(
                {
                    "kind": "size_spread",
                    "level": "info",
                    "message": (
                        f"chunks per video range {min(sizes)}–{max(sizes)} "
                        f"(median {sorted(sizes)[len(sizes) // 2]})"
                    ),
                }
            )
    return insights


def list_chunks(
    chroma_path: Path,
    video_id: str,
    chunk_collection: str = "transcript_chunks",
) -> dict[str, Any]:
    """Every stored chunk for one video, ordered by chunk index.

    Reads documents and metadata straight from Chroma for the same reason as
    ``list_corpus``: browsing the corpus must never load the embedding model.
    """
    import chromadb

    try:
        client = chromadb.PersistentClient(path=str(chroma_path))
        collection = client.get_collection(chunk_collection)
        result = collection.get(
            where={"video_id": video_id}, include=["documents", "metadatas"]
        )
    except Exception:
        return {"video_id": video_id, "chunks": [], "total": 0}

    documents = result.get("documents") or []
    metadatas = result.get("metadatas") or []
    chunks = []
    for index, meta in enumerate(metadatas):
        meta = meta or {}
        text = documents[index] if index < len(documents) else ""
        chunks.append(
            {
                "chunk_index": int(meta.get("chunk_index", index) or 0),
                "text": text or "",
                "start_seconds": meta.get("start_seconds"),
                "end_seconds": meta.get("end_seconds"),
                "start_segment_index": meta.get("start_segment_index"),
                "end_segment_index": meta.get("end_segment_index"),
                "segment_count": meta.get("segment_count") or 0,
                "source_url": meta.get("source_url") or None,
            }
        )
    chunks.sort(key=lambda chunk: chunk["chunk_index"])
    return {"video_id": video_id, "chunks": chunks, "total": len(chunks)}


def load_chunk_embeddings(
    chroma_path: Path,
    chunk_collection: str = "transcript_chunks",
) -> list[dict[str, Any]]:
    """Every chunk with its stored embedding, for similarity-graph building.

    Reads vectors straight from Chroma rather than going through
    ``TranscriptChunkStore``, because constructing that store instantiates the
    embedding model — and drawing the graph never needs to embed anything new.
    """
    import chromadb
    from chromadb.errors import NotFoundError

    client = chromadb.PersistentClient(path=str(chroma_path))
    try:
        collection = client.get_collection(chunk_collection)
    except NotFoundError:
        return []
    result = collection.get(include=["embeddings", "documents", "metadatas"])

    embeddings = result.get("embeddings")
    embeddings = [] if embeddings is None else list(embeddings)
    documents = result.get("documents") or []
    metadatas = result.get("metadatas") or []
    records: list[dict[str, Any]] = []
    for index, meta in enumerate(metadatas):
        meta = meta or {}
        if index >= len(embeddings):
            continue
        video_id = str(meta.get("video_id", ""))
        chunk_index = int(meta.get("chunk_index", index) or 0)
        records.append(
            {
                "chunk_id": f"chunk:{video_id}:{chunk_index}",
                "video_id": video_id,
                "chunk_index": chunk_index,
                "channel_id": meta.get("channel_id") or None,
                "channel_name": meta.get("channel_name") or None,
                "title": meta.get("title") or None,
                "text": documents[index] if index < len(documents) else "",
                "start_seconds": meta.get("start_seconds"),
                "end_seconds": meta.get("end_seconds"),
                "source_url": meta.get("source_url") or None,
                "embedding": [float(value) for value in embeddings[index]],
            }
        )
    return records


def load_chunk_corpus(
    chroma_path: Path,
    chunk_collection: str = "transcript_chunks",
    video_id: str | None = None,
) -> list[dict[str, Any]]:
    """All chunk texts (optionally for one video) for keyword ranking.

    Returns records shaped for BM25 scoring: text plus the identity needed to
    align a keyword hit with the same chunk from semantic search.
    """
    import chromadb

    where = {"video_id": video_id} if video_id else None
    try:
        client = chromadb.PersistentClient(path=str(chroma_path))
        collection = client.get_collection(chunk_collection)
        result = collection.get(where=where, include=["documents", "metadatas"])
    except Exception:
        return []

    documents = result.get("documents") or []
    metadatas = result.get("metadatas") or []
    records = []
    for index, meta in enumerate(metadatas):
        meta = meta or {}
        text = documents[index] if index < len(documents) else ""
        if not text:
            continue
        records.append(
            {
                "video_id": str(meta.get("video_id", "")),
                "chunk_index": int(meta.get("chunk_index", index) or 0),
                "text": text,
                "start_seconds": meta.get("start_seconds"),
                "end_seconds": meta.get("end_seconds"),
                "source_url": meta.get("source_url") or None,
            }
        )
    return records

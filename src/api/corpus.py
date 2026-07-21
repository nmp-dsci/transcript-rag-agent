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
        return {"videos": [], "totals": {"videos": 0, "chunks": 0}}

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
    return {
        "videos": videos,
        "totals": {"videos": len(videos), "chunks": total_chunks},
    }


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

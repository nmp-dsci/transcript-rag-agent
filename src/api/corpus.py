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
            client.get_collection(chunk_collection).get(include=["metadatas"]).get("metadatas")
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

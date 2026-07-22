"""Stamp video-level identity onto chunks that were indexed without it.

Chunks originally carried only ``video_id``, so retrieval could not filter by
channel. This copies ``channel_id`` / ``channel_name`` / ``title`` /
``upload_date`` from the ``raw_transcripts`` collection onto every chunk of the
matching video, using Chroma's metadata update — no transcripts are re-fetched.

Embeddings are left untouched by default. Pass ``--re-embed`` to also rebuild
chunk vectors with the contextual header (channel/title/timestamp) prepended,
which is a separate, slower change: it loads the embedding model and rewrites
every vector.

    uv run python scripts/backfill_chunk_metadata.py --dry-run
    uv run python scripts/backfill_chunk_metadata.py
    uv run python scripts/backfill_chunk_metadata.py --re-embed
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_settings  # noqa: E402
from src.rag.chunking import build_context_header  # noqa: E402

BATCH = 200


def video_identity(chroma_path: Path, raw_collection: str) -> dict[str, dict[str, str]]:
    """video_id -> the identity fields chunks should inherit."""
    import chromadb

    client = chromadb.PersistentClient(path=str(chroma_path))
    raw = client.get_collection(raw_collection)
    identity: dict[str, dict[str, str]] = {}
    for meta in raw.get(include=["metadatas"]).get("metadatas") or []:
        meta = meta or {}
        video_id = str(meta.get("video_id", ""))
        if not video_id:
            continue
        channel_name = str(meta.get("channel_name") or "")
        fields: dict[str, str] = {}
        # Fall back to the channel name when no id was captured, so a channel
        # filter still has a stable key to match on.
        channel_id = str(meta.get("channel_id") or "") or (
            f"name:{channel_name}" if channel_name else ""
        )
        if channel_id:
            fields["channel_id"] = channel_id
        if channel_name:
            fields["channel_name"] = channel_name
        if meta.get("title"):
            fields["title"] = str(meta["title"])
        if meta.get("upload_date"):
            fields["upload_date"] = str(meta["upload_date"])
        identity[video_id] = fields
    return identity


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="report only")
    parser.add_argument(
        "--re-embed",
        action="store_true",
        help="also rebuild embeddings with contextual headers (slow)",
    )
    args = parser.parse_args(argv)

    settings = load_settings(require_keys=False)
    import chromadb

    client = chromadb.PersistentClient(path=str(settings.chroma_path))
    identity = video_identity(settings.chroma_path, settings.raw_transcript_collection)
    chunks = client.get_collection(settings.chunk_collection)
    stored = chunks.get(include=["documents", "metadatas"])

    ids = stored.get("ids") or []
    documents = stored.get("documents") or []
    metadatas = stored.get("metadatas") or []

    updated_ids: list[str] = []
    updated_metas: list[dict] = []
    updated_docs: list[str] = []
    skipped_no_identity = 0
    metadata_changed = 0

    for index, chunk_id in enumerate(ids):
        meta = dict(metadatas[index] or {})
        text = documents[index] if index < len(documents) else ""
        fields = identity.get(str(meta.get("video_id", "")))
        if not fields:
            skipped_no_identity += 1
            continue
        merged = {**meta, **fields}
        header = build_context_header(
            channel_name=fields.get("channel_name"),
            title=fields.get("title"),
            start_seconds=_as_float(meta.get("start_seconds")),
            end_seconds=_as_float(meta.get("end_seconds")),
        )
        if header:
            merged["context_header"] = header
        needs_metadata_write = merged != meta
        if needs_metadata_write:
            metadata_changed += 1
        if not needs_metadata_write and not args.re_embed:
            continue
        updated_ids.append(chunk_id)
        updated_metas.append(merged)
        updated_docs.append(text)

    channels = sorted({fields.get("channel_name", "?") for fields in identity.values()})
    print(f"chunks stored          : {len(ids)}")
    print(f"videos with identity   : {len(identity)}")
    print(f"channels               : {len(channels)} — {', '.join(channels)}")
    print(f"chunks with metadata changes : {metadata_changed}")
    print(f"chunks needing update  : {len(updated_ids)}")
    if skipped_no_identity:
        print(f"chunks with no raw doc : {skipped_no_identity} (left unchanged)")

    if args.dry_run:
        print("\ndry run — nothing written")
        return 0
    if not updated_ids:
        print("\nnothing to do")
        return 0

    if args.re_embed:
        from src.rag.embeddings import HuggingFaceEmbeddingModel

        print(f"\nre-embedding {len(updated_ids)} chunks with contextual headers ...")
        model = HuggingFaceEmbeddingModel(settings.embedding_model)
        for start in range(0, len(updated_ids), BATCH):
            stop = start + BATCH
            batch_metas = updated_metas[start:stop]
            texts = [
                f"{meta['context_header']}\n{doc}" if meta.get("context_header") else doc
                for meta, doc in zip(batch_metas, updated_docs[start:stop])
            ]
            chunks.update(
                ids=updated_ids[start:stop],
                metadatas=batch_metas,
                embeddings=model.embed_documents(texts),
            )
            print(f"  {min(stop, len(updated_ids))}/{len(updated_ids)}")
    else:
        for start in range(0, len(updated_ids), BATCH):
            stop = start + BATCH
            chunks.update(
                ids=updated_ids[start:stop], metadatas=updated_metas[start:stop]
            )

    print(f"\nupdated {len(updated_ids)} chunks")
    if not args.re_embed:
        print("embeddings unchanged — re-run with --re-embed to apply contextual headers")
    return 0


def _as_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    raise SystemExit(main())

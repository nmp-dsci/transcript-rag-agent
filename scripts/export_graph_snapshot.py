"""Export the Neo4j knowledge graph as JSON for the demo deployment.

The demo container runs no graph database, but the Knowledge graph view is
part of what the demo exists to show. This script reads the *dev* graph (the
one ``index-graph`` built) and writes the three payloads the API's knowledge
graph GETs serve, so demo mode can answer them from disk:

* ``knowledge.json``     — ``GET /api/graph/knowledge``
* ``entities.json``      — ``GET /api/graph/knowledge/entities/{id}``, keyed by id
* ``video_chunks.json``  — ``GET /api/graph/knowledge/videos/{id}/chunks``, keyed by video

Run with Neo4j up (``docker compose up -d neo4j``):

    PYTHONPATH=. uv run python scripts/export_graph_snapshot.py

Output goes to ``YT_AGENT_GRAPH_SNAPSHOT_PATH`` (default
``.yt-agent/graph_snapshot``), which the demo image build copies in beside the
Chroma snapshot. Derived state: safe to delete, regenerate any time.
"""

from __future__ import annotations

import json
import sys
from typing import Any

from src.api.corpus import list_corpus
from src.config import load_settings
from src.rag.graph_store import GraphStore
from src.rag.graph_view import build_knowledge_graph, chunk_enrichment_for_video, entity_claims


def main() -> int:
    settings = load_settings(require_keys=False)
    out_dir = settings.graph_snapshot_dir
    if out_dir is None:
        print("YT_AGENT_GRAPH_SNAPSHOT_PATH resolved to nothing", file=sys.stderr)
        return 1
    store = GraphStore.from_settings(settings)

    knowledge = build_knowledge_graph(store)
    node_ids = [str(node["id"]) for node in knowledge.get("nodes", [])]

    entities: dict[str, Any] = {}
    for entity_id in node_ids:
        detail = entity_claims(store, entity_id)
        if detail.get("entity") is not None:
            entities[entity_id] = detail

    corpus = list_corpus(
        settings.chroma_path,
        settings.raw_transcript_collection,
        settings.chunk_collection,
    )
    video_chunks: dict[str, Any] = {}
    for video in corpus.get("videos", []):
        video_id = str(video["video_id"])
        enrichment = chunk_enrichment_for_video(store, video_id)
        # Skip empty enrichments: the API's demo path answers a missing key
        # with the same empty shape, and the file stays readable in review.
        if enrichment.get("chunks"):
            video_chunks[video_id] = enrichment

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in (
        ("knowledge.json", knowledge),
        ("entities.json", entities),
        ("video_chunks.json", video_chunks),
    ):
        path = out_dir / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        print(f"wrote {path} ({path.stat().st_size:,} bytes)")

    print(
        f"snapshot: {len(node_ids)} nodes · {len(entities)} entity details · "
        f"{len(video_chunks)} enriched videos"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

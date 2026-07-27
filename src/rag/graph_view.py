"""The Knowledge Graph sub-tab's data source: a real, laid-out entity graph.

GraphRAG's Neo4j store is otherwise invisible from the RAG Pipeline tab — you
can watch chunk retrieval and the chunk similarity graph, but the entity/
claim/community graph a `graph_rag` answer actually reasons over is a black
box. This module gives it the same visual treatment as the chunk graph: a
deterministic 2-D layout computed server-side, so the frontend only has to
project and render.

Chunks get their layout from PCA over embeddings (:mod:`src.rag.graph`);
entities have no embedding, so the natural layout is topological — a
force-directed placement over the same weighted entity graph
(:meth:`~src.rag.graph_store.GraphStore.entity_edges`, RELATES + co-mention)
that Leiden clusters. ``igraph`` already ships as a dependency for
community detection, so Fruchterman-Reingold reuses it rather than adding one.
"""

from __future__ import annotations

import math
from typing import Any

from src.rag.graph_store import GraphStore

_ROUND_TO = 6


def _fr_layout_coordinates(
    nodes: list[str], edges: list[tuple[str, str, float]]
) -> dict[str, tuple[float, float]]:
    """Fruchterman-Reingold layout, centred and scaled into [-1, 1]."""
    if not nodes:
        return {}
    if len(nodes) == 1:
        return {nodes[0]: (0.0, 0.0)}

    import random

    import igraph

    index_of = {node: index for index, node in enumerate(nodes)}
    graph = igraph.Graph(n=len(nodes))
    edge_pairs = [
        (index_of[source], index_of[target])
        for source, target, _weight in edges
        if source in index_of and target in index_of
    ]
    graph.add_edges(edge_pairs)
    # Fruchterman-Reingold starts from a random placement, so an unseeded run
    # reshuffles every node on each request and an entity cannot be re-found
    # by position. Seeding igraph's own generator (rather than the module-level
    # `random`) keeps the layout deterministic without making the rest of the
    # process deterministic too — same reasoning as
    # :func:`~src.rag.communities.detect_communities`.
    igraph.set_random_number_generator(random.Random(0))
    coords = list(graph.layout("fr"))
    return dict(zip(nodes, _normalise(coords)))


def _normalise(coords: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Centre and scale into [-1, 1], preserving the layout's aspect ratio.

    Both axes share one divisor — scaling them independently would stretch a
    narrow axis to full width and imply structure the layout did not find.
    """
    safe = [(x, y) if math.isfinite(x) and math.isfinite(y) else (0.0, 0.0) for x, y in coords]
    xs = [x for x, _ in safe]
    ys = [y for _, y in safe]
    x_mid = (min(xs) + max(xs)) / 2
    y_mid = (min(ys) + max(ys)) / 2
    half_span = max((max(xs) - min(xs)) / 2, (max(ys) - min(ys)) / 2)
    if half_span <= 0:
        return [(0.0, 0.0)] * len(safe)
    return [
        (round((x - x_mid) / half_span, _ROUND_TO), round((y - y_mid) / half_span, _ROUND_TO))
        for x, y in safe
    ]


def build_knowledge_graph(store: GraphStore) -> dict[str, Any]:
    """Entities placed by topology, edges, and the communities they belong to."""
    entities = store.all_entities()
    nodes, raw_edges = store.entity_edges()
    positions = _fr_layout_coordinates(nodes, raw_edges)

    entity_nodes = [
        {
            "id": entity.id,
            "name": entity.name,
            "type": entity.type,
            "mentions": entity.mentions,
            "community_id": entity.community_id,
            "x": positions.get(entity.id, (0.0, 0.0))[0],
            "y": positions.get(entity.id, (0.0, 0.0))[1],
        }
        for entity in entities
    ]
    edges = [
        {"source": source, "target": target, "weight": weight}
        for source, target, weight in raw_edges
    ]
    communities = [
        {
            "id": community.id,
            "summary": community.summary,
            "entity_count": len(community.entity_ids),
            "claim_count": community.claim_count,
        }
        for community in store.communities()
    ]
    return {"nodes": entity_nodes, "edges": edges, "communities": communities}


def chunk_enrichment_for_video(store: GraphStore, video_id: str) -> dict[str, Any]:
    """Every chunk's entity/claim extraction for one video, grouped by chunk index.

    The RAG Pipeline tab's raw-chunk view otherwise shows exactly what vector
    RAG sees — text and nothing else. This is the GraphRAG counterpart: what
    the extraction pass additionally read into that same chunk, so the two
    can sit side by side.
    """
    claims = store.claims_for_video(video_id)
    by_chunk: dict[str, dict[str, Any]] = {}
    for claim in claims:
        bucket = by_chunk.setdefault(str(claim.chunk_index), {"entities": [], "claims": []})
        for name in claim.entities:
            if name not in bucket["entities"]:
                bucket["entities"].append(name)
        bucket["claims"].append({"id": claim.id, "text": claim.text, "polarity": claim.polarity})
    return {"chunks": by_chunk}


def entity_claims(store: GraphStore, entity_id: str, limit: int = 30) -> dict[str, Any]:
    """One entity's metadata plus its dated claim timeline, for the detail panel."""
    entity = store.get_entity(entity_id)
    claims = store.claims_about([entity_id], limit=limit, hops=0)
    return {
        "entity": entity.model_dump() if entity is not None else None,
        "claims": [claim.model_dump() for claim in claims],
    }


def rank_chunks_by_graph(
    store: GraphStore,
    query: str,
    records: list[dict[str, Any]],
    top_k: int,
    video_id: str | None = None,
) -> list[dict[str, Any]]:
    """How ``graph_rag`` would retrieve for this query, for the Retrieval Lab.

    Resolves the query to entities (the same fallback-tolerant resolution the
    local answer path uses), gathers claims about them, and ranks each
    claim's *source chunk* by how many of the resolved entities its claims
    cover — a chunk whose claims mention more of what the query asked about
    ranks higher, the graph analogue of a relevance score. ``records`` (the
    same chunk records BM25/semantic rank) supplies the actual chunk text for
    the preview, so all three modes render identically in the UI.

    ``video_id`` scopes the ranking *before* anything is cut: it is handed to
    :meth:`~src.rag.graph_store.GraphStore.claims_about`, which applies it in
    Cypher ahead of its own ``LIMIT``. Narrowing any later — after the claim
    limit or after ``top_k`` — ranks the corpus and then discards everything
    outside the selected video, leaving the column near-empty even when that
    video has plenty of matching claims.
    """
    from src.agents.graph_agent import question_terms

    terms = question_terms(query)
    entities = store.resolve_entities(terms, limit=8) if terms else []
    if not entities:
        entities = store.resolve_entities(query.lower().split(), limit=8)
    if not entities:
        return []

    entity_names = {entity.name for entity in entities}
    claims = store.claims_about(
        [entity.id for entity in entities], limit=200, hops=0, video_id=video_id
    )

    by_chunk: dict[str, dict[str, Any]] = {}
    for claim in claims:
        # Cheap guard: the store already scoped the query, so this only ever
        # fires for a store that ignores the argument.
        if video_id is not None and claim.video_id != video_id:
            continue
        matched = {name for name in claim.entities if name in entity_names}
        if not matched:
            continue
        # Keyed like ranking.py's chunk_id() ("<video_id>:<chunk_index>"), not
        # claim.chunk_id ("chunk:<video_id>:<index>") — this must match the
        # records lookup below, which uses the same records the other two
        # ranking modes already key that way.
        key = f"{claim.video_id}:{claim.chunk_index}"
        bucket = by_chunk.setdefault(
            key,
            {
                "matched": set(),
                "claims": [],
                "video_id": claim.video_id,
                "chunk_index": claim.chunk_index,
            },
        )
        bucket["matched"] |= matched
        bucket["claims"].append(claim.text)
    if not by_chunk:
        return []

    records_by_id = {
        f"{record.get('video_id')}:{record.get('chunk_index')}": record for record in records
    }
    ranked = sorted(
        by_chunk.items(),
        key=lambda item: (len(item[1]["matched"]), len(item[1]["claims"])),
        reverse=True,
    )[:top_k]

    rows: list[dict[str, Any]] = []
    for chunk_id, bucket in ranked:
        record = records_by_id.get(chunk_id, {})
        rows.append(
            {
                "video_id": bucket["video_id"],
                "chunk_index": bucket["chunk_index"],
                "text": record.get("text") or bucket["claims"][0],
                "start_seconds": record.get("start_seconds"),
                "end_seconds": record.get("end_seconds"),
                "source_url": record.get("source_url"),
                "score": round(len(bucket["matched"]) / len(entity_names), 4),
                "matched_entities": sorted(bucket["matched"]),
            }
        )
    return rows

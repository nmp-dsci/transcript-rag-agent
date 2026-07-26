"""Neo4j-backed store for the GraphRAG knowledge graph (P4).

Schema (all MERGE-idempotent, so re-running index-graph converges):

    (:Entity {id, name, type, aliases, mentions, community_id})
    (:Claim  {id, text, chunk_id, video_id, source_url, video_title,
              upload_date, start_seconds, end_seconds, polarity})
    (:Community {id, summary, entity_count})
    (:Entity)-[:RELATES {type, weight}]->(:Entity)
    (:Claim)-[:ABOUT]->(:Entity)
    (:Entity)-[:IN_COMMUNITY]->(:Community)

The graph is *derived* state: it is rebuilt from the chunk corpus by
``index-graph`` (extraction results are cached by chunk hash, so a rebuild is
cheap). Nothing here is a source of truth — Chroma keeps the chunks, git keeps
the eval runs, and this store keeps the retrieval surface.

Chosen over networkx/Postgres in the s05 review: Neo4j runs as a docker-compose
service (``docker compose up -d neo4j``) and Cypher expresses the two query
shapes GraphRAG needs — entity-anchored subgraphs and per-entity claim
timelines — as single statements.
"""

from __future__ import annotations

import logging
from typing import Any

from src.config import Settings
from src.rag.graph_models import (
    ChunkExtraction,
    GraphClaim,
    GraphCommunity,
    GraphEntity,
    claim_id_for,
    entity_id_for,
)

logger = logging.getLogger(__name__)


class GraphStoreError(RuntimeError):
    pass


class GraphStore:
    """Thin Cypher layer over the neo4j driver.

    The driver import is deferred so modules that only need the record types
    (tests, the extraction contract) never pay for or require the driver.
    """

    def __init__(self, uri: str, user: str, password: str) -> None:
        try:
            from neo4j import GraphDatabase
        except ImportError as exc:  # pragma: no cover - dependency is in pyproject
            raise GraphStoreError("The neo4j driver is not installed (uv sync)") from exc
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self.uri = uri

    @classmethod
    def from_settings(cls, settings: Settings) -> "GraphStore":
        return cls(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password)

    def close(self) -> None:
        self._driver.close()

    def _run(self, query: str, **params: Any) -> list[dict[str, Any]]:
        try:
            with self._driver.session() as session:
                return [record.data() for record in session.run(query, **params)]
        except Exception as exc:
            raise GraphStoreError(
                f"Neo4j query failed ({self.uri}) — is the container up? "
                f"Start it with: docker compose up -d neo4j. Cause: {exc}"
            ) from exc

    # ── construction ──────────────────────────────────────────────────────

    def ensure_schema(self) -> None:
        for statement in (
            "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
            "CREATE CONSTRAINT claim_id IF NOT EXISTS FOR (c:Claim) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT community_id IF NOT EXISTS FOR (m:Community) REQUIRE m.id IS UNIQUE",
        ):
            self._run(statement)

    def wipe(self) -> None:
        """Delete the derived graph (entities, claims, communities) only."""
        self._run("MATCH (n) WHERE n:Entity OR n:Claim OR n:Community DETACH DELETE n")

    def upsert_extraction(self, extraction: ChunkExtraction) -> None:
        """Write one chunk's entities, relations and claims. Idempotent."""
        entities = [
            {
                "id": entity.entity_id,
                "name": entity.name,
                "type": entity.type,
                "aliases": entity.aliases,
            }
            for entity in extraction.entities
        ]
        if entities:
            # The ON MATCH alias expression is a list union — no APOC in the
            # community image, so it is spelled out inline.
            self._run(
                """
                UNWIND $entities AS row
                MERGE (e:Entity {id: row.id})
                ON CREATE SET e.name = row.name, e.type = row.type,
                              e.aliases = row.aliases, e.mentions = 1
                ON MATCH SET e.mentions = e.mentions + 1,
                             e.aliases = [x IN e.aliases WHERE NOT x IN row.aliases]
                                         + row.aliases
                """,
                entities=entities,
            )
        known_ids = {entity.entity_id for entity in extraction.entities}
        relations = [
            {
                "source": entity_id_for(relation.source),
                "target": entity_id_for(relation.target),
                "type": relation.type,
                "weight": relation.weight,
            }
            for relation in extraction.relations
            if entity_id_for(relation.source) in known_ids
            and entity_id_for(relation.target) in known_ids
            and entity_id_for(relation.source) != entity_id_for(relation.target)
        ]
        if relations:
            self._run(
                """
                UNWIND $relations AS row
                MATCH (a:Entity {id: row.source}), (b:Entity {id: row.target})
                MERGE (a)-[r:RELATES {type: row.type}]->(b)
                ON CREATE SET r.weight = row.weight
                ON MATCH SET r.weight = CASE WHEN row.weight > r.weight
                                             THEN row.weight ELSE r.weight END
                """,
                relations=relations,
            )
        claims = [
            {
                "id": claim_id_for(extraction.chunk_id, claim.text),
                "text": claim.text,
                "polarity": claim.polarity,
                "chunk_id": extraction.chunk_id,
                "video_id": extraction.video_id,
                "source_url": extraction.source_url,
                "video_title": extraction.video_title,
                "upload_date": extraction.upload_date,
                "start_seconds": extraction.start_seconds,
                "end_seconds": extraction.end_seconds,
                "entity_ids": [
                    entity_id_for(name)
                    for name in claim.entities
                    if entity_id_for(name) in known_ids
                ],
            }
            for claim in extraction.claims
        ]
        if claims:
            self._run(
                """
                UNWIND $claims AS row
                MERGE (c:Claim {id: row.id})
                SET c.text = row.text, c.polarity = row.polarity,
                    c.chunk_id = row.chunk_id, c.video_id = row.video_id,
                    c.source_url = row.source_url, c.video_title = row.video_title,
                    c.upload_date = row.upload_date,
                    c.start_seconds = row.start_seconds,
                    c.end_seconds = row.end_seconds
                WITH c, row
                UNWIND row.entity_ids AS entity_id
                MATCH (e:Entity {id: entity_id})
                MERGE (c)-[:ABOUT]->(e)
                """,
                claims=claims,
            )

    def counts(self) -> dict[str, int]:
        rows = self._run(
            """
            RETURN count { MATCH (e:Entity) RETURN e } AS entities,
                   count { MATCH (c:Claim) RETURN c } AS claims,
                   count { MATCH ()-[r:RELATES]->() RETURN r } AS relations,
                   count { MATCH (m:Community) RETURN m } AS communities
            """
        )
        return rows[0] if rows else {"entities": 0, "claims": 0, "relations": 0, "communities": 0}

    # ── query ─────────────────────────────────────────────────────────────

    def resolve_entities(self, terms: list[str], limit: int = 6) -> list[GraphEntity]:
        """Entities whose name or aliases contain any query term, by mentions.

        This is the query-time scoping decided in the s05 review: the graph is
        built over the whole corpus and each question anchors its own subgraph
        here instead of any domain pre-filter.
        """
        cleaned = [term.strip().lower() for term in terms if term and term.strip()]
        if not cleaned:
            return []
        rows = self._run(
            """
            UNWIND $terms AS term
            MATCH (e:Entity)
            WHERE toLower(e.name) CONTAINS term
               OR any(alias IN e.aliases WHERE toLower(alias) CONTAINS term)
            WITH DISTINCT e
            RETURN e.id AS id, e.name AS name, e.type AS type,
                   e.aliases AS aliases, e.mentions AS mentions,
                   e.community_id AS community_id
            ORDER BY e.mentions DESC
            LIMIT $limit
            """,
            terms=cleaned,
            limit=limit,
        )
        return [GraphEntity.model_validate(row) for row in rows]

    def claims_about(
        self, entity_ids: list[str], limit: int = 40, hops: int = 1
    ) -> list[GraphClaim]:
        """Claims about the entities (and, with hops=1, their direct neighbours).

        Ordered by ``upload_date`` so the same result set serves both the local
        answer path and the temporal timeline — the temporal layer is a sort,
        not a second store.
        """
        if not entity_ids:
            return []
        anchor = "MATCH (c:Claim)-[:ABOUT]->(e:Entity) WHERE e.id IN $ids"
        if hops >= 1:
            anchor = (
                "MATCH (seed:Entity) WHERE seed.id IN $ids "
                "MATCH (c:Claim)-[:ABOUT]->(e:Entity) "
                "WHERE e.id IN $ids OR (e)-[:RELATES]-(seed)"
            )
        rows = self._run(
            f"""
            {anchor}
            WITH DISTINCT c
            OPTIONAL MATCH (c)-[:ABOUT]->(about:Entity)
            WITH c, collect(about.name) AS entity_names
            RETURN c.id AS id, c.text AS text, entity_names AS entities,
                   c.chunk_id AS chunk_id, c.video_id AS video_id,
                   c.source_url AS source_url, c.video_title AS video_title,
                   c.upload_date AS upload_date, c.start_seconds AS start_seconds,
                   c.end_seconds AS end_seconds, c.polarity AS polarity
            ORDER BY coalesce(c.upload_date, '') ASC, c.video_id, c.start_seconds
            LIMIT $limit
            """,
            ids=entity_ids,
            limit=limit,
        )
        return [GraphClaim.model_validate(row) for row in rows]

    def entity_edges(self) -> tuple[list[str], list[tuple[str, str, float]]]:
        """The entity graph for community detection.

        Edges combine explicit RELATES weight with co-mention counts (two
        entities named by the same claim), because extraction reliably
        co-mentions entities it does not always bother to relate.
        """
        node_rows = self._run("MATCH (e:Entity) RETURN e.id AS id")
        nodes = [row["id"] for row in node_rows]
        edges: dict[tuple[str, str], float] = {}
        for row in self._run(
            """
            MATCH (a:Entity)-[r:RELATES]->(b:Entity)
            RETURN a.id AS source, b.id AS target, r.weight AS weight
            """
        ):
            key = tuple(sorted((row["source"], row["target"])))
            edges[key] = edges.get(key, 0.0) + float(row["weight"] or 0.5)
        for row in self._run(
            """
            MATCH (a:Entity)<-[:ABOUT]-(c:Claim)-[:ABOUT]->(b:Entity)
            WHERE a.id < b.id
            RETURN a.id AS source, b.id AS target, count(c) AS co_mentions
            """
        ):
            key = (row["source"], row["target"])
            edges[key] = edges.get(key, 0.0) + float(row["co_mentions"])
        return nodes, [(source, target, weight) for (source, target), weight in edges.items()]

    # ── communities ───────────────────────────────────────────────────────

    def store_communities(self, assignments: dict[str, int]) -> None:
        """Replace community membership with a fresh Leiden assignment."""
        self._run("MATCH (m:Community) DETACH DELETE m")
        rows = [
            {"entity_id": entity_id, "community_id": community_id}
            for entity_id, community_id in assignments.items()
        ]
        self._run(
            """
            UNWIND $rows AS row
            MATCH (e:Entity {id: row.entity_id})
            SET e.community_id = row.community_id
            MERGE (m:Community {id: row.community_id})
            MERGE (e)-[:IN_COMMUNITY]->(m)
            """,
            rows=rows,
        )

    def set_community_summary(self, community_id: int, summary: str) -> None:
        self._run(
            "MATCH (m:Community {id: $id}) SET m.summary = $summary",
            id=community_id,
            summary=summary,
        )

    def communities(self) -> list[GraphCommunity]:
        rows = self._run(
            """
            MATCH (m:Community)
            OPTIONAL MATCH (e:Entity)-[:IN_COMMUNITY]->(m)
            OPTIONAL MATCH (c:Claim)-[:ABOUT]->(e)
            WITH m, collect(DISTINCT e.id) AS entity_ids,
                 collect(DISTINCT e.name) AS entity_names,
                 count(DISTINCT c) AS claim_count
            RETURN m.id AS id, m.summary AS summary, entity_ids, entity_names,
                   claim_count
            ORDER BY claim_count DESC
            """
        )
        return [GraphCommunity.model_validate(row) for row in rows]

    def top_claims_for_community(self, community_id: int, limit: int = 12) -> list[GraphClaim]:
        rows = self._run(
            """
            MATCH (c:Claim)-[:ABOUT]->(e:Entity)-[:IN_COMMUNITY]->(m:Community {id: $id})
            WITH DISTINCT c, count(e) AS anchored
            OPTIONAL MATCH (c)-[:ABOUT]->(about:Entity)
            WITH c, anchored, collect(about.name) AS entity_names
            RETURN c.id AS id, c.text AS text, entity_names AS entities,
                   c.chunk_id AS chunk_id, c.video_id AS video_id,
                   c.source_url AS source_url, c.video_title AS video_title,
                   c.upload_date AS upload_date, c.start_seconds AS start_seconds,
                   c.end_seconds AS end_seconds, c.polarity AS polarity
            ORDER BY anchored DESC, coalesce(c.upload_date, '') ASC
            LIMIT $limit
            """,
            id=community_id,
            limit=limit,
        )
        return [GraphClaim.model_validate(row) for row in rows]

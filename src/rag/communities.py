"""Community detection and summaries over the entity graph (GraphRAG P4).

Leiden communities via ``igraph`` (its built-in implementation — no
``leidenalg`` dependency) over the weighted entity graph the store exports:
RELATES edges plus claim co-mention counts.

Per the s05 review decision this is **Full GraphRAG**: every community gets an
LLM summary up-front at index time (the measured whole-corpus bill is cents),
so global questions are answered by reducing over pre-built summaries instead
of paying map calls per query.
"""

from __future__ import annotations

import logging
from typing import Callable, Protocol

from src.agents.prompts import build_community_summary_prompt
from src.rag.graph_models import GraphClaim
from src.rag.graph_store import GraphStore

logger = logging.getLogger(__name__)


class ChatModel(Protocol):
    def invoke(self, messages: list) -> object: ...


def detect_communities(nodes: list[str], edges: list[tuple[str, str, float]]) -> dict[str, int]:
    """Leiden assignment for every node, isolated nodes included.

    Isolated entities (extracted once, related to nothing) each land in their
    own community; they carry no theme, and downstream summaries skip
    single-entity communities with no claims.
    """
    if not nodes:
        return {}
    import random

    import igraph

    index_of = {node: index for index, node in enumerate(nodes)}
    graph = igraph.Graph(n=len(nodes))
    edge_pairs = [
        (index_of[source], index_of[target])
        for source, target, _weight in edges
        if source in index_of and target in index_of
    ]
    weights = [
        weight for source, target, weight in edges if source in index_of and target in index_of
    ]
    graph.add_edges(edge_pairs)
    # A fixed seed keeps community assignments (and their downstream LLM
    # summaries) stable across re-runs of an unchanged corpus.
    random.seed(0)
    clustering = graph.community_leiden(
        objective_function="modularity",
        weights=weights or None,
        n_iterations=5,
    )
    return {node: clustering.membership[index_of[node]] for node in nodes}


def format_claims_block(claims: list[GraphClaim]) -> str:
    lines = [f"- [{claim.upload_date or 'undated'}] {claim.text}" for claim in claims]
    return "\n".join(lines)


class CommunitySummarizer:
    """LLM summaries for detected communities (the Full-GraphRAG map pass)."""

    def __init__(self, llm: ChatModel) -> None:
        self.llm = llm

    def summarize(self, entity_names: list[str], claims: list[GraphClaim]) -> str:
        from langchain_core.messages import HumanMessage

        prompt = build_community_summary_prompt(entity_names, format_claims_block(claims))
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return str(getattr(response, "content", response) or "").strip()


def build_communities(
    store: GraphStore,
    summarizer: CommunitySummarizer | None,
    on_progress: Callable[[str], None] | None = None,
    min_entities: int = 2,
    claims_per_summary: int = 12,
) -> int:
    """Detect Leiden communities and (with a summarizer) summarize each.

    Returns the number of communities summarized. Communities below
    ``min_entities`` with no claims are stored but not summarized — a lone
    entity is not a theme.
    """
    nodes, edges = store.entity_edges()
    assignments = detect_communities(nodes, edges)
    store.store_communities(assignments)
    if on_progress is not None:
        community_count = len(set(assignments.values()))
        on_progress(
            f"Leiden: {len(nodes)} entities, {len(edges)} edges → {community_count} communities"
        )
    if summarizer is None:
        return 0
    summarized = 0
    for community in store.communities():
        claims = store.top_claims_for_community(community.id, limit=claims_per_summary)
        if len(community.entity_ids) < min_entities and not claims:
            continue
        summary = summarizer.summarize(community.entity_names, claims)
        if not summary:
            continue
        store.set_community_summary(community.id, summary)
        summarized += 1
        if on_progress is not None:
            on_progress(
                f"community {community.id}: {len(community.entity_ids)} entities, "
                f"{len(claims)} claims → summarized"
            )
    return summarized

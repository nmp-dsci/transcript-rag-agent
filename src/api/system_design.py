"""The System Design tab's data source: a node/edge graph of the whole app.

Where :mod:`src.api.prompts` groups prompts by *system* for a flat list, this
module places those same systems on a graph alongside the infrastructure they
depend on — the Chroma collections, the Neo4j store, the embedding/rerank/LLM
models — so the Prompts tab's "what does this say" becomes "how does the app
fit together, and what does each part say and run with".

Every node carries the prompts scoped to it (from :data:`PROMPT_REGISTRY`) and
a ``config`` dict of the live :class:`~src.config.Settings` values that shape
its behaviour — read directly off the running ``Settings`` instance, so this
can never drift from what the server is actually configured with. Layout is a
fixed, hand-placed grid rather than a client-side force layout: the graph is
small and stable (new nodes only appear when a new answer path or store is
added), so a computed layout would add a dependency for no real benefit.

Agent nodes also carry ``flow``: the ordered, numbered steps a question
actually takes through that answer path, with the real ``top_k``/iteration
caps interpolated from settings — "what happens when a question comes in",
answering it the same way ``config`` answers "what does this run with": from
the live values, not a paraphrase that can drift from the code.
"""

from __future__ import annotations

from typing import Any

from src.agents.graph_agent import (
    DEFAULT_MAX_CLAIMS,
    DEFAULT_MAX_COMMUNITIES,
    DEFAULT_TIMELINE_CLAIMS,
)
from src.api.prompts import load_prompts
from src.config import Settings


def _prompts_by_system() -> dict[str, list[dict[str, Any]]]:
    """Reuse the Prompts tab's own grouping so both surfaces agree by construction."""
    return {system["key"]: system["prompts"] for system in load_prompts()["systems"]}


def _step(label: str, detail: str, branch: str | None = None) -> dict[str, Any]:
    return {"label": label, "detail": detail, "branch": branch}


def _chat_flow() -> list[dict[str, Any]]:
    return [
        _step(
            "Load transcript",
            "Fetch the full raw transcript for the given video — no chunking, no retrieval.",
        ),
        _step("Answer", "One LLM call over the complete transcript text."),
    ]


def _vector_rag_flow(settings: Settings) -> list[dict[str, Any]]:
    candidates = max(settings.rag_top_k, settings.retrieval_candidates)
    widen = settings.rerank_enabled or settings.retrieval_mode == "hybrid"
    steps = [
        _step(
            "Summary filter (optional)",
            f"If enabled: narrow to the {settings.transcript_filter_top_k} most relevant videos "
            "by per-video summary before retrieving chunks.",
        ),
        _step(
            "Retrieve",
            "{mode} search over transcript_chunks — top {n} candidates{widened}.".format(
                mode="Semantic + BM25, fused with RRF"
                if settings.retrieval_mode == "hybrid"
                else "Semantic",
                n=candidates if widen else settings.rag_top_k,
                widened=", widened for reranking" if widen else "",
            ),
        ),
    ]
    if settings.rerank_enabled:
        steps.append(
            _step(
                "Rerank",
                f"Cross-encoder ({settings.rerank_model}) reorders the candidates down to top_k={settings.rag_top_k}.",
            )
        )
    steps.append(
        _step(
            "Answer",
            f"One LLM call over the {settings.rag_top_k} retrieved chunks, citing chunk ids.",
        )
    )
    return steps


def _recursive_rag_flow(settings: Settings) -> list[dict[str, Any]]:
    return [
        _step(
            "Retrieve",
            f"Same as vector_rag: top_k={settings.rag_top_k} chunks for the question as asked.",
        ),
        _step(
            "First-pass answer",
            "One LLM call answers from that first retrieval and proposes follow-up subtopics.",
        ),
        _step(
            "Follow-up retrieval",
            f"Up to {settings.rag_max_followups} subtopics each get their own retrieval "
            f"(skipped if a subtopic would add fewer than {settings.rag_novelty_min_chunks} new chunks).",
        ),
        _step("Merge", "Deduplicate and merge every retrieval's chunks into one context."),
        _step("Synthesize", "One final LLM call answers over the merged context."),
    ]


def _agentic_rag_flow(settings: Settings) -> list[dict[str, Any]]:
    return [
        _step("ReAct loop", "A LangGraph loop: the model decides whether to retrieve or answer."),
        _step(
            "Retrieve (tool call)",
            "Each retrieval is a tool call for one sub-query the model chose; it can call this "
            f"repeatedly, up to {settings.rag_agent_max_iterations} iterations.",
        ),
        _step(
            "Answer",
            "Once the model judges it has enough evidence, it emits the final cited answer.",
        ),
    ]


def _graph_rag_flow() -> list[dict[str, Any]]:
    return [
        _step(
            "Route",
            "One LLM call classifies the question as local, global, or temporal, and names its entities.",
        ),
        _step(
            "Resolve entities",
            "Match the named entities (or, if none matched, the question's content words) against the graph.",
            branch="local",
        ),
        _step(
            "Retrieve",
            f"Up to {DEFAULT_MAX_CLAIMS} graph claims about those entities, PLUS a normal vector "
            "retrieval of transcript chunks (same as vector_rag) — both feed the same answer call.",
            branch="local",
        ),
        _step(
            "Reduce",
            "Read the pre-built community summaries (Full GraphRAG summarizes every community "
            f"at index time) — up to {DEFAULT_MAX_COMMUNITIES} communities, 2 representative claims each.",
            branch="global",
        ),
        _step(
            "Resolve entities",
            "Match the named entities against the graph, same as the local route.",
            branch="temporal",
        ),
        _step(
            "Retrieve",
            f"Up to {DEFAULT_TIMELINE_CLAIMS} dated claims about those entities, ordered oldest first.",
            branch="temporal",
        ),
        _step(
            "Answer",
            "One LLM call narrates over whichever route's evidence, citing claims/chunks/communities.",
        ),
    ]


def build_system_design(settings: Settings) -> dict[str, Any]:
    """Nodes (agents, models, stores) + edges (what depends on what)."""
    prompts_by_system = _prompts_by_system()
    rerank_config: dict[str, Any] = {"enabled": settings.rerank_enabled}
    if settings.rerank_enabled:
        rerank_config["model"] = settings.rerank_model

    nodes = [
        # ── answer paths ────────────────────────────────────────────────
        {
            "id": "chat",
            "label": "Chat",
            "kind": "agent",
            "x": 120,
            "y": 70,
            "description": "Direct-transcript Q&A and summarization over one full video.",
            "prompts": prompts_by_system.get("chat", []),
            "flow": _chat_flow(),
            "config": {
                "model": settings.deepseek_model,
                "context": "one full raw transcript, no chunking",
            },
        },
        {
            "id": "vector_rag",
            "label": "Vector RAG",
            "kind": "agent",
            "x": 340,
            "y": 70,
            "description": "Single-hop RAG: one retrieval across the corpus, then one answer call.",
            "prompts": prompts_by_system.get("vector_rag", []),
            "flow": _vector_rag_flow(settings),
            "config": {
                "model": settings.deepseek_model,
                "embedding_model": settings.embedding_model,
                "retrieval_mode": settings.retrieval_mode,
                "top_k": settings.rag_top_k,
                "retrieval_candidates": settings.retrieval_candidates,
                "neighbor_span": settings.neighbor_span,
                **rerank_config,
            },
        },
        {
            "id": "recursive_rag",
            "label": "Recursive RAG",
            "kind": "agent",
            "x": 560,
            "y": 70,
            "description": "vector_rag plus a fan-out of follow-up retrievals and a synthesis call.",
            "prompts": prompts_by_system.get("recursive_rag", []),
            "flow": _recursive_rag_flow(settings),
            "config": {
                "model": settings.deepseek_model,
                "max_depth": settings.rag_max_depth,
                "max_followups": settings.rag_max_followups,
                "novelty_min_chunks": settings.rag_novelty_min_chunks,
            },
        },
        {
            "id": "agentic_rag",
            "label": "Agentic RAG",
            "kind": "agent",
            "x": 780,
            "y": 70,
            "description": "LangGraph ReAct loop: retrieves across sub-topics until it has enough evidence.",
            "prompts": prompts_by_system.get("agentic_rag", []),
            "flow": _agentic_rag_flow(settings),
            "config": {
                "model": settings.deepseek_model,
                "max_iterations": settings.rag_agent_max_iterations,
            },
        },
        {
            "id": "graph_rag",
            "label": "GraphRAG",
            "kind": "agent",
            "x": 1000,
            "y": 70,
            "description": (
                "Routes local/global/temporal, answers over the Neo4j "
                "entity/claim knowledge graph plus vector retrieval."
            ),
            "prompts": prompts_by_system.get("graph_rag", []),
            "flow": _graph_rag_flow(),
            "config": {
                "model": settings.deepseek_model,
                "neo4j_uri": settings.neo4j_uri,
                "graph_cache_dir": str(settings.graph_cache_dir)
                if settings.graph_cache_dir
                else None,
            },
        },
        {
            "id": "summary_filter",
            "label": "Summary filter",
            "kind": "stage",
            "x": 560,
            "y": 220,
            "description": (
                "Optional pre-filter used by the RAG agents: picks the most "
                "relevant videos by per-video summary before chunk retrieval."
            ),
            "prompts": prompts_by_system.get("summary_filter", []),
            "flow": [],
            "config": {
                "filter_top_k": settings.transcript_filter_top_k,
                "min_score": settings.transcript_filter_min_score,
            },
        },
        # ── shared models ───────────────────────────────────────────────
        {
            "id": "deepseek",
            "label": "DeepSeek LLM",
            "kind": "model",
            "x": 340,
            "y": 380,
            "description": "The chat-completion model every agent and the extractor answers with.",
            "prompts": [],
            "flow": [],
            "config": {
                "model": settings.deepseek_model,
                "base_url": settings.deepseek_base_url,
            },
        },
        {
            "id": "embeddings",
            "label": "Embedding model",
            "kind": "model",
            "x": 560,
            "y": 380,
            "description": "Local dense embedding model used to vectorize chunks and queries.",
            "prompts": [],
            "flow": [],
            "config": {"model": settings.embedding_model},
        },
        {
            "id": "reranker",
            "label": "Cross-encoder reranker",
            "kind": "model",
            "x": 780,
            "y": 380,
            "description": "Reorders wide-retrieved candidates down to top_k before answering.",
            "prompts": [],
            "flow": [],
            "config": rerank_config,
        },
        # ── stores ──────────────────────────────────────────────────────
        {
            "id": "chroma_raw",
            "label": "Chroma · raw_transcripts",
            "kind": "store",
            "x": 120,
            "y": 520,
            "description": "Full raw transcript text per video, keyed by video id.",
            "prompts": [],
            "flow": [],
            "config": {
                "collection": settings.raw_transcript_collection,
                "path": str(settings.chroma_path),
            },
        },
        {
            "id": "chroma_chunks",
            "label": "Chroma · transcript_chunks",
            "kind": "store",
            "x": 340,
            "y": 520,
            "description": "Timestamp-aware transcript chunks with dense embeddings.",
            "prompts": [],
            "flow": [],
            "config": {
                "collection": settings.chunk_collection,
                "path": str(settings.chroma_path),
                "chunk_target_chars": settings.chunk_target_chars,
                "chunk_overlap_chars": settings.chunk_overlap_chars,
            },
        },
        {
            "id": "chroma_summaries",
            "label": "Chroma · transcript_summaries",
            "kind": "store",
            "x": 560,
            "y": 520,
            "description": "One embedded summary per video, used by the summary filter.",
            "prompts": [],
            "flow": [],
            "config": {
                "collection": settings.transcript_summary_collection,
                "path": str(settings.chroma_path),
            },
        },
        {
            "id": "neo4j",
            "label": "Neo4j · knowledge graph",
            "kind": "store",
            "x": 1000,
            "y": 520,
            "description": (
                "Entities, relations, claims and communities extracted from "
                "the chunk corpus (P4 GraphRAG). Derived state, rebuilt by "
                "index-graph."
            ),
            "prompts": [],
            "flow": [],
            "config": {"uri": settings.neo4j_uri, "user": settings.neo4j_user},
        },
    ]

    edges = [
        {"source": "chat", "target": "deepseek"},
        {"source": "chat", "target": "chroma_raw"},
        {"source": "vector_rag", "target": "deepseek"},
        {"source": "vector_rag", "target": "embeddings"},
        {"source": "vector_rag", "target": "chroma_chunks"},
        {"source": "vector_rag", "target": "reranker"},
        {"source": "vector_rag", "target": "summary_filter"},
        {"source": "recursive_rag", "target": "deepseek"},
        {"source": "recursive_rag", "target": "chroma_chunks"},
        {"source": "agentic_rag", "target": "deepseek"},
        {"source": "agentic_rag", "target": "chroma_chunks"},
        {"source": "graph_rag", "target": "deepseek"},
        {"source": "graph_rag", "target": "neo4j"},
        {"source": "graph_rag", "target": "chroma_chunks"},
        {"source": "summary_filter", "target": "chroma_summaries"},
        {"source": "summary_filter", "target": "deepseek"},
    ]

    return {"nodes": nodes, "edges": edges}

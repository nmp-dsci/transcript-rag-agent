"""The System Design graph: every node resolvable, edges point at real nodes,
prompts scoped correctly, and config reflects the live Settings instance."""

from __future__ import annotations

from src.api.system_design import build_system_design
from src.config import load_settings


def _settings():
    return load_settings(require_keys=False)


def test_every_edge_endpoint_is_a_real_node() -> None:
    design = build_system_design(_settings())
    node_ids = {node["id"] for node in design["nodes"]}
    for edge in design["edges"]:
        assert edge["source"] in node_ids, edge
        assert edge["target"] in node_ids, edge


def test_node_ids_are_unique() -> None:
    design = build_system_design(_settings())
    ids = [node["id"] for node in design["nodes"]]
    assert len(ids) == len(set(ids))


def test_agent_nodes_carry_their_own_prompts_only() -> None:
    design = build_system_design(_settings())
    by_id = {node["id"]: node for node in design["nodes"]}

    graph_rag_names = {p["name"] for p in by_id["graph_rag"]["prompts"]}
    assert "GRAPH_ANSWER_SYSTEM_PROMPT" in graph_rag_names
    assert "RAG_SYSTEM_PROMPT" not in graph_rag_names  # belongs to vector_rag

    vector_rag_names = {p["name"] for p in by_id["vector_rag"]["prompts"]}
    assert "RAG_SYSTEM_PROMPT" in vector_rag_names
    assert "GRAPH_ANSWER_SYSTEM_PROMPT" not in vector_rag_names


def test_store_and_model_nodes_carry_no_prompts() -> None:
    design = build_system_design(_settings())
    by_id = {node["id"]: node for node in design["nodes"]}
    for node_id in ("chroma_chunks", "chroma_raw", "neo4j", "deepseek", "embeddings"):
        assert by_id[node_id]["prompts"] == []


def test_config_reflects_live_settings() -> None:
    settings = _settings()
    design = build_system_design(settings)
    by_id = {node["id"]: node for node in design["nodes"]}

    assert by_id["vector_rag"]["config"]["embedding_model"] == settings.embedding_model
    assert by_id["vector_rag"]["config"]["top_k"] == settings.rag_top_k
    assert by_id["graph_rag"]["config"]["neo4j_uri"] == settings.neo4j_uri
    assert by_id["neo4j"]["config"]["uri"] == settings.neo4j_uri
    assert by_id["chroma_chunks"]["config"]["collection"] == settings.chunk_collection


def test_neo4j_password_is_never_exposed() -> None:
    design = build_system_design(_settings())
    for node in design["nodes"]:
        assert "password" not in str(node["config"]).lower()


def test_graph_rag_node_has_no_reranker_edge() -> None:
    """graph_rag answers over the knowledge graph, not the cross-encoder path."""
    design = build_system_design(_settings())
    graph_rag_targets = {
        edge["target"] for edge in design["edges"] if edge["source"] == "graph_rag"
    }
    assert "reranker" not in graph_rag_targets
    assert "neo4j" in graph_rag_targets


def test_non_agent_nodes_carry_no_flow() -> None:
    design = build_system_design(_settings())
    by_id = {node["id"]: node for node in design["nodes"]}
    for node_id in ("summary_filter", "chroma_chunks", "neo4j", "deepseek", "embeddings"):
        assert by_id[node_id]["flow"] == []


def test_every_agent_flow_ends_in_an_answer_step() -> None:
    """Every path terminates in one identifiable final call — the point the
    judge scores and the point a reader looks for first. recursive_rag's is
    "Synthesize" rather than "Answer", precise about what that call does."""
    design = build_system_design(_settings())
    for node in design["nodes"]:
        if node["kind"] != "agent":
            continue
        assert node["flow"], node["id"]
        assert node["flow"][-1]["label"] in {"Answer", "Synthesize"}, node["id"]


def test_vector_rag_flow_cites_the_live_top_k_and_reranker() -> None:
    settings = _settings()
    design = build_system_design(settings)
    by_id = {node["id"]: node for node in design["nodes"]}
    detail = " ".join(step["detail"] for step in by_id["vector_rag"]["flow"])
    assert f"top_k={settings.rag_top_k}" in detail
    if settings.rerank_enabled:
        assert settings.rerank_model in detail
        assert any(step["label"] == "Rerank" for step in by_id["vector_rag"]["flow"])


def test_vector_rag_flow_omits_rerank_step_when_disabled() -> None:
    settings = _settings()
    object.__setattr__(settings, "rerank_enabled", False)
    design = build_system_design(settings)
    by_id = {node["id"]: node for node in design["nodes"]}
    labels = [step["label"] for step in by_id["vector_rag"]["flow"]]
    assert "Rerank" not in labels


def test_recursive_rag_flow_cites_live_followup_settings() -> None:
    settings = _settings()
    design = build_system_design(settings)
    by_id = {node["id"]: node for node in design["nodes"]}
    detail = " ".join(step["detail"] for step in by_id["recursive_rag"]["flow"])
    assert str(settings.rag_max_followups) in detail
    assert str(settings.rag_novelty_min_chunks) in detail


def test_agentic_rag_flow_cites_the_live_iteration_cap() -> None:
    settings = _settings()
    design = build_system_design(settings)
    by_id = {node["id"]: node for node in design["nodes"]}
    detail = " ".join(step["detail"] for step in by_id["agentic_rag"]["flow"])
    assert str(settings.rag_agent_max_iterations) in detail


def test_graph_rag_flow_covers_all_three_routes() -> None:
    """The whole point: a reader can see local retrieves chunks AND claims,
    while global/temporal never touch the vector store."""
    design = build_system_design(_settings())
    by_id = {node["id"]: node for node in design["nodes"]}
    flow = by_id["graph_rag"]["flow"]
    branches = {step["branch"] for step in flow if step["branch"]}
    assert branches == {"local", "global", "temporal"}

    local_detail = " ".join(step["detail"] for step in flow if step["branch"] == "local")
    assert "graph claims" in local_detail
    assert "vector retrieval" in local_detail

    global_detail = " ".join(step["detail"] for step in flow if step["branch"] == "global")
    assert "communities" in global_detail

    temporal_detail = " ".join(step["detail"] for step in flow if step["branch"] == "temporal")
    assert "dated claims" in temporal_detail


def test_graph_rag_flow_evidence_caps_match_the_agent_defaults() -> None:
    """A hardcoded number here that drifted from the agent's real caps would
    describe a flow the code doesn't actually run."""
    from src.agents.graph_agent import (
        DEFAULT_MAX_CLAIMS,
        DEFAULT_MAX_COMMUNITIES,
        DEFAULT_TIMELINE_CLAIMS,
    )

    design = build_system_design(_settings())
    by_id = {node["id"]: node for node in design["nodes"]}
    detail = " ".join(step["detail"] for step in by_id["graph_rag"]["flow"])
    assert str(DEFAULT_MAX_CLAIMS) in detail
    assert str(DEFAULT_MAX_COMMUNITIES) in detail
    assert str(DEFAULT_TIMELINE_CLAIMS) in detail

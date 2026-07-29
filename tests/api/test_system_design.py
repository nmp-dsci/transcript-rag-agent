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


def test_vector_rag_config_keeps_the_answering_model_when_rerank_is_on() -> None:
    """The rerank config is merged into vector_rag's, so its model key must not
    collide with the answering LLM — otherwise the node's only ``model`` row
    names the cross-encoder and the LLM disappears from the panel."""
    settings = _settings()
    object.__setattr__(settings, "rerank_enabled", True)
    design = build_system_design(settings)
    by_id = {node["id"]: node for node in design["nodes"]}

    config = by_id["vector_rag"]["config"]
    assert config["model"] == settings.deepseek_model
    assert config["rerank_model"] == settings.rerank_model
    assert by_id["reranker"]["config"]["rerank_model"] == settings.rerank_model


def test_vector_rag_flow_names_the_reason_the_retrieval_is_widened() -> None:
    """Widening happens for hybrid fusion *or* reranking. Crediting it to
    reranking while reranking is off contradicts the same panel's config."""
    settings = _settings()
    object.__setattr__(settings, "retrieval_mode", "hybrid")
    object.__setattr__(settings, "rerank_enabled", False)
    design = build_system_design(settings)
    by_id = {node["id"]: node for node in design["nodes"]}
    detail = " ".join(step["detail"] for step in by_id["vector_rag"]["flow"])
    assert "widened for hybrid fusion" in detail
    assert "widened for reranking" not in detail

    object.__setattr__(settings, "rerank_enabled", True)
    design = build_system_design(settings)
    by_id = {node["id"]: node for node in design["nodes"]}
    detail = " ".join(step["detail"] for step in by_id["vector_rag"]["flow"])
    assert "widened for reranking" in detail

    object.__setattr__(settings, "retrieval_mode", "semantic")
    object.__setattr__(settings, "rerank_enabled", False)
    design = build_system_design(settings)
    by_id = {node["id"]: node for node in design["nodes"]}
    detail = " ".join(step["detail"] for step in by_id["vector_rag"]["flow"])
    assert "widened" not in detail


def test_vector_rag_flow_accounts_for_neighbor_expansion() -> None:
    """neighbor_span widens the context *after* the top_k cut, and is shown in
    the same node's config table — a bare "top_k chunks" would contradict it."""
    settings = _settings()
    object.__setattr__(settings, "neighbor_span", 2)
    design = build_system_design(settings)
    by_id = {node["id"]: node for node in design["nodes"]}
    detail = " ".join(step["detail"] for step in by_id["vector_rag"]["flow"])
    assert "neighbor_span" in detail
    assert "2 chunks either side" in detail

    object.__setattr__(settings, "neighbor_span", 0)
    design = build_system_design(settings)
    by_id = {node["id"]: node for node in design["nodes"]}
    detail = " ".join(step["detail"] for step in by_id["vector_rag"]["flow"])
    assert "neighbor_span" not in detail


def test_recursive_rag_flow_accounts_for_neighbor_expansion() -> None:
    """recursive_rag retrieves through the same provider, so the same widening
    applies to its first retrieval."""
    settings = _settings()
    object.__setattr__(settings, "neighbor_span", 1)
    design = build_system_design(settings)
    by_id = {node["id"]: node for node in design["nodes"]}
    detail = " ".join(step["detail"] for step in by_id["recursive_rag"]["flow"])
    assert "1 chunk either side" in detail


def test_recursive_rag_flow_drops_the_fan_out_when_max_depth_is_zero() -> None:
    """max_depth=0 makes _answer_recursive return the first-pass answer, so the
    follow-up/merge/synthesize steps describe calls that never happen."""
    settings = _settings()
    object.__setattr__(settings, "rag_max_depth", 0)
    design = build_system_design(settings)
    by_id = {node["id"]: node for node in design["nodes"]}
    labels = [step["label"] for step in by_id["recursive_rag"]["flow"]]
    assert labels == ["Retrieve", "First-pass answer", "Answer"]


def test_recursive_rag_flow_drops_the_fan_out_when_total_followups_is_zero() -> None:
    """max_total_followups=0 blocks every fan-out retrieval, so _answer_recursive
    returns the first pass with reason max_total_followups_reached."""
    settings = _settings()
    object.__setattr__(settings, "rag_max_total_followups", 0)
    design = build_system_design(settings)
    by_id = {node["id"]: node for node in design["nodes"]}
    flow = by_id["recursive_rag"]["flow"]
    assert [step["label"] for step in flow] == ["Retrieve", "First-pass answer", "Answer"]
    assert "max_total_followups=0" in flow[-1]["detail"]


def test_recursive_rag_flow_keeps_the_fan_out_for_a_positive_total_cap() -> None:
    settings = _settings()
    object.__setattr__(settings, "rag_max_total_followups", 2)
    design = build_system_design(settings)
    by_id = {node["id"]: node for node in design["nodes"]}
    labels = [step["label"] for step in by_id["recursive_rag"]["flow"]]
    assert "Follow-up retrieval" in labels
    assert "Synthesize" in labels


def test_recursive_rag_flow_notes_the_one_round_cap_above_max_depth_one() -> None:
    """The agent clamps max_depth to a single round; a config table showing 3
    beside a flow silently running 1 is exactly the drift this flow prevents."""
    settings = _settings()
    object.__setattr__(settings, "rag_max_depth", 3)
    design = build_system_design(settings)
    by_id = {node["id"]: node for node in design["nodes"]}
    detail = " ".join(step["detail"] for step in by_id["recursive_rag"]["flow"])
    assert "max_depth=3" in detail
    assert "one round" in detail


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


# ── retrieval variants ────────────────────────────────────────────────────────


def test_the_retrieval_variants_appear_as_their_own_agent_nodes() -> None:
    """Two more answer paths exist in the Chat tab; the map must show them."""
    design = build_system_design(_settings())
    ids = {node["id"] for node in design["nodes"]}

    assert {"hyde_rag", "contextual_rag"} <= ids


def test_the_contextual_variant_reads_the_contextual_store_and_not_the_baseline() -> None:
    design = build_system_design(_settings())
    targets = {edge["target"] for edge in design["edges"] if edge["source"] == "contextual_rag"}

    assert "chroma_contextual" in targets
    assert "chroma_chunks" not in targets


def test_hyde_reads_the_baseline_store_since_only_the_query_changes() -> None:
    design = build_system_design(_settings())
    targets = {edge["target"] for edge in design["edges"] if edge["source"] == "hyde_rag"}

    assert "chroma_chunks" in targets
    assert "chroma_contextual" not in targets


def test_the_hyde_flow_leads_with_writing_the_probe() -> None:
    design = build_system_design(_settings())
    by_id = {node["id"]: node for node in design["nodes"]}

    assert by_id["hyde_rag"]["flow"][0]["label"] == "Write hypothetical passage"


def test_the_contextual_flow_names_the_collection_it_searches() -> None:
    settings = _settings()
    by_id = {node["id"]: node for node in build_system_design(settings)["nodes"]}

    assert settings.contextual_chunk_collection in by_id["contextual_rag"]["flow"][0]["detail"]
    assert by_id["chroma_contextual"]["config"]["collection"] == (
        settings.contextual_chunk_collection
    )

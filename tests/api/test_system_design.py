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

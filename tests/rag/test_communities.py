"""Leiden detection and community-summary formatting, no Neo4j required."""

from __future__ import annotations

from src.rag.communities import detect_communities, format_claims_block
from src.rag.graph_models import GraphClaim


def test_detect_communities_separates_disconnected_clusters() -> None:
    nodes = ["a", "b", "c", "x", "y", "z"]
    edges = [
        ("a", "b", 3.0),
        ("b", "c", 3.0),
        ("a", "c", 2.0),
        ("x", "y", 3.0),
        ("y", "z", 3.0),
        ("x", "z", 2.0),
    ]
    assignments = detect_communities(nodes, edges)
    assert set(assignments) == set(nodes)
    assert assignments["a"] == assignments["b"] == assignments["c"]
    assert assignments["x"] == assignments["y"] == assignments["z"]
    assert assignments["a"] != assignments["x"]


def test_detect_communities_handles_empty_and_isolated_nodes() -> None:
    assert detect_communities([], []) == {}
    lonely = detect_communities(["solo", "duo1", "duo2"], [("duo1", "duo2", 1.0)])
    assert lonely["duo1"] == lonely["duo2"]
    assert lonely["solo"] != lonely["duo1"]


def test_detect_communities_is_seeded_without_touching_the_global_rng() -> None:
    """Assignments stay stable across runs, but index-graph shares its process
    with the rest of the pipeline, which must not inherit a fixed seed."""
    import random

    nodes = ["a", "b", "c", "x", "y", "z"]
    edges = [("a", "b", 3.0), ("b", "c", 3.0), ("x", "y", 3.0), ("y", "z", 3.0)]

    random.seed(1234)
    expected = random.random()
    random.seed(1234)
    first = detect_communities(nodes, edges)

    assert random.random() == expected
    assert detect_communities(nodes, edges) == first


def test_format_claims_block_carries_dates() -> None:
    claims = [
        GraphClaim(id="c1", text="Rates on hold.", upload_date="2026-03-01"),
        GraphClaim(id="c2", text="One cut priced in."),
    ]
    block = format_claims_block(claims)
    assert "- [2026-03-01] Rates on hold." in block
    assert "- [undated] One cut priced in." in block

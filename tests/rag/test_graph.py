from __future__ import annotations

import math
from typing import Any

import pytest

from src.rag.graph import build_chunk_graph, nearest_chunks

# Unit vectors placed at known angles, so every cosine similarity is cos(delta)
# and can be verified by hand. Chosen to avoid 90-degree gaps, whose cosine is
# floating-point noise around zero and would make min_similarity tests flaky.
ANGLES = {"a": 0.0, "b": 10.0, "c": 25.0, "d": 80.0, "e": 160.0}


def _vector(degrees: float) -> list[float]:
    radians = math.radians(degrees)
    return [math.cos(radians), math.sin(radians)]


def _record(chunk_id: str, embedding: list[float], **overrides: Any) -> dict[str, Any]:
    record = {
        "chunk_id": chunk_id,
        "video_id": f"video-{chunk_id}",
        "chunk_index": 0,
        "channel_id": "channel-1",
        "channel_name": "Channel One",
        "title": f"Title {chunk_id}",
        "text": f"Transcript text for {chunk_id}.",
        "start_seconds": 0.0,
        "end_seconds": 30.0,
        "source_url": f"https://youtu.be/{chunk_id}",
        "embedding": embedding,
    }
    record.update(overrides)
    return record


@pytest.fixture
def fan_records() -> list[dict[str, Any]]:
    """Five chunks fanned out by angle; nearest neighbours form an a-b-c-d-e chain."""
    return [
        _record(
            chunk_id,
            _vector(angle),
            channel_id="channel-1" if chunk_id in {"a", "b", "c"} else "channel-2",
        )
        for chunk_id, angle in ANGLES.items()
    ]


def _edge_pairs(graph: dict[str, Any]) -> set[frozenset[str]]:
    return {frozenset((edge["source"], edge["target"])) for edge in graph["edges"]}


def _degrees(graph: dict[str, Any]) -> dict[str, int]:
    return {node["id"]: node["degree"] for node in graph["nodes"]}


def test_reciprocal_neighbours_collapse_into_one_edge(fan_records) -> None:
    graph = build_chunk_graph(fan_records, k=1)

    # a picks b and b picks a: that is one undirected edge, not two.
    assert _edge_pairs(graph) == {
        frozenset(("a", "b")),
        frozenset(("b", "c")),
        frozenset(("c", "d")),
        frozenset(("d", "e")),
    }
    assert len(graph["edges"]) == 4
    assert len(graph["edges"]) == len(_edge_pairs(graph))


def test_no_self_edges(fan_records) -> None:
    graph = build_chunk_graph(fan_records, k=4, min_similarity=-1.0)

    assert all(edge["source"] != edge["target"] for edge in graph["edges"])


def test_degree_counts_distinct_neighbours_after_dedup(fan_records) -> None:
    graph = build_chunk_graph(fan_records, k=1)

    # Chain a-b-c-d-e: the interior nodes have two neighbours even though k=1,
    # because degree is measured on the merged undirected graph.
    assert _degrees(graph) == {"a": 1, "b": 2, "c": 2, "d": 2, "e": 1}
    assert sum(_degrees(graph).values()) == 2 * len(graph["edges"])


def test_k_caps_neighbours_and_saturates_at_complete_graph(fan_records) -> None:
    none = build_chunk_graph(fan_records, k=0, min_similarity=-1.0)
    assert none["edges"] == []
    assert all(node["degree"] == 0 for node in none["nodes"])
    assert none["stats"]["isolated_nodes"] == 5

    capped = build_chunk_graph(fan_records, k=2, min_similarity=-1.0)
    assert len(capped["edges"]) <= 5 * 2

    # k larger than the corpus cannot invent neighbours: 5 nodes, 10 pairs.
    saturated = build_chunk_graph(fan_records, k=10, min_similarity=-1.0)
    assert len(saturated["edges"]) == 10
    assert all(node["degree"] == 4 for node in saturated["nodes"])


def test_min_similarity_filters_weak_edges(fan_records) -> None:
    # cos(55 deg) = 0.5736 links c-d at threshold 0.5 but not at 0.6.
    kept = build_chunk_graph(fan_records, k=1, min_similarity=0.5)
    assert _edge_pairs(kept) == {
        frozenset(("a", "b")),
        frozenset(("b", "c")),
        frozenset(("c", "d")),
    }

    strict = build_chunk_graph(fan_records, k=1, min_similarity=0.6)
    assert _edge_pairs(strict) == {frozenset(("a", "b")), frozenset(("b", "c"))}
    assert all(edge["similarity"] >= 0.6 for edge in strict["edges"])
    assert _degrees(strict) == {"a": 1, "b": 2, "c": 1, "d": 0, "e": 0}
    assert strict["stats"]["isolated_nodes"] == 2


def test_edge_similarities_are_true_cosines(fan_records) -> None:
    graph = build_chunk_graph(fan_records, k=1)
    scores = {
        frozenset((edge["source"], edge["target"])): edge["similarity"]
        for edge in graph["edges"]
    }

    assert scores[frozenset(("a", "b"))] == pytest.approx(math.cos(math.radians(10)), abs=1e-6)
    assert scores[frozenset(("b", "c"))] == pytest.approx(math.cos(math.radians(15)), abs=1e-6)
    assert scores[frozenset(("c", "d"))] == pytest.approx(math.cos(math.radians(55)), abs=1e-6)
    assert scores[frozenset(("d", "e"))] == pytest.approx(math.cos(math.radians(80)), abs=1e-6)


def test_stats_summarise_the_graph(fan_records) -> None:
    graph = build_chunk_graph(fan_records, k=1, min_similarity=0.25)
    stats = graph["stats"]

    assert stats["nodes"] == 5
    assert stats["edges"] == len(graph["edges"])
    assert stats["k"] == 1
    assert stats["min_similarity"] == 0.25
    assert stats["channels"] == 2
    assert stats["mean_similarity"] == pytest.approx(
        sum(edge["similarity"] for edge in graph["edges"]) / len(graph["edges"]), abs=1e-6
    )
    assert stats["isolated_nodes"] == sum(
        1 for node in graph["nodes"] if node["degree"] == 0
    )


def test_layout_is_deterministic_across_calls(fan_records) -> None:
    first = build_chunk_graph(fan_records, k=2)
    second = build_chunk_graph(fan_records, k=2)

    assert first == second
    assert [(node["x"], node["y"]) for node in first["nodes"]] == [
        (node["x"], node["y"]) for node in second["nodes"]
    ]


def test_layout_coordinates_are_normalised_into_unit_range(fan_records) -> None:
    graph = build_chunk_graph(fan_records, k=2)
    xs = [node["x"] for node in graph["nodes"]]
    ys = [node["y"] for node in graph["nodes"]]

    assert all(-1.0 <= value <= 1.0 for value in xs + ys)
    # One axis is scaled to the full range; the other keeps the aspect ratio.
    assert max(max(xs), max(ys)) == pytest.approx(1.0, abs=1e-6)
    assert min(min(xs), min(ys)) == pytest.approx(-1.0, abs=1e-6)


def test_layout_disabled_returns_origin_coordinates(fan_records) -> None:
    graph = build_chunk_graph(fan_records, k=2, layout=False)

    assert all((node["x"], node["y"]) == (0.0, 0.0) for node in graph["nodes"])
    assert len(graph["edges"]) > 0


def test_nodes_carry_metadata_and_truncated_preview() -> None:
    long_text = "word " * 200
    graph = build_chunk_graph([_record("a", _vector(0), text=long_text)], k=1)
    node = graph["nodes"][0]

    assert node["id"] == "a"
    assert node["video_id"] == "video-a"
    assert node["channel_name"] == "Channel One"
    assert node["title"] == "Title a"
    assert node["source_url"] == "https://youtu.be/a"
    assert node["start_seconds"] == 0.0
    assert node["end_seconds"] == 30.0
    assert len(node["preview"]) <= 123
    assert node["preview"].endswith("...")

    collapsed = build_chunk_graph([_record("a", _vector(0), text="one\n\n  two\ttree")], k=1)
    assert collapsed["nodes"][0]["preview"] == "one two tree"


def test_empty_records_produce_an_empty_graph() -> None:
    graph = build_chunk_graph([], k=5)

    assert graph["nodes"] == []
    assert graph["edges"] == []
    assert graph["stats"] == {
        "nodes": 0,
        "edges": 0,
        "k": 5,
        "min_similarity": 0.0,
        "channels": 0,
        "mean_similarity": 0.0,
        "isolated_nodes": 0,
    }


def test_single_record_has_no_edges() -> None:
    graph = build_chunk_graph([_record("a", _vector(0))], k=5)

    assert len(graph["nodes"]) == 1
    assert graph["edges"] == []
    assert graph["nodes"][0]["degree"] == 0
    assert (graph["nodes"][0]["x"], graph["nodes"][0]["y"]) == (0.0, 0.0)
    assert graph["stats"]["isolated_nodes"] == 1


def test_identical_vectors_collapse_to_origin_without_error() -> None:
    records = [_record(chunk_id, [1.0, 0.0, 0.0]) for chunk_id in ("a", "b", "c", "d")]

    graph = build_chunk_graph(records, k=2)

    assert len(graph["edges"]) >= 1
    assert all(edge["similarity"] == 1.0 for edge in graph["edges"])
    # PCA finds no spread, so there is nothing to normalise against.
    assert all((node["x"], node["y"]) == (0.0, 0.0) for node in graph["nodes"])
    assert graph["stats"]["mean_similarity"] == 1.0


def test_records_without_usable_embeddings_are_skipped() -> None:
    records = [
        _record("good-1", _vector(0)),
        _record("good-2", _vector(10)),
        _record("missing", None),
        _record("empty", []),
        _record("wrong-width", [1.0, 0.0, 0.0]),
        {"chunk_id": "no-key", "text": "no embedding at all"},
    ]

    graph = build_chunk_graph(records, k=2)

    assert [node["id"] for node in graph["nodes"]] == ["good-1", "good-2"]
    assert graph["stats"]["nodes"] == 2


def test_zero_vectors_score_zero_rather_than_dividing_by_zero() -> None:
    records = [_record("a", _vector(0)), _record("zero", [0.0, 0.0])]

    graph = build_chunk_graph(records, k=1, min_similarity=-1.0)

    assert len(graph["nodes"]) == 2
    assert graph["edges"][0]["similarity"] == 0.0


def test_nearest_chunks_ranks_by_true_cosine_similarity(fan_records) -> None:
    ranked = nearest_chunks(fan_records, _vector(0.0), top_k=3)

    assert [item["chunk_id"] for item in ranked] == ["a", "b", "c"]
    assert ranked[0]["similarity"] == pytest.approx(1.0, abs=1e-6)
    assert ranked[1]["similarity"] == pytest.approx(math.cos(math.radians(10)), abs=1e-6)
    assert ranked[2]["similarity"] == pytest.approx(math.cos(math.radians(25)), abs=1e-6)
    scores = [item["similarity"] for item in ranked]
    assert scores == sorted(scores, reverse=True)


def test_nearest_chunks_scores_the_original_embedding_space() -> None:
    # Ranking is driven entirely by the third dimension here. Scoring in a 2-D
    # projection would have to discard one axis and could not reproduce this
    # order, so matching an independently computed full-space cosine is the
    # assertion that pins the behaviour.
    vectors = {
        "flat": [1.0, 0.0, 0.0],
        "wide": [0.0, 0.96, 0.28],
        "deep": [0.6, 0.0, 0.8],
    }
    records = [_record(name, vector) for name, vector in vectors.items()]
    query = [0.0, 0.0, 1.0]

    ranked = nearest_chunks(records, query, top_k=3)

    def cosine(vector: list[float]) -> float:
        norm = math.sqrt(sum(value * value for value in vector))
        return sum(a * b for a, b in zip(vector, query)) / norm

    expected = sorted(vectors.items(), key=lambda item: -cosine(item[1]))
    assert [item["chunk_id"] for item in ranked] == [name for name, _ in expected]
    assert ranked[0]["chunk_id"] == "deep"
    assert ranked[0]["similarity"] == pytest.approx(0.8, abs=1e-6)


def test_nearest_chunks_edge_cases(fan_records) -> None:
    assert nearest_chunks([], _vector(0.0), top_k=3) == []
    assert nearest_chunks(fan_records, _vector(0.0), top_k=0) == []
    # top_k beyond the corpus returns everything, not padding.
    assert len(nearest_chunks(fan_records, _vector(0.0), top_k=99)) == 5

    with pytest.raises(ValueError):
        nearest_chunks(fan_records, [1.0, 0.0, 0.0], top_k=1)


def test_pure_python_fallback_matches_the_numpy_path(fan_records, monkeypatch) -> None:
    """Without numpy the module must still produce the same graph topology."""
    expected = build_chunk_graph(fan_records, k=2)
    monkeypatch.setattr("src.rag.graph._HAS_NUMPY", False)
    fallback = build_chunk_graph(fan_records, k=2)

    assert fallback["edges"] == expected["edges"]
    assert _degrees(fallback) == _degrees(expected)
    assert fallback["stats"] == expected["stats"]
    # Layout differs by design: PCA is unavailable, so it rings the nodes.
    coords = [(node["x"], node["y"]) for node in fallback["nodes"]]
    assert all(-1.0 <= x <= 1.0 and -1.0 <= y <= 1.0 for x, y in coords)
    assert len(set(coords)) == len(coords)

    ranked = nearest_chunks(fan_records, _vector(0.0), top_k=3)
    assert [item["chunk_id"] for item in ranked] == ["a", "b", "c"]


def test_graph_is_json_serialisable(fan_records) -> None:
    import json

    graph = build_chunk_graph(fan_records, k=2)

    assert json.loads(json.dumps(graph)) == graph

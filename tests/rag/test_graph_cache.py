from __future__ import annotations

import math
from typing import Any

import pytest

from src.rag import graph


@pytest.fixture(autouse=True)
def clear_graph_cache():
    graph.clear_cache()
    yield
    graph.clear_cache()


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


RECORDS = [
    _record("a", _vector(0.0)),
    _record("b", _vector(10.0)),
    _record("c", _vector(25.0)),
]


def test_cap_rejects_corpora_over_the_limit() -> None:
    with pytest.raises(ValueError, match="4 chunks, exceeding the 3-chunk"):
        graph.build_chunk_graph_cached(RECORDS + [_record("d", _vector(80.0))], max_chunks=3)


def test_cap_allows_corpora_at_or_under_the_limit() -> None:
    result = graph.build_chunk_graph_cached(RECORDS, max_chunks=3)
    assert len(result["nodes"]) == 3


def test_second_call_with_same_args_reuses_the_cache(monkeypatch) -> None:
    calls = []
    original = graph.build_chunk_graph

    def spy(records, k=5, min_similarity=0.0, layout=True):
        calls.append(len(records))
        return original(records, k=k, min_similarity=min_similarity, layout=layout)

    monkeypatch.setattr(graph, "build_chunk_graph", spy)

    first = graph.build_chunk_graph_cached(RECORDS, k=1, min_similarity=0.0)
    second = graph.build_chunk_graph_cached(RECORDS, k=1, min_similarity=0.0)

    assert calls == [3]
    assert first == second


def test_cache_key_changes_with_corpus_size(monkeypatch) -> None:
    calls = []
    original = graph.build_chunk_graph

    def spy(records, k=5, min_similarity=0.0, layout=True):
        calls.append(len(records))
        return original(records, k=k, min_similarity=min_similarity, layout=layout)

    monkeypatch.setattr(graph, "build_chunk_graph", spy)

    graph.build_chunk_graph_cached(RECORDS, k=1)
    graph.build_chunk_graph_cached(RECORDS + [_record("d", _vector(80.0))], k=1)

    assert calls == [3, 4]


def test_mutating_a_cached_result_does_not_leak_into_later_calls() -> None:
    second = graph.build_chunk_graph_cached(RECORDS, k=1, min_similarity=0.0)
    second["query"] = {"text": "leaked", "nearest": []}

    third = graph.build_chunk_graph_cached(RECORDS, k=1, min_similarity=0.0)

    assert "query" not in third

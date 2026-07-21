from __future__ import annotations

import pytest

from src.rag import bm25


@pytest.fixture(autouse=True)
def clear_index_cache():
    bm25.clear_cache()
    yield
    bm25.clear_cache()


def record(index: int, text: str, video_id: str = "vid1") -> dict:
    return {"video_id": video_id, "chunk_index": index, "text": text}


CORPUS = [
    record(0, "capital gains tax discount for property investors"),
    record(1, "negative gearing rules and rental deductions"),
    record(2, "the capital gains tax discount is being grandfathered"),
    record(3, "interest rates and the housing market outlook"),
]


def test_tokenize_lowercases_and_drops_punctuation():
    assert bm25.tokenize("Capital-Gains, TAX!") == ["capital", "gains", "tax"]


def test_search_ranks_matching_chunks_first():
    results = bm25.search(CORPUS, "capital gains tax", top_k=10)
    assert [r["chunk_index"] for r in results] == [0, 2] or [
        r["chunk_index"] for r in results
    ] == [2, 0]
    assert all(r["score"] >= 0 for r in results)
    assert [r["rank"] for r in results] == [1, 2]


def test_search_excludes_chunks_without_any_query_term():
    results = bm25.search(CORPUS, "gearing", top_k=10)
    assert [r["chunk_index"] for r in results] == [1]


def test_search_returns_hits_even_when_idf_is_zero():
    """A term in half a tiny corpus scores 0 but is still a real match."""
    corpus = [record(0, "capital gains tax"), record(1, "unrelated content")]
    results = bm25.search(corpus, "capital gains tax", top_k=5)
    assert [r["chunk_index"] for r in results] == [0]


def test_search_respects_top_k():
    assert len(bm25.search(CORPUS, "capital gains tax discount", top_k=1)) == 1


def test_search_handles_empty_corpus_and_query():
    assert bm25.search([], "anything", top_k=5) == []
    assert bm25.search(CORPUS, "   ", top_k=5) == []
    assert bm25.search(CORPUS, "capital", top_k=0) == []


def test_search_ignores_blank_documents():
    results = bm25.search([record(0, "   "), record(1, "capital gains")], "capital", 5)
    assert [r["chunk_index"] for r in results] == [1]


def test_index_is_cached_per_corpus_size():
    built = []
    original = bm25.Bm25Index.build

    def counting_build(records):
        built.append(len(records))
        return original(records)

    bm25.Bm25Index.build = staticmethod(counting_build)  # type: ignore[method-assign]
    try:
        bm25.search(CORPUS, "capital", top_k=3, cache_key="corpus")
        bm25.search(CORPUS, "gearing", top_k=3, cache_key="corpus")
        assert built == [4]  # second query reuses the index
        # Indexing new content changes the chunk count, invalidating the cache.
        bm25.search(CORPUS + [record(4, "new chunk")], "capital", 3, cache_key="corpus")
        assert built == [4, 5]
    finally:
        bm25.Bm25Index.build = original  # type: ignore[method-assign]

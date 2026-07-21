from __future__ import annotations

import pytest

from src.rag.models import RetrievedChunk
from src.rag.rerank import DEFAULT_MODEL, CrossEncoderReranker, NullReranker


def chunk(index: int, text: str, score: float | None = None) -> RetrievedChunk:
    return RetrievedChunk(
        transcript_id="t-vid1",
        video_id="vid1",
        source_url="https://www.youtube.com/watch?v=vid1",
        chunk_index=index,
        text=text,
        score=score,
    )


class FakeCrossEncoder:
    """Stands in for sentence_transformers.CrossEncoder — no model download.

    Scores a pair by how many query terms appear in the chunk text, which is
    enough to assert ordering without pretending to be a real cross-encoder.
    """

    def __init__(self, name: str = "fake") -> None:
        self.name = name
        self.calls: list[list[tuple[str, str]]] = []

    def predict(self, pairs):
        self.calls.append(list(pairs))
        return [
            float(sum(term in text.lower() for term in query.lower().split()))
            for query, text in pairs
        ]


@pytest.fixture
def fake_model() -> FakeCrossEncoder:
    return FakeCrossEncoder()


@pytest.fixture
def reranker(fake_model: FakeCrossEncoder) -> CrossEncoderReranker:
    return CrossEncoderReranker(loader=lambda name: fake_model)


CHUNKS = [
    chunk(0, "interest rates and the housing market"),
    chunk(1, "the capital gains tax discount explained"),
    chunk(2, "capital city travel notes"),
]


# --- laziness ---------------------------------------------------------------


def test_model_is_not_loaded_at_construction():
    loaded: list[str] = []
    reranker = CrossEncoderReranker(loader=lambda name: loaded.append(name))
    assert loaded == []
    assert reranker.model_loaded is False


def test_from_model_name_does_not_load_the_model():
    assert CrossEncoderReranker.from_model_name("some/model").model_loaded is False


def test_from_model_name_defaults_to_the_ms_marco_cross_encoder():
    assert CrossEncoderReranker.from_model_name().model_name == DEFAULT_MODEL


def test_model_loads_on_first_rerank_and_is_reused():
    loads: list[str] = []

    def loader(name: str) -> FakeCrossEncoder:
        loads.append(name)
        return FakeCrossEncoder(name)

    reranker = CrossEncoderReranker(model_name="some/model", loader=loader)
    reranker.rerank("capital gains", CHUNKS, top_k=2)
    assert loads == ["some/model"]
    assert reranker.model_loaded is True

    reranker.rerank("housing", CHUNKS, top_k=2)
    assert loads == ["some/model"]  # second call reuses the loaded model


def test_no_model_is_loaded_when_there_is_nothing_to_rerank():
    loads: list[str] = []
    reranker = CrossEncoderReranker(loader=lambda name: loads.append(name))
    assert reranker.rerank("q", [], top_k=5) == []
    assert reranker.rerank("q", CHUNKS, top_k=0) == []
    assert loads == []


def test_default_loader_is_not_invoked_at_import_or_construction():
    """A missing/broken model must not break importing or building the object."""
    reranker = CrossEncoderReranker(model_name="definitely/not-a-real-model")
    assert reranker.model_loaded is False


# --- reranking --------------------------------------------------------------


def test_rerank_orders_chunks_by_cross_encoder_score(reranker):
    ranked = reranker.rerank("capital gains", CHUNKS, top_k=3)
    assert [c.chunk_index for c in ranked] == [1, 2, 0]


def test_rerank_scores_query_and_chunk_text_pairs(reranker, fake_model):
    reranker.rerank("capital gains", CHUNKS, top_k=3)
    assert fake_model.calls == [[("capital gains", c.text) for c in CHUNKS]]


def test_rerank_respects_top_k(reranker):
    ranked = reranker.rerank("capital gains", CHUNKS, top_k=1)
    assert [c.chunk_index for c in ranked] == [1]


def test_rerank_top_k_larger_than_input_returns_everything(reranker):
    assert len(reranker.rerank("capital", CHUNKS, top_k=99)) == 3


def test_rerank_returns_retrieved_chunks_untouched(reranker):
    original = chunk(0, "capital gains tax", score=0.42)
    ranked = reranker.rerank("capital gains", [original], top_k=1)
    assert ranked[0] is original
    assert ranked[0].score == 0.42  # retrieval score survives reranking


def test_rerank_ties_keep_the_input_order(reranker):
    tied = [chunk(0, "capital one"), chunk(1, "capital two")]
    assert [c.chunk_index for c in reranker.rerank("capital", tied, top_k=2)] == [0, 1]


def test_rerank_handles_empty_inputs_and_non_positive_top_k(reranker):
    assert reranker.rerank("capital", [], top_k=5) == []
    assert reranker.rerank("capital", CHUNKS, top_k=0) == []
    assert reranker.rerank("capital", CHUNKS, top_k=-1) == []


def test_rerank_raises_when_the_model_returns_the_wrong_score_count():
    reranker = CrossEncoderReranker(loader=lambda name: _ShortModel())
    with pytest.raises(ValueError, match="1 scores for 3 chunks"):
        reranker.rerank("capital", CHUNKS, top_k=3)


class _ShortModel:
    def predict(self, pairs):
        return [1.0]


def test_rerank_coerces_numpy_like_scores():
    """CrossEncoder.predict returns a numpy array, not a list of floats."""

    class NumpyLike:
        def __init__(self, values):
            self._values = values

        def __iter__(self):
            return iter(self._values)

        def __len__(self):
            return len(self._values)

    class ArrayModel:
        def predict(self, pairs):
            return NumpyLike([0.1, 9.9, 0.2])

    reranker = CrossEncoderReranker(loader=lambda name: ArrayModel())
    ranked = reranker.rerank("q", CHUNKS, top_k=1)
    assert ranked[0].chunk_index == 1


# --- scores -----------------------------------------------------------------


def test_rerank_with_scores_pairs_each_chunk_with_its_score(reranker):
    scored = reranker.rerank_with_scores("capital gains", CHUNKS, top_k=3)
    assert [(c.chunk_index, score) for c, score in scored] == [(1, 2.0), (2, 1.0), (0, 0.0)]


def test_rerank_matches_the_chunks_from_rerank_with_scores(reranker):
    scored = reranker.rerank_with_scores("capital gains", CHUNKS, top_k=2)
    assert reranker.rerank("capital gains", CHUNKS, top_k=2) == [c for c, _ in scored]


# --- NullReranker -----------------------------------------------------------


def test_null_reranker_preserves_order():
    assert [c.chunk_index for c in NullReranker().rerank("q", CHUNKS, top_k=3)] == [0, 1, 2]


def test_null_reranker_respects_top_k():
    assert [c.chunk_index for c in NullReranker().rerank("q", CHUNKS, top_k=2)] == [0, 1]
    assert NullReranker().rerank("q", CHUNKS, top_k=0) == []


def test_null_reranker_handles_empty_input():
    assert NullReranker().rerank("q", [], top_k=5) == []
    assert NullReranker().rerank_with_scores("q", [], top_k=5) == []


def test_null_reranker_scores_descend_with_the_incoming_order():
    """Flat/equal scores would let a later sort silently reshuffle the ranking."""
    scored = NullReranker().rerank_with_scores("q", CHUNKS, top_k=3)
    assert [c.chunk_index for c, _ in scored] == [0, 1, 2]
    assert [score for _, score in scored] == [3.0, 2.0, 1.0]


def test_null_reranker_is_interchangeable_with_the_cross_encoder(reranker):
    """Callers switch on config without branching, so the shapes must match."""
    for impl in (reranker, NullReranker()):
        ranked = impl.rerank("capital gains", CHUNKS, top_k=2)
        scored = impl.rerank_with_scores("capital gains", CHUNKS, top_k=2)
        assert len(ranked) == len(scored) == 2
        assert all(isinstance(c, RetrievedChunk) for c in ranked)
        assert all(isinstance(score, float) for _, score in scored)

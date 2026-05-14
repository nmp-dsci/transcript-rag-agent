from __future__ import annotations

from src.rag.eval import compare_answers, estimate_tokens


class FakeEmbeddingModel:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] if "same" in text else [0.0, 1.0] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]


def test_compare_answers_uses_embedding_cosine_similarity_and_token_estimates() -> None:
    comparison = compare_answers(
        question="q",
        raw_answer="same answer",
        rag_answer="same response",
        raw_prompt_context="x" * 100,
        rag_prompt_context="x" * 20,
        embedding_model=FakeEmbeddingModel(),
    )

    assert comparison.semantic_similarity == 1.0
    assert comparison.raw_prompt_tokens_estimate == 25
    assert comparison.rag_prompt_tokens_estimate == 5
    assert comparison.token_savings_percent == 80.0


def test_estimate_tokens_rounds_up() -> None:
    assert estimate_tokens("12345") == 2

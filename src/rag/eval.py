from __future__ import annotations

from math import ceil

from src.rag.embeddings import EmbeddingModel, cosine_similarity
from src.rag.models import ContextComparisonResult


def estimate_tokens(text: str) -> int:
    return ceil(len(text) / 4)


def compare_answers(
    question: str,
    raw_answer: str,
    rag_answer: str,
    raw_prompt_context: str,
    rag_prompt_context: str,
    embedding_model: EmbeddingModel,
) -> ContextComparisonResult:
    raw_embedding, rag_embedding = embedding_model.embed_documents(
        [raw_answer, rag_answer]
    )
    raw_tokens = estimate_tokens(raw_prompt_context)
    rag_tokens = estimate_tokens(rag_prompt_context)
    if raw_tokens <= 0:
        savings = 0.0
    else:
        savings = ((raw_tokens - rag_tokens) / raw_tokens) * 100
    return ContextComparisonResult(
        question=question,
        raw_answer=raw_answer,
        rag_answer=rag_answer,
        semantic_similarity=cosine_similarity(raw_embedding, rag_embedding),
        raw_prompt_tokens_estimate=raw_tokens,
        rag_prompt_tokens_estimate=rag_tokens,
        token_savings_percent=savings,
    )

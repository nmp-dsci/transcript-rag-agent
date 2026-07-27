"""Committed-matrix-run fixtures for the Scoreboard and Experiments tests.

Only imported by ``*_test``/``test_*`` modules. Kept in one place so the
on-disk run shape has a single definition to update when the eval schema
grows — the same reason ``frontend/src/pipeline/fixtures.ts`` exists.
"""

from __future__ import annotations

from typing import Any

MATRIX_CONFIG: dict[str, Any] = {
    "answer_model": "deepseek-v4",
    "embedding_model": "fake-embeddings",
    "retrieval_mode": "semantic",
    "top_k": 10,
    "judge_model": "deepseek-v4",
    "judge_samples": 1,
}


def matrix_cell(
    entry_id: str,
    *,
    composite: float | None = 0.8,
    faithfulness: float = 0.9,
    elapsed: float = 10.0,
    tokens: int = 100,
    error: str | None = None,
    question_type: str = "local",
    domain: str = "property",
) -> dict[str, Any]:
    """One (setup, golden entry) cell. ``composite=None`` means unjudged."""
    scores: dict[str, float] = {}
    if composite is not None:
        scores = {
            "faithfulness": faithfulness,
            "answer_relevancy": 0.7,
            "context_precision": 0.6,
            "composite": composite,
        }
    return {
        "id": entry_id,
        "question": f"question {entry_id}?",
        "domain": domain,
        "question_type": question_type,
        "answer": f"answer for {entry_id}",
        "error": error,
        "scores": scores,
        "retrieved_chunk_ids": ["chunk:v1:0"],
        "elapsed_seconds": elapsed,
        "token_estimate": tokens,
    }


def matrix_run(
    run_id: str = "matrix-20260727-010101",
    *,
    created_at: str = "2026-07-27T01:01:01+00:00",
    setups: dict[str, list[dict[str, Any]]] | None = None,
    judged: bool = True,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """A whole committed run: two setups over two questions, rag_agent winning."""
    setups = setups or {
        "rag_llm": [matrix_cell("g001", composite=0.5), matrix_cell("g002", composite=0.5)],
        "rag_agent": [matrix_cell("g001", composite=0.9), matrix_cell("g002", composite=0.9)],
    }
    resolved = config or MATRIX_CONFIG
    return {
        "run_id": run_id,
        "created_at": created_at,
        "kind": "matrix",
        "setups": list(setups),
        "config": resolved,
        "judged": judged,
        "reference_scored": False,
        "entry_count": max((len(cells) for cells in setups.values()), default=0),
        "question_types": {"local": 2},
        "runs": {
            setup: {
                "run_id": f"{run_id}-{setup}",
                "created_at": created_at,
                "setup": setup,
                "config": resolved,
                "entries": cells,
                "summary": {},
            }
            for setup, cells in setups.items()
        },
        "comparison": {},
    }

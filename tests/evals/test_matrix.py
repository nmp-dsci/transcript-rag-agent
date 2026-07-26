"""The head-to-head matrix pivot: overall, per question type, and ops columns."""

from __future__ import annotations

from src.evals.matrix import build_comparison, format_matrix_table


def run_for(setup: str, entries: list[dict]) -> dict:
    return {"setup": setup, "entries": entries, "config": {}}


def entry(
    question_type: str,
    scores: dict,
    *,
    error: str | None = None,
    elapsed: float = 10.0,
    tokens: int = 1000,
) -> dict:
    return {
        "id": "x",
        "question_type": question_type,
        "scores": scores,
        "error": error,
        "elapsed_seconds": elapsed,
        "token_estimate": tokens,
    }


def test_pivot_splits_overall_and_by_question_type() -> None:
    runs = {
        "rag_llm": run_for(
            "rag_llm",
            [
                entry("local", {"answer_correctness": 0.8}),
                entry("global", {"answer_correctness": 0.4}),
            ],
        ),
        "graph_rag": run_for(
            "graph_rag",
            [
                entry("local", {"answer_correctness": 0.7}),
                entry("global", {"answer_correctness": 0.9}),
            ],
        ),
    }
    comparison = build_comparison(runs)

    assert comparison["overall"]["answer_correctness"] == {
        "rag_llm": 0.6,
        "graph_rag": 0.8,
    }
    by_type = comparison["by_question_type"]
    assert by_type["global"]["answer_correctness"]["graph_rag"] == 0.9
    assert by_type["local"]["answer_correctness"]["rag_llm"] == 0.8


def test_failed_entries_count_in_ops_but_not_in_averages() -> None:
    runs = {
        "rag_llm": run_for(
            "rag_llm",
            [
                entry("local", {"answer_correctness": 1.0}, elapsed=4.0),
                entry("local", {}, error="boom", elapsed=99.0),
            ],
        )
    }
    comparison = build_comparison(runs)
    assert comparison["overall"]["answer_correctness"]["rag_llm"] == 1.0
    assert comparison["ops"]["rag_llm"] == {
        "avg_elapsed_seconds": 4.0,
        "avg_token_estimate": 1000.0,
        "answered": 1,
        "failed": 1,
    }


def test_missing_metrics_are_omitted_not_zeroed() -> None:
    runs = {
        "rag_llm": run_for("rag_llm", [entry("local", {"faithfulness": None})]),
    }
    comparison = build_comparison(runs)
    assert "faithfulness" not in comparison["overall"]


def test_format_matrix_table_renders_all_sections() -> None:
    runs = {
        "rag_llm": run_for("rag_llm", [entry("local", {"answer_correctness": 0.5})]),
        "graph_rag": run_for("graph_rag", [entry("local", {"answer_correctness": 0.6})]),
    }
    result = {
        "setups": ["rag_llm", "graph_rag"],
        "comparison": build_comparison(runs),
    }
    table = format_matrix_table(result)
    assert "overall" in table
    assert "question_type = local" in table
    assert "graph_rag" in table
    assert "avg_elapsed_seconds" in table

from __future__ import annotations

from src.evals.evaluation import EvaluationRun, render_html_report
from src.rag.models import RetrievedChunk


def test_render_html_report_compares_three_input_types() -> None:
    chunk = RetrievedChunk(
        transcript_id="raw_transcript:abc",
        video_id="abc",
        source_url="https://www.youtube.com/watch?v=abc",
        chunk_index=1,
        text="retrieved evidence",
        start_seconds=10,
        end_seconds=20,
        segment_count=1,
        score=0.9,
    )
    runs = [
        EvaluationRun(
            name="raw_single",
            input_type="raw",
            source_url="https://www.youtube.com/watch?v=abc",
            answer="raw answer",
            context_text="x" * 100,
            retrieved_chunks=[],
        ),
        EvaluationRun(
            name="rag_single",
            input_type="rag single",
            source_url="https://www.youtube.com/watch?v=abc",
            answer="rag single answer",
            context_text="x" * 40,
            retrieved_chunks=[chunk],
        ),
        EvaluationRun(
            name="rag_all",
            input_type="rag all",
            source_url=None,
            answer="rag all answer",
            context_text="x" * 20,
            retrieved_chunks=[chunk],
        ),
    ]

    output = render_html_report(
        question="q",
        source_url="https://www.youtube.com/watch?v=abc",
        top_k=10,
        runs=runs,
        similarities={
            "raw_single__rag_single": 0.8,
            "raw_single__rag_all": 0.7,
            "rag_single__rag_all": 0.9,
        },
    )

    assert "Transcript Agent Evaluation" in output
    assert "raw_single" in output
    assert "rag_single" in output
    assert "rag_all" in output
    assert "raw_single__rag_single" in output
    assert "https://www.youtube.com/watch?v=abc&amp;t=10s" in output
    assert "<details>" in output

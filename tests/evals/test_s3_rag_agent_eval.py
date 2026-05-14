from __future__ import annotations

from src.agents.models import RagTranscriptAnswer
from src.evals.s3_rag_agent_eval import EvalRun, render_html_report
from src.rag.models import RetrievedChunk


def test_render_html_report_contains_question_answers_details_and_metrics() -> None:
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
        EvalRun(
            name="url_1_only",
            source_url="https://www.youtube.com/watch?v=abc",
            answer=RagTranscriptAnswer(question="q", answer="answer [1]"),
            context_text="context",
            retrieved_chunks=[chunk],
        ),
        EvalRun(
            name="all_indexed",
            source_url=None,
            answer=RagTranscriptAnswer(question="q", answer="answer [1]"),
            context_text="context",
            retrieved_chunks=[chunk],
        ),
    ]

    html = render_html_report(
        question="q",
        top_k=10,
        runs=runs,
        similarities={"url_1_only__all_indexed": 0.95},
    )

    assert "S3 RAG Agent Evaluation" in html
    assert "url_1_only" in html
    assert "all_indexed" in html
    assert "<details>" in html
    assert "https://www.youtube.com/watch?v=abc&amp;t=10s" in html
    assert "0.950" in html

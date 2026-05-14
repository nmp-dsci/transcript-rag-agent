from __future__ import annotations

from datetime import datetime, timezone

from src.agents.context import TranscriptContext
from src.agents.models import TranscriptAnswer
from src.evals.s2_context_eval import DEFAULT_QUESTION, build_payload
from src.rag.models import RetrievedChunk
from src.transcripts.models import Transcript


def test_build_payload_includes_answers_token_metrics_and_chunks() -> None:
    transcript = Transcript(
        video_id="video",
        url="https://www.youtube.com/watch?v=video",
        raw_text="raw transcript",
        fetched_at=datetime.now(timezone.utc),
    )
    raw_context = TranscriptContext(transcript=transcript, cache_status="hit")
    chunk = RetrievedChunk(
        transcript_id="raw_transcript:video",
        video_id="video",
        source_url="https://www.youtube.com/watch?v=video",
        chunk_index=3,
        text="retrieved evidence",
        start_seconds=10,
        end_seconds=20,
        segment_count=1,
        score=0.9,
    )
    rag_context = TranscriptContext(
        transcript=transcript,
        cache_status="hit",
        context_mode="rag",
        retrieved_chunks=[chunk],
        top_k=10,
    )

    payload = build_payload(
        video_id="video",
        source_url="https://www.youtube.com/watch?v=video",
        top_k=10,
        raw_answer=TranscriptAnswer(
            question=DEFAULT_QUESTION,
            answer="raw answer",
            source_video_id="video",
        ),
        rag_answer=TranscriptAnswer(
            question=DEFAULT_QUESTION,
            answer="rag answer",
            source_video_id="video",
        ),
        raw_context=raw_context,
        rag_context=rag_context,
        comparison={
            "raw_prompt_tokens_estimate": 100,
            "rag_prompt_tokens_estimate": 25,
            "semantic_similarity": 0.87,
            "token_savings_percent": 75.0,
        },
    )

    assert payload["question"] == DEFAULT_QUESTION
    assert payload["raw"]["prompt_tokens_estimate"] == 100
    assert payload["rag"]["prompt_tokens_estimate"] == 25
    assert payload["comparison"]["semantic_similarity"] == 0.87
    assert payload["rag"]["retrieved_chunks"][0]["chunk_index"] == 3

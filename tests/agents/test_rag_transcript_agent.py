from __future__ import annotations

from datetime import datetime, timezone

from langchain_core.messages import AIMessage

from src.agents.context import TranscriptContext
from src.agents.models import RagQuestionRequest
from src.agents.rag_transcript_agent import RagTranscriptAgent
from src.rag.models import RetrievedChunk
from src.transcripts.models import Transcript


class FakeLlm:
    def __init__(self, response: str) -> None:
        self.response = response
        self.messages = None

    def invoke(self, messages):
        self.messages = messages
        return AIMessage(content=self.response)


class FakeProvider:
    def __init__(self) -> None:
        self.calls = []

    def get_context(self, question: str, source_url: str | None = None, top_k: int = 10):
        self.calls.append((question, source_url, top_k))
        transcript = Transcript(
            video_id="all",
            url="https://www.youtube.com/watch?v=abc",
            raw_text="chunk text",
            fetched_at=datetime.now(timezone.utc),
        )
        chunk = RetrievedChunk(
            transcript_id="raw_transcript:abc",
            video_id="abc",
            source_url="https://www.youtube.com/watch?v=abc",
            chunk_index=4,
            text="capital gains tax",
            start_seconds=593,
            end_seconds=665,
            segment_count=1,
        )
        return TranscriptContext(
            transcript=transcript,
            cache_status="hit",
            context_text="[1] video=abc url=https://www.youtube.com/watch?v=abc&t=593s\ncapital gains tax",
            context_mode="rag",
            retrieved_chunks=[chunk],
            top_k=top_k,
        )


def test_rag_transcript_agent_answers_and_backfills_references() -> None:
    llm = FakeLlm('{"question": "q", "answer": "answer from chunk [1]"}')
    provider = FakeProvider()
    agent = RagTranscriptAgent(llm, provider)

    answer = agent.answer(
        RagQuestionRequest(
            question="q",
            source_url="https://www.youtube.com/watch?v=abc",
            top_k=3,
        )
    )

    assert answer.answer == "answer from chunk [1]"
    assert answer.references[0].timestamp_url.unicode_string().endswith("t=593s")
    assert provider.calls == [("q", "https://www.youtube.com/watch?v=abc", 3)]
    assert "retrieved transcript chunks" in llm.messages[0].content

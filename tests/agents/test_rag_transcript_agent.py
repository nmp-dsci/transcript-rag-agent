from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

import pytest
from langchain_core.messages import AIMessage

from src.agents.context import TranscriptContext
from src.agents.models import RagQuestionRequest, TraceStep
from src.agents.rag_transcript_agent import RagTranscriptAgent
from src.rag.context import RetrievalError
from src.rag.models import RetrievedChunk
from src.transcripts.models import Transcript


class FakeLlm:
    def __init__(self, response: str | list[str]) -> None:
        self.responses = response if isinstance(response, list) else [response]
        self.messages = None
        self.calls = []

    def invoke(self, messages):
        self.messages = messages
        self.calls.append(messages)
        response = self.responses[min(len(self.calls) - 1, len(self.responses) - 1)]
        return AIMessage(content=response)


class FakeProvider:
    def __init__(self) -> None:
        self.calls = []

    def get_context(
        self,
        question: str,
        source_url: str | None = None,
        top_k: int = 10,
        filter_transcripts: bool = False,
        transcript_filter_top_k: int = 5,
        transcript_filter_min_score: float = 0.25,
        channel_id: str | None = None,
        retrieval_mode: str | None = None,
    ):
        self.calls.append(
            (
                question,
                source_url,
                top_k,
                filter_transcripts,
                transcript_filter_top_k,
                transcript_filter_min_score,
            )
        )
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
    assert provider.calls == [("q", "https://www.youtube.com/watch?v=abc", 3, False, 5, 0.25)]
    assert "retrieved transcript chunks" in llm.messages[0].content


def test_answer_clears_the_previous_questions_retrieval_state() -> None:
    """Agents are reused across questions, so a failed run must not inherit one.

    ``last_context``/``last_rewrite`` are what the chat runner persists as the
    trace; leaving the prior answer's values in place would attribute that
    retrieval to a question that never got that far.
    """

    class FailingProvider(FakeProvider):
        def get_context(self, *args, **kwargs):
            raise RuntimeError("chroma is down")

    llm = FakeLlm('{"question": "q", "answer": "first [1]"}')
    agent = RagTranscriptAgent(llm, FakeProvider())
    agent.answer(RagQuestionRequest(question="first", top_k=3))
    assert agent.last_context is not None

    agent.context_provider = FailingProvider()
    try:
        agent.answer(RagQuestionRequest(question="second", top_k=3))
    except RuntimeError:
        pass

    assert agent.last_context is None
    assert agent.last_rewrite is None
    assert agent.last_retrievals == []


def test_single_hop_surfaces_followups_without_extra_retrieval() -> None:
    llm = FakeLlm(
        """
        {
          "question": "q",
          "answer": "answer from chunk [1]",
          "followups_requested": true,
          "subtopics": [
            {
              "topic": "detail",
              "rationale": "thin evidence",
              "followup_query": "specific detail query",
              "confidence": 0.8
            }
          ]
        }
        """
    )
    provider = FakeProvider()
    agent = RagTranscriptAgent(llm, provider)

    answer = agent.answer(RagQuestionRequest(question="q", top_k=3))

    assert answer.followups_requested is True
    assert answer.subtopics[0].followup_query == "specific detail query"
    assert answer.recursion is None
    assert len(provider.calls) == 1
    assert len(llm.calls) == 1


def test_recursive_retrieves_followups_and_synthesizes_answer() -> None:
    first_response = """
    {
      "question": "q",
      "answer": "first answer [1]",
      "references": [
        {
          "label": "[1]",
          "source_url": "https://www.youtube.com/watch?v=abc",
          "timestamp_url": "https://www.youtube.com/watch?v=abc&t=593s",
          "start_seconds": 593,
          "end_seconds": 665,
          "chunk_index": 4,
          "video_id": "abc"
        }
      ],
      "followups_requested": true,
      "subtopics": [
        {
          "topic": "detail",
          "rationale": "thin evidence",
          "followup_query": "specific detail query",
          "confidence": 0.8
        }
      ]
    }
    """
    synthesis_response = """
    {
      "preserved_answer": "first answer [1]",
      "preserved_references": [
        {
          "label": "[1]",
          "source_url": "https://www.youtube.com/watch?v=abc",
          "timestamp_url": "https://www.youtube.com/watch?v=abc&t=593s",
          "start_seconds": 593,
          "end_seconds": 665,
          "chunk_index": 4,
          "video_id": "abc"
        }
      ],
      "subtopic_answers": [
        {
          "subtopic_index": 1,
          "topic": "detail",
          "followup_query": "specific detail query",
          "answer": "detail answer [s1.1]",
          "references": [
            {
              "label": "[s1.1]",
              "source_url": "https://www.youtube.com/watch?v=abc",
              "timestamp_url": "https://www.youtube.com/watch?v=abc&t=700s",
              "start_seconds": 700,
              "end_seconds": 760,
              "chunk_index": 5,
              "video_id": "abc"
            }
          ]
        }
      ],
      "layered_answer_markdown": "first answer [1]\\n\\n## detail\\ndetail answer [s1.1]"
    }
    """

    class NovelProvider(FakeProvider):
        def get_context(self, *args, **kwargs):
            context = super().get_context(*args, **kwargs)
            if len(self.calls) > 1:
                chunk = context.retrieved_chunks[0].model_copy(
                    update={
                        "chunk_index": 5,
                        "text": "follow-up detail",
                        "start_seconds": 700,
                        "end_seconds": 760,
                    }
                )
                context = replace(
                    context,
                    context_text="[1] 11:40-12:40\nfollow-up detail",
                    retrieved_chunks=[chunk],
                )
            return context

    llm = FakeLlm([first_response, synthesis_response])
    provider = NovelProvider()
    agent = RagTranscriptAgent(llm, provider)

    answer = agent.answer(
        RagQuestionRequest(
            question="q",
            recursive=True,
            recursion_options={"novelty_min_chunks": 1},
        )
    )

    assert answer.answer.startswith("first answer")
    assert answer.recursion is not None
    assert answer.recursion.terminated_reason == "completed"
    assert answer.recursion.total_followups_executed == 1
    assert provider.calls[1][0] == "specific detail query"
    assert len(llm.calls) == 2


TWO_SUBTOPIC_RESPONSE = """
{
  "question": "q",
  "answer": "first answer [1]",
  "followups_requested": true,
  "subtopics": [
    {
      "topic": "first detail",
      "rationale": "thin evidence",
      "followup_query": "first detail query",
      "confidence": 0.9
    },
    {
      "topic": "second detail",
      "rationale": "thin evidence",
      "followup_query": "second detail query",
      "confidence": 0.8
    }
  ]
}
"""


def test_every_retrieval_is_recorded_under_the_pass_it_ran_for() -> None:
    """A fan-out failure must not let a follow-up stand in for the first retrieval.

    ``last_context`` holds only the newest retrieval, so the record the chat
    runner persists has to come from the per-pass list instead.
    """

    class FanOutProvider(FakeProvider):
        def get_context(self, *args, **kwargs):
            call = len(self.calls) + 1
            if call == 3:
                raise RetrievalError(
                    "No indexed chunks found for that channel.",
                    [
                        TraceStep(
                            phase="retrieve",
                            label="Retrieve candidates",
                            detail="0 candidates",
                        )
                    ],
                )
            context = super().get_context(*args, **kwargs)
            if call > 1:
                chunk = context.retrieved_chunks[0].model_copy(
                    update={"chunk_index": 4 + call, "text": "follow-up detail"}
                )
                context = replace(context, retrieved_chunks=[chunk])
            return replace(
                context,
                trace=[
                    TraceStep(phase="retrieve", label="Retrieve candidates", detail=f"call {call}")
                ],
            )

    agent = RagTranscriptAgent(FakeLlm(TWO_SUBTOPIC_RESPONSE), FanOutProvider())

    with pytest.raises(RetrievalError):
        agent.answer(
            RagQuestionRequest(
                question="q",
                recursive=True,
                recursion_options={"novelty_min_chunks": 1},
            )
        )

    assert [retrieval.label for retrieval in agent.last_retrievals] == [
        "first retrieval",
        'follow-up 1 "first detail query"',
        'follow-up 2 "second detail query"',
    ]
    assert [step.detail for retrieval in agent.last_retrievals for step in retrieval.steps] == [
        "call 1",
        "call 2",
        "0 candidates",
    ]


def test_merged_context_keeps_every_retrievals_trace_when_synthesis_fails() -> None:
    """A synthesis that blows up must not erase the retrievals that already ran.

    ``last_context`` becomes the merged context before the synthesis call, and
    that is what the chat runner persists as the trace of a failed answer.
    """

    class TracingProvider(FakeProvider):
        def get_context(self, *args, **kwargs):
            context = super().get_context(*args, **kwargs)
            call_index = len(self.calls)
            if call_index > 1:
                chunk = context.retrieved_chunks[0].model_copy(
                    update={"chunk_index": 5, "text": "follow-up detail"}
                )
                context = replace(context, retrieved_chunks=[chunk])
            return replace(
                context,
                trace=[
                    TraceStep(
                        phase="retrieve",
                        label=f"Retrieve candidates ({call_index})",
                        detail="semantic search over the whole corpus",
                    )
                ],
            )

    class ExplodingLlm(FakeLlm):
        def invoke(self, messages):
            if self.calls:
                raise RuntimeError("deepseek 503")
            return super().invoke(messages)

    llm = ExplodingLlm(
        """
        {
          "question": "q",
          "answer": "first answer [1]",
          "followups_requested": true,
          "subtopics": [
            {
              "topic": "detail",
              "rationale": "thin evidence",
              "followup_query": "specific detail query",
              "confidence": 0.8
            }
          ]
        }
        """
    )
    agent = RagTranscriptAgent(llm, TracingProvider())

    with pytest.raises(RuntimeError):
        agent.answer(
            RagQuestionRequest(
                question="q",
                recursive=True,
                recursion_options={"novelty_min_chunks": 1},
            )
        )

    assert [step.label for step in agent.last_context.trace] == [
        "Retrieve candidates (1)",
        "Retrieve candidates (2)",
    ]


# ── reviewing a shared document ───────────────────────────────────────────────


DOC_CONTEXT = "DOCUMENT: A resume\nURL: https://example.com/cv\n\n[§1] Experience\nLed a team."


def _answer_json(answer: str = "Feedback on [§1], per [1].") -> str:
    import json

    return json.dumps(
        {
            "answer": answer,
            "references": [],
            "answer_confidence": 0.8,
            "followups_requested": False,
            "subtopics": [],
        }
    )


def test_a_document_changes_the_system_prompt_to_the_review_one() -> None:
    """The corpus stops being the subject and becomes the criteria, so the
    model is told that rather than left to infer it from an extra block."""
    from src.agents.prompts import DOC_REVIEW_SYSTEM_PROMPT

    llm = FakeLlm(_answer_json())
    agent = RagTranscriptAgent(llm, FakeProvider())

    agent.answer(RagQuestionRequest(question="any feedback?", document_context=DOC_CONTEXT))

    assert llm.messages[0].content == DOC_REVIEW_SYSTEM_PROMPT


def test_no_document_keeps_the_ordinary_rag_prompt() -> None:
    from src.agents.prompts import RAG_SYSTEM_PROMPT

    llm = FakeLlm(_answer_json())
    agent = RagTranscriptAgent(llm, FakeProvider())

    agent.answer(RagQuestionRequest(question="what changed?"))

    assert llm.messages[0].content == RAG_SYSTEM_PROMPT


def test_the_document_reaches_the_model_ahead_of_the_transcript_chunks() -> None:
    """The subject before the standards it is judged against."""
    llm = FakeLlm(_answer_json())
    agent = RagTranscriptAgent(llm, FakeProvider())

    agent.answer(RagQuestionRequest(question="any feedback?", document_context=DOC_CONTEXT))

    contents = [str(message.content) for message in llm.messages]
    document_at = next(i for i, text in enumerate(contents) if "[§1]" in text)
    chunks_at = next(i for i, text in enumerate(contents) if "Transcript context" in text)
    assert document_at < chunks_at


def test_the_corpus_is_still_retrieved_for_a_document_question() -> None:
    """The document supplies what is reviewed; the corpus supplies the advice."""
    provider = FakeProvider()
    agent = RagTranscriptAgent(FakeLlm(_answer_json()), provider)

    agent.answer(
        RagQuestionRequest(question="is my resume ATS-friendly?", document_context=DOC_CONTEXT)
    )

    assert provider.calls, "a document review still retrieves guidance from the corpus"
    assert provider.calls[0][0] == "is my resume ATS-friendly?"


def test_the_document_answer_comes_back_through_the_normal_contract() -> None:
    agent = RagTranscriptAgent(FakeLlm(_answer_json("Rewrite [§1].")), FakeProvider())

    answer = agent.answer(
        RagQuestionRequest(question="any feedback?", document_context=DOC_CONTEXT)
    )

    assert answer.answer == "Rewrite [§1]."
    assert answer.question == "any feedback?"


def test_a_document_changes_the_user_prompt_too() -> None:
    """The RAG template says "using only the retrieved transcript chunks",
    which is the wrong instruction when the subject is the user's document."""
    llm = FakeLlm(_answer_json())
    agent = RagTranscriptAgent(llm, FakeProvider())

    agent.answer(RagQuestionRequest(question="any feedback?", document_context=DOC_CONTEXT))

    user_prompt = str(llm.messages[-1].content)
    assert "only the retrieved" not in user_prompt
    assert "[§N]" in user_prompt


def test_no_document_keeps_the_ordinary_rag_user_prompt() -> None:
    llm = FakeLlm(_answer_json())
    agent = RagTranscriptAgent(llm, FakeProvider())

    agent.answer(RagQuestionRequest(question="what changed?"))

    assert "only the retrieved" in str(llm.messages[-1].content)


def test_a_caller_supplied_retrieval_query_is_what_the_corpus_sees() -> None:
    """The URL in "review https://... for me" must not be what gets embedded."""
    provider = FakeProvider()
    agent = RagTranscriptAgent(FakeLlm(_answer_json()), provider)

    agent.answer(
        RagQuestionRequest(
            question="review https://example.com/cv for me",
            document_context=DOC_CONTEXT,
            retrieval_query="review for me resume experience",
        )
    )

    assert provider.calls[0][0] == "review for me resume experience"


def test_the_answering_prompt_still_gets_the_users_own_wording() -> None:
    llm = FakeLlm(_answer_json())
    agent = RagTranscriptAgent(llm, FakeProvider())

    agent.answer(
        RagQuestionRequest(
            question="review https://example.com/cv for me",
            document_context=DOC_CONTEXT,
            retrieval_query="resume experience section",
        )
    )

    assert "review https://example.com/cv for me" in str(llm.messages[-1].content)


def test_a_retrieval_query_override_costs_no_rewrite_call() -> None:
    """The caller already knows the query; asking the model again would be a
    second LLM call charged for nothing."""
    llm = FakeLlm(_answer_json())
    agent = RagTranscriptAgent(llm, FakeProvider())

    agent.answer(
        RagQuestionRequest(
            question="and the education section?",
            history=["earlier turn"],
            document_context=DOC_CONTEXT,
            retrieval_query="education section resume",
        )
    )

    assert agent.last_rewrite is None
    assert len(llm.calls) == 1


# ── citation metadata is derived, never taken from the model ──────────────────


def _answer_json_with_references(answer: str, references: list) -> str:
    import json

    return json.dumps(
        {
            "answer": answer,
            "references": references,
            "answer_confidence": 0.8,
            "followups_requested": False,
            "subtopics": [],
        }
    )


def test_a_citations_metadata_comes_from_the_chunk_not_the_model() -> None:
    """The model picks which chunk to cite; it does not get to say what it is."""
    invented = [
        {
            "label": "[1]",
            "source_url": "https://www.youtube.com/watch?v=WRONG",
            "timestamp_url": "https://www.youtube.com/watch?v=WRONG&t=999s",
            "start_seconds": 999.0,
            "end_seconds": 1000.0,
            "chunk_index": 42,
            "video_id": "WRONG",
        }
    ]
    provider = FakeProvider()
    agent = RagTranscriptAgent(
        FakeLlm(_answer_json_with_references("Point [1].", invented)), provider
    )

    answer = agent.answer(RagQuestionRequest(question="what changed?"))

    chunk = agent.last_context.retrieved_chunks[0]
    reference = answer.references[0]
    assert reference.label == "[1]"
    assert reference.chunk_index == chunk.chunk_index
    assert reference.video_id == chunk.video_id
    assert reference.start_seconds == chunk.start_seconds


def test_a_citation_of_a_chunk_that_was_never_retrieved_is_dropped() -> None:
    """Repairing it would be the same invention one layer down."""
    out_of_range = [{"label": "[99]", "video_id": "x", "source_url": "https://y", "chunk_index": 0}]
    agent = RagTranscriptAgent(
        FakeLlm(_answer_json_with_references("Point [99].", out_of_range)), FakeProvider()
    )

    answer = agent.answer(RagQuestionRequest(question="what changed?"))

    assert all(reference.label != "[99]" for reference in answer.references)


def test_duplicate_labels_collapse_to_one_citation() -> None:
    duplicated = [
        {"label": "[1]", "video_id": "a", "source_url": "https://a", "chunk_index": 0},
        {"label": "[1]", "video_id": "b", "source_url": "https://b", "chunk_index": 7},
    ]
    agent = RagTranscriptAgent(
        FakeLlm(_answer_json_with_references("Point [1].", duplicated)), FakeProvider()
    )

    answer = agent.answer(RagQuestionRequest(question="what changed?"))

    assert [reference.label for reference in answer.references] == ["[1]"]


def test_a_subtopic_label_resolves_within_its_own_subtopic() -> None:
    """[s1.2] is the second chunk of subtopic 1, not the first chunk overall."""
    from src.agents.models import RagAnswerReference
    from src.agents.rag_transcript_agent import reconcile_references

    class Chunk:
        def __init__(self, index: int) -> None:
            self.source_url = "https://www.youtube.com/watch?v=vid"
            self.start_seconds = float(index * 10)
            self.end_seconds = float(index * 10 + 5)
            self.chunk_index = index
            self.video_id = "vid"

    chunks = [Chunk(0), Chunk(1), Chunk(2)]
    model_refs = [
        RagAnswerReference(
            label="[s1.2]",
            source_url="https://www.youtube.com/watch?v=wrong",
            timestamp_url="https://www.youtube.com/watch?v=wrong&t=999s",
            video_id="wrong",
            chunk_index=99,
        )
    ]

    resolved = reconcile_references(model_refs, chunks, label_of=lambda index: f"[s1.{index}]")

    assert [reference.label for reference in resolved] == ["[s1.2]"]
    assert resolved[0].chunk_index == 1

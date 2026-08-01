"""Follow-up questions are rewritten to stand alone before retrieval."""

from __future__ import annotations

import json

from src.agents.models import RagQuestionRequest
from src.agents.prompts import build_rag_question_prompt, build_rewrite_prompt


class FakeLLM:
    def __init__(self, content: str = '{"query": "rewritten"}', raises=False):
        self.content = content
        self.raises = raises
        self.prompts: list[str] = []

    def invoke(self, messages):
        self.prompts.append(str(messages[-1].content))
        if self.raises:
            raise RuntimeError("llm down")
        return type("R", (), {"content": self.content})()


def agent(llm):
    from src.agents.rag_transcript_agent import RagTranscriptAgent

    return RagTranscriptAgent(llm, context_provider=None)


def test_question_without_history_is_not_rewritten_and_costs_no_call():
    llm = FakeLLM()
    query = agent(llm).retrieval_query(
        RagQuestionRequest(question="What changed for investors?")
    )
    assert query == "What changed for investors?"
    assert llm.prompts == []


def test_follow_up_is_rewritten_to_stand_alone():
    llm = FakeLLM(json.dumps({"query": "capital gains tax discount changes"}))
    query = agent(llm).retrieval_query(
        RagQuestionRequest(
            question="what about the second one?",
            history=["What tax changes were announced?", "Four changes were announced."],
        )
    )
    assert query == "capital gains tax discount changes"


def test_rewrite_prompt_carries_the_prior_turns():
    prompt = build_rewrite_prompt("and the second?", ["First turn", "Second turn"])
    assert "First turn" in prompt and "Second turn" in prompt
    assert "and the second?" in prompt


def test_failed_rewrite_degrades_to_the_raw_question():
    """A rewrite failure must never block the answer."""
    query = agent(FakeLLM(raises=True)).retrieval_query(
        RagQuestionRequest(question="what about it?", history=["earlier turn"])
    )
    assert query == "what about it?"


def test_unparseable_rewrite_degrades_to_the_raw_question():
    query = agent(FakeLLM("not json at all")).retrieval_query(
        RagQuestionRequest(question="what about it?", history=["earlier turn"])
    )
    assert query == "what about it?"


def test_empty_rewrite_degrades_to_the_raw_question():
    query = agent(FakeLLM(json.dumps({"query": "   "}))).retrieval_query(
        RagQuestionRequest(question="what about it?", history=["earlier turn"])
    )
    assert query == "what about it?"


def test_no_history_records_no_rewrite_at_all():
    """No record means no call was made — the trace must not invent one."""
    rag = agent(FakeLLM())
    rag.retrieval_query(RagQuestionRequest(question="What changed?"))
    assert rag.last_rewrite is None


def test_successful_rewrite_is_recorded_for_the_trace():
    rag = agent(FakeLLM(json.dumps({"query": "capital gains tax discount"})))
    rag.retrieval_query(
        RagQuestionRequest(question="what about the second one?", history=["earlier turn"])
    )
    assert rag.last_rewrite is not None
    assert rag.last_rewrite.query == "capital gains tax discount"
    assert rag.last_rewrite.degraded is False
    assert isinstance(rag.last_rewrite.elapsed_ms, int)


def test_degraded_rewrite_is_recorded_as_degraded():
    """A failed rewrite still cost a call, and must not look like a success."""
    for llm in (FakeLLM(raises=True), FakeLLM("not json"), FakeLLM(json.dumps({"query": " "}))):
        rag = agent(llm)
        rag.retrieval_query(RagQuestionRequest(question="what about it?", history=["earlier turn"]))
        assert rag.last_rewrite is not None
        assert rag.last_rewrite.degraded is True
        assert rag.last_rewrite.query == "what about it?"


def test_answer_prompt_includes_history_but_marks_it_as_not_evidence():
    prompt = build_rag_question_prompt("follow up?", ["earlier turn"])
    assert "earlier turn" in prompt
    # Prior turns must not be citable as sources.
    assert "not" in prompt.lower() and "evidence" in prompt.lower()


def test_answer_prompt_without_history_is_unchanged():
    assert build_rag_question_prompt("q") == build_rag_question_prompt("q", [])

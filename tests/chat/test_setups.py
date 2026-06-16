from __future__ import annotations

import pytest

from src.agents.models import RagTranscriptAnswer
from src.chat.setups import (
    SETUP_KEYS,
    RagSetupRunner,
    command_for,
    select_setups,
)


class FakeContext:
    def __init__(self, text: str = "some retrieved context", chunks: int = 3) -> None:
        self.context_text = text
        self.retrieved_chunks = [object()] * chunks


class FakeRagLlm:
    def __init__(self) -> None:
        self.last_context = FakeContext(chunks=3)
        self.requests: list = []

    def answer(self, request):
        self.requests.append(request)
        return RagTranscriptAnswer(question=request.question, answer="llm answer")


class FakeRagAgent:
    def __init__(self) -> None:
        self.last_context = FakeContext(chunks=5)
        self.last_iteration_count = 4
        self.last_terminated_reason = "completed"
        self.requests: list = []

    def answer(self, request):
        self.requests.append(request)
        return RagTranscriptAnswer(question=request.question, answer="agent answer")


class BrokenAgent:
    last_context = None

    def answer(self, request):
        raise RuntimeError("boom")


def _runner(settings, *, rag_llm=None, rag_agent=None) -> RagSetupRunner:
    runner = RagSetupRunner(settings, provider=None)
    runner._rag_llm_agent = rag_llm
    runner._rag_agent = rag_agent
    return runner


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("a", SETUP_KEYS),
        ("all", SETUP_KEYS),
        ("1,3", ["rag_llm", "rag_agent"]),
        ("1 2", ["rag_llm", "rag_llm_recursive"]),
        ("rag_agent", ["rag_agent"]),
        ("1,1", ["rag_llm"]),
    ],
)
def test_select_setups_parses_choices(raw, expected) -> None:
    assert select_setups(raw) == expected


@pytest.mark.parametrize("raw", ["", "  ", "9", "foo", "1,bad"])
def test_select_setups_rejects_bad_input(raw) -> None:
    with pytest.raises(ValueError):
        select_setups(raw)


def test_command_for_includes_flags_and_url() -> None:
    assert "--rag_llm --recursive" in command_for("rag_llm_recursive")
    assert command_for("rag_agent", url="https://x")[-1] == '"'
    assert '--url "https://x"' in command_for("rag_agent", url="https://x")


def test_run_rag_llm_single_hop(settings) -> None:
    fake = FakeRagLlm()
    runner = _runner(settings, rag_llm=fake)

    result = runner.run("rag_llm", "what about X?", top_k=7)

    assert result.answer == "llm answer"
    assert result.llm_calls == 1
    assert result.chunk_count == 3
    assert result.token_estimate > 0
    assert result.error is None
    assert fake.requests[-1].top_k == 7
    assert fake.requests[-1].recursive is False


def test_run_rag_llm_recursive_sets_request_flag(settings) -> None:
    fake = FakeRagLlm()
    runner = _runner(settings, rag_llm=fake)

    result = runner.run("rag_llm_recursive", "what about X?")

    assert result.key == "rag_llm_recursive"
    assert fake.requests[-1].recursive is True
    assert fake.requests[-1].recursion_options is not None


def test_run_rag_agent_reports_iterations(settings) -> None:
    fake = FakeRagAgent()
    runner = _runner(settings, rag_agent=fake)

    result = runner.run("rag_agent", "what about X?")

    assert result.answer == "agent answer"
    assert result.iterations == 4
    assert result.terminated_reason == "completed"
    assert result.chunk_count == 5


def test_run_captures_setup_error(settings) -> None:
    runner = _runner(settings, rag_llm=BrokenAgent())

    result = runner.run("rag_llm", "q")

    assert result.error == "boom"
    assert result.answer == ""


def test_run_many_reports_progress(settings) -> None:
    runner = _runner(settings, rag_llm=FakeRagLlm(), rag_agent=FakeRagAgent())
    messages: list[str] = []

    results = runner.run_many(
        ["rag_llm", "rag_agent"], "q", on_progress=messages.append
    )

    assert [r.key for r in results] == ["rag_llm", "rag_agent"]
    assert len(messages) == 2

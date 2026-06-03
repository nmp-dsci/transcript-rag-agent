from __future__ import annotations

import json
import sys
from datetime import datetime, timezone

import pytest
from langchain_core.messages import AIMessage

from src import cli
from src.agents.context import TranscriptContext
from src.agents.models import (
    AgentProgressEvent,
    RagQuestionRequest,
    RagTranscriptAnswer,
)
from src.agents.rag_agent import RagAgent
from src.config import Settings
from src.rag.models import RetrievedChunk
from src.transcripts.models import Transcript

# --- Canned final answers ----------------------------------------------------

STRUCTURED_ANSWER_TEXT = (
    "## Key Findings\n"
    "1. AI engineers use Claude for spec-driven development [1].\n"
    "2. The main risk is silent regression [1].\n\n"
    "## Finding 1: Spec-driven development\n"
    "Engineers write a short spec before delegating [1].\n\n"
    "## Finding 2: Silent regression risk\n"
    "Claude-generated code can break untested paths [1]."
)

STRUCTURED_ANSWER_JSON = (
    '{"question": "q", '
    f'"answer": {json.dumps(STRUCTURED_ANSWER_TEXT)}, '
    '"references": [{"label": "[1]", '
    '"source_url": "https://www.youtube.com/watch?v=abc", '
    '"timestamp_url": "https://www.youtube.com/watch?v=abc&t=593s", '
    '"start_seconds": 593.0, "end_seconds": 665.0, '
    '"chunk_index": 4, "video_id": "abc"}]}'
)

FLAT_ANSWER_TEXT = "The corpus says engineers use Claude for features [1]."


# --- Fakes -------------------------------------------------------------------


class FakeBoundLlm:
    """The object returned by ``llm.bind_tools(...)``; yields scripted messages."""

    def __init__(self, parent: "FakeLlm") -> None:
        self.parent = parent

    def invoke(self, messages):
        self.parent.invocations.append(messages)
        index = min(len(self.parent.invocations) - 1, len(self.parent.responses) - 1)
        return self.parent.responses[index]


class FakeLlm:
    """Mock chat model that supports ``bind_tools`` and returns canned messages.

    ``responses`` is a list of ``AIMessage`` objects. Each call to the bound
    model's ``invoke`` returns the next message in order (clamping at the last).
    """

    def __init__(self, responses: list[AIMessage]) -> None:
        self.responses = responses
        self.invocations: list = []
        self.bound_tools: list | None = None

    def bind_tools(self, tools):
        self.bound_tools = tools
        return FakeBoundLlm(self)

    def invoke(self, messages):  # pragma: no cover - graph uses bound model
        return FakeBoundLlm(self).invoke(messages)


class FakeProvider:
    """Mock ``MultiTranscriptRagContextProvider`` returning canned contexts.

    ``chunk_count`` controls how many ``\\n\\n``-separated blocks the returned
    ``context_text`` has so ``_chunk_count_from_tool_message`` reports it back.
    """

    def __init__(self, chunk_count: int = 2) -> None:
        self.calls: list[dict] = []
        self.chunk_count = chunk_count

    def get_context(
        self,
        question: str,
        source_url: str | None = None,
        top_k: int = 10,
        filter_transcripts: bool = False,
        transcript_filter_top_k: int = 5,
        transcript_filter_min_score: float = 0.25,
    ) -> TranscriptContext:
        call_index = len(self.calls)
        self.calls.append(
            {
                "question": question,
                "source_url": source_url,
                "top_k": top_k,
                "filter_transcripts": filter_transcripts,
                "transcript_filter_top_k": transcript_filter_top_k,
                "transcript_filter_min_score": transcript_filter_min_score,
            }
        )
        # Each call returns a distinct chunk so the union across calls grows.
        chunks = [
            RetrievedChunk(
                transcript_id="raw_transcript:abc",
                video_id="abc",
                source_url="https://www.youtube.com/watch?v=abc",
                chunk_index=call_index * 10 + offset,
                text=f"chunk text {call_index}-{offset}",
                start_seconds=float(593 + offset),
                end_seconds=float(665 + offset),
                segment_count=1,
            )
            for offset in range(self.chunk_count)
        ]
        context_text = "\n\n".join(
            f"[{i + 1}] video=abc\n{chunk.text}" for i, chunk in enumerate(chunks)
        )
        transcript = Transcript(
            video_id="all",
            url="https://www.youtube.com/watch?v=abc",
            provider="rag",
            raw_text="chunk text",
            fetched_at=datetime.now(timezone.utc),
        )
        return TranscriptContext(
            transcript=transcript,
            cache_status="hit",
            context_text=context_text,
            context_mode="rag",
            retrieved_chunks=chunks,
            top_k=top_k,
        )


# --- Helpers -----------------------------------------------------------------


def _tool_call_message(query: str, call_id: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "retrieve_transcript_chunks",
                "args": {"query": query},
                "id": call_id,
            }
        ],
    )


def _final_message(content: str) -> AIMessage:
    return AIMessage(content=content)


def _retrieve_then_answer(num_retrievals: int, final: str) -> list[AIMessage]:
    messages = [
        _tool_call_message(f"focused query {i}", f"call_{i}")
        for i in range(num_retrievals)
    ]
    messages.append(_final_message(final))
    return messages


def _request(**overrides) -> RagQuestionRequest:
    base = {"question": "q", "top_k": 10}
    base.update(overrides)
    return RagQuestionRequest(**base)


# --- answer() ----------------------------------------------------------------


def test_answer_returns_valid_rag_transcript_answer() -> None:
    llm = FakeLlm(_retrieve_then_answer(1, STRUCTURED_ANSWER_JSON))
    provider = FakeProvider()
    agent = RagAgent(llm, provider)

    result = agent.answer(_request())

    assert isinstance(result, RagTranscriptAnswer)
    assert result.question
    assert result.answer
    assert result.references
    assert result.references[0].video_id == "abc"


def test_no_tool_call_first_turn_returns_llm_message_immediately() -> None:
    llm = FakeLlm([_final_message(STRUCTURED_ANSWER_JSON)])
    provider = FakeProvider()
    agent = RagAgent(llm, provider)

    result = agent.answer(_request())

    assert agent.last_iteration_count == 0
    assert provider.calls == []
    assert "## Key Findings" in result.answer


def test_single_retrieve_executes_tool_and_loops_back() -> None:
    llm = FakeLlm(_retrieve_then_answer(1, STRUCTURED_ANSWER_JSON))
    provider = FakeProvider()
    agent = RagAgent(llm, provider)

    agent.answer(_request())

    # One tool execution and a loop back means the LLM node ran twice
    # (initial tool call + final answer).
    assert agent.last_iteration_count == 1
    assert len(provider.calls) == 1
    assert len(llm.invocations) == 2


def test_three_retrieve_calls_union_and_iteration_count() -> None:
    llm = FakeLlm(_retrieve_then_answer(3, STRUCTURED_ANSWER_JSON))
    provider = FakeProvider(chunk_count=2)
    agent = RagAgent(llm, provider)

    agent.answer(_request())

    assert agent.last_iteration_count == 3
    assert len(provider.calls) == 3
    assert agent.last_context is not None
    # Union of three retrievals, each returning 2 distinct chunks = 6.
    assert len(agent.last_context.retrieved_chunks) == 6


def test_max_iterations_guard_returns_answer_not_exception() -> None:
    # The LLM never stops calling the tool, so the recursion limit fires.
    never_ending = [_tool_call_message(f"q{i}", f"id{i}") for i in range(50)]
    llm = FakeLlm(never_ending)
    provider = FakeProvider()
    agent = RagAgent(llm, provider, max_iterations=3)

    result = agent.answer(_request())

    assert isinstance(result, RagTranscriptAnswer)
    assert agent.last_terminated_reason == "max_iterations_reached"


def test_tool_closure_passes_request_params_regardless_of_query() -> None:
    llm = FakeLlm(_retrieve_then_answer(2, STRUCTURED_ANSWER_JSON))
    provider = FakeProvider()
    agent = RagAgent(llm, provider)

    request = _request(
        source_url="https://www.youtube.com/watch?v=abc",
        top_k=7,
        filter_transcripts=False,
        transcript_filter_top_k=4,
        transcript_filter_min_score=0.3,
    )
    agent.answer(request)

    assert len(provider.calls) == 2
    # Query differs per call; everything else is fixed from the request.
    assert provider.calls[0]["question"] != provider.calls[1]["question"]
    for call in provider.calls:
        assert call["source_url"] == "https://www.youtube.com/watch?v=abc"
        assert call["top_k"] == 7
        assert call["filter_transcripts"] is False
        assert call["transcript_filter_top_k"] == 4
        assert call["transcript_filter_min_score"] == 0.3


# --- Answer parsing ----------------------------------------------------------


def test_malformed_final_message_invokes_fallback_references() -> None:
    # Non-JSON final message with an inline citation -> fallback references.
    llm = FakeLlm(_retrieve_then_answer(1, FLAT_ANSWER_TEXT))
    provider = FakeProvider()
    agent = RagAgent(llm, provider)

    result = agent.answer(_request())

    assert isinstance(result, RagTranscriptAnswer)
    assert result.answer == FLAT_ANSWER_TEXT
    # Fallback pulls references from the union context (chunk cited as [1]).
    assert len(result.references) == 1
    assert result.references[0].label == "[1]"


def test_structured_answer_contains_key_findings_and_finding_headings() -> None:
    llm = FakeLlm(_retrieve_then_answer(1, STRUCTURED_ANSWER_JSON))
    provider = FakeProvider()
    agent = RagAgent(llm, provider)

    result = agent.answer(_request())

    assert "## Key Findings" in result.answer
    assert "## Finding 1:" in result.answer


def test_flat_answer_without_headings_parses_without_raising() -> None:
    llm = FakeLlm([_final_message(FLAT_ANSWER_TEXT)])
    provider = FakeProvider()
    agent = RagAgent(llm, provider)

    result = agent.answer(_request())

    assert isinstance(result, RagTranscriptAnswer)
    assert result.answer == FLAT_ANSWER_TEXT
    assert "## " not in result.answer


# --- answer_streaming() ------------------------------------------------------


def test_streaming_emits_alternating_events_before_answer_start() -> None:
    llm = FakeLlm(_retrieve_then_answer(3, STRUCTURED_ANSWER_JSON))
    provider = FakeProvider()
    agent = RagAgent(llm, provider)

    events: list[AgentProgressEvent] = []
    result = agent.answer_streaming(_request(), on_event=events.append)

    types = [event.event_type for event in events]
    assert types == [
        "retrieval_start",
        "retrieval_complete",
        "retrieval_start",
        "retrieval_complete",
        "retrieval_start",
        "retrieval_complete",
        "answer_start",
    ]
    assert sum(1 for t in types if t == "retrieval_start") == 3
    assert sum(1 for t in types if t == "retrieval_complete") == 3

    # Same answer as answer() on identical inputs.
    llm2 = FakeLlm(_retrieve_then_answer(3, STRUCTURED_ANSWER_JSON))
    nonstream = RagAgent(llm2, FakeProvider()).answer(_request())
    assert result == nonstream


def test_streaming_chunk_count_matches_provider() -> None:
    llm = FakeLlm(_retrieve_then_answer(2, STRUCTURED_ANSWER_JSON))
    provider = FakeProvider(chunk_count=5)
    agent = RagAgent(llm, provider)

    events: list[AgentProgressEvent] = []
    agent.answer_streaming(_request(), on_event=events.append)

    completes = [e for e in events if e.event_type == "retrieval_complete"]
    assert completes
    for event in completes:
        assert event.chunk_count == 5


def test_streaming_without_on_event_does_not_raise() -> None:
    llm = FakeLlm(_retrieve_then_answer(2, STRUCTURED_ANSWER_JSON))
    provider = FakeProvider()
    agent = RagAgent(llm, provider)

    result = agent.answer_streaming(_request(), on_event=None)

    assert isinstance(result, RagTranscriptAnswer)
    assert result.references


def test_answer_and_streaming_produce_identical_answers() -> None:
    request = _request()

    llm_a = FakeLlm(_retrieve_then_answer(2, STRUCTURED_ANSWER_JSON))
    invoke_result = RagAgent(llm_a, FakeProvider()).answer(request)

    llm_b = FakeLlm(_retrieve_then_answer(2, STRUCTURED_ANSWER_JSON))
    stream_result = RagAgent(llm_b, FakeProvider()).answer_streaming(request)

    assert invoke_result == stream_result


# --- CLI routing -------------------------------------------------------------


class FakeCliRagAgent:
    instantiated = False
    last_request = None

    def __init__(self) -> None:
        self.last_context = None
        self.last_iteration_count = 3
        self.last_terminated_reason = "completed"
        self.max_iterations = 10

    @classmethod
    def from_settings(cls, settings, context_provider=None):
        cls.instantiated = True
        return cls()

    def answer_streaming(self, request, on_event=None):
        FakeCliRagAgent.last_request = request
        if on_event is not None:
            on_event(
                AgentProgressEvent(iteration=1, event_type="retrieval_start", query="q")
            )
            on_event(
                AgentProgressEvent(
                    iteration=1,
                    event_type="retrieval_complete",
                    query="q",
                    chunk_count=4,
                )
            )
            on_event(AgentProgressEvent(iteration=1, event_type="answer_start"))
        return RagTranscriptAnswer(question=request.question, answer="agent answer")


class FakePipelineRagAgent:
    instantiated = False
    last_request = None

    def __init__(self) -> None:
        self.last_context = None

    @classmethod
    def from_settings(cls, settings, context_provider=None):
        cls.instantiated = True
        return cls()

    def answer(self, request):
        FakePipelineRagAgent.last_request = request
        return RagTranscriptAnswer(question=request.question, answer="pipeline answer")


class FakeEmbeddingModel:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name


class FakeChunkStore:
    def __init__(self, *args, **kwargs) -> None:
        pass


class FakeIndexer:
    def __init__(self, *args, **kwargs) -> None:
        pass


class FakeMultiProvider:
    def __init__(self, *args, **kwargs) -> None:
        pass


class _NullRun:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


def _patch_cli_for_agent(monkeypatch, tmp_path) -> None:
    settings = Settings(
        superdata_api_key="super",
        deepseek_api_key="deep",
        deepseek_model="deepseek-v4",
        deepseek_base_url=None,
        chroma_path=tmp_path / "chroma",
        mlflow_tracking_uri=f"file:{tmp_path / 'mlruns'}",
        mlflow_experiment_name="test-cli",
        log_transcript_artifacts=False,
    )
    monkeypatch.setattr(cli, "load_settings", lambda require_keys=True: settings)
    monkeypatch.setattr(cli, "SuperdataTranscriptFetcher", lambda *a, **k: object())
    monkeypatch.setattr(cli, "RawTranscriptStore", lambda *a, **k: object())
    monkeypatch.setattr(cli, "RawTranscriptContextProvider", lambda *a, **k: object())
    monkeypatch.setattr(cli, "cli_run", _NullRun)
    monkeypatch.setattr(cli, "HuggingFaceEmbeddingModel", FakeEmbeddingModel)
    monkeypatch.setattr(cli, "TranscriptChunkStore", FakeChunkStore)
    monkeypatch.setattr(cli, "RagIndexer", FakeIndexer)
    monkeypatch.setattr(cli, "MultiTranscriptRagContextProvider", FakeMultiProvider)
    monkeypatch.setattr(cli, "_build_summary_store", lambda *a, **k: None)
    monkeypatch.setattr(cli, "log_context_details", lambda *a, **k: None)
    monkeypatch.setattr(cli, "log_transcript_filter_details", lambda *a, **k: None)
    monkeypatch.setattr(cli, "log_recursion_trace", lambda *a, **k: None)
    monkeypatch.setattr(cli, "RagAgent", FakeCliRagAgent)
    monkeypatch.setattr(cli, "RagTranscriptAgent", FakePipelineRagAgent)


def test_cli_agent_flag_instantiates_rag_agent(monkeypatch, tmp_path, capsys) -> None:
    _patch_cli_for_agent(monkeypatch, tmp_path)
    FakeCliRagAgent.instantiated = False
    FakePipelineRagAgent.instantiated = False

    result = cli.main(["rag-ask", "question", "--rag_agent"])

    assert result == 0
    assert FakeCliRagAgent.instantiated is True
    assert FakePipelineRagAgent.instantiated is False
    capsys.readouterr()


def test_cli_without_agent_flag_instantiates_pipeline_agent(
    monkeypatch, tmp_path, capsys
) -> None:
    _patch_cli_for_agent(monkeypatch, tmp_path)
    FakeCliRagAgent.instantiated = False
    FakePipelineRagAgent.instantiated = False

    result = cli.main(["rag-ask", "question"])

    assert result == 0
    assert FakePipelineRagAgent.instantiated is True
    assert FakeCliRagAgent.instantiated is False
    capsys.readouterr()


def test_cli_rag_llm_flag_instantiates_pipeline_agent(
    monkeypatch, tmp_path, capsys
) -> None:
    _patch_cli_for_agent(monkeypatch, tmp_path)
    FakeCliRagAgent.instantiated = False
    FakePipelineRagAgent.instantiated = False

    result = cli.main(["rag-ask", "question", "--rag_llm"])

    assert result == 0
    assert FakePipelineRagAgent.instantiated is True
    assert FakeCliRagAgent.instantiated is False
    capsys.readouterr()


def test_cli_rag_llm_and_rag_agent_are_mutually_exclusive(
    monkeypatch, tmp_path, capsys
) -> None:
    _patch_cli_for_agent(monkeypatch, tmp_path)

    with pytest.raises(SystemExit):
        cli.main(["rag-ask", "question", "--rag_llm", "--rag_agent"])

    capsys.readouterr()


def test_cli_agent_footer_present_with_agent(monkeypatch, tmp_path, capsys) -> None:
    _patch_cli_for_agent(monkeypatch, tmp_path)

    cli.main(["rag-ask", "question", "--rag_agent"])

    output = capsys.readouterr().out
    assert "Agent: 3 iterations (rag_agent)" in output


def test_cli_agent_footer_absent_without_agent(monkeypatch, tmp_path, capsys) -> None:
    _patch_cli_for_agent(monkeypatch, tmp_path)

    cli.main(["rag-ask", "question"])

    output = capsys.readouterr().out
    assert "rag_agent" not in output
    assert "Agent:" not in output


# --- CLI streaming / TTY / color --------------------------------------------


def test_cli_tty_iteration_one_has_cyan_and_reset() -> None:
    header_state = {"printed": False}
    printer = cli._make_agent_progress_printer(is_tty=True, header_state=header_state)

    captured: list[str] = []

    class _Recorder:
        def write(self, text):
            captured.append(text)

        def flush(self):
            pass

    original = sys.stdout
    sys.stdout = _Recorder()
    try:
        printer(
            AgentProgressEvent(iteration=1, event_type="retrieval_start", query="q1")
        )
        printer(
            AgentProgressEvent(
                iteration=1,
                event_type="retrieval_complete",
                query="q1",
                chunk_count=8,
            )
        )
    finally:
        sys.stdout = original

    output = "".join(captured)
    assert "\033[96m" in output  # bright cyan for iteration 1
    assert output.rstrip("\n").endswith("\033[0m")


def test_cli_non_tty_has_no_ansi_codes() -> None:
    header_state = {"printed": False}
    printer = cli._make_agent_progress_printer(is_tty=False, header_state=header_state)

    captured: list[str] = []

    class _Recorder:
        def write(self, text):
            captured.append(text)

        def flush(self):
            pass

    original = sys.stdout
    sys.stdout = _Recorder()
    try:
        printer(
            AgentProgressEvent(iteration=1, event_type="retrieval_start", query="q1")
        )
        printer(
            AgentProgressEvent(
                iteration=1,
                event_type="retrieval_complete",
                query="q1",
                chunk_count=8,
            )
        )
        printer(AgentProgressEvent(iteration=1, event_type="answer_start"))
    finally:
        sys.stdout = original

    output = "".join(captured)
    assert "\033[" not in output


def test_color_cycle_iteration_seven_matches_iteration_one() -> None:
    assert cli._color_for(1) == "\033[96m"
    assert cli._color_for(2) == "\033[93m"
    assert cli._color_for(3) == "\033[92m"
    assert cli._color_for(4) == "\033[95m"
    assert cli._color_for(5) == "\033[94m"
    assert cli._color_for(6) == "\033[97m"
    # (N - 1) % 6 cycling: iteration 7 -> same as iteration 1.
    assert cli._color_for(7) == cli._color_for(1)
    for iteration in range(1, 13):
        expected = cli._ITERATION_COLORS[(iteration - 1) % 6]
        assert cli._color_for(iteration) == expected


def test_answer_references_footer_lines_have_no_ansi(monkeypatch, tmp_path) -> None:
    # On a TTY the retrieval lines are colored, but the Answer / References /
    # Agent footer lines must never contain ANSI codes.
    _patch_cli_for_agent(monkeypatch, tmp_path)
    monkeypatch.setattr(cli.sys.stdout, "isatty", lambda: True, raising=False)

    captured: list[str] = []

    class _Recorder:
        def write(self, text):
            captured.append(text)

        def flush(self):
            pass

        def isatty(self):
            return True

    original = sys.stdout
    sys.stdout = _Recorder()
    try:
        cli.main(["rag-ask", "question", "--rag_agent"])
    finally:
        sys.stdout = original

    output = "".join(captured)
    for line in output.splitlines():
        if (
            line.startswith("Answer")
            or line.startswith("References")
            or line.startswith("Agent:")
        ):
            assert "\033[" not in line, line


@pytest.mark.parametrize("iteration", [1, 7, 13])
def test_color_for_wraps_palette(iteration) -> None:
    assert cli._color_for(iteration) == cli._ITERATION_COLORS[(iteration - 1) % 6]

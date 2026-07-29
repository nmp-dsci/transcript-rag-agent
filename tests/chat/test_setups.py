from __future__ import annotations

import pytest

from src.agents.models import RagTranscriptAnswer
from src.chat.setups import (
    SETUP_KEYS,
    RagSetupRunner,
    command_for,
    select_setups,
)


class FakeChunk:
    def __init__(self, text: str) -> None:
        self.text = text


class FakeContext:
    def __init__(self, text: str = "some retrieved context", chunks: int = 3) -> None:
        self.context_text = text
        self.retrieved_chunks = [FakeChunk(f"chunk {i}") for i in range(chunks)]


class FakeRagLlm:
    """A stand-in that publishes per-answer state where a real agent does.

    ``RagTranscriptAgent`` only records ``last_context``/``last_rewrite`` while
    ``answer`` runs, and the runner clears both before every run, so a fake that
    set them in ``__init__`` would model state no real agent can hold.
    ``context``/``rewrite`` are what this fake will publish when asked.
    """

    def __init__(self) -> None:
        self.context = FakeContext(chunks=3)
        self.rewrite = None
        self.last_context = None
        self.last_rewrite = None
        self.requests: list = []

    def _publish(self) -> None:
        self.last_context = self.context
        self.last_rewrite = self.rewrite

    def answer(self, request):
        self.requests.append(request)
        self._publish()
        return RagTranscriptAnswer(question=request.question, answer="llm answer")


class FakeRagAgent:
    def __init__(self) -> None:
        self.context = FakeContext(chunks=5)
        self.last_context = None
        self.last_iteration_count = 4
        self.last_terminated_reason = "completed"
        self.requests: list = []

    def answer(self, request):
        self.requests.append(request)
        self.last_context = self.context
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
    assert result.contexts == ["chunk 0", "chunk 1", "chunk 2"]
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
    # Nothing was recorded before the failure, so an empty trace is the honest one.
    assert result.trace == []


def test_run_many_reports_progress(settings) -> None:
    runner = _runner(settings, rag_llm=FakeRagLlm(), rag_agent=FakeRagAgent())
    messages: list[str] = []

    results = runner.run_many(["rag_llm", "rag_agent"], "q", on_progress=messages.append)

    assert [r.key for r in results] == ["rag_llm", "rag_agent"]
    assert len(messages) == 2


def test_followups_serialise_from_agent_subtopics():
    from src.agents.models import FollowupSubtopic
    from src.chat.setups import _followups_to_dicts

    dicts = _followups_to_dicts([FollowupSubtopic(topic="t", followup_query="q", confidence=0.5)])
    assert dicts[0]["followup_query"] == "q"


def test_followups_tolerate_plain_dicts_and_junk():
    from src.chat.setups import _followups_to_dicts

    assert _followups_to_dicts([{"topic": "t"}, None, "junk"]) == [{"topic": "t"}]
    assert _followups_to_dicts([]) == []


# ── persisted traces ─────────────────────────────────────────────────────────


def _chunk(video_id: str, index: int):
    from src.rag.models import RetrievedChunk

    return RetrievedChunk(
        transcript_id=f"raw_transcript:{video_id}",
        video_id=video_id,
        source_url=f"https://www.youtube.com/watch?v={video_id}",
        chunk_index=index,
        text=f"text {index}",
    )


class TracingContext(FakeContext):
    """A context whose provider recorded retrieval-stage TraceSteps."""

    def __init__(self) -> None:
        from src.agents.models import TraceStep

        super().__init__(chunks=2)
        self.trace = [
            TraceStep(
                phase="retrieve",
                label="Retrieve candidates",
                detail="semantic search over the whole corpus — 2 candidates",
                chunk_ids=["chunk:vid00000001:0", "chunk:vid00000001:1"],
                elapsed_ms=12,
            )
        ]


def test_single_hop_trace_combines_context_stages_and_llm_step(settings) -> None:
    fake = FakeRagLlm()
    fake.context = TracingContext()
    runner = _runner(settings, rag_llm=fake)

    result = runner.run("rag_llm", "what about X?")

    assert [step["phase"] for step in result.trace] == ["retrieve", "llm"]
    assert result.trace[0]["chunk_ids"] == ["chunk:vid00000001:0", "chunk:vid00000001:1"]
    assert result.trace[1]["model"] == settings.deepseek_model


def test_single_hop_trace_records_the_history_driven_rewrite(settings) -> None:
    """A follow-up costs an extra LLM call; the trace must not under-report it."""
    from src.agents.models import QueryRewrite
    from src.chat.setups import AskScope

    fake = FakeRagLlm()
    fake.context = TracingContext()
    fake.rewrite = QueryRewrite(query="capital gains discount changes", elapsed_ms=90)
    runner = _runner(settings, rag_llm=fake)

    result = runner.run(
        "rag_llm", "what about the second one?", scope=AskScope(history=["earlier turn"])
    )

    assert [step["phase"] for step in result.trace] == ["llm", "retrieve", "llm"]
    rewrite_step = result.trace[0]
    assert rewrite_step["label"] == "Rewrite query"
    assert "capital gains discount changes" in rewrite_step["detail"]
    assert rewrite_step["elapsed_ms"] == 90


def test_llm_calls_count_the_rewrite_the_follow_up_actually_made(settings) -> None:
    """The LLM-call chip must agree with the trace printed beneath it."""
    from src.agents.models import QueryRewrite
    from src.chat.setups import AskScope

    fake = FakeRagLlm()
    fake.context = TracingContext()
    fake.rewrite = QueryRewrite(query="capital gains discount changes")
    runner = _runner(settings, rag_llm=fake)

    result = runner.run(
        "rag_llm", "what about the second one?", scope=AskScope(history=["earlier turn"])
    )

    assert result.llm_calls == 2
    assert sum(1 for step in result.trace if step["phase"] == "llm") == 2


def test_recursive_llm_calls_count_the_rewrite_on_top_of_the_stages(settings) -> None:
    from src.agents.models import QueryRewrite, RecursionStage, RecursionTrace
    from src.chat.setups import AskScope

    recursion = RecursionTrace(
        stages=[
            RecursionStage(name="first_pass", llm_calls=1, retrievals=1),
            RecursionStage(name="final_synthesis", llm_calls=1, retrievals=0),
        ],
        terminated_reason="completed",
    )

    class RecursiveFake(FakeRagLlm):
        def answer(self, request):
            self._publish()
            return RagTranscriptAnswer(
                question=request.question, answer="synth", recursion=recursion
            )

    fake = RecursiveFake()
    fake.rewrite = QueryRewrite(query="negative gearing cap timing")
    runner = _runner(settings, rag_llm=fake)

    result = runner.run(
        "rag_llm_recursive", "and when does it start?", scope=AskScope(history=["earlier turn"])
    )

    assert result.llm_calls == 3


def test_trace_marks_a_degraded_rewrite_as_degraded(settings) -> None:
    from src.agents.models import QueryRewrite

    fake = FakeRagLlm()
    fake.context = TracingContext()
    fake.rewrite = QueryRewrite(query="what about it?", degraded=True)
    runner = _runner(settings, rag_llm=fake)

    result = runner.run("rag_llm", "what about it?")

    assert "rewrite failed" in result.trace[0]["detail"]


def test_recursive_trace_names_what_the_first_retrieval_embedded(settings) -> None:
    from src.agents.models import QueryRewrite, RecursionStage, RecursionTrace

    recursion = RecursionTrace(
        stages=[RecursionStage(name="first_pass", llm_calls=1, retrievals=1)],
        terminated_reason="no_followups_requested",
    )

    class RecursiveFake(FakeRagLlm):
        def answer(self, request):
            self._publish()
            return RagTranscriptAnswer(
                question=request.question, answer="first", recursion=recursion
            )

    fake = RecursiveFake()
    fake.rewrite = QueryRewrite(query="negative gearing cap timing")
    runner = _runner(settings, rag_llm=fake)

    result = runner.run("rag_llm_recursive", "and when does it start?")

    labels = [step["label"] for step in result.trace]
    assert labels[:2] == ["Rewrite query", "First retrieval"]
    assert "negative gearing cap timing" in result.trace[1]["detail"]
    assert "as asked" not in result.trace[1]["detail"]


def test_recursive_trace_without_history_says_the_question_was_used(settings) -> None:
    from src.agents.models import RecursionStage, RecursionTrace

    recursion = RecursionTrace(
        stages=[RecursionStage(name="first_pass", llm_calls=1, retrievals=1)],
        terminated_reason="no_followups_requested",
    )

    class RecursiveFake(FakeRagLlm):
        def answer(self, request):
            self._publish()
            return RagTranscriptAnswer(
                question=request.question, answer="first", recursion=recursion
            )

    runner = _runner(settings, rag_llm=RecursiveFake())
    result = runner.run("rag_llm_recursive", "q")

    assert result.trace[0]["label"] == "First retrieval"
    assert result.trace[0]["detail"] == "initial retrieval for the question as asked"


def test_recursive_trace_flattens_the_recursion_trace(settings) -> None:
    from src.agents.models import (
        FollowupSubtopic,
        RecursionStage,
        RecursionTrace,
        SubtopicEvidence,
    )

    recursion = RecursionTrace(
        stages=[
            RecursionStage(name="first_pass", llm_calls=1, retrievals=1),
            RecursionStage(name="fan_out", llm_calls=0, retrievals=1),
            RecursionStage(name="final_synthesis", llm_calls=1, retrievals=0),
        ],
        subtopic_evidence=[
            SubtopicEvidence(
                subtopic_index=1,
                subtopic=FollowupSubtopic(
                    topic="CGT discount", followup_query="how does the CGT discount change?"
                ),
                chunks=[_chunk("vid00000002", 4)],
                outcome="merged 1 new chunk",
            )
        ],
        terminated_reason="completed",
        total_followups_proposed=2,
        total_followups_executed=1,
    )

    class RecursiveFake(FakeRagLlm):
        def answer(self, request):
            self.requests.append(request)
            self._publish()
            return RagTranscriptAnswer(
                question=request.question, answer="synth", recursion=recursion
            )

    runner = _runner(settings, rag_llm=RecursiveFake())
    result = runner.run("rag_llm_recursive", "what about X?")

    labels = [step["label"] for step in result.trace]
    assert labels == [
        "First retrieval",
        "First-pass answer",
        "Follow-up: CGT discount",
        "Merge evidence",
        "Synthesize",
    ]
    followup = result.trace[2]
    assert followup["chunk_ids"] == ["chunk:vid00000002:4"]
    assert "merged 1 new chunk" in followup["detail"]


def test_recursive_trace_records_a_kept_first_answer(settings) -> None:
    from src.agents.models import RecursionStage, RecursionTrace

    recursion = RecursionTrace(
        stages=[RecursionStage(name="first_pass", llm_calls=1, retrievals=1)],
        terminated_reason="max_total_followups_reached",
        total_followups_proposed=3,
        total_followups_executed=0,
    )

    class RecursiveFake(FakeRagLlm):
        def answer(self, request):
            self._publish()
            return RagTranscriptAnswer(
                question=request.question, answer="first", recursion=recursion
            )

    runner = _runner(settings, rag_llm=RecursiveFake())
    result = runner.run("rag_llm_recursive", "q")

    assert result.trace[-1]["label"] == "First-pass answer kept"
    assert "max_total_followups_reached" in result.trace[-1]["detail"]
    assert all(step["label"] != "Synthesize" for step in result.trace)


def test_agent_trace_built_from_streamed_events(settings) -> None:
    from src.agents.models import AgentProgressEvent

    class StreamingFakeAgent(FakeRagAgent):
        def answer_streaming(self, request, on_event):
            on_event(
                AgentProgressEvent(
                    iteration=1,
                    event_type="retrieval_complete",
                    query="sub-question one",
                    chunk_count=5,
                )
            )
            on_event(AgentProgressEvent(iteration=2, event_type="answer_start"))
            return self.answer(request)

    runner = _runner(settings, rag_agent=StreamingFakeAgent())
    result = runner.run("rag_agent", "q", on_agent_event=lambda event: None)

    assert [step["phase"] for step in result.trace] == ["retrieve", "llm"]
    assert "sub-question one" in result.trace[0]["detail"]
    assert result.trace[0]["iteration"] == 1


def test_agent_trace_summarizes_when_events_were_not_streamed(settings) -> None:
    runner = _runner(settings, rag_agent=FakeRagAgent())

    result = runner.run("rag_agent", "q")

    assert len(result.trace) == 1
    assert "4 retrieval iterations" in result.trace[0]["detail"]


def test_failed_answer_keeps_the_retrieval_steps_already_recorded(settings) -> None:
    """Retrieval succeeded and the answer call blew up — that is the diagnostic."""

    class FailsAfterRetrieval(FakeRagLlm):
        def __init__(self) -> None:
            super().__init__()
            self.context = TracingContext()

        def answer(self, request):
            self._publish()
            raise RuntimeError("deepseek 503")

    runner = _runner(settings, rag_llm=FailsAfterRetrieval())
    result = runner.run("rag_llm", "what about X?")

    assert result.error == "deepseek 503"
    assert [step["label"] for step in result.trace] == ["Retrieve candidates"]
    assert result.trace[0]["chunk_ids"] == ["chunk:vid00000001:0", "chunk:vid00000001:1"]
    # The answer call never returned, so no answer step may appear.
    assert all(step["phase"] != "llm" for step in result.trace)


def test_failed_graph_answer_keeps_the_route_and_evidence_steps(settings) -> None:
    from src.agents.models import TraceStep

    class FailingGraphAgent:
        def __init__(self) -> None:
            self.last_context = None
            self.last_trace = []

        def answer(self, request):
            # A real agent records the route and its evidence before the
            # answer call, so the failure leaves those steps behind.
            self.last_trace = [
                TraceStep(phase="route", label="Route → global", detail="no entities"),
                TraceStep(phase="retrieve", label="Community summaries", detail="12 summaries"),
            ]
            raise RuntimeError("neo4j unavailable")

    runner = _runner(settings)
    runner._graph_rag_agent = FailingGraphAgent()
    result = runner.run("graph_rag", "q")

    assert result.error == "neo4j unavailable"
    assert [step["label"] for step in result.trace] == [
        "Route → global",
        "Community summaries",
    ]


def test_failed_agentic_answer_keeps_the_events_already_streamed(settings) -> None:
    from src.agents.models import AgentProgressEvent

    class FailsMidStream(FakeRagAgent):
        def answer_streaming(self, request, on_event):
            on_event(
                AgentProgressEvent(
                    iteration=1,
                    event_type="retrieval_complete",
                    query="sub-question one",
                    chunk_count=5,
                )
            )
            raise RuntimeError("tool loop crashed")

    runner = _runner(settings, rag_agent=FailsMidStream())
    result = runner.run("rag_agent", "q", on_agent_event=lambda event: None)

    assert result.error == "tool loop crashed"
    assert [step["label"] for step in result.trace] == ["Retrieve (iteration 1)"]


def test_failed_agentic_answer_without_events_reports_no_steps(settings) -> None:
    """A half-run iteration count would summarise a loop that never finished."""

    class FailsBeforeStreaming(FakeRagAgent):
        def answer(self, request):
            raise RuntimeError("boom")

    runner = _runner(settings, rag_agent=FailsBeforeStreaming())
    result = runner.run("rag_agent", "q")

    assert result.error == "boom"
    assert result.trace == []


def test_a_failure_before_the_answer_never_persists_the_previous_questions_trace(
    settings,
) -> None:
    """Agents are cached; a neighbouring question's steps are worse than none."""
    fake = FakeRagLlm()
    fake.context = TracingContext()
    runner = _runner(settings, rag_llm=fake)

    runner.run("rag_llm", "what about X?")
    assert fake.last_context is not None

    # A malformed url is rejected while the request is built, before answer()
    # runs — and therefore before the agent clears its own record.
    result = runner.run("rag_llm", "an unrelated question", url="not-a-url")

    assert result.error is not None
    assert result.trace == []


def test_a_failed_retrieval_keeps_the_stages_it_measured(settings) -> None:
    """The steps that explain the failure ride out on the error itself."""
    from src.agents.models import TraceStep
    from src.rag.context import RetrievalError

    class FailingRetrieval(FakeRagLlm):
        def answer(self, request):
            raise RetrievalError(
                "No transcript summaries matched the question.",
                [
                    TraceStep(
                        phase="filter",
                        label="Summary filter",
                        detail="0 videos matched by per-video summary",
                        elapsed_ms=31,
                    )
                ],
            )

    runner = _runner(settings, rag_llm=FailingRetrieval())
    result = runner.run("rag_llm", "q")

    assert [step["label"] for step in result.trace] == ["Summary filter"]
    assert result.trace[0]["elapsed_ms"] == 31


class RecordingRagLlm(FakeRagLlm):
    """A stand-in that keeps the per-retrieval record a real agent keeps.

    ``RagTranscriptAgent`` appends one ``RetrievalPass`` per retrieval, so a
    recursive answer's fan-out is on the agent even though ``last_context``
    holds only the newest one.
    """

    def __init__(self, passes) -> None:
        super().__init__()
        self.passes = passes
        self.last_retrievals: list = []

    def _publish(self) -> None:
        super()._publish()
        self.last_retrievals = list(self.passes)
        # ``_retrieve`` overwrites last_context per retrieval, so what it holds
        # after a fan-out is whichever pass ran last — never the whole answer.
        self.last_context.trace = list(self.passes[-1].steps) if self.passes else []


def _pass(label: str, *steps):
    from src.agents.models import RetrievalPass

    return RetrievalPass(label=label, steps=list(steps))


def _retrieve_step(detail: str, **fields):
    from src.agents.models import TraceStep

    return TraceStep(phase="retrieve", label="Retrieve candidates", detail=detail, **fields)


def test_a_failed_fan_out_attributes_every_retrieval_to_its_pass(settings) -> None:
    """A follow-up's stages must never read as the answer's first retrieval."""

    class FailsMidFanOut(RecordingRagLlm):
        def answer(self, request):
            self._publish()
            raise RuntimeError("deepseek 503")

    fake = FailsMidFanOut(
        [
            _pass("first retrieval", _retrieve_step("30 candidates", chunk_ids=["chunk:v1:0"])),
            _pass(
                'follow-up 1 "how does the CGT discount change?"', _retrieve_step("12 candidates")
            ),
            _pass('follow-up 2 "when does it start?"', _retrieve_step("0 candidates")),
        ]
    )
    runner = _runner(settings, rag_llm=fake)

    result = runner.run("rag_llm_recursive", "what about X?")

    assert result.error == "deepseek 503"
    assert [step["detail"] for step in result.trace] == [
        "first retrieval — 30 candidates",
        'follow-up 1 "how does the CGT discount change?" — 12 candidates',
        'follow-up 2 "when does it start?" — 0 candidates',
    ]
    assert result.trace[0]["chunk_ids"] == ["chunk:v1:0"]


def test_a_single_retrieval_is_reported_exactly_as_the_provider_measured_it(
    settings,
) -> None:
    """One pass has nothing to disambiguate, so nothing is added to its wording."""

    class FailsAfterRetrieval(RecordingRagLlm):
        def answer(self, request):
            self._publish()
            raise RuntimeError("deepseek 503")

    fake = FailsAfterRetrieval([_pass("retrieval", _retrieve_step("30 candidates"))])
    runner = _runner(settings, rag_llm=fake)

    result = runner.run("rag_llm", "q")

    assert [step["detail"] for step in result.trace] == ["30 candidates"]


def test_recursive_trace_prefers_the_first_retrievals_measured_stages(settings) -> None:
    """Measured chunk ids and timings beat a prose description of the same pass."""
    from src.agents.models import RecursionStage, RecursionTrace

    recursion = RecursionTrace(
        stages=[RecursionStage(name="first_pass", llm_calls=1, retrievals=1)],
        terminated_reason="no_followups_requested",
    )

    class RecursiveFake(RecordingRagLlm):
        def answer(self, request):
            self._publish()
            return RagTranscriptAnswer(
                question=request.question, answer="first", recursion=recursion
            )

    fake = RecursiveFake(
        [
            _pass(
                "first retrieval",
                _retrieve_step(
                    "semantic search over the whole corpus — 30 candidates",
                    chunk_ids=["chunk:vid00000001:0"],
                    elapsed_ms=770,
                ),
            )
        ]
    )
    runner = _runner(settings, rag_llm=fake)

    result = runner.run("rag_llm_recursive", "what about X?")

    assert result.trace[0]["label"] == "Retrieve candidates"
    assert result.trace[0]["chunk_ids"] == ["chunk:vid00000001:0"]
    assert result.trace[0]["elapsed_ms"] == 770
    assert all(step["label"] != "First retrieval" for step in result.trace)
    assert [step["label"] for step in result.trace[1:]] == [
        "First-pass answer",
        "First-pass answer kept",
    ]


def test_a_real_fan_out_failure_persists_every_retrieval_in_order(settings) -> None:
    """The same case end to end, across the seam the trace is assembled over.

    The follow-up here returns one chunk against a novelty floor of two, so it
    never reaches the merged context — its retrieval still ran, and the trace of
    the failed answer has to say so.
    """
    from datetime import datetime, timezone

    from langchain_core.messages import AIMessage

    from src.agents.context import TranscriptContext
    from src.agents.models import TraceStep
    from src.agents.rag_transcript_agent import RagTranscriptAgent
    from src.rag.context import RetrievalError
    from src.transcripts.models import Transcript

    subtopics = """
    {
      "question": "q",
      "answer": "first answer [1]",
      "followups_requested": true,
      "subtopics": [
        {"topic": "a", "rationale": "thin", "followup_query": "first detail", "confidence": 0.9},
        {"topic": "b", "rationale": "thin", "followup_query": "second detail", "confidence": 0.8}
      ]
    }
    """

    class Llm:
        def invoke(self, messages):
            return AIMessage(content=subtopics)

    class FanOutProvider:
        def __init__(self) -> None:
            self.calls = 0

        def get_context(self, question, **kwargs):
            self.calls += 1
            step = TraceStep(
                phase="retrieve",
                label="Retrieve candidates",
                detail=f"{self.calls} candidates",
            )
            if self.calls == 3:
                raise RetrievalError("No indexed chunks found for that channel.", [step])
            return TranscriptContext(
                transcript=Transcript(
                    video_id="all",
                    url="https://www.youtube.com/watch?v=vid00000001",
                    provider="rag",
                    raw_text="text",
                    fetched_at=datetime.now(timezone.utc),
                ),
                cache_status="hit",
                context_text=f"[1] chunk {self.calls}",
                context_mode="rag",
                retrieved_chunks=[_chunk("vid00000001", self.calls)],
                trace=[step],
            )

    runner = _runner(settings, rag_llm=RagTranscriptAgent(Llm(), FanOutProvider()))

    result = runner.run("rag_llm_recursive", "q")

    assert result.error == "No indexed chunks found for that channel."
    assert [step["detail"] for step in result.trace] == [
        "first retrieval — 1 candidates",
        'follow-up 1 "first detail" — 2 candidates',
        'follow-up 2 "second detail" — 3 candidates',
    ]


def test_an_unrelated_error_carrying_a_trace_attribute_is_not_spliced(settings) -> None:
    """Only RetrievalError promises TraceSteps; anything else must not be trusted."""

    class Exploded(RuntimeError):
        trace = "a stack trace, not a list of steps"

    class Raises(FakeRagLlm):
        def answer(self, request):
            raise Exploded("neo4j driver blew up")

    runner = _runner(settings, rag_llm=Raises())

    result = runner.run("rag_llm", "q")

    assert result.error == "neo4j driver blew up"
    assert result.trace == []


def test_graph_trace_passes_through_the_agents_recorded_steps(settings) -> None:
    from src.agents.models import TraceStep

    class FakeGraphAgent:
        def __init__(self) -> None:
            self.last_context = None
            self.last_llm_calls = 2
            self.last_route = "temporal"
            self.last_trace = []

        def answer(self, request):
            self.last_context = FakeContext(chunks=1)
            self.last_trace = [
                TraceStep(phase="route", label="Route → temporal", detail="entities: x"),
                TraceStep(phase="retrieve", label="Claim timeline", detail="12 dated claims"),
                TraceStep(phase="llm", label="Answer", detail="one narration call"),
            ]
            return RagTranscriptAnswer(question=request.question, answer="graph answer")

    runner = _runner(settings)
    runner._graph_rag_agent = FakeGraphAgent()
    result = runner.run("graph_rag", "q")

    assert [step["label"] for step in result.trace] == [
        "Route → temporal",
        "Claim timeline",
        "Answer",
    ]
    assert result.terminated_reason == "temporal"

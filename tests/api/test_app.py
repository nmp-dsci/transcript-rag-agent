from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from src.api import main as api_main
from src.api.main import create_app
from src.chat.setups import SETUP_KEYS, SetupResult, setup_spec
from src.config import Settings
from src.evals.matrix import DEFAULT_MATRIX_SETUPS
from src.rag.models import RetrievedChunk
from tests.api.matrix_fixtures import matrix_cell, matrix_run


class FakeChunkStore:
    """Stands in for TranscriptChunkStore: scoped vs. global chunk queries."""

    def __init__(self, by_video: dict[str, list[RetrievedChunk]]) -> None:
        self.by_video = by_video
        self.calls: list[tuple] = []

    def query_by_video_id(self, video_id: str, query: str, top_k: int) -> list[RetrievedChunk]:
        self.calls.append(("scoped", video_id, query, top_k))
        return self.by_video.get(video_id, [])[:top_k]

    def query_all(self, query: str, top_k: int) -> list[RetrievedChunk]:
        self.calls.append(("all", query, top_k))
        all_chunks = [chunk for chunks in self.by_video.values() for chunk in chunks]
        return all_chunks[:top_k]


class FakeProvider:
    """Stands in for MultiTranscriptRagContextProvider: only what /api/rank uses."""

    def __init__(self, chunk_store: FakeChunkStore) -> None:
        self.chunk_store = chunk_store

    def get_context(
        self, *, question: str, source_url: str | None, top_k: int, **kwargs
    ) -> SimpleNamespace:
        return SimpleNamespace(retrieved_chunks=self.chunk_store.query_all(question, top_k))


class FakeRunner:
    """Stands in for RagSetupRunner: canned answers, recorded calls."""

    def __init__(self, agent_steps: list | None = None) -> None:
        self.calls: list[tuple[str, str, str | None]] = []
        self.agent_steps = agent_steps or []
        self.provider = FakeProvider(FakeChunkStore({}))
        self.scopes: list = []

    def run(
        self,
        key: str,
        question: str,
        *,
        url: str | None = None,
        top_k: int | None = None,
        on_agent_event=None,
        scope=None,
    ) -> SetupResult:
        self.calls.append((key, question, url))
        self.top_ks: list[int | None] = getattr(self, "top_ks", [])
        self.top_ks.append(top_k)
        self.scopes.append(scope)
        if key == "rag_agent" and on_agent_event is not None:
            for event in self.agent_steps:
                on_agent_event(event)
        return SetupResult(
            key=key,
            title=setup_spec(key).title,
            command=f"fake {key}",
            answer=f"Answer from {key} [1]",
            references=[
                {
                    "label": "[1]",
                    "video_id": "abc123",
                    "timestamp_url": "https://youtu.be/abc123?t=5",
                }
            ],
            token_estimate=42,
            chunk_count=3,
            elapsed_seconds=0.1,
            contexts=[f"context one for {key}", f"context two for {key}"],
            model="deepseek-v4",
            embedding_model="all-MiniLM-L6-v2",
            top_k=top_k or 10,
        )


class FakeJudge:
    """Stands in for RagasJudge: deterministic scores, recorded calls."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, list[str]]] = []
        self.answer_models: list[str | None] = []

    def score(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        answer_model: str | None = None,
    ) -> dict:
        self.calls.append((question, answer, list(contexts)))
        self.answer_models.append(answer_model)
        value = 0.9 if "rag_agent" in answer else 0.5
        return {
            "judge": "ragas",
            "judge_model": "fake-judge",
            "rubric_version": "ragas-v1",
            "scores": {
                "faithfulness": value,
                "answer_relevancy": value,
                "context_precision": value,
            },
            "composite": value,
            "elapsed_seconds": 0.1,
            "scored_at": "2026-07-20T00:00:00+00:00",
            "error": None,
        }


FAKE_CORPUS = {
    "videos": [
        {
            "video_id": "abc123",
            "title": "Tax changes explained",
            "channel_name": "Finance Weekly",
            "source_url": "https://youtu.be/abc123",
            "duration_seconds": 812.0,
            "upload_date": "2026-06-01",
            "view_count": 1200,
            "summary": "A summary.",
            "fetched_at": "2026-06-10T00:00:00+00:00",
            "chunk_count": 42,
        }
    ],
    "totals": {"videos": 1, "chunks": 42},
}

FAKE_CHUNKS = {
    "abc123": [
        {
            "chunk_index": 0,
            "text": "capital gains tax discount rules explained",
            "start_seconds": 0.0,
            "end_seconds": 60.0,
            "segment_count": 4,
            "source_url": "https://youtu.be/abc123",
        },
        {
            "chunk_index": 1,
            "text": "negative gearing and property investors",
            "start_seconds": 60.0,
            "end_seconds": 120.0,
            "segment_count": 5,
            "source_url": "https://youtu.be/abc123",
        },
    ]
}


class Harness:
    def __init__(self, settings: Settings, tmp_path: Path, agent_steps: list | None = None) -> None:
        self.runner = FakeRunner(agent_steps)
        self.judge = FakeJudge()
        self.factory_calls = 0
        self.judge_factory_calls = 0
        self.index_argv: list[list[str]] = []
        self.history_path = tmp_path / "chat_history.json"
        self.chat_html_path = tmp_path / "chat.html"
        # The Scoreboard and Experiments tabs read committed eval runs, not the
        # live chat history — seed via ``seed_matrix_run``.
        self.runs_dir = tmp_path / "runs"

        def factory() -> FakeRunner:
            self.factory_calls += 1
            return self.runner

        def judge_factory() -> FakeJudge:
            self.judge_factory_calls += 1
            return self.judge

        def index_fn(argv: list[str]) -> int:
            self.index_argv.append(argv)
            return 0

        app = create_app(
            settings,
            runner_factory=factory,  # type: ignore[arg-type]
            judge_factory=judge_factory,  # type: ignore[arg-type]
            corpus_fn=lambda: FAKE_CORPUS,
            chunks_fn=lambda video_id: {
                "video_id": video_id,
                "chunks": FAKE_CHUNKS.get(video_id, []),
                "total": len(FAKE_CHUNKS.get(video_id, [])),
            },
            chunk_records_fn=lambda video_id: [
                {**chunk, "video_id": "abc123"}
                for chunk in FAKE_CHUNKS["abc123"]
                if video_id in (None, "abc123")
            ],
            history_path=self.history_path,
            chat_html_path=self.chat_html_path,
            index_fn=index_fn,
            frontend_dist=tmp_path / "no-bundle",
            runs_dir=self.runs_dir,
        )
        self.client = TestClient(app)

    def ask(self, question: str = "What is agentic RAG?", **kwargs) -> str:
        """Ask and return the saved entry id."""
        response = self.client.post("/api/ask", json={"question": question, **kwargs})
        events = sse_events(response.text)
        assert events[-1][0] == "done", events
        return events[-1][1]["id"]

    def seed_matrix_run(self, data: dict) -> None:
        """Commit a matrix run for the Scoreboard/Experiments tabs to read."""
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        (self.runs_dir / f"{data['run_id']}.json").write_text(json.dumps(data), encoding="utf-8")


@pytest.fixture
def harness(settings: Settings, tmp_path: Path) -> Harness:
    return Harness(settings, tmp_path)


def sse_events(text: str) -> list[tuple[str, dict]]:
    events = []
    for block in text.strip().split("\n\n"):
        event, data = "message", ""
        for line in block.split("\n"):
            if line.startswith("event: "):
                event = line[len("event: ") :]
            elif line.startswith("data: "):
                data += line[len("data: ") :]
        events.append((event, json.loads(data)))
    return events


def test_health(harness: Harness) -> None:
    response = harness.client.get("/api/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["runner_loaded"] is False
    assert payload["judge_loaded"] is False
    assert payload["judge_model"] == "deepseek-v4"
    assert payload["answer_model"] == "deepseek-v4"
    assert payload["ui"] == "legacy"  # no built bundle in the test harness


def test_setups_lists_all(harness: Harness) -> None:
    payload = harness.client.get("/api/setups").json()
    assert [spec["key"] for spec in payload["setups"]] == SETUP_KEYS
    assert all(spec["title"] and spec["description"] for spec in payload["setups"])


def test_history_starts_empty(harness: Harness) -> None:
    assert harness.client.get("/api/history").json() == {"conversations": []}


def test_ask_streams_answers_and_persists(harness: Harness) -> None:
    response = harness.client.post(
        "/api/ask",
        json={"question": "What is agentic RAG?", "setups": ["rag_llm", "rag_agent"]},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    events = sse_events(response.text)
    kinds = [event for event, _ in events]
    # Loading notice, then progress + answer per setup, then the saved entry.
    assert kinds == ["progress", "progress", "answer", "progress", "answer", "done"]
    answers = [data for event, data in events if event == "answer"]
    assert [a["key"] for a in answers] == ["rag_llm", "rag_agent"]
    assert answers[0]["answer"] == "Answer from rag_llm [1]"

    done = events[-1][1]
    assert done["question"] == "What is agentic RAG?"
    assert len(done["answers"]) == 2

    assert harness.runner.calls == [
        ("rag_llm", "What is agentic RAG?", None),
        ("rag_agent", "What is agentic RAG?", None),
    ]
    saved = json.loads(harness.history_path.read_text(encoding="utf-8"))
    assert len(saved["conversations"]) == 1
    assert "What is agentic RAG?" in harness.chat_html_path.read_text(encoding="utf-8")

    history = harness.client.get("/api/history").json()
    assert len(history["conversations"]) == 1


def test_ask_defaults_to_all_setups(harness: Harness) -> None:
    response = harness.client.post("/api/ask", json={"question": "Hello?"})
    answers = [d for e, d in sse_events(response.text) if e == "answer"]
    assert [a["key"] for a in answers] == SETUP_KEYS


def test_ask_passes_url_filter(harness: Harness) -> None:
    harness.client.post(
        "/api/ask",
        json={
            "question": "Hello?",
            "setups": ["rag_llm"],
            "url": " https://youtu.be/abc123 ",
        },
    )
    assert harness.runner.calls == [("rag_llm", "Hello?", "https://youtu.be/abc123")]


def test_ask_rejects_unknown_setup(harness: Harness) -> None:
    response = harness.client.post("/api/ask", json={"question": "Hello?", "setups": ["nope"]})
    assert response.status_code == 422
    assert "nope" in response.json()["detail"]


@pytest.mark.parametrize("question", ["", "   "])
def test_ask_rejects_blank_question(harness: Harness, question: str) -> None:
    response = harness.client.post("/api/ask", json={"question": question})
    assert response.status_code == 422


def test_runner_built_once_across_questions(harness: Harness) -> None:
    first = harness.client.post("/api/ask", json={"question": "One?", "setups": ["rag_llm"]})
    second = harness.client.post("/api/ask", json={"question": "Two?", "setups": ["rag_llm"]})
    assert harness.factory_calls == 1
    loading = [
        d["message"]
        for e, d in sse_events(second.text)
        if e == "progress" and "Loading" in d.get("message", "")
    ]
    assert loading == []  # only the first stream announces the stack load
    assert "Loading" in sse_events(first.text)[0][1]["message"]
    assert harness.client.get("/api/health").json()["runner_loaded"] is True


def test_stream_reports_stack_failure(settings: Settings, tmp_path: Path) -> None:
    def broken_factory():  # noqa: ANN202 - test double
        raise RuntimeError("stack exploded")

    app = create_app(
        settings,
        runner_factory=broken_factory,
        history_path=tmp_path / "h.json",
        chat_html_path=tmp_path / "c.html",
        index_fn=lambda argv: 0,
    )
    response = TestClient(app).post("/api/ask", json={"question": "Hello?"})
    assert response.status_code == 200
    events = sse_events(response.text)
    assert events[-1][0] == "error"
    assert "stack exploded" in events[-1][1]["message"]


def test_index_video_invokes_cli_path(harness: Harness) -> None:
    response = harness.client.post(
        "/api/index", json={"mode": "video", "url": "https://youtu.be/abc123"}
    )
    assert response.status_code == 200
    assert response.json()["ok"] is True
    assert harness.index_argv == [["index-rag", "https://youtu.be/abc123"]]


def test_index_channel_invokes_bulk_path(harness: Harness) -> None:
    harness.client.post("/api/index", json={"mode": "channel", "channel": "@some", "latest": 3})
    assert harness.index_argv == [["bulk-index", "channel", "--channel", "@some", "--latest", "3"]]


def test_index_requires_target(harness: Harness) -> None:
    assert harness.client.post("/api/index", json={"mode": "video"}).status_code == 422
    assert harness.client.post("/api/index", json={"mode": "channel"}).status_code == 422


def test_index_stream_reports_discover_fetch_processing_then_done(
    harness: Harness,
) -> None:
    response = harness.client.post(
        "/api/index/stream", json={"mode": "video", "url": "https://youtu.be/abc123"}
    )
    assert response.status_code == 200
    events = sse_events(response.text)
    stages = [data["stage"] for event, data in events if event == "stage"]
    assert stages == ["discover", "fetch", "processing"]
    assert events[-1][0] == "done"
    done = events[-1][1]
    assert done["ok"] is True
    assert done["target"] == "https://youtu.be/abc123"
    assert "added_videos" in done
    assert "added_video_count" in done
    assert "added_chunk_count" in done
    assert "totals" in done
    assert "insights" in done
    assert "channels" in done


def test_index_stream_reports_added_videos_and_totals(settings: Settings, tmp_path: Path) -> None:
    before_corpus = {"videos": [], "totals": {"videos": 0, "chunks": 0}}
    after_corpus = {
        "videos": [{"video_id": "abc123"}],
        "totals": {"videos": 1, "chunks": 5},
        "insights": ["insight"],
        "channels": ["chan"],
    }
    calls = {"n": 0}

    def corpus_fn() -> dict:
        calls["n"] += 1
        return before_corpus if calls["n"] == 1 else after_corpus

    app = create_app(
        settings,
        runner_factory=lambda: None,
        history_path=tmp_path / "h.json",
        chat_html_path=tmp_path / "c.html",
        index_fn=lambda argv: 0,
        corpus_fn=corpus_fn,
    )
    client = TestClient(app)
    response = client.post(
        "/api/index/stream", json={"mode": "video", "url": "https://youtu.be/abc123"}
    )
    events = sse_events(response.text)
    done = events[-1]
    assert done[0] == "done"
    assert done[1]["added_video_count"] == 1
    assert done[1]["added_chunk_count"] == 5
    assert done[1]["insights"] == ["insight"]
    assert done[1]["channels"] == ["chan"]


def test_index_stream_emits_heartbeats_while_indexing_is_slow(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(api_main, "_INDEX_HEARTBEAT_INTERVAL_SECONDS", 0.05)
    release = threading.Event()

    def slow_index_fn(argv: list[str]) -> int:
        release.wait(timeout=2)
        return 0

    app = create_app(
        settings,
        runner_factory=lambda: None,
        history_path=tmp_path / "h.json",
        chat_html_path=tmp_path / "c.html",
        index_fn=slow_index_fn,
    )
    client = TestClient(app)
    timer = threading.Timer(0.2, release.set)
    timer.start()
    try:
        response = client.post(
            "/api/index/stream",
            json={"mode": "video", "url": "https://youtu.be/abc123"},
        )
    finally:
        timer.cancel()
    events = sse_events(response.text)
    stage_events = [data for event, data in events if event == "stage"]
    heartbeats = [data for data in stage_events if data["message"] == "Still indexing..."]
    assert len(heartbeats) >= 1
    assert all(data["stage"] == "processing" for data in heartbeats)
    assert events[-1][0] == "done"


def test_index_stream_reports_nonzero_exit_code_as_error(
    settings: Settings, tmp_path: Path
) -> None:
    app = create_app(
        settings,
        runner_factory=lambda: None,
        history_path=tmp_path / "h.json",
        chat_html_path=tmp_path / "c.html",
        index_fn=lambda argv: 1,
    )
    client = TestClient(app)
    response = client.post(
        "/api/index/stream", json={"mode": "video", "url": "https://youtu.be/abc123"}
    )
    events = sse_events(response.text)
    assert events[-1][0] == "error"
    assert "exit 1" in events[-1][1]["message"]


def test_index_stream_reports_exception_from_index_fn_as_error(
    settings: Settings, tmp_path: Path
) -> None:
    def boom(argv: list[str]) -> int:
        raise RuntimeError("index blew up")

    app = create_app(
        settings,
        runner_factory=lambda: None,
        history_path=tmp_path / "h.json",
        chat_html_path=tmp_path / "c.html",
        index_fn=boom,
    )
    client = TestClient(app)
    response = client.post(
        "/api/index/stream", json={"mode": "video", "url": "https://youtu.be/abc123"}
    )
    events = sse_events(response.text)
    assert events[-1][0] == "error"
    assert "index blew up" in events[-1][1]["message"]


def test_ask_persists_contexts(harness: Harness) -> None:
    harness.ask(setups=["rag_llm"])
    saved = json.loads(harness.history_path.read_text(encoding="utf-8"))
    answer = saved["conversations"][0]["answers"][0]
    assert answer["contexts"] == [
        "context one for rag_llm",
        "context two for rag_llm",
    ]
    assert answer["evaluation"] is None


def test_ask_passes_top_k(harness: Harness) -> None:
    harness.ask(setups=["rag_llm"], top_k=25)
    assert harness.runner.top_ks == [25]


def test_judge_streams_scores_and_persists(harness: Harness) -> None:
    entry_id = harness.ask(setups=["rag_llm", "rag_agent"])
    response = harness.client.post("/api/judge", json={"entry_id": entry_id})
    assert response.status_code == 200

    events = sse_events(response.text)
    scored = [data for event, data in events if event == "scored"]
    assert [s["key"] for s in scored] == ["rag_llm", "rag_agent"]
    assert scored[0]["evaluation"]["composite"] == 0.5
    assert scored[1]["evaluation"]["composite"] == 0.9
    assert events[-1][0] == "done"

    # Judge received the stored contexts for each answer.
    assert harness.judge.calls[0][2] == [
        "context one for rag_llm",
        "context two for rag_llm",
    ]
    saved = json.loads(harness.history_path.read_text(encoding="utf-8"))
    evaluations = [a["evaluation"] for a in saved["conversations"][0]["answers"]]
    assert all(e and e["judge"] == "ragas" for e in evaluations)


def test_ask_appends_setups_to_an_existing_entry(harness: Harness) -> None:
    entry_id = harness.ask(setups=["rag_llm"])
    response = harness.client.post(
        "/api/ask",
        json={
            "question": "What is agentic RAG?",
            "setups": ["rag_agent"],
            "entry_id": entry_id,
        },
    )
    done = sse_events(response.text)[-1][1]
    assert done["id"] == entry_id
    assert [a["key"] for a in done["answers"]] == ["rag_llm", "rag_agent"]

    saved = json.loads(harness.history_path.read_text(encoding="utf-8"))
    assert len(saved["conversations"]) == 1  # appended, not duplicated


def test_ask_replaces_a_rerun_setup_in_place(harness: Harness) -> None:
    entry_id = harness.ask(setups=["rag_llm"])
    response = harness.client.post(
        "/api/ask",
        json={
            "question": "What is agentic RAG?",
            "setups": ["rag_llm"],
            "entry_id": entry_id,
        },
    )
    assert response.status_code == 200
    saved = json.loads(harness.history_path.read_text(encoding="utf-8"))
    answers = saved["conversations"][0]["answers"]
    assert [a["key"] for a in answers] == ["rag_llm"]


def test_ask_rejects_mismatched_question_for_entry_id(harness: Harness) -> None:
    entry_id = harness.ask(setups=["rag_llm"])
    response = harness.client.post(
        "/api/ask",
        json={
            "question": "A completely different question?",
            "setups": ["rag_agent"],
            "entry_id": entry_id,
        },
    )
    assert response.status_code == 422

    saved = json.loads(harness.history_path.read_text(encoding="utf-8"))
    answers = saved["conversations"][0]["answers"]
    assert [a["key"] for a in answers] == ["rag_llm"]


def test_ask_unknown_entry_404(harness: Harness) -> None:
    response = harness.client.post(
        "/api/ask", json={"question": "q", "setups": ["rag_llm"], "entry_id": "nope"}
    )
    assert response.status_code == 404


def test_judge_unknown_entry_404(harness: Harness) -> None:
    response = harness.client.post("/api/judge", json={"entry_id": "nope"})
    assert response.status_code == 404


def test_judge_skips_already_judged_unless_forced(harness: Harness) -> None:
    entry_id = harness.ask(setups=["rag_llm"])
    harness.client.post("/api/judge", json={"entry_id": entry_id})
    second = harness.client.post("/api/judge", json={"entry_id": entry_id})
    assert len(harness.judge.calls) == 1  # second run had nothing to score
    assert sse_events(second.text)[-1][0] == "done"

    harness.client.post("/api/judge", json={"entry_id": entry_id, "force": True})
    assert len(harness.judge.calls) == 2


def test_judge_marks_answers_without_contexts(settings: Settings, tmp_path: Path) -> None:
    harness = Harness(settings, tmp_path)
    entry_id = harness.ask(setups=["rag_llm"])
    # Simulate a pre-persistence record: strip the stored contexts.
    saved = json.loads(harness.history_path.read_text(encoding="utf-8"))
    saved["conversations"][0]["answers"][0]["contexts"] = []
    harness.history_path.write_text(json.dumps(saved), encoding="utf-8")

    response = harness.client.post("/api/judge", json={"entry_id": entry_id})
    scored = [d for e, d in sse_events(response.text) if e == "scored"]
    assert harness.judge.calls == []  # never called without contexts
    assert "no stored retrieval contexts" in scored[0]["evaluation"]["error"]


def test_scoreboard_aggregates_a_committed_matrix_run(harness: Harness) -> None:
    harness.seed_matrix_run(matrix_run())

    board = harness.client.get("/api/scoreboard").json()
    assert board["entries_total"] == 2
    assert board["entries_judged"] == 2
    # The run's own judge, not the server's configured one.
    assert board["judge_model"] == "deepseek-v4"
    assert board["run_id"] == "matrix-20260727-010101"

    rows = {row["key"]: row for row in board["setups"]}
    assert rows["rag_agent"]["avg_composite"] == 0.9
    assert rows["rag_llm"]["avg_composite"] == 0.5
    assert rows["rag_agent"]["win_rate"] == 1.0
    assert rows["rag_llm"]["win_rate"] == 0.0
    assert rows["rag_agent"]["judged"] == 2
    # Sorted best-first.
    assert board["setups"][0]["key"] == "rag_agent"


def test_scoreboard_lists_the_judged_questions_for_the_selected_run(harness: Harness) -> None:
    harness.seed_matrix_run(matrix_run())

    board = harness.client.get("/api/scoreboard").json()
    assert [q["id"] for q in board["questions"]] == ["g001", "g002"]
    setups = {s["key"]: s for s in board["questions"][0]["setups"]}
    assert setups["rag_agent"]["composite"] == 0.9
    assert setups["rag_llm"]["composite"] == 0.5


def test_scoreboard_survives_a_run_whose_config_is_null(harness: Harness) -> None:
    """A snapshot without a config must not take the whole tab down.

    ``config`` is only how the run describes the judge it used; the rows come
    from the cells. Reading it must fall back the way the rest of the module
    tolerates malformed snapshots, not 500 the request.
    """
    run = matrix_run()
    run["config"] = None
    harness.seed_matrix_run(run)

    response = harness.client.get("/api/scoreboard")
    assert response.status_code == 200
    board = response.json()
    # No recorded judge, so the server's configured one describes the numbers.
    assert board["judge_model"] == harness.client.get("/api/health").json()["judge_model"]
    assert {row["key"] for row in board["setups"]} == {"rag_llm", "rag_agent"}


def test_scoreboard_is_empty_without_any_committed_run(harness: Harness) -> None:
    # Asking and judging live must NOT feed the leaderboard any more — chat
    # history is the live set, the matrix run is the eval set.
    entry_id = harness.ask(setups=["rag_llm"])
    harness.client.post("/api/judge", json={"entry_id": entry_id})

    board = harness.client.get("/api/scoreboard").json()
    assert board["setups"] == []
    assert board["entries_total"] == 0
    assert board["run_id"] is None
    assert board["runs"] == []
    assert board["questions"] == []


def test_scoreboard_lists_runs_and_selects_one_by_id(harness: Harness) -> None:
    harness.seed_matrix_run(matrix_run("matrix-older", created_at="2026-07-01T00:00:00+00:00"))
    harness.seed_matrix_run(
        matrix_run(
            "matrix-newer",
            created_at="2026-07-27T00:00:00+00:00",
            setups={"rag_llm": [matrix_cell("g001", composite=0.42)]},
        )
    )

    default = harness.client.get("/api/scoreboard").json()
    assert [run["run_id"] for run in default["runs"]] == ["matrix-newer", "matrix-older"]
    assert default["run_id"] == "matrix-newer"  # newest by default
    assert default["setups"][0]["avg_composite"] == 0.42

    picked = harness.client.get("/api/scoreboard", params={"run_id": "matrix-older"}).json()
    assert picked["run_id"] == "matrix-older"
    assert {row["key"] for row in picked["setups"]} == {"rag_llm", "rag_agent"}


def test_scoreboard_reads_the_runs_directory_once_per_request(
    harness: Harness, monkeypatch
) -> None:
    """The picker's list and the selected run come from one pass.

    A committed matrix run is a large document, and the Scoreboard reloads on
    every group-by, judge-filter and run-picker change — parsing the whole
    directory twice per request makes ordinary interaction cost double.
    """
    from src.api import matrix_runs

    passes = 0
    original = matrix_runs._iter_matrix_files

    def counting(runs_dir=None):
        nonlocal passes
        passes += 1
        return original(runs_dir)

    monkeypatch.setattr(matrix_runs, "_iter_matrix_files", counting)
    harness.seed_matrix_run(matrix_run())

    board = harness.client.get("/api/scoreboard").json()

    assert passes == 1
    assert board["run_id"] == "matrix-20260727-010101"
    assert [run["run_id"] for run in board["runs"]] == ["matrix-20260727-010101"]


def test_scoreboard_unknown_run_id_renders_empty_rather_than_erroring(
    harness: Harness,
) -> None:
    harness.seed_matrix_run(matrix_run())
    board = harness.client.get("/api/scoreboard", params={"run_id": "nope"}).json()
    assert board["run_id"] is None
    assert board["setups"] == []
    # The picker still lists what *is* available, so the user can recover.
    assert [run["run_id"] for run in board["runs"]] == ["matrix-20260727-010101"]


def test_scoreboard_groups_by_model(harness: Harness) -> None:
    harness.seed_matrix_run(matrix_run())

    board = harness.client.get("/api/scoreboard", params={"group_by": "setup_model"}).json()
    assert board["group_by"] == "setup_model"
    row = board["setups"][0]
    assert row["model"] == "deepseek-v4"
    assert row["legacy"] is False


def test_scoreboard_reports_provenance(harness: Harness) -> None:
    harness.seed_matrix_run(matrix_run())

    provenance = harness.client.get("/api/scoreboard").json()["provenance"]
    assert provenance["judge_models"] == ["deepseek-v4"]
    assert provenance["embedding_models"] == ["fake-embeddings"]
    assert provenance["last_judged"] == "2026-07-27T01:01:01+00:00"
    assert "faithfulness" in provenance["metrics"]


def test_scoreboard_filters_by_judge(harness: Harness) -> None:
    harness.seed_matrix_run(matrix_run())

    board = harness.client.get("/api/scoreboard", params={"judge_model": "someone-else"}).json()
    assert board["entries_judged"] == 0


def test_scoreboard_judge_filter_keeps_answers_count(harness: Harness) -> None:
    # An answer judged by a *different* judge than the filter must still count
    # toward "answers" (it exists), even though it's excluded from "judged"/
    # win-rate accounting because that judge's scale isn't comparable.
    harness.seed_matrix_run(matrix_run(setups={"rag_llm": [matrix_cell("g001")]}))

    unfiltered = harness.client.get("/api/scoreboard").json()
    filtered = harness.client.get("/api/scoreboard", params={"judge_model": "someone-else"}).json()

    unfiltered_row = next(r for r in unfiltered["setups"] if r["key"] == "rag_llm")
    filtered_row = next(r for r in filtered["setups"] if r["key"] == "rag_llm")
    assert unfiltered_row["answers"] == filtered_row["answers"] == 1
    assert unfiltered_row["judged"] == 1
    assert filtered_row["judged"] == 0


def test_eval_matrix_endpoint_starts_a_run_and_reports_it(
    tmp_path: Path, settings: Settings
) -> None:
    calls: list[list[str]] = []

    def matrix_run_fn(setups, on_cell):
        calls.append(setups)
        on_cell({"setup": setups[0], "entry_id": "g001", "cached": True, "done": 1, "total": 1})
        return {"run_id": "matrix-from-ui", "cache_hits": 1, "cache_misses": 0}

    app = create_app(
        settings,
        runner_factory=lambda: FakeRunner(),  # type: ignore[arg-type]
        judge_factory=lambda: FakeJudge(),  # type: ignore[arg-type]
        corpus_fn=lambda: FAKE_CORPUS,
        history_path=tmp_path / "history.json",
        chat_html_path=tmp_path / "chat.html",
        index_fn=lambda argv: 0,
        frontend_dist=tmp_path / "no-bundle",
        runs_dir=tmp_path / "runs",
        matrix_run_fn=matrix_run_fn,
    )
    client = TestClient(app)

    started = client.post("/api/eval/matrix", json={"setups": ["rag_llm"]}).json()
    assert started["status"] == "running"

    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        job = client.get("/api/eval/matrix").json()["job"]
        if job["status"] == "done":
            break
        time.sleep(0.01)
    assert job["status"] == "done"
    assert job["run_id"] == "matrix-from-ui"
    assert job["cells_done"] == 1
    assert calls == [["rag_llm"]]


def test_eval_matrix_defaults_to_every_setup_in_the_matrix(
    tmp_path: Path, settings: Settings
) -> None:
    calls: list[list[str]] = []

    app = create_app(
        settings,
        runner_factory=lambda: FakeRunner(),  # type: ignore[arg-type]
        judge_factory=lambda: FakeJudge(),  # type: ignore[arg-type]
        corpus_fn=lambda: FAKE_CORPUS,
        history_path=tmp_path / "history.json",
        chat_html_path=tmp_path / "chat.html",
        index_fn=lambda argv: 0,
        frontend_dist=tmp_path / "no-bundle",
        runs_dir=tmp_path / "runs",
        matrix_run_fn=lambda setups, on_cell: calls.append(setups) or {"run_id": "matrix-all"},
    )
    client = TestClient(app)
    client.post("/api/eval/matrix", json={})

    deadline = time.monotonic() + 2
    while time.monotonic() < deadline and not calls:
        time.sleep(0.01)
    assert calls == [DEFAULT_MATRIX_SETUPS]


def test_eval_matrix_rejects_an_unknown_setup_before_starting_a_run(
    tmp_path: Path, settings: Settings
) -> None:
    """A typo'd engine would fail every cell, and the run function still commits
    the all-errors matrix-*.json the Scoreboard's picker then offers as a run."""
    calls: list[list[str]] = []

    app = create_app(
        settings,
        runner_factory=lambda: FakeRunner(),  # type: ignore[arg-type]
        judge_factory=lambda: FakeJudge(),  # type: ignore[arg-type]
        corpus_fn=lambda: FAKE_CORPUS,
        history_path=tmp_path / "history.json",
        chat_html_path=tmp_path / "chat.html",
        index_fn=lambda argv: 0,
        frontend_dist=tmp_path / "no-bundle",
        runs_dir=tmp_path / "runs",
        matrix_run_fn=lambda setups, on_cell: (
            calls.append(setups) or {"run_id": "matrix-should-not-exist"}
        ),
    )
    client = TestClient(app)

    response = client.post("/api/eval/matrix", json={"setups": ["rag_lmm"]})

    assert response.status_code == 422
    assert "rag_lmm" in response.json()["detail"]
    time.sleep(0.05)
    assert calls == []
    assert client.get("/api/eval/matrix").json() == {"job": None}


def test_eval_matrix_snapshot_is_null_before_any_run(harness: Harness) -> None:
    assert harness.client.get("/api/eval/matrix").json() == {"job": None}


def test_corpus_endpoint(harness: Harness) -> None:
    payload = harness.client.get("/api/corpus").json()
    assert payload["totals"] == {"videos": 1, "chunks": 42}
    assert payload["videos"][0]["title"] == "Tax changes explained"


def test_corpus_chunks_endpoint(harness: Harness) -> None:
    payload = harness.client.get("/api/corpus/abc123/chunks").json()
    assert payload["video_id"] == "abc123"
    assert payload["total"] == 2
    assert payload["chunks"][0]["chunk_index"] == 0
    assert "capital gains" in payload["chunks"][0]["text"]


def test_corpus_chunks_unknown_video_is_empty(harness: Harness) -> None:
    payload = harness.client.get("/api/corpus/nope/chunks").json()
    assert payload == {"video_id": "nope", "chunks": [], "total": 0}


def test_ask_streams_agent_steps(settings: Settings, tmp_path: Path) -> None:
    from src.agents.models import AgentProgressEvent

    steps = [
        AgentProgressEvent(iteration=1, event_type="retrieval_start", query="capital gains"),
        AgentProgressEvent(
            iteration=1,
            event_type="retrieval_complete",
            query="capital gains",
            chunk_count=7,
        ),
    ]
    harness = Harness(settings, tmp_path, agent_steps=steps)
    response = harness.client.post("/api/ask", json={"question": "Why?", "setups": ["rag_agent"]})
    emitted = [d for e, d in sse_events(response.text) if e == "agent_step"]
    assert [s["event_type"] for s in emitted] == [
        "retrieval_start",
        "retrieval_complete",
    ]
    assert emitted[0]["key"] == "rag_agent"
    assert emitted[1]["chunk_count"] == 7


def test_ask_emits_no_agent_steps_for_pipeline_setups(settings: Settings, tmp_path: Path) -> None:
    from src.agents.models import AgentProgressEvent

    harness = Harness(
        settings,
        tmp_path,
        agent_steps=[AgentProgressEvent(iteration=1, event_type="answer_start")],
    )
    response = harness.client.post("/api/ask", json={"question": "Why?", "setups": ["rag_llm"]})
    assert [e for e, _ in sse_events(response.text) if e == "agent_step"] == []


def test_rank_returns_aligned_modes(harness: Harness) -> None:
    payload = harness.client.post(
        "/api/rank",
        json={"query": "capital gains tax", "modes": ["bm25"], "top_k": 5},
    ).json()
    bm25_rows = payload["modes"]["bm25"]
    assert bm25_rows, "keyword search should match the seeded chunk"
    assert bm25_rows[0]["chunk_id"] == "abc123:0"
    assert bm25_rows[0]["rank"] == 1
    # Score can legitimately be 0 on a corpus this small (see test_bm25.py);
    # membership, not score, decides what counts as a keyword hit.
    assert bm25_rows[0]["score"] >= 0
    # A single mode has nothing to align against.
    assert bm25_rows[0]["other_rank"] is None


def _chunk(video_id: str, chunk_index: int, text: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        transcript_id=f"{video_id}-t",
        video_id=video_id,
        source_url=f"https://youtu.be/{video_id}",
        chunk_index=chunk_index,
        text=text,
        score=score,
    )


def test_rank_semantic_scopes_to_video_id_beyond_global_top_k(
    harness: Harness,
) -> None:
    # "other" dominates the unscoped top-k ranking; "abc123" has only one,
    # lower-scoring chunk that a global query_all(top_k=2) would truncate away
    # before any post-hoc video_id filter ever saw it.
    by_video = {
        "other": [
            _chunk("other", 0, "unrelated chunk one", 0.9),
            _chunk("other", 1, "unrelated chunk two", 0.8),
        ],
        "abc123": [_chunk("abc123", 0, "capital gains tax explained", 0.1)],
    }
    harness.runner.provider = FakeProvider(FakeChunkStore(by_video))

    payload = harness.client.post(
        "/api/rank",
        json={
            "query": "capital gains tax",
            "modes": ["semantic"],
            "top_k": 2,
            "video_id": "abc123",
        },
    ).json()

    semantic_rows = payload["modes"]["semantic"]
    assert [row["video_id"] for row in semantic_rows] == ["abc123"]
    assert harness.runner.provider.chunk_store.calls == [
        ("scoped", "abc123", "capital gains tax", 2)
    ]


def test_rank_reports_an_unreachable_graph_without_losing_the_other_modes(
    settings: Settings, tmp_path: Path
) -> None:
    class BrokenGraphStore:
        def resolve_entities(self, terms, limit=6):
            raise ConnectionError("neo4j unreachable")

    app = create_app(
        settings,
        runner_factory=lambda: FakeRunner(),  # type: ignore[arg-type]
        history_path=tmp_path / "h.json",
        chat_html_path=tmp_path / "c.html",
        index_fn=lambda argv: 0,
        frontend_dist=tmp_path / "no-bundle",
        chunk_records_fn=lambda video_id: [
            {**chunk, "video_id": "abc123"} for chunk in FAKE_CHUNKS["abc123"]
        ],
        graph_store_factory=lambda: BrokenGraphStore(),
    )

    payload = (
        TestClient(app)
        .post(
            "/api/rank",
            json={"query": "capital gains tax", "modes": ["bm25", "graph"], "top_k": 5},
        )
        .json()
    )

    # The query still succeeds and BM25 still ranks — but the graph column says
    # why it is empty instead of passing for "nothing matched".
    assert payload["modes"]["bm25"]
    assert payload["modes"]["graph"] == []
    assert "neo4j unreachable" in payload["errors"]["graph"]


def test_rank_reads_the_chunk_corpus_once_for_bm25_and_graph(
    settings: Settings, tmp_path: Path
) -> None:
    """Both keyword and graph ranking need every chunk; one read serves both.

    Each corpus read opens the Chroma collection and pulls every document and
    its metadata, so reading per mode would double that work for every "All 3"
    query.
    """
    calls: list[str | None] = []

    def records(video_id: str | None) -> list[dict]:
        calls.append(video_id)
        return [{**chunk, "video_id": "abc123"} for chunk in FAKE_CHUNKS["abc123"]]

    class EmptyGraphStore:
        def resolve_entities(self, terms, limit=6):
            return []

    app = create_app(
        settings,
        runner_factory=lambda: FakeRunner(),  # type: ignore[arg-type]
        history_path=tmp_path / "h.json",
        chat_html_path=tmp_path / "c.html",
        index_fn=lambda argv: 0,
        frontend_dist=tmp_path / "no-bundle",
        chunk_records_fn=records,
        graph_store_factory=lambda: EmptyGraphStore(),
    )

    response = TestClient(app).post(
        "/api/rank",
        json={"query": "capital gains tax", "modes": ["bm25", "graph"], "top_k": 5},
    )

    assert response.status_code == 200
    assert response.json()["modes"]["bm25"]
    assert calls == [None]


def test_rank_rejects_blank_query(harness: Harness) -> None:
    assert harness.client.post("/api/rank", json={"query": "  "}).status_code == 422


def test_chunk_graph_404_when_no_records(harness: Harness) -> None:
    response = harness.client.post("/api/chunk-graph", json={})
    assert response.status_code == 404
    assert "Index a video first" in response.json()["detail"]


def test_chunk_graph_returns_graph_for_records(settings: Settings, tmp_path: Path) -> None:
    records = [
        {
            "chunk_id": "chunk:abc123:0",
            "video_id": "abc123",
            "chunk_index": 0,
            "channel_id": "UC1",
            "channel_name": "Channel One",
            "title": "Title",
            "text": "some text",
            "start_seconds": 0.0,
            "end_seconds": 30.0,
            "source_url": "https://youtu.be/abc123",
            "embedding": [1.0, 0.0],
        },
        {
            "chunk_id": "chunk:abc123:1",
            "video_id": "abc123",
            "chunk_index": 1,
            "channel_id": "UC1",
            "channel_name": "Channel One",
            "title": "Title",
            "text": "more text",
            "start_seconds": 30.0,
            "end_seconds": 60.0,
            "source_url": "https://youtu.be/abc123",
            "embedding": [0.0, 1.0],
        },
    ]
    app = create_app(
        settings,
        runner_factory=lambda: None,
        history_path=tmp_path / "h.json",
        chat_html_path=tmp_path / "c.html",
        index_fn=lambda argv: 0,
        graph_records_fn=lambda: records,
    )
    response = TestClient(app).post("/api/chunk-graph", json={})
    assert response.status_code == 200
    payload = response.json()
    assert len(payload["nodes"]) == 2


def test_chunk_graph_500_on_backend_read_failure(settings: Settings, tmp_path: Path) -> None:
    def broken_records_fn() -> list[dict]:
        raise RuntimeError("chroma store corrupted")

    app = create_app(
        settings,
        runner_factory=lambda: None,
        history_path=tmp_path / "h.json",
        chat_html_path=tmp_path / "c.html",
        index_fn=lambda argv: 0,
        graph_records_fn=broken_records_fn,
    )
    response = TestClient(app).post("/api/chunk-graph", json={})
    assert response.status_code == 500
    assert "chroma store corrupted" in response.json()["detail"]


def test_ui_and_assets_served(harness: Harness) -> None:
    page = harness.client.get("/")
    assert page.status_code == 200
    assert "RAG Evaluation Workbench" in page.text
    assert "/assets/render.js" in page.text
    for marker in ["/api/judge", "/api/scoreboard", "/api/corpus", "auto-judge"]:
        assert marker in page.text, marker

    render_js = harness.client.get("/assets/render.js")
    assert "function answerBubble" in render_js.text
    assert "function renderAnswer" in render_js.text

    answer_css = harness.client.get("/assets/answer.css")
    assert ".bubble" in answer_css.text


def test_built_bundle_is_served_when_present(settings: Settings, tmp_path: Path) -> None:
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text('<div id="root"></div>', encoding="utf-8")
    (dist / "assets" / "index-abc.js").write_text("console.log(1)", encoding="utf-8")

    app = create_app(
        settings,
        runner_factory=lambda: None,
        history_path=tmp_path / "h.json",
        chat_html_path=tmp_path / "c.html",
        index_fn=lambda argv: 0,
        corpus_fn=lambda: FAKE_CORPUS,
        frontend_dist=dist,
    )
    client = TestClient(app)

    assert 'id="root"' in client.get("/").text
    assert client.get("/api/health").json()["ui"] == "react"
    assert client.get("/assets/index-abc.js").status_code == 200
    # The legacy shared renderer keeps its route despite the /assets mount.
    assert "function renderAnswer" in client.get("/assets/render.js").text


def test_ask_scopes_retrieval_to_a_channel(harness: Harness) -> None:
    harness.ask(setups=["rag_llm"], channel_id="UC_finance")
    assert harness.runner.scopes[-1].channel_id == "UC_finance"


def test_ask_prefers_video_scope_over_channel(harness: Harness) -> None:
    """A pinned video is narrower, so sending both must not widen the scope."""
    harness.ask(
        setups=["rag_llm"],
        url="https://youtu.be/abc123",
        channel_id="UC_finance",
    )
    assert harness.runner.scopes[-1].channel_id is None


def test_ask_passes_retrieval_mode_and_filter(harness: Harness) -> None:
    harness.ask(setups=["rag_llm"], retrieval_mode="hybrid", filter_transcripts=True)
    scope = harness.runner.scopes[-1]
    assert scope.retrieval_mode == "hybrid"
    assert scope.filter_transcripts is True


def test_ask_rejects_unknown_retrieval_mode(harness: Harness) -> None:
    response = harness.client.post("/api/ask", json={"question": "q", "retrieval_mode": "magic"})
    assert response.status_code == 422


def test_judge_receives_the_model_that_wrote_each_answer(harness: Harness) -> None:
    """self_graded must reflect the actual pairing, not the current config."""
    entry_id = harness.ask(setups=["rag_llm"])
    harness.client.post("/api/judge", json={"entry_id": entry_id})
    assert harness.judge.answer_models == ["deepseek-v4"]


def test_ask_persists_retrieved_chunk_ids(harness: Harness) -> None:
    """References cover only cited chunks, so recall needs the full list."""
    entry_id = harness.ask(setups=["rag_llm"])
    entry = next(
        e for e in harness.client.get("/api/history").json()["conversations"] if e["id"] == entry_id
    )
    assert "retrieved_chunk_ids" in entry["answers"][0]


def test_index_queue_enqueue_returns_immediately_while_a_job_runs(
    settings: Settings, tmp_path: Path
) -> None:
    """The whole point of the queue: a second submission never blocks."""
    release = threading.Event()

    def slow_index_fn(argv: list[str]) -> int:
        release.wait(timeout=2)
        return 0

    app = create_app(
        settings,
        runner_factory=lambda: None,
        history_path=tmp_path / "h.json",
        chat_html_path=tmp_path / "c.html",
        index_fn=slow_index_fn,
        frontend_dist=tmp_path / "no-bundle",
    )
    client = TestClient(app)
    try:
        first = client.post("/api/index/queue", json={"mode": "video", "url": "https://youtu.be/a"})
        second = client.post(
            "/api/index/queue", json={"mode": "video", "url": "https://youtu.be/b"}
        )
        assert first.status_code == 200
        assert second.status_code == 200
        assert first.json()["status"] == "queued"
        assert second.json()["status"] == "queued"
        assert first.json()["id"] != second.json()["id"]
    finally:
        release.set()


def test_index_queue_snapshot_lists_jobs_in_order(settings: Settings, tmp_path: Path) -> None:
    app = create_app(
        settings,
        runner_factory=lambda: None,
        history_path=tmp_path / "h.json",
        chat_html_path=tmp_path / "c.html",
        index_fn=lambda argv: 0,
        frontend_dist=tmp_path / "no-bundle",
    )
    client = TestClient(app)
    client.post("/api/index/queue", json={"mode": "video", "url": "https://youtu.be/a"})
    client.post("/api/index/queue", json={"mode": "video", "url": "https://youtu.be/b"})

    def snapshot() -> list[dict]:
        return client.get("/api/index/queue").json()["jobs"]

    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        jobs = snapshot()
        if len(jobs) == 2 and all(j["status"] == "done" for j in jobs):
            break
        time.sleep(0.02)
    jobs = snapshot()
    assert [job["target"] for job in jobs] == [
        "https://youtu.be/a",
        "https://youtu.be/b",
    ]


def test_index_queue_channel_mode_carries_latest(settings: Settings, tmp_path: Path) -> None:
    app = create_app(
        settings,
        runner_factory=lambda: None,
        history_path=tmp_path / "h.json",
        chat_html_path=tmp_path / "c.html",
        index_fn=lambda argv: 0,
        frontend_dist=tmp_path / "no-bundle",
    )
    client = TestClient(app)
    response = client.post(
        "/api/index/queue",
        json={"mode": "channel", "channel": "@some", "latest": 7},
    )
    assert response.json()["mode"] == "channel"
    assert response.json()["latest"] == 7
    assert response.json()["target"] == "@some"


def test_index_queue_stream_seeds_then_broadcasts_over_a_real_server(
    settings: Settings, tmp_path: Path
) -> None:
    """/api/index/queue/stream never ends, so FastAPI's TestClient can't run it.

    ``TestClient`` is built on ``httpx.ASGITransport``, which fully drains a
    StreamingResponse's generator before returning control — fine for the
    finite ``/api/ask``/``/api/judge`` streams, but an infinite queue feed
    (by design: it stays open for the life of a browser tab) hangs it
    forever. A real uvicorn server plus a real ``httpx`` network connection
    supports true incremental reads, so this one test runs the app for real
    on an ephemeral port instead, with a hard client-side timeout so a
    regression fails loudly instead of hanging the suite.
    """
    import socket
    import time as time_module

    import httpx
    import uvicorn

    app = create_app(
        settings,
        runner_factory=lambda: None,
        history_path=tmp_path / "h.json",
        chat_html_path=tmp_path / "c.html",
        index_fn=lambda argv: 0,
        frontend_dist=tmp_path / "no-bundle",
    )
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        deadline = time_module.monotonic() + 5
        while not server.started and time_module.monotonic() < deadline:
            time_module.sleep(0.02)
        assert server.started, "uvicorn did not start in time"

        base_url = f"http://127.0.0.1:{port}"
        with httpx.Client(timeout=5.0) as client:
            client.post(
                f"{base_url}/api/index/queue",
                json={"mode": "video", "url": "https://youtu.be/a"},
            )
            with client.stream("GET", f"{base_url}/api/index/queue/stream") as response:
                assert response.status_code == 200
                lines = response.iter_lines()
                assert next(lines) == "event: snapshot"
                assert "https://youtu.be/a" in next(lines)

                client.post(
                    f"{base_url}/api/index/queue",
                    json={"mode": "video", "url": "https://youtu.be/b"},
                )
                seen_event_names: list[str] = []
                for line in lines:
                    if line.startswith("event: "):
                        seen_event_names.append(line[len("event: ") :])
                    if len(seen_event_names) >= 2:
                        break
                assert seen_event_names[0] == "job"
    finally:
        server.should_exit = True
        thread.join(timeout=5)


def test_subscription_stream_forwards_events_as_sse() -> None:
    import queue as queue_module

    subscriber: queue_module.Queue = queue_module.Queue()
    subscriber.put({"type": "job", "job": {"id": "j1"}})
    stream = api_main._subscription_stream(subscriber, lambda _sub: None)
    try:
        chunk = next(stream)
    finally:
        stream.close()
    assert chunk.startswith("event: job\n")
    assert '"id": "j1"' in chunk


def test_subscription_stream_keeps_alive_so_a_closed_tab_is_noticed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An idle stream must still reach a ``yield``.

    Starlette drives these sync generators on a worker thread, so a ``get()``
    that blocks until the next broadcast pins that thread — and the subscriber
    registration — for the life of the process when no job is running, which
    is exactly when a tab is most likely to be closed. Emitting a keepalive
    comment on the way round is what lets the disconnect close the generator
    and run its unsubscribe.
    """
    import queue as queue_module

    monkeypatch.setattr(api_main, "_SSE_KEEPALIVE_INTERVAL_SECONDS", 0.01)
    subscriber: queue_module.Queue = queue_module.Queue()
    unsubscribed: list = []
    stream = api_main._subscription_stream(subscriber, unsubscribed.append)

    # Nothing is ever broadcast, and the generator still yields ...
    assert next(stream).startswith(":")
    assert unsubscribed == []

    # ... so closing it (what a client disconnect does) deregisters instead of
    # leaking the subscriber and its thread.
    stream.close()
    assert unsubscribed == [subscriber]


def test_system_design_route_returns_nodes_and_edges(settings: Settings, tmp_path: Path) -> None:
    app = create_app(
        settings,
        runner_factory=lambda: None,
        history_path=tmp_path / "h.json",
        chat_html_path=tmp_path / "c.html",
        index_fn=lambda argv: 0,
        frontend_dist=tmp_path / "no-bundle",
    )
    client = TestClient(app)
    response = client.get("/api/system-design")
    assert response.status_code == 200
    body = response.json()
    node_ids = {node["id"] for node in body["nodes"]}
    assert "graph_rag" in node_ids
    assert "vector_rag" in node_ids
    assert "neo4j" in node_ids


def test_knowledge_graph_route_returns_nodes_edges_and_communities(
    settings: Settings, tmp_path: Path
) -> None:
    class FakeGraphStore:
        def all_entities(self, limit: int = 2000):
            from src.rag.graph_models import GraphEntity

            return [GraphEntity(id="a", name="Alpha", mentions=3, community_id=0)]

        def entity_edges(self):
            return ["a"], []

        def communities(self):
            from src.rag.graph_models import GraphCommunity

            return [GraphCommunity(id=0, entity_ids=["a"], entity_names=["Alpha"], claim_count=1)]

        def get_entity(self, entity_id: str):
            from src.rag.graph_models import GraphEntity

            return (
                GraphEntity(id="a", name="Alpha", mentions=3, community_id=0)
                if entity_id == "a"
                else None
            )

        def claims_about(self, entity_ids, limit=40, hops=1):
            from src.rag.graph_models import GraphClaim

            return [GraphClaim(id="c1", text="Alpha does things.", upload_date="2026-01-01")]

    app = create_app(
        settings,
        runner_factory=lambda: None,
        history_path=tmp_path / "h.json",
        chat_html_path=tmp_path / "c.html",
        index_fn=lambda argv: 0,
        frontend_dist=tmp_path / "no-bundle",
        graph_store_factory=lambda: FakeGraphStore(),
    )
    client = TestClient(app)

    response = client.get("/api/graph/knowledge")
    assert response.status_code == 200
    body = response.json()
    assert body["nodes"] == [
        {
            "id": "a",
            "name": "Alpha",
            "type": "concept",
            "mentions": 3,
            "community_id": 0,
            "x": 0.0,
            "y": 0.0,
        }
    ]
    assert body["communities"][0]["summary"] is None

    entity_response = client.get("/api/graph/knowledge/entities/a")
    assert entity_response.status_code == 200
    assert entity_response.json()["entity"]["name"] == "Alpha"
    assert len(entity_response.json()["claims"]) == 1

    missing_response = client.get("/api/graph/knowledge/entities/nope")
    assert missing_response.status_code == 404


def test_knowledge_graph_route_reports_503_when_store_unreachable(
    settings: Settings, tmp_path: Path
) -> None:
    class BrokenGraphStore:
        def all_entities(self, limit: int = 2000):
            raise ConnectionError("neo4j unreachable")

    app = create_app(
        settings,
        runner_factory=lambda: None,
        history_path=tmp_path / "h.json",
        chat_html_path=tmp_path / "c.html",
        index_fn=lambda argv: 0,
        frontend_dist=tmp_path / "no-bundle",
        graph_store_factory=lambda: BrokenGraphStore(),
    )
    client = TestClient(app)
    response = client.get("/api/graph/knowledge")
    assert response.status_code == 503
    assert "neo4j unreachable" in response.json()["detail"]

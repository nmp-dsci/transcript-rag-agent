from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.main import create_app
from src.chat.setups import SETUP_KEYS, SetupResult, setup_spec
from src.config import Settings


class FakeRunner:
    """Stands in for RagSetupRunner: canned answers, recorded calls."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str | None]] = []

    def run(
        self,
        key: str,
        question: str,
        *,
        url: str | None = None,
        top_k: int | None = None,
    ) -> SetupResult:
        self.calls.append((key, question, url))
        self.top_ks: list[int | None] = getattr(self, "top_ks", [])
        self.top_ks.append(top_k)
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
        )


class FakeJudge:
    """Stands in for RagasJudge: deterministic scores, recorded calls."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, list[str]]] = []

    def score(self, question: str, answer: str, contexts: list[str]) -> dict:
        self.calls.append((question, answer, list(contexts)))
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


class Harness:
    def __init__(self, settings: Settings, tmp_path: Path) -> None:
        self.runner = FakeRunner()
        self.judge = FakeJudge()
        self.factory_calls = 0
        self.judge_factory_calls = 0
        self.index_argv: list[list[str]] = []
        self.history_path = tmp_path / "chat_history.json"
        self.chat_html_path = tmp_path / "chat.html"

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
            history_path=self.history_path,
            chat_html_path=self.chat_html_path,
            index_fn=index_fn,
        )
        self.client = TestClient(app)

    def ask(self, question: str = "What is agentic RAG?", **kwargs) -> str:
        """Ask and return the saved entry id."""
        response = self.client.post(
            "/api/ask", json={"question": question, **kwargs}
        )
        events = sse_events(response.text)
        assert events[-1][0] == "done", events
        return events[-1][1]["id"]


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
    assert response.json() == {
        "status": "ok",
        "runner_loaded": False,
        "judge_loaded": False,
        "judge_model": "deepseek-v4",
    }


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
    response = harness.client.post(
        "/api/ask", json={"question": "Hello?", "setups": ["nope"]}
    )
    assert response.status_code == 422
    assert "nope" in response.json()["detail"]


@pytest.mark.parametrize("question", ["", "   "])
def test_ask_rejects_blank_question(harness: Harness, question: str) -> None:
    response = harness.client.post("/api/ask", json={"question": question})
    assert response.status_code == 422


def test_runner_built_once_across_questions(harness: Harness) -> None:
    first = harness.client.post(
        "/api/ask", json={"question": "One?", "setups": ["rag_llm"]}
    )
    second = harness.client.post(
        "/api/ask", json={"question": "Two?", "setups": ["rag_llm"]}
    )
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
    harness.client.post(
        "/api/index", json={"mode": "channel", "channel": "@some", "latest": 3}
    )
    assert harness.index_argv == [
        ["bulk-index", "channel", "--channel", "@some", "--latest", "3"]
    ]


def test_index_requires_target(harness: Harness) -> None:
    assert harness.client.post("/api/index", json={"mode": "video"}).status_code == 422
    assert (
        harness.client.post("/api/index", json={"mode": "channel"}).status_code == 422
    )


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


def test_judge_marks_answers_without_contexts(
    settings: Settings, tmp_path: Path
) -> None:
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


def test_scoreboard_aggregates_by_setup(harness: Harness) -> None:
    for _ in range(2):
        entry_id = harness.ask(setups=["rag_llm", "rag_agent"])
        harness.client.post("/api/judge", json={"entry_id": entry_id})

    board = harness.client.get("/api/scoreboard").json()
    assert board["entries_total"] == 2
    assert board["entries_judged"] == 2
    assert board["judge_model"] == "deepseek-v4"

    rows = {row["key"]: row for row in board["setups"]}
    assert rows["rag_agent"]["avg_composite"] == 0.9
    assert rows["rag_llm"]["avg_composite"] == 0.5
    assert rows["rag_agent"]["win_rate"] == 1.0
    assert rows["rag_llm"]["win_rate"] == 0.0
    assert rows["rag_agent"]["judged"] == 2
    # Sorted best-first.
    assert board["setups"][0]["key"] == "rag_agent"


def test_corpus_endpoint(harness: Harness) -> None:
    payload = harness.client.get("/api/corpus").json()
    assert payload["totals"] == {"videos": 1, "chunks": 42}
    assert payload["videos"][0]["title"] == "Tax changes explained"


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

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
        )


class Harness:
    def __init__(self, settings: Settings, tmp_path: Path) -> None:
        self.runner = FakeRunner()
        self.factory_calls = 0
        self.index_argv: list[list[str]] = []
        self.history_path = tmp_path / "chat_history.json"
        self.chat_html_path = tmp_path / "chat.html"

        def factory() -> FakeRunner:
            self.factory_calls += 1
            return self.runner

        def index_fn(argv: list[str]) -> int:
            self.index_argv.append(argv)
            return 0

        app = create_app(
            settings,
            runner_factory=factory,  # type: ignore[arg-type]
            history_path=self.history_path,
            chat_html_path=self.chat_html_path,
            index_fn=index_fn,
        )
        self.client = TestClient(app)


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
    assert response.json() == {"status": "ok", "runner_loaded": False}


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


def test_ui_and_assets_served(harness: Harness) -> None:
    page = harness.client.get("/")
    assert page.status_code == 200
    assert "Transcript RAG Chat" in page.text
    assert "/assets/render.js" in page.text

    render_js = harness.client.get("/assets/render.js")
    assert "function answerBubble" in render_js.text
    assert "function renderAnswer" in render_js.text

    answer_css = harness.client.get("/assets/answer.css")
    assert ".bubble" in answer_css.text

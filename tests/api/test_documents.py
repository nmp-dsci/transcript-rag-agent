"""Chat-native document review over the API: fetch, review, card, reload."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.main import create_app
from src.config import Settings
from src.documents.fetch import DocumentFetchError, UnsafeUrlError
from src.documents.models import FetchedPage
from src.documents.store import DocumentStore
from tests.api.test_app import FakeRunner, sse_events

URL = "https://example.com/cv"
RESUME = (
    "<html><head><title>A resume</title></head><body>"
    "<h2>Experience</h2><p>Led a team of six engineers for four years running.</p>"
    "<h2>Education</h2><p>BSc in computer science, thesis on retrieval evaluation.</p>"
    "</body></html>"
)


class CountingFetch:
    def __init__(self, body: str = RESUME) -> None:
        self.body = body
        self.calls: list[str] = []

    def __call__(self, url: str) -> FetchedPage:
        self.calls.append(url)
        return FetchedPage(
            requested_url=url, url=url, status_code=200, content_type="text/html", body=self.body
        )


class Docs:
    """An app wired to a temp document store and a scripted fetcher."""

    def __init__(self, settings: Settings, tmp_path: Path, fetch=None) -> None:
        self.runner = FakeRunner()
        self.store = DocumentStore(tmp_path / "documents")
        self.fetch = fetch or CountingFetch()
        self.history_path = tmp_path / "history.json"
        app = create_app(
            settings,
            runner_factory=lambda: self.runner,  # type: ignore[arg-type]
            history_path=self.history_path,
            chat_html_path=tmp_path / "chat.html",
            index_fn=lambda argv: 0,
            frontend_dist=tmp_path / "no-bundle",
            document_store=self.store,
            document_fetch_fn=self.fetch,
        )
        self.client = TestClient(app)

    def ask(self, question: str, **kwargs) -> list[tuple[str, dict]]:
        response = self.client.post("/api/ask", json={"question": question, **kwargs})
        assert response.status_code == 200
        return sse_events(response.text)


@pytest.fixture
def docs(settings: Settings, tmp_path: Path) -> Docs:
    return Docs(settings, tmp_path)


def _events_of(events: list[tuple[str, dict]], name: str) -> list[dict]:
    return [data for event, data in events if event == name]


# ── the default is still text ─────────────────────────────────────────────────


def test_a_plain_question_behaves_exactly_as_before(docs: Docs) -> None:
    events = docs.ask("what are the tax changes?")

    assert _events_of(events, "document") == []
    assert docs.fetch.calls == []
    assert events[-1][0] == "done"
    assert events[-1][1]["document_id"] is None


def test_a_plain_question_still_runs_every_requested_setup(docs: Docs) -> None:
    docs.ask("what changed?", setups=["rag_llm", "rag_agent"])

    assert [call[0] for call in docs.runner.calls] == ["rag_llm", "rag_agent"]


# ── a URL turns the answer into a review ──────────────────────────────────────


def test_a_url_in_the_message_is_fetched_and_announced(docs: Docs) -> None:
    events = docs.ask(f"any feedback on {URL}?")

    assert docs.fetch.calls == [URL]
    card = _events_of(events, "document")[0]
    assert card["title"] == "A resume"
    assert card["url"] == URL
    # This page opens straight on a heading, so there is no unheaded preamble.
    assert [section["heading"] for section in card["sections"]] == ["Experience", "Education"]


def test_the_card_arrives_before_the_answer_it_belongs_to(docs: Docs) -> None:
    """So it can render while the answer is still being written."""
    names = [event for event, _data in docs.ask(f"feedback on {URL}")]

    assert names.index("document") < names.index("answer")


def test_a_review_answers_with_the_single_hop_path_only(docs: Docs) -> None:
    """The others would ignore the document and produce a corpus answer
    dressed as a review."""
    docs.ask(f"feedback on {URL}", setups=["rag_llm", "rag_agent", "graph_rag"])

    assert [call[0] for call in docs.runner.calls] == ["rag_llm"]


def test_the_document_reaches_the_runner_as_answer_context(docs: Docs) -> None:
    docs.ask(f"feedback on {URL}")

    scope = docs.runner.scopes[-1]
    assert scope.document_context is not None
    assert "[§1] Experience" in scope.document_context


def test_the_entry_records_the_document_by_id_only(docs: Docs) -> None:
    """The history file is committed; the fetched text must not be in it."""
    events = docs.ask(f"feedback on {URL}")
    document_id = events[-1][1]["document_id"]

    assert document_id is not None
    written = json.loads(docs.history_path.read_text(encoding="utf-8"))
    assert "Led a team of six engineers" not in json.dumps(written)


# ── reuse across a thread ─────────────────────────────────────────────────────


def test_a_follow_up_in_the_same_entry_reuses_the_document(docs: Docs) -> None:
    events = docs.ask(f"feedback on {URL}")
    entry_id = events[-1][1]["id"]

    follow_up = docs.ask(f"feedback on {URL}", entry_id=entry_id, setups=["rag_llm"])

    assert docs.fetch.calls == [URL], "the page must not be fetched twice"
    assert _events_of(follow_up, "document")[0]["reused"] is True


# ── failures are reported, never silently answered ────────────────────────────


def test_a_refused_url_reports_why_and_answers_nothing(settings, tmp_path) -> None:
    def refuse(url: str):
        raise UnsafeUrlError("that host resolves to a non-public address")

    docs = Docs(settings, tmp_path, fetch=refuse)

    events = docs.ask(f"review http://169.254.169.254/latest and {URL}")

    assert events[-1][0] == "error"
    assert "non-public address" in events[-1][1]["message"]
    assert docs.runner.calls == [], "no answer is written for a page nobody read"


def test_a_failed_fetch_reports_why(settings, tmp_path) -> None:
    def fail(url: str):
        raise DocumentFetchError("HTTP 404")

    docs = Docs(settings, tmp_path, fetch=fail)

    events = docs.ask(f"review {URL}")

    assert events[-1][0] == "error"
    assert "404" in events[-1][1]["message"]


# ── reading the document back ─────────────────────────────────────────────────


def test_a_stored_document_can_be_read_back_for_a_card_after_reload(docs: Docs) -> None:
    events = docs.ask(f"feedback on {URL}")
    document_id = events[-1][1]["document_id"]

    response = docs.client.get(f"/api/documents/{document_id}")

    assert response.status_code == 200
    assert response.json()["title"] == "A resume"


def test_an_unknown_document_is_a_404_not_a_crash(docs: Docs) -> None:
    """A cleared store means the card is absent; the conversation still reads."""
    assert docs.client.get("/api/documents/doc:nope").status_code == 404

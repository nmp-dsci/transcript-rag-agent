"""Deciding, once, whether a question reviews a document and which one."""

from __future__ import annotations

import pytest

from src.documents.extract import document_id_for
from src.documents.fetch import DocumentFetchError, UnsafeUrlError
from src.documents.models import Document, DocumentSection, FetchedPage
from src.documents.resolve import describe_failure, resolve_document
from src.documents.store import DocumentStore

URL = "https://example.com/cv"
BODY = "<html><head><title>A resume</title></head><body><h2>Experience</h2><p>Led a team of six engineers for four years.</p></body></html>"


def _page(url: str = URL, body: str = BODY) -> FetchedPage:
    return FetchedPage(
        requested_url=url, url=url, status_code=200, content_type="text/html", body=body
    )


class CountingFetch:
    def __init__(self, page: FetchedPage | None = None) -> None:
        self.page = page or _page()
        self.calls: list[str] = []

    def __call__(self, url: str) -> FetchedPage:
        self.calls.append(url)
        return self.page


# ── ordinary questions ────────────────────────────────────────────────────────


def test_a_question_without_a_url_resolves_to_no_document(tmp_path) -> None:
    fetch = CountingFetch()

    resolved = resolve_document(
        "what are the tax changes?", store=DocumentStore(tmp_path), fetch=fetch
    )

    assert resolved is None
    assert fetch.calls == []


def test_a_youtube_link_is_not_a_document(tmp_path) -> None:
    fetch = CountingFetch()

    resolved = resolve_document(
        "summarise https://www.youtube.com/watch?v=3hk7nO_q0a8",
        store=DocumentStore(tmp_path),
        fetch=fetch,
    )

    assert resolved is None
    assert fetch.calls == []


# ── fetching a new document ───────────────────────────────────────────────────


def test_a_url_is_fetched_extracted_and_stored(tmp_path) -> None:
    store = DocumentStore(tmp_path)
    fetch = CountingFetch()

    resolved = resolve_document(f"any feedback on {URL}?", store=store, fetch=fetch)

    assert resolved is not None
    assert resolved.reused is False
    assert resolved.document.title == "A resume"
    assert fetch.calls == [URL]
    # Stored, so the next turn does not hit the page again.
    assert store.get(document_id_for(URL)) is not None


def test_the_resolved_context_carries_citable_sections(tmp_path) -> None:
    resolved = resolve_document(
        f"feedback on {URL}", store=DocumentStore(tmp_path), fetch=CountingFetch()
    )

    assert resolved is not None
    assert "[§1]" in resolved.context
    assert URL in resolved.context


# ── reuse ─────────────────────────────────────────────────────────────────────


def test_a_pinned_document_is_reused_without_refetching(tmp_path) -> None:
    """A follow-up must read the same text the first answer read."""
    store = DocumentStore(tmp_path)
    store.save(
        Document(
            id="doc:pinned",
            url=URL,
            requested_url=URL,
            title="A resume",
            sections=[DocumentSection(index=0, heading="Experience", text="Led a team.")],
        )
    )
    fetch = CountingFetch()

    resolved = resolve_document(
        "now check the experience section",
        store=store,
        pinned_document_id="doc:pinned",
        fetch=fetch,
    )

    assert resolved is not None
    assert resolved.reused is True
    assert resolved.document.id == "doc:pinned"
    assert fetch.calls == []


def test_the_same_url_asked_twice_is_fetched_once(tmp_path) -> None:
    store = DocumentStore(tmp_path)
    fetch = CountingFetch()

    resolve_document(f"feedback on {URL}", store=store, fetch=fetch)
    second = resolve_document(f"and the tone of {URL}?", store=store, fetch=fetch)

    assert fetch.calls == [URL]
    assert second is not None and second.reused is True


def test_a_pin_that_has_been_cleared_falls_back_to_the_message(tmp_path) -> None:
    """The store is derived state; a missing document is not a dead thread."""
    fetch = CountingFetch()

    resolved = resolve_document(
        f"feedback on {URL}",
        store=DocumentStore(tmp_path),
        pinned_document_id="doc:gone",
        fetch=fetch,
    )

    assert resolved is not None
    assert resolved.reused is False
    assert fetch.calls == [URL]


def test_a_cleared_pin_with_no_url_in_the_message_is_an_ordinary_question(tmp_path) -> None:
    resolved = resolve_document(
        "now check the experience section",
        store=DocumentStore(tmp_path),
        pinned_document_id="doc:gone",
        fetch=CountingFetch(),
    )

    assert resolved is None


# ── failures reach the caller ─────────────────────────────────────────────────


def test_a_refused_url_raises_rather_than_answering_from_the_corpus(tmp_path) -> None:
    """Answering anyway would look like a review of a page nobody read."""

    def refuse(url: str):
        raise UnsafeUrlError("resolves to a non-public address")

    with pytest.raises(UnsafeUrlError):
        resolve_document(f"review {URL}", store=DocumentStore(tmp_path), fetch=refuse)


def test_a_failed_fetch_raises(tmp_path) -> None:
    def fail(url: str):
        raise DocumentFetchError("HTTP 404")

    with pytest.raises(DocumentFetchError):
        resolve_document(f"review {URL}", store=DocumentStore(tmp_path), fetch=fail)


def test_a_policy_refusal_is_described_verbatim() -> None:
    message = describe_failure(UnsafeUrlError("that host resolves to a non-public address"))

    assert "non-public address" in message


def test_an_unexpected_error_is_summarised_not_leaked() -> None:
    message = describe_failure(RuntimeError("KeyError at line 41 of internals.py"))

    assert "internals.py" not in message
    assert "could not be reviewed" in message


# ── what the trace says ───────────────────────────────────────────────────────


def test_the_detail_says_whether_the_page_was_hit(tmp_path) -> None:
    store = DocumentStore(tmp_path)
    fetched = resolve_document(f"feedback on {URL}", store=store, fetch=CountingFetch())
    reused = resolve_document(f"feedback on {URL}", store=store, fetch=CountingFetch())

    assert fetched is not None and reused is not None
    assert fetched.detail().startswith("fetched")
    assert reused.detail().startswith("reused from this thread")

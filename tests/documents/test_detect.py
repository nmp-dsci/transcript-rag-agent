"""Deciding whether a chat message is asking for a document review."""

from __future__ import annotations

import pytest

from src.documents.detect import find_document_url, find_urls, is_youtube_url


# ── the default is text ───────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "message",
    [
        "What are the key tax changes?",
        "compare this to example.com's approach",
        "how do I make my resume ATS-friendly?",
        "",
    ],
)
def test_a_message_without_a_url_is_an_ordinary_question(message: str) -> None:
    """The chat stays a chat: no mode switch, nothing to turn on."""
    assert find_document_url(message) is None


def test_a_bare_domain_is_prose_not_a_fetch_request() -> None:
    """A fetch is a side effect; it is not triggered on a guess."""
    assert find_document_url("have a look at example.com and tell me") is None


# ── a URL adds the behaviour ──────────────────────────────────────────────────


def test_a_url_in_the_message_is_the_document_to_review() -> None:
    url = find_document_url("review this please https://example.com/cv")

    assert url == "https://example.com/cv"


def test_the_url_can_be_anywhere_in_the_message() -> None:
    assert find_document_url("https://example.com/cv — any feedback?") == "https://example.com/cv"


@pytest.mark.parametrize("scheme", ["http", "https", "HTTPS"])
def test_both_schemes_are_recognised(scheme: str) -> None:
    assert find_document_url(f"{scheme}://example.com/x") is not None


# ── YouTube keeps its existing meaning ────────────────────────────────────────


@pytest.mark.parametrize(
    "url",
    [
        "https://www.youtube.com/watch?v=3hk7nO_q0a8",
        "https://youtu.be/3hk7nO_q0a8",
        "https://m.youtube.com/watch?v=3hk7nO_q0a8",
    ],
)
def test_youtube_links_are_not_documents_to_review(url: str) -> None:
    """They already scope retrieval to one video — a behaviour people rely on."""
    assert is_youtube_url(url) is True
    assert find_document_url(f"summarise {url}") is None


def test_a_youtube_link_does_not_hide_a_real_document_later_in_the_message() -> None:
    message = "compare https://www.youtube.com/watch?v=3hk7nO_q0a8 with https://example.com/cv"

    assert find_document_url(message) == "https://example.com/cv"


def test_a_non_youtube_host_is_not_treated_as_youtube() -> None:
    assert is_youtube_url("https://notyoutube.com/watch?v=x") is False


# ── shape of the extracted URL ────────────────────────────────────────────────


@pytest.mark.parametrize(
    "message,expected",
    [
        ("see https://example.com/cv.", "https://example.com/cv"),
        ("see https://example.com/cv, then", "https://example.com/cv"),
        ("see https://example.com/cv!", "https://example.com/cv"),
        ("(https://example.com/cv)", "https://example.com/cv"),
    ],
)
def test_sentence_punctuation_is_not_part_of_the_url(message: str, expected: str) -> None:
    assert find_document_url(message) == expected


def test_a_query_string_survives_trimming() -> None:
    url = "https://example.com/page?ref=chat&id=7"

    assert find_document_url(f"look at {url}") == url


def test_only_the_first_document_url_is_taken() -> None:
    """One card, one set of anchors — a three-link message reviews the first."""
    message = "https://example.com/one and https://example.com/two"

    assert find_document_url(message) == "https://example.com/one"


def test_find_urls_reports_every_url_in_order() -> None:
    message = (
        "https://a.example.com and https://www.youtube.com/watch?v=x then https://b.example.com"
    )

    assert find_urls(message) == [
        "https://a.example.com",
        "https://www.youtube.com/watch?v=x",
        "https://b.example.com",
    ]

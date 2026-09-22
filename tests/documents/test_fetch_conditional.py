"""Per-call content types and conditional GETs, added for the corpus poller.

The property these tests exist to protect is that the *other four* bounds in
``fetch.py`` — scheme, address re-validation on each hop, redirect count, byte
cap — are not negotiable and gained no parameter, because those four are what
make this module not an SSRF proxy.
"""

from __future__ import annotations

import contextlib

import httpx
import pytest

from src.documents.fetch import (
    ALLOWED_CONTENT_TYPES,
    FEED_CONTENT_TYPES,
    FEED_MAX_BYTES,
    MARKDOWN_CONTENT_TYPES,
    DEFAULT_MAX_BYTES,
    UnsafeUrlError,
    fetch_document,
)


@pytest.fixture(autouse=True)
def public_dns(monkeypatch):
    """Resolve every host to one public address, as the sibling suite does."""
    monkeypatch.setattr(
        "src.documents.fetch.socket.getaddrinfo",
        lambda host, port, **kwargs: [(2, 1, 6, "", ("93.184.216.34", port))],
    )


@contextlib.contextmanager
def client(handler):
    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport, follow_redirects=False) as http:
        yield http


def xml_handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200,
        text="<rss><channel/></rss>",
        headers={"content-type": "application/xml", "etag": '"v1"'},
    )


def test_the_default_policy_still_refuses_a_feed() -> None:
    # The chat's paste-a-link path must not start accepting XML just because
    # the corpus poller needs to.
    with client(xml_handler) as http:
        with pytest.raises(UnsafeUrlError, match="application/xml"):
            fetch_document("https://example.com/f.xml", client=http)


def test_a_caller_can_opt_into_feed_content_types() -> None:
    with client(xml_handler) as http:
        page = fetch_document(
            "https://example.com/f.xml", client=http, allowed_content_types=FEED_CONTENT_TYPES
        )
    assert page.status_code == 200
    assert page.content_type == "application/xml"


def test_opting_into_feeds_does_not_also_allow_html() -> None:
    # The parameter narrows as well as widens; it is not a bypass.
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html/>", headers={"content-type": "text/html"})

    with client(handler) as http:
        with pytest.raises(UnsafeUrlError, match="text/html"):
            fetch_document(
                "https://example.com/page", client=http, allowed_content_types=FEED_CONTENT_TYPES
            )


def test_the_refusal_message_names_the_allowed_types_it_was_given() -> None:
    with client(xml_handler) as http:
        with pytest.raises(UnsafeUrlError, match="text/html"):
            fetch_document(
                "https://example.com/f.xml",
                client=http,
                allowed_content_types=ALLOWED_CONTENT_TYPES,
            )


def test_validators_are_sent_as_conditional_headers() -> None:
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(request.headers)
        return httpx.Response(304, headers={"etag": '"v1"'})

    with client(handler) as http:
        fetch_document(
            "https://example.com/f.xml",
            client=http,
            allowed_content_types=FEED_CONTENT_TYPES,
            etag='"v1"',
            last_modified="Mon, 01 Sep 2026 00:00:00 GMT",
        )
    assert seen["if-none-match"] == '"v1"'
    assert seen["if-modified-since"] == "Mon, 01 Sep 2026 00:00:00 GMT"


def test_no_conditional_headers_are_sent_when_there_are_no_validators() -> None:
    seen: list[httpx.Headers] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers)
        return xml_handler(request)

    with client(handler) as http:
        fetch_document(
            "https://example.com/f.xml", client=http, allowed_content_types=FEED_CONTENT_TYPES
        )
    assert "if-none-match" not in seen[0]
    assert "if-modified-since" not in seen[0]


def test_a_304_is_a_successful_fetch_with_an_empty_body() -> None:
    # For a refresh, "nothing changed" is the most common and by far the
    # cheapest successful outcome, not a failure.
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(304, headers={"etag": '"v1"'})

    with client(handler) as http:
        page = fetch_document(
            "https://example.com/f.xml",
            client=http,
            allowed_content_types=FEED_CONTENT_TYPES,
            etag='"v1"',
        )
    assert page.not_modified
    assert page.status_code == 304
    assert page.body == ""


def test_a_304_is_not_mistaken_for_a_redirect() -> None:
    # httpx classifies 304 as a 3xx, so the redirect handler would otherwise
    # claim it and complain about a missing Location header.
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(304)

    with client(handler) as http:
        page = fetch_document("https://example.com/f.xml", client=http, etag='"v1"')
    assert page.not_modified


def test_a_304_carries_the_validators_forward_so_the_next_poll_is_conditional() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(304)

    with client(handler) as http:
        page = fetch_document(
            "https://example.com/f.xml",
            client=http,
            etag='"v1"',
            last_modified="Mon, 01 Sep 2026 00:00:00 GMT",
        )
    assert page.etag == '"v1"'
    assert page.last_modified == "Mon, 01 Sep 2026 00:00:00 GMT"


def test_a_200_records_the_validators_the_server_offered() -> None:
    with client(xml_handler) as http:
        page = fetch_document(
            "https://example.com/f.xml", client=http, allowed_content_types=FEED_CONTENT_TYPES
        )
    assert page.etag == '"v1"'
    assert page.not_modified is False


def test_a_custom_user_agent_replaces_the_default_on_the_client_it_builds() -> None:
    # Only exercisable on the client-less path: a caller that supplies its own
    # httpx.Client owns that client's headers.
    import src.documents.fetch as module

    captured: dict[str, object] = {}
    real_client = httpx.Client

    def recording_client(**kwargs):
        captured.update(kwargs)
        return real_client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200, text="hi", headers={"content-type": "text/plain"}
                )
            ),
            follow_redirects=False,
        )

    original = httpx.Client
    httpx.Client = recording_client  # type: ignore[misc]
    try:
        module.fetch_document("https://example.com/x", user_agent="yt-agent-corpus/1.0")
    finally:
        httpx.Client = original  # type: ignore[misc]
    assert captured["headers"]["User-Agent"] == "yt-agent-corpus/1.0"
    # Everything else the module sends is unchanged.
    assert "Accept" in captured["headers"]


def test_the_address_and_scheme_bounds_take_no_parameter() -> None:
    # Named explicitly so a future change that adds one has to delete a test.
    import inspect

    parameters = set(inspect.signature(fetch_document).parameters)
    assert parameters == {
        "url",
        "client",
        "max_bytes",
        "timeout_seconds",
        "max_redirects",
        "allowed_content_types",
        "etag",
        "last_modified",
        "user_agent",
    }


def test_private_addresses_are_still_refused_when_feed_types_are_allowed(monkeypatch) -> None:
    # Opting into feed content types must not loosen the address bound.
    monkeypatch.setattr(
        "src.documents.fetch.socket.getaddrinfo",
        lambda host, port, **kwargs: [(2, 1, 6, "", ("169.254.169.254", port))],
    )
    with pytest.raises(UnsafeUrlError, match="non-public address"):
        fetch_document(
            "http://metadata.example/latest/meta-data/",
            allowed_content_types=FEED_CONTENT_TYPES,
        )


def test_feeds_get_a_larger_cap_than_pages_but_are_still_bounded() -> None:
    # A full-text feed is one document holding twenty whole articles, so it is
    # legitimately bigger than any single page. Measured: one registered feed
    # is 2.9 MB, which the page cap cut mid-CDATA.
    assert FEED_MAX_BYTES > DEFAULT_MAX_BYTES
    assert FEED_MAX_BYTES <= 16_000_000


def test_markdown_content_types_are_declared_but_not_in_the_default_policy() -> None:
    assert "text/markdown" in MARKDOWN_CONTENT_TYPES
    assert not set(MARKDOWN_CONTENT_TYPES) & set(ALLOWED_CONTENT_TYPES)

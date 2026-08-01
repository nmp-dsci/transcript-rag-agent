"""The SSRF bounds on fetching a user-supplied URL."""

from __future__ import annotations

import httpx
import pytest

from src.documents.fetch import (
    DocumentFetchError,
    UnsafeUrlError,
    assert_fetchable,
    fetch_document,
    is_public_address,
)


@pytest.fixture(autouse=True)
def public_dns(monkeypatch):
    """Resolve every host to one public address unless a test says otherwise."""
    monkeypatch.setattr(
        "src.documents.fetch.socket.getaddrinfo",
        lambda host, port, **kwargs: [(2, 1, 6, "", ("93.184.216.34", port))],
    )


def _client(handler) -> httpx.Client:
    return httpx.Client(
        transport=httpx.MockTransport(handler),
        follow_redirects=False,
        timeout=5.0,
    )


def _html(text: str = "<html><body><p>hello</p></body></html>") -> httpx.Response:
    return httpx.Response(200, text=text, headers={"content-type": "text/html; charset=utf-8"})


# ── address classification ────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",  # loopback — the server's own services
        "10.0.0.5",  # private
        "192.168.1.10",  # private
        "172.16.0.1",  # private
        "169.254.169.254",  # cloud instance metadata
        "0.0.0.0",  # unspecified
        "::1",  # IPv6 loopback
        "fd00::1",  # IPv6 unique-local
        "fe80::1",  # IPv6 link-local
        "::ffff:127.0.0.1",  # IPv4-mapped loopback
        "64:ff9b::7f00:1",  # NAT64 well-known prefix wrapping loopback
        "64:ff9b::a9fe:a9fe",  # NAT64 well-known prefix wrapping link-local metadata
        "224.0.0.1",  # multicast
        "not-an-ip",
    ],
)
def test_non_public_addresses_are_rejected(address: str) -> None:
    assert is_public_address(address) is False


@pytest.mark.parametrize(
    "address",
    ["93.184.216.34", "1.1.1.1", "2606:4700::1111", "64:ff9b::101:101"],
)
def test_publicly_routable_addresses_are_allowed(address: str) -> None:
    assert is_public_address(address) is True


# ── scheme and URL shape ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "url", ["file:///etc/passwd", "gopher://example.com/", "ftp://example.com"]
)
def test_only_http_urls_can_be_fetched(url: str) -> None:
    with pytest.raises(UnsafeUrlError, match="cannot be fetched"):
        assert_fetchable(url)


def test_credentials_in_the_url_are_refused() -> None:
    """They would be forwarded to whatever the URL redirects to."""
    with pytest.raises(UnsafeUrlError, match="credentials"):
        assert_fetchable("https://user:secret@example.com/page")


def test_a_url_with_no_host_is_refused() -> None:
    with pytest.raises(UnsafeUrlError, match="no host"):
        assert_fetchable("https:///page")


# ── address checks on the real path ───────────────────────────────────────────


def test_a_host_resolving_to_a_private_address_is_refused(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.documents.fetch.socket.getaddrinfo",
        lambda host, port, **kwargs: [(2, 1, 6, "", ("169.254.169.254", port))],
    )

    with pytest.raises(UnsafeUrlError, match="non-public address"):
        assert_fetchable("https://metadata.example.com/latest/meta-data/")


def test_one_private_answer_among_several_rejects_the_whole_host(monkeypatch) -> None:
    """Otherwise the guard depends on which address the resolver happens to
    return first."""
    monkeypatch.setattr(
        "src.documents.fetch.socket.getaddrinfo",
        lambda host, port, **kwargs: [
            (2, 1, 6, "", ("93.184.216.34", port)),
            (2, 1, 6, "", ("10.0.0.5", port)),
        ],
    )

    with pytest.raises(UnsafeUrlError, match="10.0.0.5"):
        assert_fetchable("https://split-horizon.example.com/")


def test_a_host_that_does_not_resolve_is_a_fetch_error_not_a_policy_error(monkeypatch) -> None:
    import socket as socket_module

    def boom(host, port, **kwargs):
        raise socket_module.gaierror("nodename nor servname provided")

    monkeypatch.setattr("src.documents.fetch.socket.getaddrinfo", boom)

    with pytest.raises(DocumentFetchError, match="could not resolve"):
        assert_fetchable("https://nope.example.com/")


# ── redirects ─────────────────────────────────────────────────────────────────


def test_a_redirect_to_a_private_address_is_refused(monkeypatch) -> None:
    """The standard bypass: a public URL that 302s to instance metadata."""

    def resolve(host, port, **kwargs):
        address = "169.254.169.254" if host == "metadata.example.com" else "93.184.216.34"
        return [(2, 1, 6, "", (address, port))]

    monkeypatch.setattr("src.documents.fetch.socket.getaddrinfo", resolve)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "http://metadata.example.com/latest/"})

    with _client(handler) as client:
        with pytest.raises(UnsafeUrlError, match="non-public address"):
            fetch_document("https://example.com/start", client=client)


def test_a_redirect_to_a_public_page_is_followed_and_recorded() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/start":
            return httpx.Response(302, headers={"location": "https://example.com/final"})
        return _html()

    with _client(handler) as client:
        page = fetch_document("https://example.com/start", client=client)

    assert page.requested_url == "https://example.com/start"
    assert page.url == "https://example.com/final"
    assert page.redirect_chain == ["https://example.com/start", "https://example.com/final"]


def test_a_relative_redirect_is_validated_as_the_url_it_resolves_to() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/start":
            return httpx.Response(302, headers={"location": "/moved"})
        return _html()

    with _client(handler) as client:
        page = fetch_document("https://example.com/start", client=client)

    assert page.url == "https://example.com/moved"


def test_a_redirect_loop_gives_up_rather_than_spinning() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "https://example.com/round"})

    with _client(handler) as client:
        with pytest.raises(DocumentFetchError, match="redirected more than"):
            fetch_document("https://example.com/round", client=client, max_redirects=2)


def test_a_redirect_without_a_location_is_an_error_not_a_hang() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302)

    with _client(handler) as client:
        with pytest.raises(DocumentFetchError, match="no location header"):
            fetch_document("https://example.com/x", client=client)


# ── content type and size ─────────────────────────────────────────────────────


@pytest.mark.parametrize("content_type", ["application/pdf", "image/png", "application/zip"])
def test_non_text_responses_are_refused(content_type: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"binary", headers={"content-type": content_type})

    with _client(handler) as client:
        with pytest.raises(UnsafeUrlError, match="cannot be reviewed"):
            fetch_document("https://example.com/file", client=client)


def test_a_charset_parameter_does_not_defeat_the_content_type_check() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, text="hi", headers={"content-type": "TEXT/HTML; charset=ISO-8859-1"}
        )

    with _client(handler) as client:
        assert fetch_document("https://example.com/x", client=client).body == "hi"


def test_an_oversized_body_is_cut_and_says_so() -> None:
    """A review of half a page must not be presented as a review of the page."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="x" * 5000, headers={"content-type": "text/html"})

    with _client(handler) as client:
        page = fetch_document("https://example.com/big", client=client, max_bytes=1000)

    assert page.truncated is True
    assert len(page.body) == 1000


def test_a_body_within_the_cap_is_not_marked_truncated() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="short", headers={"content-type": "text/html"})

    with _client(handler) as client:
        page = fetch_document("https://example.com/small", client=client, max_bytes=1000)

    assert page.truncated is False
    assert page.body == "short"


# ── failures ──────────────────────────────────────────────────────────────────


def test_an_http_error_status_is_reported_with_the_url() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="nope", headers={"content-type": "text/html"})

    with _client(handler) as client:
        with pytest.raises(DocumentFetchError, match="HTTP 404"):
            fetch_document("https://example.com/missing", client=client)


def test_a_transport_failure_is_a_fetch_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    with _client(handler) as client:
        with pytest.raises(DocumentFetchError, match="could not fetch"):
            fetch_document("https://example.com/down", client=client)


def test_no_cookies_or_auth_headers_are_sent() -> None:
    """Nothing derived from the user's session may leave with a fetch."""
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(request.headers)
        return _html()

    with _client(handler) as client:
        fetch_document("https://example.com/x", client=client)

    assert "cookie" not in seen
    assert "authorization" not in seen

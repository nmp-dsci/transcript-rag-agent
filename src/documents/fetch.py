"""Fetching a user-supplied URL without turning the server into a proxy.

The chat lets anyone paste a link and have the server retrieve it. That is a
server-side request forgery primitive unless it is bounded, because the server
can reach things the user cannot: cloud instance-metadata endpoints
(``169.254.169.254``), databases on ``localhost``, and anything else on the
private network the process happens to sit in.

So every fetch is bounded on five axes, and each one is enforced here rather
than trusted to the caller:

* **Scheme** — ``http``/``https`` only. ``file://`` reads the disk and
  ``gopher://`` has been used to smuggle protocol payloads.
* **Address** — the hostname is resolved and *every* address it resolves to
  must be globally routable. One private answer among several is a rejection,
  not a reason to try the others.
* **Redirects** — followed manually, with the same address check on each hop.
  A public URL that 302s to ``http://169.254.169.254/`` is the standard bypass,
  and it only fails if each hop is re-validated.
* **Size** — the body is streamed and cut at a byte cap, so a multi-gigabyte
  response cannot exhaust memory.
* **Content type** — text only. The point is to read a page, and refusing
  binaries early avoids downloading them at all.

**Known residual risk, stated rather than papered over:** validation resolves
the hostname and the connection then resolves it again, so a DNS entry that
changes between the two (a rebinding attack) is not prevented. Closing it means
pinning the connection to the validated IP and carrying the ``Host`` header,
which httpx does not expose cleanly. The exposure is a same-process fetch of a
private address, which is the same thing the address check blocks in the common
case; if this ever runs anywhere untrusted, that is the gap to close first.
"""

from __future__ import annotations

import ipaddress
import socket
from typing import Any
from urllib.parse import urlparse, urlunparse

from src.documents.models import FetchedPage

ALLOWED_SCHEMES = ("http", "https")

#: Only text is fetchable — the point is to read a page. Matched against the
#: content type with parameters (``; charset=utf-8``) stripped.
ALLOWED_CONTENT_TYPES = (
    "text/html",
    "text/plain",
    "application/xhtml+xml",
)

DEFAULT_MAX_BYTES = 2_000_000
DEFAULT_TIMEOUT_SECONDS = 15.0
DEFAULT_MAX_REDIRECTS = 5

#: Sent so operators can identify the traffic. No cookies, no auth headers, and
#: nothing derived from the user's session ever goes out with a fetch.
DEFAULT_HEADERS = {
    "User-Agent": "yt-agent-doc-review/1.0 (+https://github.com/nmp-dsci/transcript-rag-agent)",
    "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9",
}


class UnsafeUrlError(ValueError):
    """The URL is refused by policy — scheme, address, redirect or content type.

    Separate from :class:`DocumentFetchError` because the two mean different
    things to a user: this one will never succeed on retry, and the message is
    safe to show verbatim.
    """


class DocumentFetchError(RuntimeError):
    """The fetch was attempted and failed — DNS, connection, timeout, HTTP status."""


def _strip_parameters(content_type: str) -> str:
    return content_type.split(";")[0].strip().lower()


#: RFC 6052 well-known prefix: on a NAT64 network the OS translates any
#: address under this prefix to the embedded IPv4 address, so it is the same
#: bypass class as ``::ffff:0:0/96`` and must be unwrapped the same way.
_NAT64_WELL_KNOWN_PREFIX = ipaddress.IPv6Network("64:ff9b::/96")


def is_public_address(address: str) -> bool:
    """Whether an IP is globally routable, and therefore fetchable.

    Every non-global category is a rejection, not just the private ranges:
    loopback reaches the server's own services, link-local reaches cloud
    instance metadata, and reserved/multicast have no business being fetched at
    all. IPv4-mapped IPv6 addresses (``::ffff:127.0.0.1``) and addresses under
    the NAT64 well-known prefix (``64:ff9b::/96``) are unwrapped first, since
    either mapped form would otherwise pass an IPv6-only check.
    """
    try:
        parsed = ipaddress.ip_address(address)
    except ValueError:
        return False
    if isinstance(parsed, ipaddress.IPv6Address) and parsed.ipv4_mapped is not None:
        parsed = parsed.ipv4_mapped
    elif isinstance(parsed, ipaddress.IPv6Address) and parsed in _NAT64_WELL_KNOWN_PREFIX:
        parsed = ipaddress.IPv4Address(int(parsed) & 0xFFFFFFFF)
    if (
        parsed.is_private
        or parsed.is_loopback
        or parsed.is_link_local
        or parsed.is_multicast
        or parsed.is_reserved
        or parsed.is_unspecified
    ):
        return False
    return parsed.is_global


def resolve_public_host(host: str, port: int) -> list[str]:
    """Every address ``host`` resolves to, or raise if any is not public.

    Rejecting when *any* answer is private is deliberate. A host that resolves
    to both a public and a private address would otherwise be fetchable
    whenever the resolver happened to order them favourably, which makes the
    guard depend on luck.
    """
    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise DocumentFetchError(f"could not resolve {host!r}: {exc}") from exc

    addresses = list(dict.fromkeys(str(info[4][0]) for info in infos))
    if not addresses:
        raise DocumentFetchError(f"could not resolve {host!r}")
    blocked = [address for address in addresses if not is_public_address(address)]
    if blocked:
        raise UnsafeUrlError(
            f"{host!r} resolves to a non-public address ({', '.join(blocked)}); "
            "only publicly routable hosts can be fetched"
        )
    return addresses


def assert_fetchable(url: str) -> str:
    """Validate one URL and return it normalized, or raise :class:`UnsafeUrlError`.

    Runs before every request *and* before every redirect hop, which is the
    whole point — a first-hop check alone is trivially bypassed by a redirect.
    """
    parsed = urlparse(url)
    if parsed.scheme.lower() not in ALLOWED_SCHEMES:
        raise UnsafeUrlError(
            f"{parsed.scheme or 'that'!r} URLs cannot be fetched; "
            f"use one of: {', '.join(ALLOWED_SCHEMES)}"
        )
    if parsed.username or parsed.password:
        # Credentials in the URL would be forwarded to whatever it redirects
        # to, so they are refused rather than stripped and silently ignored.
        raise UnsafeUrlError("URLs with embedded credentials cannot be fetched")
    host = parsed.hostname
    if not host:
        raise UnsafeUrlError(f"{url!r} has no host to fetch")
    port = parsed.port or (443 if parsed.scheme.lower() == "https" else 80)
    resolve_public_host(host, port)
    return urlunparse(parsed)


def fetch_document(
    url: str,
    *,
    client: Any | None = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    max_redirects: int = DEFAULT_MAX_REDIRECTS,
) -> FetchedPage:
    """Fetch one URL under every bound in this module's docstring.

    ``client`` accepts a pre-built ``httpx.Client`` so tests can drive this with
    a transport stub; when omitted a client is created and closed per fetch.
    Redirects are followed here rather than by httpx precisely so each hop can
    be re-validated.
    """
    import httpx

    owned = client is None
    http = client or httpx.Client(
        timeout=timeout_seconds, follow_redirects=False, headers=DEFAULT_HEADERS
    )
    try:
        return _fetch(http, url, max_bytes=max_bytes, max_redirects=max_redirects)
    finally:
        if owned:
            http.close()


def _fetch(http: Any, url: str, *, max_bytes: int, max_redirects: int) -> FetchedPage:
    import httpx

    requested_url = url
    chain: list[str] = []
    current = assert_fetchable(url)

    for _hop in range(max_redirects + 1):
        chain.append(current)
        try:
            with http.stream("GET", current, follow_redirects=False) as response:
                if response.is_redirect:
                    location = response.headers.get("location")
                    if not location:
                        raise DocumentFetchError(
                            f"{current} returned {response.status_code} with no location header"
                        )
                    # Resolved against the current URL so a relative redirect
                    # is validated as the absolute URL it will actually reach.
                    current = assert_fetchable(str(response.url.join(location)))
                    continue
                if response.status_code >= 400:
                    raise DocumentFetchError(f"{current} returned HTTP {response.status_code}")
                content_type = _strip_parameters(response.headers.get("content-type", "text/html"))
                if content_type not in ALLOWED_CONTENT_TYPES:
                    raise UnsafeUrlError(
                        f"{current} is {content_type}, which cannot be reviewed; "
                        f"fetchable types are: {', '.join(ALLOWED_CONTENT_TYPES)}"
                    )
                body, truncated = _read_capped(response, max_bytes)
                return FetchedPage(
                    requested_url=requested_url,
                    url=str(response.url),
                    status_code=response.status_code,
                    content_type=content_type,
                    body=body,
                    truncated=truncated,
                    redirect_chain=chain,
                )
        except httpx.HTTPError as exc:
            raise DocumentFetchError(f"could not fetch {current}: {exc}") from exc

    raise DocumentFetchError(
        f"{requested_url} redirected more than {max_redirects} times; giving up"
    )


def _read_capped(response: Any, max_bytes: int) -> tuple[str, bool]:
    """Stream the body up to ``max_bytes``, reporting whether it was cut.

    Streamed rather than read whole because the cap has to bound *memory*, not
    just the returned string — reading first and truncating after would already
    have allocated whatever the server chose to send.
    """
    chunks: list[bytes] = []
    total = 0
    truncated = False
    for chunk in response.iter_bytes():
        remaining = max_bytes - total
        if len(chunk) >= remaining:
            chunks.append(chunk[:remaining])
            truncated = True
            break
        chunks.append(chunk)
        total += len(chunk)
    raw = b"".join(chunks)
    encoding = response.encoding or "utf-8"
    try:
        return raw.decode(encoding, errors="replace"), truncated
    except LookupError:
        # An unknown charset in the header must not lose the page.
        return raw.decode("utf-8", errors="replace"), truncated

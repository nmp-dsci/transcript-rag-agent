"""Parsing the three XML shapes a text channel arrives in.

RSS, Atom and sitemaps are all XML, so one module handles all three and the
project gains no dependency — ``xml.etree`` is in the stdlib and these formats
need no more than element walking.

**Two things this module is careful about, because both were measured.**

*Order is not chronology.* Feeds are not reliably sorted. A real channel probed
while writing this returned its newest entry second and a three-year-old entry
first. Every parser here therefore sorts by the dates it found and never trusts
position.

*A feed may carry the whole article.* Two of the registered sources put the
full post in ``<description>`` or ``<content:encoded>`` — 27 KB and 30 KB per
item — while four ship a 21-to-1,157-character teaser. The difference decides
whether the page needs fetching at all, and on one source the feed body is not
merely cheaper but strictly better: its page extracts to a subscribe prompt.
:func:`looks_like_body` is the test, and it is a length threshold rather than a
per-site rule.

**Entity expansion.** ``xml.etree`` resolves internal entity definitions, so a
small document can expand to a large one. Legitimate feeds carry no DOCTYPE, so
one is refused outright here rather than handed to the parser; combined with the
fetch layer's byte cap that closes the practical exposure without adding a
dependency.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from xml.etree import ElementTree

logger = logging.getLogger(__name__)

ATOM_NS = "http://www.w3.org/2005/Atom"
CONTENT_NS = "http://purl.org/rss/1.0/modules/content/"
SITEMAP_NS = "http://www.sitemaps.org/schemas/sitemap/0.9"

#: Above this many characters an item body is the article; below it, a teaser.
#: Chosen from the measured split: full-text feeds ship 27,000+ characters per
#: item and teaser feeds ship under 1,200, so anywhere in between separates
#: them and nothing sits near the line.
BODY_CHAR_THRESHOLD = 2000

_DOCTYPE = re.compile(r"<!DOCTYPE", re.IGNORECASE)


class FeedParseError(ValueError):
    """The body is not XML this module can read."""


@dataclass(frozen=True)
class FeedItem:
    """One entry from a feed or one URL from a sitemap."""

    url: str
    external_id: str
    title: str | None = None
    published_at: str | None = None
    updated_at: str | None = None
    body_html: str | None = None

    @property
    def sort_key(self) -> str:
        """Newest-first ordering key. Undated items sort last, not first."""
        return self.updated_at or self.published_at or ""


def looks_like_body(text: str | None) -> bool:
    """Whether a feed item's body is the article rather than a teaser."""
    return bool(text) and len(text or "") >= BODY_CHAR_THRESHOLD


def _localname(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def parse_xml(body: str, truncated: bool = False) -> ElementTree.Element:
    """Parse XML, refusing a DOCTYPE. See the module docstring.

    ``truncated`` turns the resulting parse failure into an honest one. A feed
    cut at the byte cap fails deep inside the document — measured, at an
    "unclosed CDATA section" — and that message sends whoever reads it looking
    for a malformed feed instead of a cap that needs raising.
    """
    if _DOCTYPE.search(body[:4096]):
        raise FeedParseError("refusing XML with a DOCTYPE declaration")
    try:
        return ElementTree.fromstring(body)
    except ElementTree.ParseError as exc:
        if truncated:
            raise FeedParseError(
                f"feed was cut off at the byte cap after {len(body)} characters, "
                f"so it cannot be parsed; raise FEED_MAX_BYTES for this source"
            ) from exc
        raise FeedParseError(f"not parseable as XML: {exc}") from exc


def _text(element: ElementTree.Element | None) -> str | None:
    if element is None:
        return None
    value = "".join(element.itertext()).strip()
    return value or None


def _find(parent: ElementTree.Element, *names: str) -> ElementTree.Element | None:
    """The first child matching any of ``names``, ignoring namespaces."""
    wanted = set(names)
    for child in parent:
        if _localname(child.tag) in wanted:
            return child
    return None


def _find_all(parent: ElementTree.Element, name: str) -> list[ElementTree.Element]:
    return [child for child in parent.iter() if _localname(child.tag) == name]


def normalise_date(value: str | None) -> str | None:
    """A UTC ISO-8601 string, from either of the two formats feeds use.

    RSS dates are RFC 822 (``Tue, 21 Sep 2026 19:00:10 +0000``) and Atom dates
    are ISO 8601; normalising both here means every downstream comparison is a
    string comparison on the same shape.
    """
    if not value:
        return None
    raw = value.strip()
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        try:
            parsed = parsedate_to_datetime(raw)
        except (TypeError, ValueError):
            return None
    if parsed is None:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat()


def _rss_items(root: ElementTree.Element) -> list[FeedItem]:
    items: list[FeedItem] = []
    for node in _find_all(root, "item"):
        link = _text(_find(node, "link"))
        guid = _text(_find(node, "guid"))
        url = link or guid
        if not url or not url.startswith(("http://", "https://")):
            continue
        # content:encoded wins over description: where both exist, the former
        # is the article and the latter its summary.
        encoded = None
        for child in node:
            if child.tag == f"{{{CONTENT_NS}}}encoded":
                encoded = _text(child)
                break
        body = encoded or _text(_find(node, "description"))
        published = normalise_date(_text(_find(node, "pubDate")))
        items.append(
            FeedItem(
                url=url,
                # The guid is identity; the URL is only where it lives today.
                external_id=guid or url,
                title=_text(_find(node, "title")),
                published_at=published,
                updated_at=published,
                body_html=body if looks_like_body(body) else None,
            )
        )
    return items


def _atom_entries(root: ElementTree.Element) -> list[FeedItem]:
    items: list[FeedItem] = []
    for node in _find_all(root, "entry"):
        url = None
        for child in node:
            if _localname(child.tag) != "link":
                continue
            rel = child.attrib.get("rel", "alternate")
            href = child.attrib.get("href")
            if href and rel == "alternate":
                url = href
                break
            if href and url is None:
                url = href
        identifier = _text(_find(node, "id"))
        url = url or (identifier if (identifier or "").startswith("http") else None)
        if not url:
            continue
        body = _text(_find(node, "content")) or _text(_find(node, "summary"))
        published = normalise_date(_text(_find(node, "published")))
        updated = normalise_date(_text(_find(node, "updated")))
        items.append(
            FeedItem(
                url=url,
                external_id=identifier or url,
                title=_text(_find(node, "title")),
                published_at=published,
                updated_at=updated or published,
                body_html=body if looks_like_body(body) else None,
            )
        )
    return items


def parse_feed(body: str, truncated: bool = False) -> list[FeedItem]:
    """Every item or entry in an RSS or Atom document, newest first.

    Accepts either format without being told which: the two are distinguished
    by the elements present, and a caller that had to know would need a second
    request just to find out.
    """
    root = parse_xml(body, truncated)
    items = _rss_items(root) + _atom_entries(root)
    if not items and _localname(root.tag) not in {"rss", "feed", "RDF"}:
        raise FeedParseError(f"<{_localname(root.tag)}> is not a feed")
    items.sort(key=lambda item: item.sort_key, reverse=True)
    return items


@dataclass(frozen=True)
class Sitemap:
    """A parsed sitemap: either URLs, or the sitemaps that hold them."""

    urls: tuple[FeedItem, ...] = ()
    #: Set instead of :attr:`urls` when this was a ``<sitemapindex>``. Followed
    #: one level by the poller — one registered source needs exactly that hop.
    children: tuple[str, ...] = ()

    @property
    def is_index(self) -> bool:
        return bool(self.children) and not self.urls


def parse_sitemap(body: str, truncated: bool = False) -> Sitemap:
    """Parse a sitemap or a sitemap index.

    ``<lastmod>`` is carried through as the change signal where it exists. It
    often does not: of two registered sitemaps, one dates every one of its 535
    URLs and the other dates none of its 192, which is why the content hash
    rather than the date is the floor.
    """
    root = parse_xml(body, truncated)
    name = _localname(root.tag)
    if name == "sitemapindex":
        children = [
            text for node in _find_all(root, "sitemap") if (text := _text(_find(node, "loc")))
        ]
        return Sitemap(children=tuple(children))
    if name != "urlset":
        raise FeedParseError(f"<{name}> is not a sitemap")
    urls: list[FeedItem] = []
    for node in _find_all(root, "url"):
        loc = _text(_find(node, "loc"))
        if not loc:
            continue
        lastmod = normalise_date(_text(_find(node, "lastmod")))
        urls.append(FeedItem(url=loc, external_id=loc, updated_at=lastmod))
    urls.sort(key=lambda item: item.sort_key, reverse=True)
    return Sitemap(urls=tuple(urls))

"""Turning a fetched page into headed sections of readable text.

The chat reviews *extracted text*, never the live page. Rendering the page
itself would mean an iframe, which most real sites refuse via
``X-Frame-Options``/CSP, or a screenshot, which needs a headless browser —
and neither is what retrieval reads anyway. Extracted text is what the model
sees, so it is what the user should see too.

Sections rather than one blob, because a section is the unit of both retrieval
and citation: it is what lets feedback say "the experience section" and what
gives the ranker something smaller than the whole document to score. A heading
starts a new section; the text before the first heading is a section of its own,
which is where a document with no headings at all ends up.

Boilerplate handling is deliberately conservative. ``script``/``style`` and
friends are dropped because they are never content, but ``nav``/``header``/
``footer`` are **kept**: on the documents this feature exists for — a resume, a
personal site, an invitation — the name and contact details are very often
inside ``<header>``, and a reader-mode heuristic that strips them would delete
exactly the part a reviewer is asked about.

Stdlib ``html.parser`` rather than BeautifulSoup or readability: the job is
"headings and text runs", the parser is forgiving of malformed markup, and it
adds no dependency to a project that has deliberately few.
"""

from __future__ import annotations

import hashlib
import re
from html.parser import HTMLParser

from src.documents.models import Document, DocumentSection, FetchedPage

#: Tags whose contents are never document text. ``head`` is deliberately absent:
#: skipping it would take ``<title>`` with it, and everything in there that is
#: not the title is either skipped by name (``script``/``style``) or a void
#: element with no text of its own.
_SKIPPED_TAGS = frozenset({"script", "style", "noscript", "template", "svg", "canvas", "iframe"})

#: Tags that end the current run of text. Anything block-level, so words from
#: two paragraphs never run together into one sentence.
_BLOCK_TAGS = frozenset(
    {
        "p", "div", "section", "article", "main", "aside", "nav", "header", "footer",
        "li", "ul", "ol", "dl", "dt", "dd", "tr", "td", "th", "table",
        "blockquote", "pre", "figure", "figcaption", "br", "hr", "form", "label",
    }
)  # fmt: skip

_HEADING_TAGS = frozenset({"h1", "h2", "h3", "h4", "h5", "h6"})

#: Sections shorter than this are folded into the previous one. A heading with
#: two words under it is a label, not a section, and indexing it separately
#: gives the ranker units too small to be discriminative.
MIN_SECTION_CHARS = 40


class _SectionParser(HTMLParser):
    """Collects ``(heading, text)`` pairs in document order."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title: str | None = None
        self.sections: list[tuple[str | None, list[str]]] = [(None, [])]
        self._skip_depth = 0
        self._in_title = False
        self._heading_tag: str | None = None
        self._heading_parts: list[str] = []

    # ── tags ─────────────────────────────────────────────────────────────────

    def handle_starttag(self, tag: str, attrs: list) -> None:
        tag = tag.lower()
        if tag in _SKIPPED_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag == "title":
            self._in_title = True
        elif tag in _HEADING_TAGS:
            # An unclosed heading would otherwise capture the whole rest of the
            # document as heading text, leaving no body at all.
            self._flush_heading()
            self._heading_tag = tag
            self._heading_parts = []
        elif tag in _BLOCK_TAGS:
            self._flush_heading()
            self._break_text()

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in _SKIPPED_TAGS:
            # Clamped at zero so a stray closing tag cannot unbalance the
            # counter and start swallowing the rest of the document.
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if self._skip_depth:
            return
        if tag == "title":
            self._in_title = False
        elif tag in _HEADING_TAGS:
            self._flush_heading()
        elif tag in _BLOCK_TAGS:
            self._break_text()

    def close(self) -> None:
        # A document that ends mid-heading still has that heading's words.
        self._flush_heading()
        super().close()

    def handle_data(self, data: str) -> None:
        if self._skip_depth or not data.strip():
            return
        if self._in_title:
            self.title = _collapse((self.title or "") + " " + data)
        elif self._heading_tag is not None:
            self._heading_parts.append(data)
        else:
            self.sections[-1][1].append(data)

    # ── internals ────────────────────────────────────────────────────────────

    def _flush_heading(self) -> None:
        """End an open heading, starting the section it labels."""
        if self._heading_tag is None:
            return
        heading = _collapse("".join(self._heading_parts))
        self._heading_tag = None
        self._heading_parts = []
        if heading:
            self.sections.append((heading, []))

    def _break_text(self) -> None:
        """End the current text run so the next block starts a new line."""
        parts = self.sections[-1][1]
        if parts and parts[-1] != "\n":
            parts.append("\n")


def _collapse(text: str) -> str:
    return re.sub(r"[ \t ]+", " ", text).strip()


def _clean_block(parts: list[str]) -> str:
    """Join a section's raw data runs into paragraph-separated text."""
    joined = "".join(parts)
    lines = [_collapse(line) for line in joined.split("\n")]
    return "\n".join(line for line in lines if line)


def _sections_from_plain_text(body: str) -> list[tuple[str | None, str]]:
    """Blank-line-delimited blocks, since plain text carries no headings."""
    blocks = [_clean_block([block]) for block in re.split(r"\n\s*\n", body)]
    return [(None, block) for block in blocks if block]


def _merge_short_sections(
    pairs: list[tuple[str | None, str]],
) -> list[tuple[str | None, str]]:
    """Fold a too-short section into the one before it, heading included.

    The heading is not discarded — it is prepended to the text it labels — so a
    document whose structure is "heading, one line, heading, one line" keeps
    every word while still producing units worth ranking.
    """
    merged: list[tuple[str | None, str]] = []
    for heading, text in pairs:
        if not heading and not text:
            continue
        if merged and len(text) < MIN_SECTION_CHARS:
            previous_heading, previous_text = merged[-1]
            tail = f"{heading}\n{text}".strip() if heading else text
            merged[-1] = (previous_heading, f"{previous_text}\n{tail}".strip())
            continue
        merged.append((heading, text))
    return merged


def document_id_for(url: str) -> str:
    """A stable id for a URL, so the same link reuses the same document."""
    return "doc:" + hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]


def extract_document(page: FetchedPage, document_id: str | None = None) -> Document:
    """Extract a fetched page into a titled, sectioned :class:`Document`.

    ``truncated`` carries through from the fetch, because a document cut at the
    byte cap must not be reviewed as if it were complete.
    """
    if page.content_type == "text/plain":
        pairs = _sections_from_plain_text(page.body)
        title = None
    else:
        parser = _SectionParser()
        parser.feed(page.body)
        parser.close()
        pairs = [(heading, _clean_block(parts)) for heading, parts in parser.sections]
        title = parser.title

    pairs = _merge_short_sections([(heading, text) for heading, text in pairs])
    sections = [
        DocumentSection(index=index, heading=heading, text=text)
        for index, (heading, text) in enumerate(pairs)
    ]
    if title is None and sections and sections[0].heading:
        # No <title>: the first heading is the document's own name for itself.
        title = sections[0].heading

    return Document(
        id=document_id or document_id_for(page.url),
        url=page.url,
        requested_url=page.requested_url,
        title=title,
        sections=sections,
        truncated=page.truncated,
        fetched_at=page.fetched_at,
    )

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

**Two modes, because there are now two callers with opposite needs.**
``mode="resume"`` is the historical behaviour and stays the default, keeping
``nav``/``header``/``footer`` for the reason above. ``mode="article"`` drops
them, because on the practitioner blogs and repo pages the corpus watches that
markup is chrome: extracting a GitHub project page in resume mode yields
"Navigation Menu" and "Folders and files" before a word of content, and a
Substack post page yields a subscribe prompt. Measured, not assumed — the
numbers are in ``.lavish/s26_new-channels-ingestion.html``.
"""

from __future__ import annotations

import hashlib
import re
from html.parser import HTMLParser

from urllib.parse import urlparse

from src.documents.fetch import MARKDOWN_CONTENT_TYPES
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

#: Dropped in ``mode="article"``. ``nav``, ``footer`` and ``aside`` are never
#: article text on the sources this corpus watches.
_CHROME_TAGS = frozenset({"nav", "footer", "aside"})

#: Also dropped in article mode, but only outside ``article``/``main``: a
#: ``<header>`` at the top of the page is site chrome, while one *inside* the
#: article is usually the article's own title block.
_CONDITIONAL_CHROME_TAGS = frozenset({"header"})

#: Tags that mean "the article proper starts here".
_CONTENT_TAGS = frozenset({"article", "main"})

#: A leading section with one of these headings is a table of contents — a list
#: of links to the rest of the page. Indexing it produces a chunk that matches
#: every query about the document and answers none of them.
_TOC_HEADINGS = frozenset(
    {
        "contents",
        "table of contents",
        "on this page",
        "in this article",
        "in this post",
        "overview of contents",
    }
)

RESUME_MODE = "resume"
ARTICLE_MODE = "article"
EXTRACT_MODES = (RESUME_MODE, ARTICLE_MODE)

#: Sections shorter than this are folded into the previous one. A heading with
#: two words under it is a label, not a section, and indexing it separately
#: gives the ranker units too small to be discriminative.
MIN_SECTION_CHARS = 40


class _SectionParser(HTMLParser):
    """Collects ``(heading, text)`` pairs in document order."""

    def __init__(self, mode: str = RESUME_MODE) -> None:
        super().__init__(convert_charrefs=True)
        self.mode = mode
        self.title: str | None = None
        self.sections: list[tuple[str | None, list[str]]] = [(None, [])]
        self._skip_depth = 0
        self._in_title = False
        self._heading_tag: str | None = None
        self._heading_parts: list[str] = []
        #: Open ``article``/``main`` elements, so a ``<header>`` inside one is
        #: kept while the page's own banner header is dropped.
        self._content_depth = 0

    def _is_chrome(self, tag: str) -> bool:
        if self.mode != ARTICLE_MODE:
            return False
        if tag in _CHROME_TAGS:
            return True
        return tag in _CONDITIONAL_CHROME_TAGS and self._content_depth == 0

    # ── tags ─────────────────────────────────────────────────────────────────

    def handle_starttag(self, tag: str, attrs: list) -> None:
        tag = tag.lower()
        if tag in _SKIPPED_TAGS or self._is_chrome(tag):
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag in _CONTENT_TAGS:
            self._content_depth += 1
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
        if tag in _SKIPPED_TAGS or self._is_chrome(tag):
            # Clamped at zero so a stray closing tag cannot unbalance the
            # counter and start swallowing the rest of the document.
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if self._skip_depth:
            return
        if tag in _CONTENT_TAGS:
            self._content_depth = max(0, self._content_depth - 1)
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


#: ATX headings: one to six hashes, a space, then the heading text.
_ATX_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")

#: A fenced code block opener or closer. Hashes inside one are Python comments
#: or shell prompts, not headings, which is the whole reason to track fences.
_CODE_FENCE = re.compile(r"^\s*(```|~~~)")

#: Inline HTML in Markdown. READMEs routinely open with a ``<div align=center>``
#: of shield badges, and stored verbatim those tags are markup noise in the
#: corpus — they embed as text and retrieve as nothing. Stripped only outside
#: fenced code, where a ``<`` is far more likely to be code than a tag.
_INLINE_HTML = re.compile(r"</?[A-Za-z][^<>]*>", re.DOTALL)

#: A Markdown image, which contributes alt text at best and a badge URL at
#: worst. Matched before links, since an image is a link with a bang.
_MARKDOWN_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")

#: A Markdown link. The visible text is kept and the target dropped: a URL
#: embeds as a meaningless token and bloats the chunk it sits in, while the
#: link text is ordinary prose. Left as-is, a curated link list's headings
#: arrive carrying "[ ](https://awesome.re)".
_MARKDOWN_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")


def _strip_markdown_markup(text: str) -> str:
    """Reduce a run of Markdown prose to its readable text.

    Inline HTML tags and images go; links keep their text and lose their
    target. Applied to a whole run rather than line by line, because a
    README's badge block routinely breaks a single ``<img …>`` across several
    lines and a per-line pattern leaves those halves in the corpus as text.
    """
    without_tags = _INLINE_HTML.sub(" ", text)
    without_images = _MARKDOWN_IMAGE.sub(" ", without_tags)
    return _MARKDOWN_LINK.sub(r"\1", without_images)


def _sections_from_markdown(body: str) -> tuple[str | None, list[tuple[str | None, str]]]:
    """Split raw Markdown on ATX headings, respecting fenced code blocks.

    Needed because :func:`extract_document` is an *HTML* extractor, and the
    repo docs this corpus watches are fetched as raw Markdown from
    ``raw.githubusercontent.com``. Run through the HTML path, a README's
    ``## Factor 4`` arrives as ordinary text: measured on one real repo, that
    produced twenty sections and **zero headings**, which makes
    section-anchored citation impossible while looking like a clean ingest.

    Each section is assembled from alternating prose and fenced-code segments.
    Only the prose segments have markup stripped: inside a fence a ``<`` is far
    more likely to be an operator than a tag.

    Returns the document title — the first level-one heading, if any — and the
    heading/body pairs.
    """
    title: str | None = None
    # Each section is (heading, segments); each segment is (is_code, lines).
    sections: list[tuple[str | None, list[tuple[bool, list[str]]]]] = [(None, [(False, [])])]
    in_fence = False
    fence_marker = ""

    def segment(is_code: bool) -> list[str]:
        segments = sections[-1][1]
        if not segments or segments[-1][0] != is_code:
            segments.append((is_code, []))
        return segments[-1][1]

    for line in body.splitlines():
        fence = _CODE_FENCE.match(line)
        if fence is not None:
            marker = fence.group(1)
            if not in_fence:
                in_fence, fence_marker = True, marker
                segment(True).append(line)
            elif marker == fence_marker:
                segment(True).append(line)
                in_fence = False
            else:
                segment(True).append(line)
            continue
        if in_fence:
            segment(True).append(line)
            continue
        heading = _ATX_HEADING.match(line)
        if heading is None:
            segment(False).append(line)
            continue
        text = _collapse(_strip_markdown_markup(heading.group(2)))
        if title is None and len(heading.group(1)) == 1:
            title = text
        sections.append((text, [(False, [])]))

    pairs: list[tuple[str | None, str]] = []
    for heading, segments in sections:
        rendered: list[str] = []
        for is_code, lines in segments:
            joined = "\n".join(lines)
            if not is_code:
                joined = _strip_markdown_markup(joined)
            if joined.strip():
                rendered.append(joined)
        # ``_clean_block`` concatenates its parts directly, because the HTML
        # path hands it data runs that already carry their own line breaks;
        # ``splitlines()`` removed those, so they go back in here.
        pairs.append((heading, _clean_block(["\n".join(rendered)])))
    return title, pairs


def _looks_like_markdown(page: FetchedPage) -> bool:
    """Whether a text/plain body should be read as Markdown.

    Conservative on purpose: a ``.md`` URL or a declared Markdown type, never a
    guess from the body. A transcript pasted as plain text must keep behaving
    exactly as it does today.
    """
    if page.content_type in MARKDOWN_CONTENT_TYPES:
        return True
    path = urlparse(page.url).path.lower()
    return path.endswith((".md", ".markdown", ".mdx"))


def _without_leading_toc(
    pairs: list[tuple[str | None, str]],
) -> list[tuple[str | None, str]]:
    """Drop a leading table-of-contents section.

    Only a *leading* one, and only when something follows it: a page whose
    entire content is a contents list is a legitimate index, and deleting its
    one section would leave an empty document rather than a cleaner one.
    """
    for index, (heading, _text) in enumerate(pairs):
        if heading and heading.strip().lower() in _TOC_HEADINGS:
            if index + 1 < len(pairs):
                return pairs[:index] + pairs[index + 1 :]
            return pairs
        if index >= 2:
            break
    return pairs


def document_id_for(url: str) -> str:
    """A stable id for a URL, so the same link reuses the same document.

    Kept for the chat's paste-a-link path, where a URL genuinely is the
    identity of the thing being reviewed. The corpus does **not** use it: a
    watched source is keyed on the id its channel gave it (a feed guid, a
    sitemap loc), because a post that moves keeps its guid while this hash
    would mint a second document and orphan the first.
    """
    return "doc:" + hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]


def extract_document(
    page: FetchedPage,
    document_id: str | None = None,
    mode: str = RESUME_MODE,
) -> Document:
    """Extract a fetched page into a titled, sectioned :class:`Document`.

    ``truncated`` carries through from the fetch, because a document cut at the
    byte cap must not be reviewed as if it were complete.

    ``mode`` selects the boilerplate policy — see the module docstring. It
    defaults to ``"resume"``, so the chat's review path is byte-for-byte what
    it was.
    """
    if mode not in EXTRACT_MODES:
        raise ValueError(f"mode must be one of {', '.join(EXTRACT_MODES)}, got {mode!r}")
    if _looks_like_markdown(page):
        title, pairs = _sections_from_markdown(page.body)
    elif page.content_type == "text/plain":
        pairs = _sections_from_plain_text(page.body)
        title = None
    else:
        parser = _SectionParser(mode=mode)
        parser.feed(page.body)
        parser.close()
        pairs = [(heading, _clean_block(parts)) for heading, parts in parser.sections]
        title = parser.title

    if mode == ARTICLE_MODE:
        pairs = _without_leading_toc(pairs)
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

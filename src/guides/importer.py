"""Bring a hand-made guide page (a Lavish artifact) under ``guides/`` as v1.

Ship Like a Studio was written before the pipeline existed. Rather than
regenerate it — and lose the page that was approved — the importer rewrites
the existing HTML into the guide contract:

* the inline stylesheet becomes the shared ``guides/guide.css`` (so later
  guides inherit the same editorial system) and the page links it;
* every section, heading, paragraph, list item, quote and code block gets a
  stable, document-order ``id`` so comments can anchor to it;
* every YouTube link becomes a ``<cite data-video="…">`` around its text, so
  the verifier can resolve it — to video level, since the original page does
  not record chunk ids;
* the page is wrapped as a full HTML document and gains the reader bridge
  script, then is snapshotted as ``versions/v1.html``.

The rewrite is a token-preserving pass over the original markup: text,
entities, comments and attribute spelling come out exactly as they went in,
except where a rule above adds something.
"""

from __future__ import annotations

import datetime as dt
import json
import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from src.guides.catalog import (
    GuidePaths,
    Manifest,
    guide_paths,
    write_index,
    write_manifest,
)
from src.guides.verify import ChunkLookup, verify_html, write_claims

YOUTUBE_ID = re.compile(r"(?:youtube\.com/watch\?(?:[^\"'#]*&)?v=|youtu\.be/)([A-Za-z0-9_-]{6,})")
YOUTUBE_T = re.compile(r"[?&]t=(\d+)")

ID_TAGS = {"h2", "h3", "h4", "p", "li", "blockquote", "pre", "figure", "table"}

GUIDE_CSS_HREF = "/guides/guide.css"
BRIDGE_SRC = "/guides/guide-bridge.js"

#: Appended to the shared stylesheet: what the reader bridge needs the page to
#: be able to show (a highlighted anchor when a comment is selected) and the
#: cite element itself, which the original page never used.
READER_CSS = """
/* ---------- reader additions (workbench bridge) ---------- */
cite { font-style: inherit; }
cite[data-video] { font-family: var(--mono); font-size: 0.72em; color: var(--accent); vertical-align: 0.25em;
  margin-left: 0.15em; white-space: nowrap; }
cite[data-video]::before { content: "["; opacity: 0.6; }
cite[data-video]::after { content: "]"; opacity: 0.6; }
a > cite[data-video], li > a cite[data-video] { font-family: inherit; font-size: inherit; color: inherit; vertical-align: baseline; margin: 0; white-space: normal; }
a > cite[data-video]::before, a > cite[data-video]::after { content: none; }
.guide-hl { outline: 2px solid var(--accent, #2434C9); outline-offset: 4px; border-radius: 4px;
  transition: outline-color 300ms ease; }
/* Long section lists (agent-written guides run to ten) wrap instead of scrolling. */
nav.site .bar { flex-wrap: wrap; row-gap: 4px; overflow-x: visible; }
nav.site .links { flex-wrap: wrap; row-gap: 2px; gap: 6px 18px; justify-content: flex-end; }
"""


def video_id_from_url(url: str) -> str | None:
    match = YOUTUBE_ID.search(url)
    return match.group(1) if match else None


class _Rewriter(HTMLParser):
    """Re-emit the document, applying the import rules. See module docstring."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self.out: list[str] = []
        self.style_blocks: list[str] = []
        self._in_style = False
        self._style_buf: list[str] = []
        self._section: str = "top"
        self._counts: dict[str, int] = {}
        self._cite_depth: list[bool] = []  # per open <a>: did we open a cite?
        self.title: str | None = None
        self._in_title = False
        self._title_buf: list[str] = []

    # -- helpers -----------------------------------------------------------
    def _next_id(self, tag: str) -> str:
        key = f"{self._section}-{tag}"
        self._counts[key] = self._counts.get(key, 0) + 1
        return f"{key}{self._counts[key]}"

    @staticmethod
    def _add_attr(raw: str, name: str, value: str) -> str:
        """Insert ``name="value"`` before the closing ``>`` of a raw start tag."""
        end = raw.rfind(">")
        body = raw[:end].rstrip()
        selfclose = body.endswith("/")
        if selfclose:
            body = body[:-1].rstrip()
        return f'{body} {name}="{value}"{" /" if selfclose else ""}>'

    # -- parser callbacks ------------------------------------------------
    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        raw = self.get_starttag_text() or ""
        attributes = {key: value for key, value in attrs}
        if tag == "style":
            self._in_style = True
            self._style_buf = []
            return
        if tag == "title":
            self._in_title = True
            self._title_buf = []
        if tag in ("section", "header", "footer", "nav", "main"):
            existing_id = attributes.get("id")
            if existing_id:
                self._section = existing_id
            elif tag == "section":
                self._section = self._next_id("section")
                raw = self._add_attr(raw, "id", self._section)
        if tag in ID_TAGS and not attributes.get("id"):
            raw = self._add_attr(raw, "id", self._next_id(tag))
        self.out.append(raw)
        if tag == "a":
            href = attributes.get("href") or ""
            video_id = video_id_from_url(href)
            if video_id:
                seconds = YOUTUBE_T.search(href)
                extra = f' data-t="{seconds.group(1)}"' if seconds else ""
                self.out.append(f'<cite data-video="{video_id}"{extra}>')
                self._cite_depth.append(True)
            else:
                self._cite_depth.append(False)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.out.append(self.get_starttag_text() or "")

    def handle_endtag(self, tag: str) -> None:
        if tag in ("section", "header", "footer", "nav"):
            self._section = "top"
        if tag == "style" and self._in_style:
            self._in_style = False
            self.style_blocks.append("".join(self._style_buf))
            return
        if tag == "title":
            self._in_title = False
            self.title = "".join(self._title_buf).strip()
        if tag == "a" and self._cite_depth:
            if self._cite_depth.pop():
                self.out.append("</cite>")
        self.out.append(f"</{tag}>")

    def handle_data(self, data: str) -> None:
        if self._in_style:
            self._style_buf.append(data)
            return
        if self._in_title:
            self._title_buf.append(data)
        self.out.append(data)

    def handle_entityref(self, name: str) -> None:
        self._emit_ref(f"&{name};")

    def handle_charref(self, name: str) -> None:
        self._emit_ref(f"&#{name};")

    def _emit_ref(self, text: str) -> None:
        if self._in_style:
            self._style_buf.append(text)
            return
        if self._in_title:
            self._title_buf.append(text)
        self.out.append(text)

    def handle_comment(self, data: str) -> None:
        self.out.append(f"<!--{data}-->")

    def handle_decl(self, decl: str) -> None:
        self.out.append(f"<!{decl}>")

    def handle_pi(self, data: str) -> None:
        self.out.append(f"<?{data}>")

    def unknown_decl(self, data: str) -> None:
        self.out.append(f"<![{data}]>")


_SOURCE_LI = re.compile(
    r"<li[^>]*>\s*<a[^>]*href=\"([^\"]+)\"[^>]*>(.*?)</a>\s*(?:<span class=\"tk\">(.*?)</span>)?",
    re.S,
)
_TAG = re.compile(r"<[^>]+>")


def _text(fragment: str) -> str:
    import html as html_lib

    return " ".join(html_lib.unescape(_TAG.sub("", fragment)).split())


def extract_sources(html: str) -> list[dict[str, Any]]:
    """The Sources appendix as records: video id, title, and the line it contributed."""
    sources: list[dict[str, Any]] = []
    seen: set[str] = set()
    section = html
    match = re.search(r'<section id="sources".*?</section>', html, re.S)
    if match:
        section = match.group(0)
    for href, title, contributed in _SOURCE_LI.findall(section):
        video_id = video_id_from_url(href)
        if not video_id or video_id in seen:
            continue
        seen.add(video_id)
        sources.append(
            {
                "video_id": video_id,
                "title": _text(title),
                "contributed": _text(contributed).lstrip("—- ").strip() if contributed else "",
                "url": f"https://www.youtube.com/watch?v={video_id}",
            }
        )
    return sources


def extract_subtitle(html: str) -> str:
    match = re.search(r'<p class="sub">(.*?)</p>', html, re.S)
    return _text(match.group(1)) if match else ""


def strip_wrapper(html: str) -> str:
    """Drop any doctype/html/head/body wrapper so the rewrite works on the content."""
    body = re.sub(r"(?is)^.*?<body[^>]*>", "", html, count=1) if "<body" in html.lower() else html
    body = re.sub(r"(?is)</body>\s*</html>\s*$", "", body)
    return body


def rewrite(html: str) -> tuple[str, str, str | None]:
    """Return ``(page_html, css, title)`` for the guide contract."""
    parser = _Rewriter()
    parser.feed(strip_wrapper(html))
    parser.close()
    content = "".join(parser.out)
    css = "\n".join(block.strip("\n") for block in parser.style_blocks)
    # Head lines (title, font links) in the source stay in the head; the rest is body.
    head_parts: list[str] = []

    def take_head(pattern: str) -> None:
        nonlocal content
        for found in re.findall(pattern, content, flags=re.I | re.S):
            head_parts.append(found.strip())
        content = re.sub(pattern, "", content, flags=re.I | re.S)

    take_head(r"<title>.*?</title>")
    take_head(r"<link[^>]*>")
    take_head(r"<meta[^>]*>")
    head = "\n".join(head_parts)
    page = (
        '<!doctype html>\n<html lang="en">\n<head>\n'
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"{head}\n"
        f'<link rel="stylesheet" href="{GUIDE_CSS_HREF}">\n'
        "</head>\n<body>\n"
        f"{content.strip()}\n"
        f'<script src="{BRIDGE_SRC}"></script>\n'
        "</body>\n</html>\n"
    )
    return page, css, parser.title


def import_lavish_html(
    source: Path,
    *,
    slug: str,
    guides_dir: Path | None = None,
    topic: str | None = None,
    title: str | None = None,
    known_videos: set[str] | None = None,
    chunk_lookup: ChunkLookup | None = None,
    chunk_counts: dict[str, int] | None = None,
    compiled_at: str | None = None,
    write_css: bool = True,
) -> Manifest:
    """Import ``source`` as ``guides/<slug>/`` v1. Returns the written manifest."""
    paths: GuidePaths = guide_paths(slug, guides_dir)
    original = source.read_text(encoding="utf-8")
    page, css, found_title = rewrite(original)
    sources = extract_sources(original)
    video_ids = [item["video_id"] for item in sources]
    known = known_videos if known_videos is not None else set(video_ids)

    paths.dir.mkdir(parents=True, exist_ok=True)
    paths.versions.mkdir(parents=True, exist_ok=True)
    if write_css and css:
        (paths.root / "guide.css").write_text(css + "\n" + READER_CSS, encoding="utf-8")
    paths.html.write_text(page, encoding="utf-8")
    paths.version_html(1).write_text(page, encoding="utf-8")

    report = verify_html(page, known_videos=known, chunk_lookup=chunk_lookup)
    write_claims(paths.claims, report)

    manifest = Manifest(
        slug=slug,
        title=title or found_title or slug,
        topic=topic or title or found_title or slug,
        subtitle=extract_subtitle(original),
        status="published",
        current_version=1,
        compiled_at=compiled_at or dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d"),
        model={"composer": "hand-run (imported)"},
        video_ids=video_ids,
        sources=sources,
        chunk_count=sum((chunk_counts or {}).get(video_id, 0) for video_id in video_ids),
        cite_total=report.total,
        cite_valid=report.valid,
        provenance={
            "imported_from": str(source),
            "imported_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "cite_level": "video",
            "sections": report.sections,
        },
    )
    write_manifest(paths, manifest)
    write_index(paths.root)
    return manifest


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))

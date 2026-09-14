"""Check a guide page's citations against the corpus, with no LLM in the loop.

Every sourced claim in a guide is a ``<cite data-video="…" data-chunk="…">``
element. This module resolves each one: the video must be indexed, the chunk
(when given) must exist for that video, and the quote (when given) must occur
in that chunk's text. The result is ``claims.json`` and a pass rate the
catalog card shows — so "every claim traces to a source" is a number the
reader can check rather than a sentence the page asserts.

It also enforces the two structural rules the reader relies on: every section
carries an ``id`` (comments anchor to ids), and the page loads no script from
anywhere but ``/guides/`` and carries no inline event handlers (the page is
agent-written and renders inside the workbench).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable

#: ``chunk_lookup(video_id) -> list[dict]`` with ``chunk_index`` and ``text`` keys,
#: the shape :func:`src.api.corpus.load_chunk_corpus` returns.
ChunkLookup = Callable[[str], list[dict[str, Any]]]

ALLOWED_SCRIPT_PREFIX = "/guides/"


@dataclass
class Claim:
    cite_index: int
    section_id: str | None
    video_id: str
    chunk_index: int | None
    quote: str | None
    text: str
    video_ok: bool = False
    chunk_ok: bool | None = None
    quote_ok: bool | None = None

    @property
    def valid(self) -> bool:
        if not self.video_ok:
            return False
        if self.chunk_ok is False or self.quote_ok is False:
            return False
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "cite_index": self.cite_index,
            "section_id": self.section_id,
            "video_id": self.video_id,
            "chunk_index": self.chunk_index,
            "chunk_id": (
                f"chunk:{self.video_id}:{self.chunk_index}"
                if self.chunk_index is not None
                else None
            ),
            "quote": self.quote,
            "text": self.text,
            "video_ok": self.video_ok,
            "chunk_ok": self.chunk_ok,
            "quote_ok": self.quote_ok,
            "valid": self.valid,
        }


@dataclass
class Report:
    claims: list[Claim] = field(default_factory=list)
    structure_errors: list[str] = field(default_factory=list)
    sections: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.claims)

    @property
    def valid(self) -> int:
        return sum(1 for claim in self.claims if claim.valid)

    @property
    def invalid(self) -> list[Claim]:
        return [claim for claim in self.claims if not claim.valid]

    @property
    def pass_rate(self) -> float:
        return self.valid / self.total if self.total else 1.0

    @property
    def ok(self) -> bool:
        return not self.structure_errors and not self.invalid

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "valid": self.valid,
            "pass_rate": round(self.pass_rate, 4),
            "sections": list(self.sections),
            "structure_errors": list(self.structure_errors),
            "claims": [claim.to_dict() for claim in self.claims],
        }


class _CiteScanner(HTMLParser):
    """Collect cites, section ids and the script/handler violations."""

    SECTION_TAGS = {"section"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.cites: list[dict[str, Any]] = []
        self.sections: list[str] = []
        self.errors: list[str] = []
        self._section_stack: list[str | None] = []
        self._open_cite: dict[str, Any] | None = None
        self._cite_text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = {key: value for key, value in attrs}
        for key in attributes:
            if key.startswith("on"):
                self.errors.append(f"inline event handler <{tag} {key}>")
        if tag == "script":
            src = attributes.get("src")
            if src and not src.startswith(ALLOWED_SCRIPT_PREFIX):
                self.errors.append(f"external script {src}")
        if tag in self.SECTION_TAGS:
            section_id = attributes.get("id")
            if not section_id:
                self.errors.append("section without id")
            else:
                self.sections.append(section_id)
            self._section_stack.append(section_id)
        if tag == "cite":
            if self._open_cite is not None:
                self.errors.append(
                    f"<cite data-video={self._open_cite['video_id']!r}> was still open "
                    "when another <cite> opened"
                )
                self._close_cite()
            chunk_raw = attributes.get("data-chunk")
            chunk_index: int | None = None
            if chunk_raw not in (None, ""):
                try:
                    chunk_index = int(str(chunk_raw))
                except ValueError:
                    self.errors.append(f"cite with non-integer data-chunk {chunk_raw!r}")
            self._open_cite = {
                "video_id": attributes.get("data-video") or "",
                "chunk_index": chunk_index,
                "quote": attributes.get("data-quote") or None,
                "section_id": next((item for item in reversed(self._section_stack) if item), None),
            }
            self._cite_text = []

    def handle_endtag(self, tag: str) -> None:
        if tag in self.SECTION_TAGS and self._section_stack:
            self._section_stack.pop()
        if tag == "cite" and self._open_cite is not None:
            self._close_cite()

    def handle_data(self, data: str) -> None:
        if self._open_cite is not None:
            self._cite_text.append(data)

    def _close_cite(self) -> None:
        assert self._open_cite is not None
        self._open_cite["text"] = " ".join("".join(self._cite_text).split())
        self.cites.append(self._open_cite)
        self._open_cite = None
        self._cite_text = []

    def close(self) -> None:
        super().close()
        if self._open_cite is not None:
            self.errors.append(
                f"<cite data-video={self._open_cite['video_id']!r}> was never closed"
            )
            self._close_cite()


_WS = re.compile(r"\s+")


def _norm(text: str) -> str:
    return _WS.sub(" ", text).strip().lower()


def verify_html(
    html: str,
    *,
    known_videos: set[str],
    chunk_lookup: ChunkLookup | None = None,
) -> Report:
    """Resolve every cite in ``html``; see the module docstring for the rules."""
    scanner = _CiteScanner()
    scanner.feed(html)
    scanner.close()
    report = Report(structure_errors=list(scanner.errors), sections=list(scanner.sections))
    chunk_cache: dict[str, dict[int, str]] = {}

    def chunks_for(video_id: str) -> dict[int, str]:
        if video_id not in chunk_cache:
            records = chunk_lookup(video_id) if chunk_lookup else []
            chunk_cache[video_id] = {
                int(record["chunk_index"]): str(record.get("text") or "")
                for record in records
                if "chunk_index" in record
            }
        return chunk_cache[video_id]

    for index, raw in enumerate(scanner.cites):
        claim = Claim(
            cite_index=index,
            section_id=raw["section_id"],
            video_id=raw["video_id"],
            chunk_index=raw["chunk_index"],
            quote=raw["quote"],
            text=raw["text"],
        )
        claim.video_ok = bool(claim.video_id) and claim.video_id in known_videos
        if claim.video_ok and claim.chunk_index is not None:
            texts = chunks_for(claim.video_id)
            claim.chunk_ok = claim.chunk_index in texts
            if claim.chunk_ok and claim.quote:
                claim.quote_ok = _norm(claim.quote) in _norm(texts[claim.chunk_index])
        report.claims.append(claim)
    return report


def write_claims(path: Path, report: Report) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n", "utf-8")

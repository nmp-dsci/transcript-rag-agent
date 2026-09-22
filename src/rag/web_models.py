"""Web sources and their chunks: the corpus's second source type.

These are deliberately *not* :class:`~src.rag.models.TranscriptChunk` with some
fields left null. That shortcut was considered and rejected: chunk identity in
this project is video identity — ``chunk_id`` is ``f"chunk:{video_id}:{index}"``
and ``replace_chunks`` is keyed on a video — so an article stored that way has
to invent a ``video_id``, and 1,654 references across 142 files would then be
reasoning about a video that does not exist. Channel filters would partition on
a YouTube channel an article does not have, neighbour expansion would cross
document boundaries it could not see, and the committed eval snapshots would
change what retrieval returns without changing any number the CI gate checks.

So a web source is its own type, in its own collection, and the union happens
at query time where it is a deliberate read-side choice.

**Identity comes from the channel, never from the URL.** ``external_id`` is the
feed guid, the sitemap loc, or the pinned repo path. Hashing the URL instead —
which is what the chat's ``document_id_for`` does, correctly, for its own
purpose — would give a moved post a new id, re-index it as a second document,
and leave the first behind as a duplicate nobody notices.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone

from pydantic import BaseModel, Field

from src.channels.models import LIVE


def content_hash(text: str) -> str:
    """The hash that decides whether anything needs re-embedding.

    Taken over the *extracted text*, not the fetched bytes: a page whose only
    change is a rotating nav link or a cache-busting query string has not
    changed as a document, and re-embedding it would be pure cost.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


def source_key(external_id: str) -> str:
    """A filesystem- and Chroma-safe key for one source's stable identity."""
    return "web:" + hashlib.sha256(external_id.encode("utf-8")).hexdigest()[:20]


def anchor_for(heading: str | None) -> str | None:
    """A URL fragment for a heading, so an article citation deep-links.

    The parity that matters: a transcript citation carries ``14:22`` and links
    into the video at that second. An article citation should carry its section
    and link into the page at that heading. Most static site generators slugify
    headings exactly this way, so the fragment usually resolves; when it does
    not, the link still opens the right page and the section is still named.
    """
    if not heading:
        return None
    slug = re.sub(r"[^a-z0-9]+", "-", heading.lower()).strip("-")
    return slug or None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class WebSource(BaseModel):
    """One watched document, and everything the refresh loop needs to know."""

    external_id: str
    url: str
    channel_id: str = ""
    title: str | None = None
    #: Where the document lives now, when a redirect moved it. The
    #: ``external_id`` deliberately does not move with it.
    canonical_url: str | None = None
    #: Where a reader should be sent, when that differs from where we fetched.
    display_url: str | None = None
    #: Every URL this source has been served at, oldest first. Kept so a
    #: citation stored against the old URL can still be explained.
    url_history: list[str] = Field(default_factory=list)
    content_hash: str = ""
    #: One of :data:`src.channels.models.SOURCE_STATES`. Recorded, not
    #: inferred — an unrecorded failure is indistinguishable from untried.
    state: str = LIVE
    #: Why the state is what it is, for the states that need explaining
    #: (``blocked`` by which robots group, ``gone`` since when).
    state_reason: str | None = None
    #: Bumped every time the content hash changes, so a citation can say which
    #: revision it was taken from.
    revision: int = 1
    section_count: int = 0
    chunk_count: int = 0
    word_count: int = 0
    #: The body hit the fetch byte cap, so the stored copy is part of a
    #: document. Carried from the fetch rather than recomputed, and never
    #: allowed to read as a whole document.
    truncated: bool = False
    #: Whether this revision came from the feed body rather than a page fetch.
    from_feed: bool = False
    etag: str | None = None
    last_modified: str | None = None
    published_at: str | None = None
    updated_at: str | None = None
    first_seen_at: str = Field(default_factory=_now)
    last_fetched_at: str = Field(default_factory=_now)
    last_changed_at: str = Field(default_factory=_now)
    #: Consecutive fetches that failed or 404'd, so ``gone`` needs corroboration
    #: rather than one bad afternoon.
    consecutive_misses: int = 0

    @property
    def key(self) -> str:
        return source_key(self.external_id)

    @property
    def live_url(self) -> str:
        """Where to fetch this source from now."""
        return self.canonical_url or self.url

    @property
    def reader_url(self) -> str:
        """Where a citation should link."""
        return self.display_url or self.live_url

    @property
    def verifiable(self) -> bool:
        """Whether a citation into this source can still be re-checked.

        A ``gone`` source keeps its chunks — an answer built on them is still a
        real answer — but the citation has to say it cannot be verified rather
        than render a dead link as a live one.
        """
        return self.state not in {"gone", "blocked"}


class WebChunk(BaseModel):
    """One section-anchored run of article text."""

    source_key: str
    external_id: str
    chunk_index: int
    text: str
    channel_id: str = ""
    url: str = ""
    title: str | None = None
    #: The heading this chunk sits under. The citation unit, and the reason
    #: chunks never span two sections.
    heading: str | None = None
    #: Which section of the document, so neighbour expansion has an ordering
    #: that means something.
    section_index: int = 0
    #: Position of this chunk within its section, for sections long enough to
    #: need several.
    part_index: int = 0
    anchor: str | None = None
    published_at: str | None = None
    #: Hash of :attr:`text` alone. This is what makes a small edit cheap: on
    #: re-ingest, a chunk whose hash is unchanged keeps its existing vector
    #: instead of paying to embed it again.
    content_hash: str = ""
    revision: int = 1

    @property
    def chunk_id(self) -> str:
        return f"{self.source_key}:{self.chunk_index}"

    @property
    def citation(self) -> str:
        """How an answer names this chunk.

        The article analogue of a transcript's ``mm:ss``: the source's title
        and the section heading, which is the smallest unit a reader can be
        pointed at and find.
        """
        parts = [self.title or self.url or self.external_id]
        if self.heading:
            parts.append(f"§ {self.heading}")
        return " · ".join(parts)

    @property
    def embedding_text(self) -> str:
        """What gets embedded: the document and section context, then the text.

        Transcript chunks need a context header because a conversational
        fragment loses its subject. Article prose is more self-contained, but a
        chunk from the middle of a long section still benefits from knowing
        which document and which section it came from — and it keeps the two
        source types' embedding inputs shaped alike, which is the only way
        their scores stand a chance of being comparable.
        """
        header = " · ".join(part for part in (self.title, self.heading) if part)
        return f"{header}\n{self.text}" if header else self.text


class RetrievedWebChunk(WebChunk):
    score: float | None = None

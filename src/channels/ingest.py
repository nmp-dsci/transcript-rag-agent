"""Turning one candidate into a stored, chunked, embedded source.

This is where the three measured rules from the plan actually bite:

* **Prefer the feed body.** When a channel's feed carries the article rather
  than a teaser, the page is never fetched. On three registered sources that is
  not merely cheaper: their pages are JavaScript shells that extract to about a
  hundred words, while their feeds carry the whole post. Without this rule those
  three are unusable; with it they are among the best sources on the register.
* **Article extraction mode.** ``nav``/``header``/``footer`` and a leading
  table of contents are dropped. The chat's own paste-a-link path keeps the
  historical ``resume`` mode, so nothing it does changes.
* **A word-count floor.** Below it the ingest *fails loudly* instead of storing
  what it got. Measured across the seed set, this is what catches a
  JavaScript-rendered page: six of twenty-seven links extract to under 110
  words, and every one of them is a shell or a login wall. Silently storing a
  nav bar as an article is the failure this exists to prevent.

Nothing here raises for an expected failure. An ingest run covers many
candidates and must not abandon the rest because one host is down — the same
reason the ingestion queue records a failed job instead of taking its worker
down with it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

from src.channels.models import (
    BLOCKED,
    CHANGED,
    LIVE,
    MOVED,
    TRUNCATED,
    Candidate,
    ChannelConfig,
)
from src.channels.robots import USER_AGENT, RobotsPolicy
from src.documents.extract import ARTICLE_MODE, extract_document
from src.documents.fetch import DocumentFetchError, UnsafeUrlError, fetch_document
from src.documents.models import Document, FetchedPage
from src.rag.web_chunking import build_web_chunks
from src.rag.web_models import WebSource, content_hash, source_key
from src.rag.web_store import WebChunkStore, WebSourceStore

logger = logging.getLogger(__name__)

#: Outcomes, matching the vocabulary the transcript ingestion run records so
#: the two read the same in a log or a UI.
INDEXED = "indexed"
UPDATED = "updated"
UNCHANGED = "unchanged"
SKIPPED = "skipped"
FAILED = "failed"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class IngestOutcome:
    """What happened to one candidate, and why."""

    external_id: str
    url: str
    outcome: str
    source: WebSource | None = None
    chunk_count: int = 0
    embedded_count: int = 0
    removed_chunk_ids: list[str] = field(default_factory=list)
    words: int = 0
    reason: str | None = None

    @property
    def changed(self) -> bool:
        return self.outcome in {INDEXED, UPDATED}


@dataclass
class WebIngestor:
    """Fetches, extracts, chunks and stores one candidate at a time."""

    source_store: WebSourceStore
    chunk_store: WebChunkStore
    robots: RobotsPolicy = field(default_factory=RobotsPolicy)
    target_chars: int = 1200
    overlap_chars: int = 150
    fetch = staticmethod(fetch_document)

    def _page_from_feed(self, candidate: Candidate) -> FetchedPage:
        """A synthetic page carrying the feed's own copy of the article.

        Synthetic rather than a second code path: the body is HTML either way,
        so the extractor, the chunker and the hash all behave identically
        whether the bytes came from a feed or from the page.
        """
        return FetchedPage(
            requested_url=candidate.url,
            url=candidate.url,
            status_code=200,
            content_type="text/html",
            body=candidate.body_html or "",
        )

    def _fetch_page(
        self, candidate: Candidate, stored: WebSource | None, conditional: bool = True
    ) -> tuple[FetchedPage | None, str | None]:
        """The page, or ``None`` with a reason it was not fetched."""
        verdict = self.robots.check(candidate.url)
        if not verdict.allowed:
            return None, verdict.reason
        self.robots.wait(candidate.url)
        page = self.fetch(
            candidate.url,
            user_agent=USER_AGENT,
            etag=stored.etag if (stored and conditional) else None,
            last_modified=stored.last_modified if (stored and conditional) else None,
        )
        return page, None

    def ingest(
        self,
        candidate: Candidate,
        channel: ChannelConfig,
        refresh: bool = False,
        reextract: bool = False,
    ) -> IngestOutcome:
        """Index one candidate, reporting rather than raising on failure.

        ``reextract`` re-derives chunks from the page even when nothing about
        the page has changed. It exists because the two cheap short-circuits —
        a ``304`` and a matching content hash — both stop *before* extraction,
        which is correct when the question is "has the document changed" and
        wrong when the extractor itself has. Without it, an improvement to
        boilerplate stripping or chunking reaches only sources ingested after
        it landed, and the corpus quietly holds two vintages of chunk.

        It still costs almost nothing to run: the chunk hashes decide what is
        re-embedded, so re-extracting a document whose text comes out the same
        embeds zero chunks.
        """
        stored = self.source_store.get(candidate.external_id)
        if stored is not None and not refresh:
            # Discovery already filters on seen ids; this is the second guard,
            # for a candidate that reached here from a different channel or a
            # re-run. A re-index is an explicit refresh, never a side effect.
            return IngestOutcome(
                candidate.external_id, candidate.url, UNCHANGED, stored, stored.chunk_count
            )

        from_feed = bool(candidate.body_html) and channel.body_in_feed
        try:
            if from_feed:
                page: FetchedPage | None = self._page_from_feed(candidate)
                blocked_reason = None
            else:
                page, blocked_reason = self._fetch_page(
                    candidate, stored, conditional=not reextract
                )
        except (DocumentFetchError, UnsafeUrlError) as exc:
            return self._record_miss(candidate, stored, str(exc))

        if page is None:
            outcome = self._record_state(candidate, stored, BLOCKED, blocked_reason or "blocked")
            return IngestOutcome(
                candidate.external_id, candidate.url, SKIPPED, outcome, reason=blocked_reason
            )

        if page.not_modified and stored is not None and not reextract:
            stored.last_fetched_at = _now()
            stored.state = LIVE
            stored.consecutive_misses = 0
            self.source_store.upsert(stored)
            return IngestOutcome(
                candidate.external_id, candidate.url, UNCHANGED, stored, stored.chunk_count
            )

        document = extract_document(page, source_key(candidate.external_id), mode=ARTICLE_MODE)
        words = document.word_count
        if words < channel.min_words:
            reason = (
                f"extracted {words} words, below this channel's floor of {channel.min_words}; "
                f"the page is most likely rendered by JavaScript or behind a login"
            )
            self._record_state(candidate, stored, BLOCKED, reason)
            return IngestOutcome(
                candidate.external_id, candidate.url, FAILED, stored, words=words, reason=reason
            )

        digest = content_hash(document.text)
        if stored is not None and stored.content_hash == digest and not reextract:
            stored.last_fetched_at = _now()
            stored.state = LIVE
            stored.consecutive_misses = 0
            stored.etag = page.etag or stored.etag
            stored.last_modified = page.last_modified or stored.last_modified
            self.source_store.upsert(stored)
            return IngestOutcome(
                candidate.external_id, candidate.url, UNCHANGED, stored, stored.chunk_count
            )

        text_changed = stored is None or stored.content_hash != digest
        source = self._build_source(
            candidate, stored, page, document, digest, from_feed, text_changed
        )
        chunks = build_web_chunks(
            document,
            external_id=candidate.external_id,
            channel_id=channel.id,
            target_chars=self.target_chars,
            overlap_chars=self.overlap_chars,
            revision=source.revision,
            published_at=candidate.published_at,
            url=source.reader_url,
        )
        embedded, removed = self.chunk_store.replace(source.key, chunks)
        source.chunk_count = len(chunks)
        source.section_count = len(document.sections)
        self.source_store.upsert(source)
        return IngestOutcome(
            candidate.external_id,
            candidate.url,
            UPDATED if stored is not None else INDEXED,
            source,
            chunk_count=len(chunks),
            embedded_count=embedded,
            removed_chunk_ids=removed,
            words=words,
        )

    def _build_source(
        self,
        candidate: Candidate,
        stored: WebSource | None,
        page: FetchedPage,
        document: Document,
        digest: str,
        from_feed: bool,
        text_changed: bool = True,
    ) -> WebSource:
        moment = _now()
        # A redirect moved the document. Identity deliberately does not move
        # with it: the external_id stays, the new URL is recorded, and the old
        # one is kept so a citation stored against it can still be explained.
        moved = stored is not None and page.url != stored.live_url
        history = list(stored.url_history) if stored else []
        if stored is not None and moved and stored.live_url not in history:
            history.append(stored.live_url)
        if page.truncated:
            state = TRUNCATED
        elif moved:
            state = MOVED
        elif stored is not None and text_changed:
            state = CHANGED
        else:
            state = LIVE
        return WebSource(
            external_id=candidate.external_id,
            url=stored.url if stored else candidate.url,
            display_url=candidate.display_url or (stored.display_url if stored else None),
            canonical_url=page.url
            if page.url != (stored.url if stored else candidate.url)
            else None,
            url_history=history,
            channel_id=candidate.channel_id,
            title=document.title or (stored.title if stored else None) or candidate.title,
            content_hash=digest,
            state=state,
            # Only a genuine content change is a new revision. Re-reading the
            # same bytes with a better extractor is not a new version of the
            # document, and a citation that says "revision 4" has to mean the
            # document was edited four times.
            revision=(stored.revision + (1 if text_changed else 0)) if stored else 1,
            word_count=document.word_count,
            truncated=page.truncated,
            from_feed=from_feed,
            etag=page.etag,
            last_modified=page.last_modified,
            published_at=candidate.published_at or (stored.published_at if stored else None),
            updated_at=candidate.updated_at,
            first_seen_at=stored.first_seen_at if stored else moment,
            last_fetched_at=moment,
            last_changed_at=(
                moment if text_changed else (stored.last_changed_at if stored else moment)
            ),
            consecutive_misses=0,
        )

    def _record_state(
        self, candidate: Candidate, stored: WebSource | None, state: str, reason: str
    ) -> WebSource | None:
        """Write a state onto a stored source, without touching its chunks.

        A source that is blocked or gone keeps everything it had: an answer
        already built on those chunks is still a real answer, and deleting them
        would rewrite history to hide a fetch that failed today.
        """
        if stored is None:
            return None
        stored.state = state
        stored.state_reason = reason
        stored.last_fetched_at = _now()
        self.source_store.upsert(stored)
        return stored

    def _record_miss(
        self, candidate: Candidate, stored: WebSource | None, error: str
    ) -> IngestOutcome:
        if stored is not None:
            stored.consecutive_misses += 1
            stored.last_fetched_at = _now()
            stored.state_reason = error
            # Two misses, not one: a host having a bad afternoon is not a
            # document that has gone away, and this state is what makes a
            # citation render as unverifiable.
            if stored.consecutive_misses >= 2:
                stored.state = "gone"
            self.source_store.upsert(stored)
        return IngestOutcome(candidate.external_id, candidate.url, FAILED, stored, reason=error)

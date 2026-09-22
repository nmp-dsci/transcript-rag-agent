"""Deciding which stored sources to re-visit, and folding in what came back.

This is the second of the two refresh problems. The first — "did anything new
appear" — is a feed poll, and it lives in :mod:`src.channels.pollers`. This one
is "is what I already stored still what is there", and it is a different
question needing a different mechanism: a schedule over stored sources, and a
recorded state for each.

The state machine itself is in :class:`~src.channels.ingest.WebIngestor`, which
is where the fetch happens and therefore the only place that can observe a 304,
a redirect, a 404 or a byte cap. This module decides *when* to ask and what the
answer means for the next asking.

**The cadence is content-derived, not configured.** A source is re-visited
after roughly as long as it was stable the last time we looked, so a writeup
untouched for three months is checked monthly while one edited last week is
checked weekly. That matters because the register mixes a 2021 conference
writeup with a weekly newsletter, and one interval cannot serve both.

The interval is fixed at *fetch* time rather than recomputed from the current
clock. Computed from "now", it grows as the source sits there, which pushes the
due time away at the same rate the wait accumulates — a source whose interval
always recedes is a source that never comes due.

**Nothing is ever deleted.** A source that 404s keeps every chunk it had: an
answer already built on them is still a real answer, and removing them would
rewrite history to hide a fetch that failed this morning. What changes is that
its citations stop claiming to be verifiable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from src.channels.ingest import UNCHANGED, IngestOutcome, WebIngestor
from src.channels.models import (
    GONE,
    MAX_INTERVAL_HOURS,
    MIN_INTERVAL_HOURS,
    Candidate,
    ChannelConfig,
)
from src.rag.web_models import WebSource
from src.rag.web_store import WebSourceStore

logger = logging.getLogger(__name__)

#: How long after its last fetch a never-changed source is re-visited first.
FIRST_REVISIT_HOURS = 24.0

#: States that are not worth re-fetching on a schedule. ``blocked`` is a policy
#: decision, not a transient failure, so retrying it hourly would be both
#: pointless and rude; it is re-checked only when explicitly forced.
SKIP_STATES = frozenset({"blocked"})


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _parse(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def revisit_interval(source: WebSource) -> float:
    """Hours after the last fetch that this source is worth fetching again.

    The observed stable period *as of that fetch*: a document last edited
    three months before we last looked at it is checked in about three months,
    one edited the day before is checked in about a day. Clamped at both ends,
    so a very old source is still re-checked and a busy one is not hammered.

    Deliberately not a function of the current time. Measured from "now", the
    interval grows while the source waits, which moves the due time away as
    fast as the wait accrues and means a stable source never becomes due.

    A source not yet seen to change — including one stored a moment ago —
    falls back to :data:`FIRST_REVISIT_HOURS`, because zero stable time is an
    absence of evidence rather than evidence of volatility.
    """
    changed = _parse(source.last_changed_at)
    fetched = _parse(source.last_fetched_at)
    if changed is None or fetched is None:
        return FIRST_REVISIT_HOURS
    stable_hours = max(0.0, (fetched - changed).total_seconds() / 3600.0)
    if stable_hours <= 0:
        return FIRST_REVISIT_HOURS
    return max(MIN_INTERVAL_HOURS, min(MAX_INTERVAL_HOURS, stable_hours))


def next_revisit_at(source: WebSource) -> datetime | None:
    """The instant this source becomes due. Fixed, not receding."""
    fetched = _parse(source.last_fetched_at)
    if fetched is None:
        return None
    return fetched + timedelta(hours=revisit_interval(source))


def is_due(source: WebSource, now: datetime | None = None, force: bool = False) -> bool:
    """Whether this source should be fetched again."""
    if force:
        return True
    if source.state in SKIP_STATES:
        return False
    if source.state == GONE and source.consecutive_misses >= 4:
        # Corroborated as gone several times over. Still stored, still cited,
        # but no longer worth a request every day forever.
        return False
    due_at = next_revisit_at(source)
    return due_at is None or due_at <= (now or _now())


def due_sources(
    store: WebSourceStore,
    channel_id: str | None = None,
    now: datetime | None = None,
    force: bool = False,
    limit: int | None = None,
) -> list[WebSource]:
    """Stored sources worth re-visiting, oldest fetch first.

    Oldest first so a run cut short by a limit still makes progress on the
    staleest sources rather than re-checking the same few.
    """
    candidates = [
        source
        for source in store.all()
        if (channel_id is None or source.channel_id == channel_id) and is_due(source, now, force)
    ]
    candidates.sort(key=lambda source: source.last_fetched_at)
    return candidates[:limit] if limit else candidates


@dataclass
class RefreshReport:
    """What one refresh run did."""

    checked: int = 0
    unchanged: int = 0
    updated: int = 0
    failed: int = 0
    skipped: int = 0
    embedded: int = 0
    outcomes: list[IngestOutcome] = field(default_factory=list)

    def record(self, outcome: IngestOutcome) -> None:
        self.checked += 1
        self.embedded += outcome.embedded_count
        self.outcomes.append(outcome)
        if outcome.outcome == UNCHANGED:
            self.unchanged += 1
        elif outcome.changed:
            self.updated += 1
        elif outcome.outcome == "skipped":
            self.skipped += 1
        else:
            self.failed += 1

    def summary(self) -> str:
        return (
            f"checked {self.checked}: {self.unchanged} unchanged, {self.updated} updated, "
            f"{self.skipped} skipped, {self.failed} failed; {self.embedded} chunks embedded"
        )


def candidate_for(source: WebSource) -> Candidate:
    """The candidate that re-fetches one stored source.

    Its ``external_id`` is the stored one, never re-derived from the URL —
    which is the whole reason a moved document stays one document.
    """
    return Candidate(
        url=source.live_url,
        external_id=source.external_id,
        channel_id=source.channel_id,
        title=source.title,
        published_at=source.published_at,
        updated_at=source.updated_at,
        display_url=source.display_url,
    )


def refresh_sources(
    ingestor: WebIngestor,
    sources: list[WebSource],
    channels: dict[str, ChannelConfig],
    default_channel: ChannelConfig | None = None,
    reextract: bool = False,
) -> RefreshReport:
    """Re-fetch each source, folding every outcome into one report.

    A source whose channel has been removed from ``channels.yaml`` is still
    refreshed, under ``default_channel``, because deleting a channel is a
    statement about what to *watch* — not a decision to abandon documents
    already in the corpus.
    """
    report = RefreshReport()
    for source in sources:
        channel = channels.get(source.channel_id) or default_channel
        if channel is None:
            logger.info(
                "skipping %s: channel %r is no longer configured and no default was given",
                source.external_id,
                source.channel_id,
            )
            report.skipped += 1
            report.checked += 1
            continue
        report.record(
            ingestor.ingest(candidate_for(source), channel, refresh=True, reextract=reextract)
        )
    return report

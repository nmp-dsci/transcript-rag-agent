from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.channels.ingest import WebIngestor
from src.channels.models import MAX_INTERVAL_HOURS, MIN_INTERVAL_HOURS, ChannelConfig
from src.channels.refresh import (
    FIRST_REVISIT_HOURS,
    candidate_for,
    due_sources,
    is_due,
    next_revisit_at,
    refresh_sources,
    revisit_interval,
)
from src.documents.fetch import DocumentFetchError
from src.documents.models import FetchedPage
from src.rag.web_models import WebSource
from src.rag.web_store import WebChunkStore, WebSourceStore

NOW = datetime(2026, 9, 22, 12, tzinfo=timezone.utc)

BODY = (
    "<html><head><title>T</title></head><body><article><h2>Section</h2><p>"
    + ("prose about evaluating agents carefully. " * 70)
    + "</p></article></body></html>"
)


def source(changed_hours: float, fetched_hours: float = 0.0, **kwargs) -> WebSource:
    return WebSource(
        external_id=kwargs.pop("external_id", "guid-1"),
        url="https://a.example/post",
        channel_id=kwargs.pop("channel_id", "feed"),
        last_changed_at=(NOW - timedelta(hours=changed_hours)).isoformat(),
        last_fetched_at=(NOW - timedelta(hours=fetched_hours)).isoformat(),
        **kwargs,
    )


def test_a_long_stable_source_is_revisited_less_often_than_a_recently_edited_one() -> None:
    # The register mixes a 2021 conference writeup with a weekly newsletter;
    # one interval cannot serve both.
    quiet = revisit_interval(source(changed_hours=24 * 90, fetched_hours=0))
    recent = revisit_interval(source(changed_hours=24 * 4, fetched_hours=0))
    assert recent < quiet
    assert quiet == MAX_INTERVAL_HOURS
    assert recent == pytest.approx(24 * 4)


def test_the_revisit_interval_is_clamped_at_both_ends() -> None:
    for hours in (0.1, 1, 10, 1000, 100000):
        interval = revisit_interval(source(changed_hours=hours, fetched_hours=0))
        assert MIN_INTERVAL_HOURS <= interval <= MAX_INTERVAL_HOURS


def test_a_source_never_seen_to_change_gets_the_default_interval() -> None:
    # Zero stable time is an absence of evidence, not evidence of volatility.
    assert revisit_interval(source(changed_hours=0, fetched_hours=0)) == FIRST_REVISIT_HOURS


def test_the_due_time_does_not_recede_as_the_source_waits() -> None:
    # Computed from "now", the interval grows while the source sits there and
    # pushes the due time away as fast as the wait accrues.
    stable = source(changed_hours=24 * 10, fetched_hours=0)
    first = next_revisit_at(stable)
    assert first == next_revisit_at(stable)
    assert is_due(stable, first + timedelta(minutes=1))


def test_a_source_is_not_due_until_its_interval_since_the_last_fetch() -> None:
    stable = source(changed_hours=24 * 7, fetched_hours=0)
    assert not is_due(stable, NOW)
    due_at = next_revisit_at(stable)
    assert due_at == NOW + timedelta(hours=24 * 7)
    assert is_due(stable, due_at + timedelta(minutes=1))


def test_a_blocked_source_is_never_due_on_a_schedule_but_can_be_forced() -> None:
    # Blocked is a policy decision, not a transient failure; retrying it
    # hourly would be both pointless and rude.
    blocked = source(changed_hours=1, fetched_hours=999, state="blocked")
    assert not is_due(blocked, NOW)
    assert is_due(blocked, NOW, force=True)


def test_a_source_corroborated_as_gone_stops_being_polled_forever() -> None:
    assert is_due(
        source(changed_hours=1, fetched_hours=999, state="gone", consecutive_misses=2), NOW
    )
    assert not is_due(
        source(changed_hours=1, fetched_hours=999, state="gone", consecutive_misses=4), NOW
    )


def test_due_sources_come_back_staleest_first(tmp_path: Path, embeddings) -> None:
    # A run cut short by a limit should make progress on the stalest sources
    # rather than re-checking the same few.
    store = WebSourceStore(tmp_path, embeddings)
    for index, fetched in enumerate([1000.0, 5000.0, 3000.0]):
        store.upsert(source(changed_hours=24 * 30, fetched_hours=fetched, external_id=f"g{index}"))
    assert [s.external_id for s in due_sources(store, now=NOW)] == ["g1", "g2", "g0"]
    assert [s.external_id for s in due_sources(store, now=NOW, limit=2)] == ["g1", "g2"]


def test_due_sources_can_be_scoped_to_one_channel(tmp_path: Path, embeddings) -> None:
    store = WebSourceStore(tmp_path, embeddings)
    store.upsert(source(24 * 30, 5000, external_id="a", channel_id="one"))
    store.upsert(source(24 * 30, 5000, external_id="b", channel_id="two"))
    assert [s.external_id for s in due_sources(store, channel_id="two", now=NOW)] == ["b"]


def test_the_refresh_candidate_reuses_the_stored_identity() -> None:
    stored = source(24, display_url="https://github.com/o/r")
    candidate = candidate_for(stored)
    assert candidate.external_id == "guid-1"
    assert candidate.url == stored.live_url
    assert candidate.reader_url == "https://github.com/o/r"


@pytest.fixture
def ingestor(tmp_path: Path, embeddings, permissive_robots) -> WebIngestor:
    return WebIngestor(
        source_store=WebSourceStore(tmp_path, embeddings),
        chunk_store=WebChunkStore(tmp_path, embeddings),
        robots=permissive_robots,
    )


def test_a_refresh_run_reports_every_outcome(ingestor) -> None:
    channel = ChannelConfig(id="feed", kind="rss", url="https://a.example/f.xml")
    responses = {
        "https://a.example/alive": FetchedPage(
            requested_url="https://a.example/alive",
            url="https://a.example/alive",
            status_code=200,
            content_type="text/html",
            body=BODY,
        )
    }

    def fetch(url: str, **kwargs):
        if url not in responses:
            raise DocumentFetchError(f"no page at {url}")
        return responses[url]

    ingestor.fetch = fetch
    alive = WebSource(external_id="alive", url="https://a.example/alive", channel_id="feed")
    dead = WebSource(external_id="dead", url="https://a.example/dead", channel_id="feed")
    ingestor.source_store.upsert(alive)
    ingestor.source_store.upsert(dead)

    report = refresh_sources(ingestor, [alive, dead], {"feed": channel})
    assert report.checked == 2
    assert report.updated == 1
    assert report.failed == 1
    assert "checked 2" in report.summary()


def test_a_source_whose_channel_was_deleted_is_skipped_not_lost(ingestor) -> None:
    # Deleting a channel is a statement about what to watch, not a decision to
    # abandon documents already in the corpus.
    orphan = WebSource(external_id="orphan", url="https://a.example/x", channel_id="removed")
    ingestor.source_store.upsert(orphan)
    report = refresh_sources(ingestor, [orphan], {})
    assert report.skipped == 1
    assert ingestor.source_store.get("orphan") is not None


def test_a_refresh_run_passes_reextract_through_to_every_source(ingestor) -> None:
    channel = ChannelConfig(id="feed", kind="rss", url="https://a.example/f.xml")
    seen: list[bool] = []

    def fetch(url: str, **kwargs):
        # reextract suppresses the conditional headers, which is observable.
        seen.append(kwargs.get("etag") is None)
        return FetchedPage(
            requested_url=url, url=url, status_code=200, content_type="text/html", body=BODY
        )

    ingestor.fetch = fetch
    stored = WebSource(
        external_id="a", url="https://a.example/alive", channel_id="feed", etag='"v1"'
    )
    ingestor.source_store.upsert(stored)
    refresh_sources(ingestor, [stored], {"feed": channel})
    refresh_sources(ingestor, [stored], {"feed": channel}, reextract=True)
    assert seen == [False, True]


def test_a_default_channel_covers_orphaned_sources(ingestor) -> None:
    default = ChannelConfig(id="default", kind="url_list", urls=("https://a.example/x",))
    orphan = WebSource(external_id="orphan", url="https://a.example/alive", channel_id="removed")
    ingestor.source_store.upsert(orphan)
    ingestor.fetch = lambda url, **kwargs: FetchedPage(
        requested_url=url, url=url, status_code=200, content_type="text/html", body=BODY
    )
    report = refresh_sources(ingestor, [orphan], {}, default_channel=default)
    assert report.skipped == 0
    assert report.updated == 1

from __future__ import annotations

from pathlib import Path

import pytest

from src.channels.ingest import (
    FAILED,
    INDEXED,
    SKIPPED,
    UNCHANGED,
    UPDATED,
    WebIngestor,
)
from src.channels.models import Candidate, ChannelConfig
from src.documents.fetch import DocumentFetchError
from src.documents.models import FetchedPage
from src.rag.web_store import WebChunkStore, WebSourceStore

ARTICLE = """<html><head><title>Evals guide</title></head><body>
<nav>Navigation Menu Home About Pricing</nav>
<h2>Contents</h2><p>One. Two. Three.</p>
<article>
<h2>Error analysis first</h2>
<p>{first}</p>
<h2>Writing a judge</h2>
<p>{second}</p>
</article>
<footer>Copyright somebody</footer>
</body></html>"""

FIRST = "Read fifty real traces before you write a single judge prompt. " * 12
SECOND = "Align the judge against human labels and publish the agreement rate. " * 12


def article(first: str = FIRST, second: str = SECOND) -> str:
    return ARTICLE.format(first=first, second=second)


def page(body: str, **kwargs) -> FetchedPage:
    defaults = dict(
        requested_url="https://a.example/post",
        url="https://a.example/post",
        status_code=200,
        content_type="text/html",
        body=body,
    )
    return FetchedPage(**{**defaults, **kwargs})


@pytest.fixture
def ingestor(tmp_path: Path, embeddings, permissive_robots) -> WebIngestor:
    return WebIngestor(
        source_store=WebSourceStore(tmp_path, embeddings),
        chunk_store=WebChunkStore(tmp_path, embeddings),
        robots=permissive_robots,
    )


def with_pages(ingestor: WebIngestor, responses: dict[str, FetchedPage], calls: list | None = None):
    def fetch(url: str, **kwargs):
        if calls is not None:
            calls.append(url)
        if url not in responses:
            raise DocumentFetchError(f"no page at {url}")
        return responses[url]

    ingestor.fetch = fetch
    return ingestor


CHANNEL = ChannelConfig(id="feed", kind="rss", url="https://a.example/f.xml")
FEED_CHANNEL = ChannelConfig(
    id="feed", kind="rss", url="https://a.example/f.xml", body_in_feed=True
)


def candidate(**kwargs) -> Candidate:
    defaults = dict(url="https://a.example/post", external_id="guid-1", channel_id="feed")
    return Candidate(**{**defaults, **kwargs})


def test_a_new_source_is_fetched_extracted_chunked_and_stored(ingestor) -> None:
    with_pages(ingestor, {"https://a.example/post": page(article())})
    outcome = ingestor.ingest(candidate(), CHANNEL)
    assert outcome.outcome == INDEXED
    assert outcome.chunk_count > 0
    assert outcome.embedded_count == outcome.chunk_count
    stored = ingestor.source_store.get("guid-1")
    assert stored is not None
    assert stored.title == "Evals guide"
    assert stored.revision == 1


def test_article_mode_drops_the_navigation_and_the_table_of_contents(ingestor) -> None:
    with_pages(ingestor, {"https://a.example/post": page(article())})
    ingestor.ingest(candidate(), CHANNEL)
    text = " ".join(chunk.text for chunk in ingestor.chunk_store.all_chunks())
    assert "Navigation Menu" not in text
    assert "Copyright somebody" not in text
    headings = {chunk.heading for chunk in ingestor.chunk_store.all_chunks()}
    assert "Contents" not in headings
    assert "Error analysis first" in headings


def test_chunks_carry_a_section_anchor_so_a_citation_deep_links(ingestor) -> None:
    with_pages(ingestor, {"https://a.example/post": page(article())})
    ingestor.ingest(candidate(), CHANNEL)
    chunk = next(c for c in ingestor.chunk_store.all_chunks() if c.heading == "Writing a judge")
    assert chunk.anchor == "writing-a-judge"
    assert chunk.citation == "Evals guide · § Writing a judge"


def test_a_feed_body_is_used_and_the_page_is_never_fetched(ingestor) -> None:
    # Three registered sources are JavaScript shells whose pages extract to
    # about a hundred words while their feeds carry the whole post.
    calls: list[str] = []
    with_pages(ingestor, {}, calls)
    outcome = ingestor.ingest(candidate(body_html=article()), FEED_CHANNEL)
    assert outcome.outcome == INDEXED
    assert calls == []
    assert ingestor.source_store.get("guid-1").from_feed is True


def test_a_feed_body_is_ignored_when_the_channel_does_not_declare_one(ingestor) -> None:
    calls: list[str] = []
    with_pages(ingestor, {"https://a.example/post": page(article())}, calls)
    ingestor.ingest(candidate(body_html=article()), CHANNEL)
    assert calls == ["https://a.example/post"]
    assert ingestor.source_store.get("guid-1").from_feed is False


def test_a_thin_extract_fails_loudly_instead_of_being_stored(ingestor) -> None:
    # Measured: six of twenty-seven seed links extract to under 110 words, and
    # every one of them is a JavaScript shell or a login wall.
    with_pages(
        ingestor, {"https://a.example/post": page("<html><body><nav>Log In</nav></body></html>")}
    )
    outcome = ingestor.ingest(candidate(), CHANNEL)
    assert outcome.outcome == FAILED
    assert "below this channel's floor" in (outcome.reason or "")
    assert ingestor.chunk_store.count() == 0
    assert ingestor.source_store.get("guid-1") is None


def test_a_candidate_whose_first_ingest_fails_stays_retryable(ingestor) -> None:
    # cli.py and the API poll endpoint only mark a candidate's external_id as
    # seen once its ingest actually stores a WebSource. A transient failure on
    # first encounter (here, a thin extract) must leave the id retryable
    # rather than silently vanishing from all future polls.
    from src.channels.models import ChannelState

    state = ChannelState(channel_id="feed")
    with_pages(
        ingestor, {"https://a.example/post": page("<html><body><nav>Log In</nav></body></html>")}
    )
    outcome = ingestor.ingest(candidate(), CHANNEL)
    assert outcome.outcome == FAILED
    assert outcome.source is None
    # The caller's rule: only remember an id once something was stored.
    if outcome.source is not None:
        state.remember([candidate().external_id])
    assert "guid-1" not in state.seen_ids

    with_pages(ingestor, {"https://a.example/post": page(article())})
    outcome = ingestor.ingest(candidate(), CHANNEL)
    assert outcome.outcome == INDEXED
    assert outcome.source is not None
    if outcome.source is not None:
        state.remember([candidate().external_id])
    assert "guid-1" in state.seen_ids


def test_a_second_ingest_of_an_unchanged_source_embeds_nothing(ingestor, embeddings) -> None:
    with_pages(ingestor, {"https://a.example/post": page(article())})
    ingestor.ingest(candidate(), CHANNEL)
    before = embeddings.embedded_count
    outcome = ingestor.ingest(candidate(), CHANNEL, refresh=True)
    assert outcome.outcome == UNCHANGED
    assert embeddings.embedded_count == before


def test_an_unrefreshed_re_ingest_does_not_re_fetch_at_all(ingestor) -> None:
    calls: list[str] = []
    with_pages(ingestor, {"https://a.example/post": page(article())}, calls)
    ingestor.ingest(candidate(), CHANNEL)
    ingestor.ingest(candidate(), CHANNEL)
    # A re-index is an explicit refresh, never a side effect of being offered
    # the same candidate twice.
    assert calls == ["https://a.example/post"]


def test_a_changed_article_re_embeds_only_the_chunks_that_moved(ingestor, embeddings) -> None:
    with_pages(ingestor, {"https://a.example/post": page(article())})
    ingestor.ingest(candidate(), CHANNEL)
    before = embeddings.embedded_count

    edited = article(second=SECOND.replace("agreement rate", "agreement percentage"))
    with_pages(ingestor, {"https://a.example/post": page(edited)})
    outcome = ingestor.ingest(candidate(), CHANNEL, refresh=True)

    assert outcome.outcome == UPDATED
    assert 0 < outcome.embedded_count < outcome.chunk_count
    assert embeddings.embedded_count - before == outcome.embedded_count
    assert ingestor.source_store.get("guid-1").revision == 2


def test_a_304_leaves_the_stored_copy_and_its_vectors_alone(ingestor, embeddings) -> None:
    with_pages(ingestor, {"https://a.example/post": page(article(), etag='"v1"')})
    ingestor.ingest(candidate(), CHANNEL)
    before = embeddings.embedded_count
    with_pages(
        ingestor,
        {"https://a.example/post": page("", status_code=304, content_type="", etag='"v1"')},
    )
    outcome = ingestor.ingest(candidate(), CHANNEL, refresh=True)
    assert outcome.outcome == UNCHANGED
    assert embeddings.embedded_count == before


def test_a_moved_document_keeps_its_identity_and_records_the_new_url(ingestor) -> None:
    # Identity must not move with the URL: hashing the URL instead would mint
    # a second document and orphan the first.
    with_pages(ingestor, {"https://a.example/post": page(article())})
    ingestor.ingest(candidate(), CHANNEL)
    moved = page(article(second=SECOND + "extra sentence. "), url="https://a.example/moved")
    with_pages(ingestor, {"https://a.example/post": moved})
    ingestor.ingest(candidate(), CHANNEL, refresh=True)

    stored = ingestor.source_store.get("guid-1")
    assert stored.external_id == "guid-1"
    assert stored.canonical_url == "https://a.example/moved"
    assert "https://a.example/post" in stored.url_history
    assert ingestor.source_store.count() == 1


def test_a_robots_refusal_skips_without_deleting_what_is_stored(
    ingestor, ai_blocking_robots
) -> None:
    with_pages(ingestor, {"https://a.example/post": page(article())})
    ingestor.ingest(candidate(), CHANNEL)
    chunks_before = ingestor.chunk_store.count()

    ingestor.robots = ai_blocking_robots
    outcome = ingestor.ingest(candidate(url="https://sub.example/p/post"), CHANNEL, refresh=True)
    assert outcome.outcome == SKIPPED
    assert ingestor.chunk_store.count() == chunks_before


def test_a_dead_url_becomes_gone_only_after_corroboration(ingestor) -> None:
    with_pages(ingestor, {"https://a.example/post": page(article())})
    ingestor.ingest(candidate(), CHANNEL)
    chunks_before = ingestor.chunk_store.count()
    with_pages(ingestor, {})

    ingestor.ingest(candidate(), CHANNEL, refresh=True)
    assert ingestor.source_store.get("guid-1").state != "gone"
    ingestor.ingest(candidate(), CHANNEL, refresh=True)

    stored = ingestor.source_store.get("guid-1")
    assert stored.state == "gone"
    assert not stored.verifiable
    # Nothing is deleted: an answer already built on these chunks is still a
    # real answer, it just cannot be re-checked.
    assert ingestor.chunk_store.count() == chunks_before


def test_reextract_rebuilds_chunks_even_when_the_page_is_unchanged(ingestor) -> None:
    # Both cheap short-circuits — a 304 and a matching content hash — stop
    # before extraction, which is right when the question is "has the document
    # changed" and wrong when the extractor itself has.
    with_pages(ingestor, {"https://a.example/post": page(article(), etag='"v1"')})
    ingestor.ingest(candidate(), CHANNEL)
    with_pages(
        ingestor,
        {"https://a.example/post": page("", status_code=304, content_type="", etag='"v1"')},
    )
    assert ingestor.ingest(candidate(), CHANNEL, refresh=True).outcome == UNCHANGED

    # With reextract the conditional headers are not sent at all, so the
    # server answers 200 and the document is re-derived.
    with_pages(ingestor, {"https://a.example/post": page(article(), etag='"v1"')})
    outcome = ingestor.ingest(candidate(), CHANNEL, refresh=True, reextract=True)
    assert outcome.outcome == UPDATED


def test_reextract_does_not_send_the_stored_validators(ingestor) -> None:
    sent: list[tuple[str | None, str | None]] = []

    def fetch(url: str, **kwargs):
        sent.append((kwargs.get("etag"), kwargs.get("last_modified")))
        return page(article(), etag='"v1"', last_modified="then")

    ingestor.fetch = fetch
    ingestor.ingest(candidate(), CHANNEL)
    ingestor.ingest(candidate(), CHANNEL, refresh=True)
    ingestor.ingest(candidate(), CHANNEL, refresh=True, reextract=True)
    assert sent[0] == (None, None)
    assert sent[1] == ('"v1"', "then")
    assert sent[2] == (None, None)


def test_a_reextract_that_changes_nothing_embeds_nothing(ingestor, embeddings) -> None:
    # The chunk hashes still decide what is re-embedded, so rolling an
    # extractor change through a corpus costs only the fetches.
    with_pages(ingestor, {"https://a.example/post": page(article())})
    ingestor.ingest(candidate(), CHANNEL)
    before = embeddings.embedded_count
    outcome = ingestor.ingest(candidate(), CHANNEL, refresh=True, reextract=True)
    assert outcome.embedded_count == 0
    assert embeddings.embedded_count == before


def test_a_reextract_that_changes_nothing_is_not_a_new_revision(ingestor) -> None:
    # Re-reading the same bytes with a better extractor is not a new version
    # of the document, and "revision 4" has to mean it was edited four times.
    with_pages(ingestor, {"https://a.example/post": page(article())})
    ingestor.ingest(candidate(), CHANNEL)
    first = ingestor.source_store.get("guid-1")
    ingestor.ingest(candidate(), CHANNEL, refresh=True, reextract=True)
    after = ingestor.source_store.get("guid-1")
    assert after.revision == first.revision
    assert after.state == "live"
    assert after.last_changed_at == first.last_changed_at


def test_a_real_edit_found_during_a_reextract_is_still_a_new_revision(ingestor) -> None:
    with_pages(ingestor, {"https://a.example/post": page(article())})
    ingestor.ingest(candidate(), CHANNEL)
    edited = article(second=SECOND.replace("agreement rate", "agreement percentage"))
    with_pages(ingestor, {"https://a.example/post": page(edited)})
    ingestor.ingest(candidate(), CHANNEL, refresh=True, reextract=True)
    stored = ingestor.source_store.get("guid-1")
    assert stored.revision == 2
    assert stored.state == "changed"


def test_a_truncated_fetch_is_recorded_rather_than_read_as_a_whole_document(ingestor) -> None:
    with_pages(ingestor, {"https://a.example/post": page(article(), truncated=True)})
    ingestor.ingest(candidate(), CHANNEL)
    stored = ingestor.source_store.get("guid-1")
    assert stored.truncated is True
    assert stored.state == "truncated"

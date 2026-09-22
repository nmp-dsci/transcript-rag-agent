from __future__ import annotations

from pathlib import Path

import pytest

from src.rag.web_chunking import build_web_chunks
from src.rag.web_models import WebSource, source_key
from src.rag.web_store import WebChunkStore, WebSourceStore

from tests.channels.conftest import FakeEmbeddings
from tests.rag.test_web_chunking import document


@pytest.fixture
def embeddings() -> FakeEmbeddings:
    return FakeEmbeddings()


def chunks_for(text_a: str = "one ", text_b: str = "two ", external_id: str = "guid"):
    return build_web_chunks(
        document(("A", text_a * 60), ("B", text_b * 60)), external_id=external_id, channel_id="feed"
    )


def test_a_source_round_trips_with_its_refresh_state(tmp_path: Path, embeddings) -> None:
    store = WebSourceStore(tmp_path, embeddings)
    source = WebSource(
        external_id="guid",
        url="https://a.example/post",
        channel_id="feed",
        title="T",
        state="gone",
        state_reason="404 twice",
        url_history=["https://a.example/old"],
        revision=3,
    )
    store.upsert(source)
    restored = store.get("guid")
    assert restored is not None
    assert restored.state == "gone"
    assert restored.url_history == ["https://a.example/old"]
    assert restored.revision == 3
    assert not restored.verifiable


def test_an_unknown_source_reads_as_absent(tmp_path: Path, embeddings) -> None:
    assert WebSourceStore(tmp_path, embeddings).get("nope") is None


def test_sources_list_newest_fetch_first_and_scope_by_channel(tmp_path: Path, embeddings) -> None:
    store = WebSourceStore(tmp_path, embeddings)
    store.upsert(
        WebSource(
            external_id="a", url="https://x/a", channel_id="one", last_fetched_at="2026-01-01"
        )
    )
    store.upsert(
        WebSource(
            external_id="b", url="https://x/b", channel_id="two", last_fetched_at="2026-09-01"
        )
    )
    assert [s.external_id for s in store.all()] == ["b", "a"]
    assert [s.external_id for s in store.for_channel("one")] == ["a"]
    assert store.count() == 2


def test_deleting_a_source_reports_whether_there_was_one(tmp_path: Path, embeddings) -> None:
    store = WebSourceStore(tmp_path, embeddings)
    store.upsert(WebSource(external_id="a", url="https://x/a"))
    assert store.delete("a") is True
    assert store.delete("a") is False


def test_replacing_chunks_embeds_every_one_the_first_time(tmp_path: Path, embeddings) -> None:
    store = WebChunkStore(tmp_path, embeddings)
    chunks = chunks_for()
    embedded, removed = store.replace(source_key("guid"), chunks)
    assert embedded == len(chunks)
    assert removed == []
    assert store.count() == len(chunks)


def test_re_replacing_identical_chunks_embeds_nothing(tmp_path: Path, embeddings) -> None:
    store = WebChunkStore(tmp_path, embeddings)
    store.replace(source_key("guid"), chunks_for())
    embedded, removed = store.replace(source_key("guid"), chunks_for())
    assert embedded == 0
    assert removed == []


def test_only_the_chunks_whose_text_changed_are_re_embedded(tmp_path: Path, embeddings) -> None:
    # The rule that makes a refresh habit affordable: a typo fixed in one
    # paragraph of a ninety-chunk article re-embeds one chunk.
    store = WebChunkStore(tmp_path, embeddings)
    first = chunks_for()
    store.replace(source_key("guid"), first)
    edited = chunks_for(text_b="three ")
    embedded, _removed = store.replace(source_key("guid"), edited)
    assert 0 < embedded < len(edited)


def test_a_paragraph_that_merely_moved_keeps_its_vector(tmp_path: Path, embeddings) -> None:
    # Matching by index would re-embed the whole tail after an insertion.
    store = WebChunkStore(tmp_path, embeddings)
    store.replace(source_key("guid"), chunks_for())
    shifted = build_web_chunks(
        document(("New first", "zero " * 60), ("A", "one " * 60), ("B", "two " * 60)),
        external_id="guid",
    )
    embedded, _removed = store.replace(source_key("guid"), shifted)
    original_count = len(chunks_for())
    assert embedded == len(shifted) - original_count


def test_a_shrinking_re_chunk_removes_the_tail_it_leaves_behind(tmp_path: Path, embeddings) -> None:
    store = WebChunkStore(tmp_path, embeddings)
    store.replace(source_key("guid"), chunks_for())
    smaller = build_web_chunks(document(("A", "one " * 60)), external_id="guid")
    _embedded, removed = store.replace(source_key("guid"), smaller)
    assert removed
    assert store.count() == len(smaller)


def test_replacing_one_source_leaves_another_alone(tmp_path: Path, embeddings) -> None:
    store = WebChunkStore(tmp_path, embeddings)
    store.replace(source_key("a"), chunks_for(external_id="a"))
    store.replace(source_key("b"), chunks_for(external_id="b"))
    store.replace(source_key("a"), build_web_chunks(document(("A", "x " * 60)), external_id="a"))
    keys = {chunk.source_key for chunk in store.all_chunks()}
    assert keys == {source_key("a"), source_key("b")}
    assert len(store.chunk_ids_for(source_key("b"))) == len(chunks_for(external_id="b"))


def test_deleting_a_source_removes_only_its_chunks(tmp_path: Path, embeddings) -> None:
    store = WebChunkStore(tmp_path, embeddings)
    store.replace(source_key("a"), chunks_for(external_id="a"))
    store.replace(source_key("b"), chunks_for(external_id="b"))
    removed = store.delete_source(source_key("a"))
    assert removed
    assert {chunk.source_key for chunk in store.all_chunks()} == {source_key("b")}


def test_a_query_returns_scored_chunks_with_their_citation_intact(
    tmp_path: Path, embeddings
) -> None:
    store = WebChunkStore(tmp_path, embeddings)
    store.replace(
        source_key("guid"),
        build_web_chunks(
            document(
                ("Agents", "building agents in production " * 20),
                ("Evals", "eval harness design " * 20),
            ),
            external_id="guid",
        ),
    )
    hits = store.query("agents", top_k=2)
    assert hits
    assert all(hit.score is not None for hit in hits)
    assert all(hit.text for hit in hits)
    assert any("§" in hit.citation for hit in hits)


def test_an_empty_store_answers_a_query_without_calling_the_embedder(
    tmp_path: Path, embeddings
) -> None:
    store = WebChunkStore(tmp_path, embeddings)
    assert store.query("anything", top_k=3) == []
    assert not store.has_any()


def test_excluded_sources_are_invisible_on_every_read_path(tmp_path: Path, embeddings) -> None:
    # The held-out evaluation depends on this: exclusion has to hold on every
    # read, and a per-call argument is what one path forgets.
    plain = WebChunkStore(tmp_path, embeddings)
    plain.replace(source_key("a"), chunks_for(external_id="a"))
    plain.replace(source_key("b"), chunks_for(external_id="b"))

    scoped = WebChunkStore(tmp_path, embeddings, exclude_external_ids=["a"])
    assert {chunk.external_id for chunk in scoped.all_chunks()} == {"b"}
    assert all(hit.external_id == "b" for hit in scoped.query("one", top_k=10))


def test_the_transcript_collections_are_not_touched(tmp_path: Path, embeddings) -> None:
    # The separation is the point: transcript_chunks is what the committed
    # eval snapshots were measured against.
    store = WebChunkStore(tmp_path, embeddings)
    store.replace(source_key("guid"), chunks_for())
    names = {collection.name for collection in store.client.list_collections()}
    assert "web_chunks" in names
    assert "transcript_chunks" not in names

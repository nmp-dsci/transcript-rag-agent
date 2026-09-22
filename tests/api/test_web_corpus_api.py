"""The corpus's web half, as the pipeline tree reads it.

The tree has two roots because the corpus has two source types, stored in two
collections. These tests hold that line: what the web endpoints report comes
from the web collections and nothing else.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.main import create_app
from src.config import Settings
from src.rag.web_models import WebChunk, WebSource
from src.rag.web_store import WebChunkStore, WebSourceStore

CHANNELS_YAML = """version: 1

channels:
  - id: hamel-dev
    kind: rss
    url: https://hamel.dev/index.xml
    label: Hamel Husain
  - id: seeds
    kind: url_list
    label: Seed articles
    urls: ["https://a.example/one"]
"""


class FakeEmbeddings:
    """Enough of the protocol for the store to open and write."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text) % 5), 1.0, 0.0]


def app_for(settings: Settings, tmp_path: Path) -> TestClient:
    return TestClient(
        create_app(
            settings,
            history_path=tmp_path / "h.json",
            chat_html_path=tmp_path / "c.html",
            index_fn=lambda argv: 0,
            channels_poll_fn=lambda ids: {"polled": len(ids)},
        )
    )


@pytest.fixture
def web_settings(settings: Settings, tmp_path: Path) -> Settings:
    target = tmp_path / "channels.yaml"
    target.write_text(CHANNELS_YAML, encoding="utf-8")
    return dataclasses.replace(settings, channels_file=target)


def seed(settings: Settings, sources: list[WebSource], chunks: list[WebChunk]) -> None:
    embeddings = FakeEmbeddings()
    source_store = WebSourceStore(settings.chroma_path, embeddings, settings.web_source_collection)
    for source in sources:
        source_store.upsert(source)
    chunk_store = WebChunkStore(settings.chroma_path, embeddings, settings.web_chunk_collection)
    by_source: dict[str, list[WebChunk]] = {}
    for chunk in chunks:
        by_source.setdefault(chunk.source_key, []).append(chunk)
    for key, items in by_source.items():
        chunk_store.replace(key, items)


def source(external_id: str, channel_id: str, **overrides: object) -> WebSource:
    return WebSource(
        external_id=external_id,
        url=f"https://example.com/{external_id}",
        channel_id=channel_id,
        title=f"Post {external_id}",
        chunk_count=2,
        word_count=400,
        section_count=3,
        **overrides,  # type: ignore[arg-type]
    )


def chunk(parent: WebSource, index: int, heading: str) -> WebChunk:
    return WebChunk(
        source_key=parent.key,
        external_id=parent.external_id,
        chunk_index=index,
        text=f"Body of {heading}",
        channel_id=parent.channel_id,
        url=parent.url,
        title=parent.title,
        heading=heading,
        section_index=index,
        anchor=heading.lower().replace(" ", "-"),
    )


def test_web_documents_are_grouped_by_the_channel_that_found_them(web_settings, tmp_path) -> None:
    one, two, three = source("a", "hamel-dev"), source("b", "hamel-dev"), source("c", "seeds")
    seed(web_settings, [one, two, three], [])
    body = app_for(web_settings, tmp_path).get("/api/web/sources").json()

    # The chunk total is summed from what each source recorded at ingest, so
    # it reads the same whether or not the chunks are paged in here.
    assert body["totals"] == {"channels": 2, "sources": 3, "chunks": 6}
    by_channel = {group["channel_id"]: group for group in body["channels"]}
    assert len(by_channel["hamel-dev"]["sources"]) == 2
    assert len(by_channel["seeds"]["sources"]) == 1
    # The label comes from channels.yaml, not from the stored documents.
    assert by_channel["hamel-dev"]["label"] == "Hamel Husain"


def test_a_channel_edited_out_of_the_file_still_names_its_documents(web_settings, tmp_path) -> None:
    # Documents outlive the register entry that found them; falling back to the
    # id keeps them reachable instead of grouping them under a blank label.
    seed(web_settings, [source("a", "deleted-channel")], [])
    body = app_for(web_settings, tmp_path).get("/api/web/sources").json()
    assert body["channels"][0]["label"] == "deleted-channel"


def test_a_document_reports_the_state_the_refresh_loop_recorded(web_settings, tmp_path) -> None:
    # Recorded, never inferred — the tree badges these, so they must survive
    # the round trip rather than being recomputed from what happens to be set.
    gone = source("a", "seeds", state="gone", state_reason="404 twice", revision=4)
    seed(web_settings, [gone], [])
    body = app_for(web_settings, tmp_path).get("/api/web/sources").json()
    stored = body["channels"][0]["sources"][0]
    assert stored["state"] == "gone"
    assert stored["state_reason"] == "404 twice"
    assert stored["revision"] == 4
    # A gone source keeps its chunks but can no longer back a live citation.
    assert stored["verifiable"] is False


def test_a_live_document_is_verifiable(web_settings, tmp_path) -> None:
    seed(web_settings, [source("a", "seeds")], [])
    body = app_for(web_settings, tmp_path).get("/api/web/sources").json()
    assert body["channels"][0]["sources"][0]["verifiable"] is True


def test_a_document_cites_where_a_reader_should_go_not_where_we_fetched(
    web_settings, tmp_path
) -> None:
    # github_docs fetches raw Markdown and cites the rendered page.
    raw = source("a", "seeds", display_url="https://github.com/o/r/blob/HEAD/readme.md")
    seed(web_settings, [raw], [])
    body = app_for(web_settings, tmp_path).get("/api/web/sources").json()
    assert body["channels"][0]["sources"][0]["url"].startswith("https://github.com/")


def test_chunks_come_back_in_reading_order_with_their_headings(web_settings, tmp_path) -> None:
    parent = source("a", "seeds")
    seed(
        web_settings,
        [parent],
        [chunk(parent, 2, "Third"), chunk(parent, 0, "First"), chunk(parent, 1, "Second")],
    )
    body = app_for(web_settings, tmp_path).get(f"/api/web/sources/{parent.key}/chunks").json()
    assert body["total"] == 3
    assert [item["heading"] for item in body["chunks"]] == ["First", "Second", "Third"]
    # The anchor is what makes an article citation deep-link the way a
    # transcript citation seeks to a timestamp.
    assert body["chunks"][0]["anchor"] == "first"


def test_chunks_of_one_document_never_include_another_s(web_settings, tmp_path) -> None:
    mine, theirs = source("a", "seeds"), source("b", "seeds")
    seed(web_settings, [mine, theirs], [chunk(mine, 0, "Mine"), chunk(theirs, 0, "Theirs")])
    body = app_for(web_settings, tmp_path).get(f"/api/web/sources/{mine.key}/chunks").json()
    assert [item["heading"] for item in body["chunks"]] == ["Mine"]


def test_an_unknown_document_is_empty_rather_than_an_error(web_settings, tmp_path) -> None:
    seed(web_settings, [source("a", "seeds")], [])
    response = app_for(web_settings, tmp_path).get("/api/web/sources/web:nosuchkey/chunks")
    assert response.status_code == 200
    assert response.json()["total"] == 0


def test_an_empty_store_reports_no_channels_rather_than_failing(web_settings, tmp_path) -> None:
    body = app_for(web_settings, tmp_path).get("/api/web/sources").json()
    assert body == {"channels": [], "totals": {"channels": 0, "sources": 0, "chunks": 0}}

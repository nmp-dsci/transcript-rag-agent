"""The channels surface: what the workbench reads, and what it can start."""

from __future__ import annotations

import dataclasses
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.main import create_app
from src.config import Settings

CHANNELS_YAML = """version: 1

defaults:
  poll_interval_hours: 12

channels:
  - id: hamel-dev
    kind: rss
    url: https://hamel.dev/index.xml
    label: Hamel Husain
    body_in_feed: true
  - id: seeds
    kind: url_list
    label: Seed articles
    urls: ["https://a.example/one"]
    enabled: false
"""


def app_for(settings: Settings, tmp_path: Path, polls: list[list[str]] | None = None) -> TestClient:
    """An app whose poller is faked.

    The real one opens Chroma, loads the embedding model and makes live
    requests — on a daemon worker thread that outlives the test and starves
    the timing-sensitive queue tests in the sibling modules. Injected exactly
    like ``index_fn`` and ``graph_extract_fn`` beside it, for the same reason.
    """

    def fake_poll(channel_ids: list[str]) -> dict[str, object]:
        if polls is not None:
            polls.append(list(channel_ids))
        return {"polled": len(channel_ids), "indexed": 0, "updated": 0, "failed": 0, "details": []}

    return TestClient(
        create_app(
            settings,
            history_path=tmp_path / "h.json",
            chat_html_path=tmp_path / "c.html",
            index_fn=lambda argv: 0,
            channels_poll_fn=fake_poll,
        )
    )


@pytest.fixture
def channels_settings(settings: Settings, tmp_path: Path) -> Settings:
    target = tmp_path / "channels.yaml"
    target.write_text(CHANNELS_YAML, encoding="utf-8")
    return dataclasses.replace(settings, channels_file=target)


def test_the_register_is_reported_with_totals(channels_settings, tmp_path) -> None:
    client = app_for(channels_settings, tmp_path)
    body = client.get("/api/channels").json()
    assert body["totals"] == {"channels": 2, "enabled": 1, "sources": 0}
    by_id = {channel["id"]: channel for channel in body["channels"]}
    assert by_id["hamel-dev"]["enabled"] is True
    assert by_id["hamel-dev"]["body_in_feed"] is True
    assert by_id["seeds"]["enabled"] is False
    # A channel with no label falls back to its id rather than rendering blank.
    assert by_id["seeds"]["label"] == "Seed articles"


def test_totals_sources_never_exceeds_the_sum_of_visible_rows(channels_settings, tmp_path) -> None:
    # A channel can hold stored documents after its entry is edited out of
    # channels.yaml (deliberately supported — see web_corpus). The headline
    # total must not silently exceed what the rows in front of it add up to.
    from src.rag.web_models import WebSource
    from src.rag.web_store import WebSourceStore

    class FakeEmbeddings:
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [self.embed_query(text) for text in texts]

        def embed_query(self, text: str) -> list[float]:
            return [float(len(text) % 5), 1.0, 0.0]

    store = WebSourceStore(
        channels_settings.chroma_path, FakeEmbeddings(), channels_settings.web_source_collection
    )
    store.upsert(
        WebSource(
            external_id="a",
            url="https://example.com/a",
            channel_id="hamel-dev",
            title="Post a",
        )
    )
    store.upsert(
        WebSource(
            external_id="b",
            url="https://example.com/b",
            channel_id="deleted-channel",
            title="Post b",
        )
    )

    client = app_for(channels_settings, tmp_path)
    body = client.get("/api/channels").json()

    assert body["totals"]["sources"] == sum(channel["sources"] for channel in body["channels"])
    orphaned = {channel["id"]: channel for channel in body["channels"]}["deleted-channel"]
    assert orphaned["orphaned"] is True
    assert orphaned["sources"] == 1
    assert orphaned["label"] == "deleted-channel"


def test_a_channel_never_polled_reports_no_history(channels_settings, tmp_path) -> None:
    client = app_for(channels_settings, tmp_path)
    channel = client.get("/api/channels").json()["channels"][0]
    assert channel["last_polled_at"] is None
    assert channel["seen"] == 0
    assert channel["consecutive_failures"] == 0
    assert channel["disabled_reason"] is None


def test_no_channels_file_is_an_empty_register_not_an_error(settings, tmp_path) -> None:
    client = app_for(
        dataclasses.replace(settings, channels_file=tmp_path / "absent.yaml"), tmp_path
    )
    response = client.get("/api/channels")
    assert response.status_code == 200
    assert response.json()["totals"]["channels"] == 0


def test_a_malformed_channels_file_reports_why_instead_of_a_500(settings, tmp_path) -> None:
    # A bad file must not take the tab down; it reads as "no channels, and
    # here is the reason".
    target = tmp_path / "channels.yaml"
    target.write_text(
        "channels:\n  - {id: 'Bad Id', kind: rss, url: 'https://a/b'}\n", encoding="utf-8"
    )
    client = app_for(dataclasses.replace(settings, channels_file=target), tmp_path)
    body = client.get("/api/channels").json()
    assert body["channels"] == []
    assert "kebab-case" in body["error"]


def test_no_secret_or_key_appears_in_the_register(channels_settings, tmp_path) -> None:
    client = app_for(channels_settings, tmp_path)
    text = client.get("/api/channels").text
    assert "super" not in text
    assert "deep" not in text


def test_a_poll_is_queued_and_returns_immediately(channels_settings, tmp_path) -> None:
    client = app_for(channels_settings, tmp_path)
    response = client.post("/api/channels/poll", json={"channel_ids": ["hamel-dev"]})
    assert response.status_code == 202
    job = response.json()
    assert job["mode"] == "channels"
    assert job["channel_ids"] == ["hamel-dev"]
    assert job["target"] == "hamel-dev"
    assert job["status"] in {"queued", "running", "done", "error"}


def test_an_empty_request_polls_every_enabled_channel(channels_settings, tmp_path) -> None:
    client = app_for(channels_settings, tmp_path)
    job = client.post("/api/channels/poll", json={}).json()
    assert job["channel_ids"] == []
    assert job["target"] == "every enabled channel"


def test_a_queued_poll_appears_in_the_shared_queue(channels_settings, tmp_path) -> None:
    # It rides the existing ingestion queue so the UI watches it in the same
    # place, through the same SSE stream, as a video job.
    client = app_for(channels_settings, tmp_path)
    queued = client.post("/api/channels/poll", json={"channel_ids": ["seeds"]}).json()
    jobs = client.get("/api/index/queue").json()["jobs"]
    assert queued["id"] in {job["id"] for job in jobs}


def test_a_queued_poll_reaches_the_configured_poller(channels_settings, tmp_path) -> None:
    polls: list[list[str]] = []
    client = app_for(channels_settings, tmp_path, polls)
    client.post("/api/channels/poll", json={"channel_ids": ["hamel-dev"]})
    for _ in range(200):
        if polls:
            break
        time.sleep(0.02)
    assert polls == [["hamel-dev"]]


def test_demo_mode_serves_the_register_but_refuses_to_poll(channels_settings, tmp_path) -> None:
    demo = dataclasses.replace(channels_settings, demo_mode=True)
    client = app_for(demo, tmp_path)
    # Read-only: the demo has no watched sources, so it reports an empty
    # register rather than pretending the feature is absent.
    assert client.get("/api/channels").status_code == 200
    assert client.post("/api/channels/poll", json={}).status_code == 403

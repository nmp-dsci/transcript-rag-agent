"""``channels poll`` end to end: the real caller path, not a reimplementation.

test_ingest.py already proves ``WebIngestor.ingest`` itself leaves a failed
candidate unrecorded. This module proves the thing that actually matters for
correctness: that ``_run_poll`` — the code that decides whether to call
``state.remember`` — still gets that decision right, so a regression that
moved ``remember`` back to before ``ingest`` would fail a test here.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from src.channels.cli import _run_poll
from src.channels.ingest import WebIngestor
from src.channels.registry import ChannelStateStore, state_path
from src.channels.robots import RobotsPolicy
from src.config import Settings
from src.documents.models import FetchedPage
from src.rag.web_store import WebChunkStore, WebSourceStore

CHANNELS_YAML = """version: 1

channels:
  - id: feed
    kind: url_list
    urls: ["https://a.example/post"]
"""

THIN_PAGE = "<html><body><nav>Log In</nav></body></html>"
ARTICLE_PAGE = """<html><head><title>Evals guide</title></head><body>
<article>
<h2>Error analysis first</h2>
<p>{filler}</p>
</article>
</body></html>""".format(filler="Read fifty real traces before writing a judge. " * 40)


def poll_args(*, force: bool = False) -> argparse.Namespace:
    return argparse.Namespace(id=[], dry_run=False, force=force, reset=False)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    channels_file = tmp_path / "channels.yaml"
    channels_file.write_text(CHANNELS_YAML, encoding="utf-8")
    return Settings(
        superdata_api_key="super",
        supadata_api_keys=("super",),
        deepseek_api_key="deep",
        deepseek_model="deepseek-v4",
        deepseek_base_url=None,
        chroma_path=tmp_path / "chroma",
        mlflow_tracking_uri=f"file:{tmp_path / 'mlruns'}",
        mlflow_experiment_name="test-cli",
        log_transcript_artifacts=False,
        channels_file=channels_file,
    )


def test_a_candidate_whose_first_poll_ingest_fails_is_retried_by_the_next_poll(
    monkeypatch, settings: Settings, embeddings, permissive_robots: RobotsPolicy
) -> None:
    ingestor = WebIngestor(
        source_store=WebSourceStore(
            settings.chroma_path, embeddings, settings.web_source_collection
        ),
        chunk_store=WebChunkStore(settings.chroma_path, embeddings, settings.web_chunk_collection),
        robots=permissive_robots,
    )
    pages = {"https://a.example/post": THIN_PAGE}

    def fetch(url: str, **kwargs) -> FetchedPage:
        return FetchedPage(
            requested_url=url,
            url=url,
            status_code=200,
            content_type="text/html",
            body=pages[url],
        )

    ingestor.fetch = fetch
    monkeypatch.setattr("src.channels.cli._stores", lambda _settings: (ingestor, permissive_robots))

    # First poll: the only candidate extracts below the word floor, so nothing
    # is stored. Its id must not be recorded as seen.
    assert _run_poll(poll_args(), settings) == 0

    states = ChannelStateStore(state_path(settings.chroma_path))
    assert "https://a.example/post" not in states.get("feed").seen_ids
    assert ingestor.source_store.get("https://a.example/post") is None

    # Second poll offers the same, now-succeeding, URL again — proving it was
    # never marked seen — and this time it is recorded.
    pages["https://a.example/post"] = ARTICLE_PAGE
    assert _run_poll(poll_args(force=True), settings) == 0

    states = ChannelStateStore(state_path(settings.chroma_path))
    assert "https://a.example/post" in states.get("feed").seen_ids
    assert ingestor.source_store.get("https://a.example/post") is not None

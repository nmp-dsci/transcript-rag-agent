from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.channels.models import ChannelConfig, ChannelConfigError
from src.channels.registry import (
    ChannelStateStore,
    append_channel,
    load_registry,
    state_path,
)

SAMPLE = """# why we watch these
version: 1

defaults:
  poll_interval_hours: 12

channels:
  # measured: the feed carries the whole post
  - id: hamel-dev
    kind: rss
    url: https://hamel.dev/index.xml
    body_in_feed: true
  - id: seeds
    kind: url_list
    urls: ["https://a.example/one"]
    enabled: false
"""


def test_a_missing_file_is_an_empty_registry_not_an_error(tmp_path: Path) -> None:
    # A checkout that has never configured a channel is an ordinary state.
    registry = load_registry(tmp_path / "absent.yaml")
    assert len(registry) == 0
    assert registry.enabled() == ()


def test_channels_load_with_file_defaults_applied(tmp_path: Path) -> None:
    target = tmp_path / "channels.yaml"
    target.write_text(SAMPLE, encoding="utf-8")
    registry = load_registry(target)
    assert [channel.id for channel in registry] == ["hamel-dev", "seeds"]
    assert registry.get("hamel-dev").poll_interval_hours == 12
    assert registry.get("hamel-dev").body_in_feed is True
    assert [channel.id for channel in registry.enabled()] == ["hamel-dev"]
    assert registry.get("absent") is None


def test_a_duplicate_id_is_rejected(tmp_path: Path) -> None:
    target = tmp_path / "channels.yaml"
    target.write_text(
        "channels:\n"
        "  - {id: dup, kind: rss, url: 'https://a.example/1'}\n"
        "  - {id: dup, kind: rss, url: 'https://a.example/2'}\n",
        encoding="utf-8",
    )
    with pytest.raises(ChannelConfigError, match="appears twice"):
        load_registry(target)


def test_unknown_top_level_keys_are_rejected(tmp_path: Path) -> None:
    target = tmp_path / "channels.yaml"
    target.write_text("channelz: []\n", encoding="utf-8")
    with pytest.raises(ChannelConfigError, match="unknown top-level keys"):
        load_registry(target)


def test_invalid_yaml_names_the_file(tmp_path: Path) -> None:
    target = tmp_path / "channels.yaml"
    target.write_text("channels: [\n", encoding="utf-8")
    with pytest.raises(ChannelConfigError, match="not valid YAML"):
        load_registry(target)


def test_appending_a_channel_preserves_the_comments_already_in_the_file(tmp_path: Path) -> None:
    # The comments record what was *measured* about each source, which is half
    # the reason the file is committed. A YAML round-trip would strip them.
    target = tmp_path / "channels.yaml"
    target.write_text(SAMPLE, encoding="utf-8")
    append_channel(ChannelConfig(id="new-feed", kind="atom", url="https://b.example/a.xml"), target)
    text = target.read_text(encoding="utf-8")
    assert "# why we watch these" in text
    assert "# measured: the feed carries the whole post" in text
    assert [channel.id for channel in load_registry(target)] == [
        "hamel-dev",
        "seeds",
        "new-feed",
    ]


def test_appending_creates_the_file_when_it_does_not_exist(tmp_path: Path) -> None:
    target = tmp_path / "channels.yaml"
    append_channel(ChannelConfig(id="first", kind="rss", url="https://a.example/f.xml"), target)
    registry = load_registry(target)
    assert [channel.id for channel in registry] == ["first"]
    # The generated header records the one scope rule that must not be lost.
    assert "YouTube is deliberately absent" in target.read_text(encoding="utf-8")


def test_appending_a_duplicate_id_is_refused(tmp_path: Path) -> None:
    target = tmp_path / "channels.yaml"
    target.write_text(SAMPLE, encoding="utf-8")
    with pytest.raises(ChannelConfigError, match="already in"):
        append_channel(ChannelConfig(id="seeds", kind="url_list", urls=("https://x/y",)), target)


def test_state_lives_beside_the_store_it_describes(tmp_path: Path) -> None:
    assert state_path(tmp_path / "chroma") == tmp_path / "channels" / "state.json"


def test_state_round_trips_through_the_file(tmp_path: Path) -> None:
    path = state_path(tmp_path / "chroma")
    store = ChannelStateStore(path)
    state = store.get("hamel-dev")
    state.etag = '"abc"'
    state.interval_hours = 96.0
    state.remember(["g1", "g2"])
    store.put(state)
    store.save()

    reloaded = ChannelStateStore(path).get("hamel-dev")
    assert reloaded.etag == '"abc"'
    assert reloaded.interval_hours == 96.0
    assert reloaded.seen_ids == ["g1", "g2"]


def test_an_unknown_channel_reads_as_never_polled_rather_than_missing(tmp_path: Path) -> None:
    store = ChannelStateStore(state_path(tmp_path / "chroma"))
    state = store.get("brand-new")
    assert state.last_polled_at is None
    assert state.seen_ids == []


def test_unreadable_state_is_discarded_rather_than_raised(tmp_path: Path) -> None:
    # Derived state: a corrupt file means "poll everything again", which is
    # correct and cheap, not a reason to take a command down.
    path = state_path(tmp_path / "chroma")
    path.parent.mkdir(parents=True)
    path.write_text("{not json", encoding="utf-8")
    assert ChannelStateStore(path).get("anything").etag is None


def test_saving_leaves_no_temporary_file_behind(tmp_path: Path) -> None:
    path = state_path(tmp_path / "chroma")
    store = ChannelStateStore(path)
    store.put(store.get("a"))
    store.save()
    assert sorted(child.name for child in path.parent.iterdir()) == ["state.json"]
    assert json.loads(path.read_text(encoding="utf-8"))["version"] == 1

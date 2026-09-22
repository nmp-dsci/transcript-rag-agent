from __future__ import annotations

import pytest

from src.channels.models import (
    DEFAULT_INTERVAL_HOURS,
    SEEN_ID_LIMIT,
    ChannelConfig,
    ChannelConfigError,
    ChannelState,
)


def test_a_channel_needs_a_url_unless_it_carries_its_own_list() -> None:
    with pytest.raises(ChannelConfigError, match="needs a url"):
        ChannelConfig(id="feed", kind="rss")
    with pytest.raises(ChannelConfigError, match="needs urls"):
        ChannelConfig(id="hand-picked", kind="url_list")
    # A url_list is complete without a url, which is the whole point of it.
    assert ChannelConfig(id="hand-picked", kind="url_list", urls=("https://a.example/x",)).urls


def test_youtube_is_not_a_channel_kind() -> None:
    # Video ingestion stays the manual path; there is deliberately no poller.
    with pytest.raises(ChannelConfigError, match="is not one of"):
        ChannelConfig(id="yt", kind="youtube_channel", url="https://youtube.com/feeds/x")


@pytest.mark.parametrize("bad", ["Feed", "feed_one", "-feed", "", "a" * 70])
def test_ids_must_be_kebab_case(bad: str) -> None:
    with pytest.raises(ChannelConfigError, match="kebab-case"):
        ChannelConfig(id=bad, kind="rss", url="https://a.example/f.xml")


def test_a_bad_regex_is_rejected_when_the_channel_loads_not_when_it_polls() -> None:
    # A pattern that fails at poll time is a channel that silently yields
    # nothing, which is the failure this check exists to turn into an error.
    with pytest.raises(ChannelConfigError, match="is not a regex"):
        ChannelConfig(id="feed", kind="rss", url="https://a.example/f.xml", include="[unclosed")


def test_include_filters_on_url_and_title_independently() -> None:
    channel = ChannelConfig(
        id="feed",
        kind="sitemap",
        url="https://a.example/sitemap.xml",
        include=r"^https://a\.example/blog/",
        include_title="(?i)agent",
    )
    assert channel.accepts("https://a.example/blog/one", "Building agents")
    assert not channel.accepts("https://a.example/pricing", "Building agents")
    assert not channel.accepts("https://a.example/blog/one", "Quarterly results")
    # A missing title cannot satisfy a title filter.
    assert not channel.accepts("https://a.example/blog/one", None)


def test_unknown_keys_are_an_error_rather_than_silently_ignored() -> None:
    with pytest.raises(ChannelConfigError, match="unknown keys"):
        ChannelConfig.from_dict(
            {"id": "feed", "kind": "rss", "url": "https://a.example/f.xml", "poll_interval": 3}
        )


def test_file_defaults_sit_underneath_a_channels_own_values() -> None:
    defaults = {"poll_interval_hours": 6.0, "min_words": 100}
    channel = ChannelConfig.from_dict(
        {"id": "feed", "kind": "rss", "url": "https://a.example/f.xml", "min_words": 900},
        defaults,
    )
    assert channel.poll_interval_hours == 6.0
    assert channel.min_words == 900


def test_serialising_a_channel_omits_defaults() -> None:
    # The file is committed and read by people, so a round-trip must not bury
    # the two interesting lines under fifteen restated defaults.
    data = ChannelConfig(id="feed", kind="rss", url="https://a.example/f.xml").to_dict()
    assert data == {"id": "feed", "kind": "rss", "url": "https://a.example/f.xml"}
    assert "poll_interval_hours" not in data


def test_remembering_ids_deduplicates_within_a_batch_and_against_history() -> None:
    state = ChannelState(channel_id="feed")
    state.remember(["a", "b", "a"])
    state.remember(["b", "c"])
    assert state.seen_ids == ["a", "b", "c"]


def test_remembered_ids_are_bounded_and_drop_the_oldest_first() -> None:
    state = ChannelState(channel_id="feed")
    state.remember([str(index) for index in range(SEEN_ID_LIMIT + 10)])
    assert len(state.seen_ids) == SEEN_ID_LIMIT
    assert state.seen_ids[0] == "10"
    assert state.seen_ids[-1] == str(SEEN_ID_LIMIT + 9)


def test_state_survives_a_dict_round_trip() -> None:
    state = ChannelState(channel_id="feed", etag='"abc"', interval_hours=48.0)
    state.remember(["g1"])
    restored = ChannelState.from_dict(state.to_dict())
    assert restored == state
    assert restored.interval_hours == 48.0
    assert ChannelState(channel_id="x").interval_hours == DEFAULT_INTERVAL_HOURS

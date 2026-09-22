from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.channels.models import (
    MAX_INTERVAL_HOURS,
    MIN_INTERVAL_HOURS,
    Candidate,
    ChannelConfig,
    ChannelState,
    PollResult,
)
from src.channels.schedule import is_due, record_poll, reset

NOW = datetime(2026, 9, 22, 12, tzinfo=timezone.utc)


def channel(**kwargs) -> ChannelConfig:
    return ChannelConfig(id="feed", kind="rss", url="https://a.example/f.xml", **kwargs)


def found(count: int = 1) -> PollResult:
    return PollResult(
        channel_id="feed",
        candidates=[
            Candidate(url=f"https://a.example/{i}", external_id=f"g{i}") for i in range(count)
        ],
    )


def nothing() -> PollResult:
    return PollResult(channel_id="feed", unchanged=True)


def test_a_channel_never_polled_is_due_immediately() -> None:
    assert is_due(channel(), ChannelState(channel_id="feed"), NOW)


def test_a_channel_is_not_due_again_until_its_interval_elapses() -> None:
    state = ChannelState(channel_id="feed")
    record_poll(channel(), state, found(), NOW)
    assert not is_due(channel(), state, NOW)
    assert is_due(channel(), state, NOW + timedelta(hours=state.interval_hours + 1))


def test_polls_that_find_nothing_back_off_to_the_ceiling() -> None:
    state = ChannelState(channel_id="feed")
    intervals = []
    for _ in range(6):
        record_poll(channel(), state, nothing(), NOW)
        intervals.append(state.interval_hours)
    assert intervals[0] == 48.0
    assert intervals[-1] == MAX_INTERVAL_HOURS
    assert all(later >= earlier for earlier, later in zip(intervals, intervals[1:]))


def test_polls_that_find_something_speed_up_to_the_floor() -> None:
    state = ChannelState(channel_id="feed", interval_hours=MAX_INTERVAL_HOURS)
    for _ in range(8):
        record_poll(channel(), state, found(), NOW)
    assert state.interval_hours == MIN_INTERVAL_HOURS


def test_record_poll_does_not_mark_candidates_as_seen() -> None:
    # Marking an id as seen is the caller's job, done only once that
    # candidate's ingest has actually stored something. A candidate that
    # fails on this poll (network error, robots block, below the word floor)
    # must stay retryable, which only works if record_poll never claims it.
    state = ChannelState(channel_id="feed")
    record_poll(channel(), state, found(3), NOW)
    assert state.seen_ids == []
    assert state.last_polled_at == NOW.isoformat()


def test_validators_are_only_overwritten_when_the_poll_returned_one() -> None:
    state = ChannelState(channel_id="feed", etag='"old"', last_modified="then")
    record_poll(channel(), state, PollResult(channel_id="feed", unchanged=True), NOW)
    assert state.etag == '"old"'
    assert state.last_modified == "then"
    record_poll(channel(), state, PollResult(channel_id="feed", etag='"new"'), NOW)
    assert state.etag == '"new"'


def test_failures_back_off_and_then_disable_the_channel() -> None:
    # Five rather than one, because a feed answering 404 once and 200 a minute
    # later is ordinary — measured on a real source.
    state = ChannelState(channel_id="feed")
    config = channel()
    for attempt in range(1, config.disable_after_failures):
        record_poll(config, state, PollResult(channel_id="feed", error="boom"), NOW)
        assert state.consecutive_failures == attempt
        assert state.disabled_reason is None
        assert is_due(config, state, NOW + timedelta(days=30))
    record_poll(config, state, PollResult(channel_id="feed", error="boom"), NOW)
    assert state.disabled_reason is not None
    assert "5 consecutive failed polls" in state.disabled_reason
    assert not is_due(config, state, NOW + timedelta(days=30))


def test_one_success_clears_an_accumulated_failure_count() -> None:
    state = ChannelState(channel_id="feed")
    record_poll(channel(), state, PollResult(channel_id="feed", error="boom"), NOW)
    record_poll(channel(), state, found(), NOW)
    assert state.consecutive_failures == 0
    assert state.last_error is None


def test_a_channel_switched_off_in_the_file_is_never_due_even_forced() -> None:
    # Re-enabling is an edit to channels.yaml, a visible act, not a flag that
    # quietly resurrects a source somebody turned off.
    config = channel(enabled=False)
    state = ChannelState(channel_id="feed")
    assert not is_due(config, state, NOW)
    assert not is_due(config, state, NOW, force=True)


def test_force_polls_an_enabled_channel_that_is_merely_early() -> None:
    state = ChannelState(channel_id="feed")
    record_poll(channel(), state, found(), NOW)
    assert not is_due(channel(), state, NOW)
    assert is_due(channel(), state, NOW, force=True)


def test_a_disabled_channel_stays_disabled_until_reset() -> None:
    config = channel()
    state = ChannelState(channel_id="feed")
    for _ in range(config.disable_after_failures):
        record_poll(config, state, PollResult(channel_id="feed", error="boom"), NOW)
    assert not is_due(config, state, NOW, force=True)
    reset(state, config)
    assert is_due(config, state, NOW)
    assert state.consecutive_failures == 0
    assert state.etag is None


def test_reset_keeps_the_ids_already_seen() -> None:
    # Re-enabling a channel is not a request to re-ingest its whole archive.
    config = channel()
    state = ChannelState(channel_id="feed")
    record_poll(config, state, found(2), NOW)
    state.remember(["g0", "g1"])
    reset(state, config)
    assert state.seen_ids == ["g0", "g1"]

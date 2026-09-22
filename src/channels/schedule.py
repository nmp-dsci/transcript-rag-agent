"""When a channel is next worth polling, and what one poll did to that answer.

The cadence is adaptive for a measured reason: the registered sources differ by
three orders of magnitude in how often they publish, and one fixed interval is
either wasteful for the quiet ones or late for the busy ones. So the interval
moves with what the polls actually find — doubling on a poll that finds nothing,
halving on one that finds something — inside hard bounds, so a quiet source is
still checked weekly and a busy one is never hammered.

Failures back off the same way and then stop: five consecutive failures disables
the channel. Five rather than one because a feed answering ``404`` once and
``200`` a minute later is ordinary — measured, on a real source — while five in
a row is a source that has moved or died and will not fix itself.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.channels.models import (
    MAX_INTERVAL_HOURS,
    MIN_INTERVAL_HOURS,
    ChannelConfig,
    ChannelState,
    PollResult,
)


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


def clamp_interval(hours: float) -> float:
    return max(MIN_INTERVAL_HOURS, min(MAX_INTERVAL_HOURS, hours))


def is_due(
    channel: ChannelConfig,
    state: ChannelState,
    now: datetime | None = None,
    force: bool = False,
) -> bool:
    """Whether this channel should be polled.

    A disabled channel is never due, and ``force`` deliberately does not
    override that: re-enabling is an edit to ``channels.yaml`` or a reset, both
    of which are visible acts, rather than a flag that quietly resurrects a
    source somebody turned off.
    """
    if not channel.enabled or state.disabled_reason:
        return False
    if force:
        return True
    due_at = _parse(state.next_due_at)
    return due_at is None or due_at <= (now or _now())


def record_poll(
    channel: ChannelConfig,
    state: ChannelState,
    result: PollResult,
    now: datetime | None = None,
) -> ChannelState:
    """Fold one poll's outcome into the channel's state, in place.

    Deliberately does **not** mark ``result.candidates`` as seen: that is the
    caller's job, once each candidate's ingest has actually stored something.
    A candidate that failed on this poll — a transient network error, a
    temporary robots block, an extract below the word floor — must still be
    offered again next time, and it can only stay retryable if its id was
    never recorded as seen in the first place.

    Returns the same object it was given, so a caller can pass it straight to
    :meth:`ChannelStateStore.put` without wondering which copy is current.
    """
    moment = now or _now()
    state.last_polled_at = moment.isoformat()

    if result.error:
        state.consecutive_failures += 1
        state.last_error = result.error
        if state.consecutive_failures >= channel.disable_after_failures:
            state.disabled_reason = (
                f"{state.consecutive_failures} consecutive failed polls; last error: {result.error}"
            )
        # Back off exponentially on the way to being disabled, so a host having
        # a bad hour is not hit once an hour throughout it.
        state.interval_hours = clamp_interval(state.interval_hours * 2)
    else:
        state.consecutive_failures = 0
        state.last_error = None
        if result.etag is not None:
            state.etag = result.etag
        if result.last_modified is not None:
            state.last_modified = result.last_modified
        found = len(result.candidates)
        base = state.interval_hours or channel.poll_interval_hours
        state.interval_hours = clamp_interval(base / 2 if found else base * 2)

    state.next_due_at = (moment + timedelta(hours=state.interval_hours)).isoformat()
    return state


def reset(state: ChannelState, channel: ChannelConfig) -> ChannelState:
    """Clear a channel's memory so the next poll starts from scratch.

    Keeps ``seen_ids``: re-enabling a channel is not a request to re-ingest its
    whole archive, and forgetting what it has already offered is exactly how
    that would happen.
    """
    state.disabled_reason = None
    state.consecutive_failures = 0
    state.last_error = None
    state.etag = None
    state.last_modified = None
    state.next_due_at = None
    state.interval_hours = channel.poll_interval_hours
    return state

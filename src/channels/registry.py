"""Reading and writing the two halves of a channel: intent and memory.

``channels.yaml`` is committed, because *which* sources this corpus watches is
a decision worth reviewing in a diff. ``.yt-agent/channels/state.json`` is not,
because ETags and cursors are derived — deleting the file costs one extra poll,
not a corpus. That split is the same one ``guides/`` already uses against
``.yt-agent/``, and it is the reason a poll run never dirties the working tree.

Writing YAML back out is deliberately limited to appending a channel (the
``channels add`` path). A round-trip through PyYAML would strip every comment
in the file, and the comments are half the point of committing it — they record
what was *measured* about each source, which is exactly the knowledge that goes
missing otherwise.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.channels.models import ChannelConfig, ChannelConfigError, ChannelState

logger = logging.getLogger(__name__)

#: ``channels.yaml`` at the repo root, independent of the working directory —
#: the same convention :mod:`src.guides.catalog` uses for ``guides/``.
DEFAULT_CHANNELS_FILE = Path(__file__).resolve().parents[2] / "channels.yaml"

STATE_DIRNAME = "channels"
STATE_FILENAME = "state.json"

#: File-level keys that are not channels.
_TOP_LEVEL_KEYS = frozenset({"version", "defaults", "channels"})


@dataclass
class ChannelRegistry:
    """Every configured channel, and the state each one has accumulated."""

    channels: tuple[ChannelConfig, ...] = ()
    path: Path | None = None

    def get(self, channel_id: str) -> ChannelConfig | None:
        for channel in self.channels:
            if channel.id == channel_id:
                return channel
        return None

    def enabled(self) -> tuple[ChannelConfig, ...]:
        return tuple(channel for channel in self.channels if channel.enabled)

    def __len__(self) -> int:
        return len(self.channels)

    def __iter__(self) -> Iterator[ChannelConfig]:
        return iter(self.channels)


def load_registry(path: Path | str | None = None) -> ChannelRegistry:
    """Parse ``channels.yaml``.

    A missing file is an empty registry, not an error: a checkout that has
    never configured a channel is an ordinary state, and every command that
    reads this should say "no channels" rather than raise.
    """
    target = Path(path) if path is not None else DEFAULT_CHANNELS_FILE
    if not target.is_file():
        return ChannelRegistry(path=target)
    try:
        raw = yaml.safe_load(target.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ChannelConfigError(f"{target} is not valid YAML: {exc}") from exc
    if raw is None:
        return ChannelRegistry(path=target)
    if not isinstance(raw, dict):
        raise ChannelConfigError(f"{target} must be a mapping at the top level")
    unknown = set(raw) - _TOP_LEVEL_KEYS
    if unknown:
        raise ChannelConfigError(f"{target}: unknown top-level keys {', '.join(sorted(unknown))}")
    defaults = raw.get("defaults") or {}
    if not isinstance(defaults, dict):
        raise ChannelConfigError(f"{target}: defaults must be a mapping")
    entries = raw.get("channels") or []
    if not isinstance(entries, list):
        raise ChannelConfigError(f"{target}: channels must be a list")
    channels: list[ChannelConfig] = []
    seen: set[str] = set()
    for entry in entries:
        channel = ChannelConfig.from_dict(entry, defaults)
        if channel.id in seen:
            raise ChannelConfigError(f"{target}: channel id {channel.id!r} appears twice")
        seen.add(channel.id)
        channels.append(channel)
    return ChannelRegistry(channels=tuple(channels), path=target)


def append_channel(channel: ChannelConfig, path: Path | str | None = None) -> Path:
    """Add one channel to ``channels.yaml``, preserving the existing text.

    The file is appended to as text rather than re-serialised, so every comment
    already in it survives. The new block is written with PyYAML so the quoting
    is correct, then indented into the ``channels:`` list.
    """
    target = Path(path) if path is not None else DEFAULT_CHANNELS_FILE
    existing = load_registry(target)
    if existing.get(channel.id) is not None:
        raise ChannelConfigError(f"channel {channel.id!r} is already in {target}")

    block = yaml.safe_dump([channel.to_dict()], sort_keys=False, allow_unicode=True, width=100)
    # safe_dump writes a top-level sequence at column 0; the channels list is
    # indented by two, so every line shifts right by two.
    indented = "".join(f"  {line}\n" if line.strip() else "\n" for line in block.splitlines())

    if not target.is_file():
        target.write_text(
            "# channels.yaml — the sources this corpus watches.\n"
            "#\n"
            "# Text sources only. YouTube is deliberately absent: videos are added by hand\n"
            "# through the existing ingest form, so nothing here can spend a Supadata credit.\n"
            "version: 1\n\nchannels:\n" + indented,
            encoding="utf-8",
        )
        return target

    text = target.read_text(encoding="utf-8")
    if "channels:" not in text:
        text = text.rstrip("\n") + "\n\nchannels:\n"
    target.write_text(text.rstrip("\n") + "\n" + indented, encoding="utf-8")
    return target


def state_path(chroma_path: Path | str) -> Path:
    """Where channel state lives, beside the store it describes."""
    return Path(chroma_path).parent / STATE_DIRNAME / STATE_FILENAME


class ChannelStateStore:
    """Every channel's runtime memory, in one JSON file.

    One file rather than one per channel: the whole map is small (a few
    thousand ids at most), a poll run reads all of it anyway to decide what is
    due, and a single atomic replace is easier to reason about than N writes
    that can half-succeed.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._states: dict[str, ChannelState] | None = None

    def _load(self) -> dict[str, ChannelState]:
        if self._states is not None:
            return self._states
        states: dict[str, ChannelState] = {}
        if self.path.is_file():
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                # Derived state: an unreadable file means "poll everything
                # again", which is correct and cheap, not a reason to fail.
                logger.warning("discarding unreadable channel state at %s", self.path)
                raw = {}
            for channel_id, data in (raw.get("channels") or {}).items():
                if isinstance(data, dict):
                    states[channel_id] = ChannelState.from_dict({**data, "channel_id": channel_id})
        self._states = states
        return states

    def get(self, channel_id: str) -> ChannelState:
        """This channel's state, or a fresh one. Never ``None``."""
        states = self._load()
        if channel_id not in states:
            states[channel_id] = ChannelState(channel_id=channel_id)
        return states[channel_id]

    def all(self) -> dict[str, ChannelState]:
        return dict(self._load())

    def put(self, state: ChannelState) -> None:
        self._load()[state.channel_id] = state

    def save(self) -> Path:
        states = self._load()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "version": 1,
            "channels": {
                channel_id: {
                    key: value for key, value in state.to_dict().items() if key != "channel_id"
                }
                for channel_id, state in sorted(states.items())
            },
        }
        # Write-then-replace: a kill mid-write leaves the previous state
        # intact rather than a truncated file that reads as "never polled".
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.path)
        return self.path

"""What a channel is, and what one poll of it produced.

A channel is a *watched source of URLs*. That is the whole abstraction: it is
asked "what is here now", it answers with candidates, and it remembers just
enough to make the next ask cheap. Fetching, extracting, chunking and embedding
are someone else's job — the same someone that already does all four for
transcripts.

Two objects, deliberately separated, because they have different lifetimes and
different homes:

* :class:`ChannelConfig` is *intent* — which sources we follow, and how. It is
  committed to ``channels.yaml`` so that adding a source shows up in a diff.
* :class:`ChannelState` is *runtime memory* — ETags, cursors, due times, the
  ids already seen. It lives under ``.yt-agent/`` which is gitignored, so a
  poll never dirties the tree and a cleared cache costs one extra poll rather
  than a lost corpus.

Text sources only. There is no YouTube poller: videos are added by hand through
the existing ingest form, so nothing here can spend a Supadata credit.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

#: The poller kinds. ``url_list`` is included for symmetry but polls nothing —
#: it is a hand-curated set of URLs that only changes when someone edits the
#: file, which makes it the safest possible place to start a corpus.
CHANNEL_KINDS = ("rss", "atom", "sitemap", "github_docs", "url_list")

#: Where a channel's "has this changed" answer comes from.
#:
#: ``etag`` a conditional GET, the cheapest signal and the one 8 of 11 probed
#: sources offer | ``lastmod`` a sitemap's per-URL date | ``content_hash`` no
#: signal at all, so the only way to know is to fetch and hash. That last case
#: is not exotic: two of the eleven registered sources have nothing else.
CHANGE_SIGNALS = ("etag", "lastmod", "content_hash")

#: The recorded states of one stored source. Recorded, never inferred — this
#: project already learned that lesson from summaries, where "no summary" could
#: not distinguish a video nobody had summarised from one whose provider
#: returned 402 and lost the attempt.
LIVE = "live"
CHANGED = "changed"
MOVED = "moved"
GONE = "gone"
BLOCKED = "blocked"
TRUNCATED = "truncated"
SOURCE_STATES = (LIVE, CHANGED, MOVED, GONE, BLOCKED, TRUNCATED)

#: Cadence bounds. A poll that finds nothing doubles the interval; one that
#: finds something halves it. Both are clamped, so a quiet source is checked
#: weekly rather than never and a busy one is not hammered.
MIN_INTERVAL_HOURS = 6.0
MAX_INTERVAL_HOURS = 168.0
DEFAULT_INTERVAL_HOURS = 24.0

#: Consecutive failed polls before a channel is disabled. Five, because a feed
#: that answers 404 once and 200 a minute later is ordinary (measured), while
#: five in a row is a source that has actually moved or died.
DEFAULT_DISABLE_AFTER = 5

#: Items accepted from a single poll. Uncapped, a first poll of one registered
#: feed would ingest 1,215 archive items, most of them off-topic.
DEFAULT_MAX_ITEMS = 25

#: Below this many words an extract is chrome — a nav bar, a cookie wall, or
#: the shell of a page whose content is rendered by JavaScript we do not run.
#: Failing loudly beats storing it as an article.
DEFAULT_MIN_WORDS = 250

#: How many external ids one channel remembers. Bounded because a 863-item feed
#: would otherwise grow the state file without limit; generous enough that an
#: id is never forgotten while it is still in the feed window.
SEEN_ID_LIMIT = 5000


class ChannelConfigError(ValueError):
    """``channels.yaml`` says something this code cannot act on."""


def _compiled(pattern: str | None, field_name: str, channel_id: str) -> re.Pattern[str] | None:
    if not pattern:
        return None
    try:
        return re.compile(pattern)
    except re.error as exc:
        raise ChannelConfigError(
            f"channel {channel_id!r}: {field_name} is not a regex: {exc}"
        ) from exc


@dataclass(frozen=True)
class ChannelConfig:
    """One watched source, as declared in ``channels.yaml``."""

    id: str
    kind: str
    label: str = ""
    #: The feed or sitemap to poll. Absent for ``url_list``, which carries
    #: :attr:`urls` instead, and for ``github_docs`` it is the repo page.
    url: str | None = None
    enabled: bool = True
    poll_interval_hours: float = DEFAULT_INTERVAL_HOURS
    max_items_per_poll: int = DEFAULT_MAX_ITEMS
    min_words: int = DEFAULT_MIN_WORDS
    disable_after_failures: int = DEFAULT_DISABLE_AFTER
    #: Keep only candidate URLs matching this. The registered Anthropic
    #: sitemap carries 535 locs of which 25 are the engineering blog; without
    #: a filter a sitemap channel is a whole-site crawl.
    include: str | None = None
    #: Keep only candidates whose title matches. For the 863-item Hugging Face
    #: feed this is the difference between a topic and a firehose.
    include_title: str | None = None
    #: The feed carries the article itself rather than a teaser. Measured per
    #: source; when true the page is never fetched, which on one registered
    #: source is not merely cheaper but the difference between the essay and a
    #: subscribe button.
    body_in_feed: bool = False
    change_signal: str = "etag"
    #: ``github_docs`` only: repo-relative file globs, pulled as raw Markdown
    #: rather than as the rendered HTML page.
    paths: tuple[str, ...] = ()
    #: ``url_list`` only.
    urls: tuple[str, ...] = ()
    topics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.id or not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,62}", self.id):
            raise ChannelConfigError(f"channel id {self.id!r} must be lowercase kebab-case")
        if self.kind not in CHANNEL_KINDS:
            raise ChannelConfigError(
                f"channel {self.id!r}: kind {self.kind!r} is not one of {', '.join(CHANNEL_KINDS)}"
            )
        if self.change_signal not in CHANGE_SIGNALS:
            raise ChannelConfigError(
                f"channel {self.id!r}: change_signal {self.change_signal!r} is not one of "
                f"{', '.join(CHANGE_SIGNALS)}"
            )
        if self.kind == "url_list":
            if not self.urls:
                raise ChannelConfigError(f"channel {self.id!r}: a url_list needs urls")
        elif not self.url:
            raise ChannelConfigError(f"channel {self.id!r}: kind {self.kind!r} needs a url")
        if self.max_items_per_poll < 1:
            raise ChannelConfigError(f"channel {self.id!r}: max_items_per_poll must be at least 1")
        # Validate the regexes at load time. A bad pattern discovered mid-poll
        # is a channel that silently yields nothing.
        _compiled(self.include, "include", self.id)
        _compiled(self.include_title, "include_title", self.id)

    @property
    def include_re(self) -> re.Pattern[str] | None:
        return _compiled(self.include, "include", self.id)

    @property
    def include_title_re(self) -> re.Pattern[str] | None:
        return _compiled(self.include_title, "include_title", self.id)

    def accepts(self, url: str, title: str | None = None) -> bool:
        """Whether a candidate survives this channel's filters."""
        pattern = self.include_re
        if pattern is not None and not pattern.search(url):
            return False
        title_pattern = self.include_title_re
        if title_pattern is not None and not title_pattern.search(title or ""):
            return False
        return True

    def to_dict(self) -> dict[str, Any]:
        """The YAML form, with defaults omitted so the file stays readable."""
        data: dict[str, Any] = {"id": self.id, "kind": self.kind}
        if self.label:
            data["label"] = self.label
        if self.url:
            data["url"] = self.url
        for name, default in (
            ("enabled", True),
            ("poll_interval_hours", DEFAULT_INTERVAL_HOURS),
            ("max_items_per_poll", DEFAULT_MAX_ITEMS),
            ("min_words", DEFAULT_MIN_WORDS),
            ("disable_after_failures", DEFAULT_DISABLE_AFTER),
            ("body_in_feed", False),
            ("change_signal", "etag"),
        ):
            value = getattr(self, name)
            if value != default:
                data[name] = value
        for name in ("include", "include_title"):
            value = getattr(self, name)
            if value:
                data[name] = value
        for name in ("paths", "urls", "topics"):
            value = getattr(self, name)
            if value:
                data[name] = list(value)
        return data

    @classmethod
    def from_dict(
        cls, data: dict[str, Any], defaults: dict[str, Any] | None = None
    ) -> "ChannelConfig":
        """One channel, with file-level ``defaults`` filled in underneath it."""
        if not isinstance(data, dict):
            raise ChannelConfigError(f"a channel must be a mapping, got {type(data).__name__}")
        merged: dict[str, Any] = {}
        known = set(cls.__dataclass_fields__)
        for source in (defaults or {}, data):
            for key, value in source.items():
                if key in known:
                    merged[key] = value
        for name in ("paths", "urls", "topics"):
            if name in merged and merged[name] is not None:
                merged[name] = tuple(merged[name])
        unknown_defaults = set(defaults or {}) - known
        if unknown_defaults:
            raise ChannelConfigError(
                f"channels.yaml defaults: unknown keys {', '.join(sorted(unknown_defaults))}"
            )
        unknown = set(data) - known
        if unknown:
            raise ChannelConfigError(
                f"channel {data.get('id', '?')!r}: unknown keys {', '.join(sorted(unknown))}"
            )
        return cls(**merged)


@dataclass
class ChannelState:
    """What one channel remembers between polls. Derived state, gitignored."""

    channel_id: str
    etag: str | None = None
    last_modified: str | None = None
    last_polled_at: str | None = None
    #: When this channel is next worth polling. ``None`` means "now".
    next_due_at: str | None = None
    #: The adaptive interval, which starts at the configured one and then
    #: doubles or halves with what the polls find.
    interval_hours: float = DEFAULT_INTERVAL_HOURS
    consecutive_failures: int = 0
    disabled_reason: str | None = None
    #: External ids already offered by this channel, newest last. This is the
    #: only discovery signal available for a feed with no ETag and no dates.
    seen_ids: list[str] = field(default_factory=list)
    last_error: str | None = None

    def remember(self, external_ids: list[str]) -> None:
        """Record ids as seen, keeping the newest :data:`SEEN_ID_LIMIT`.

        Deduplicates within the batch as well as against what is already
        known: a feed that lists the same guid twice is not two items.
        """
        known = set(self.seen_ids)
        for external_id in external_ids:
            if external_id in known:
                continue
            known.add(external_id)
            self.seen_ids.append(external_id)
        if len(self.seen_ids) > SEEN_ID_LIMIT:
            del self.seen_ids[: len(self.seen_ids) - SEEN_ID_LIMIT]

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel_id": self.channel_id,
            "etag": self.etag,
            "last_modified": self.last_modified,
            "last_polled_at": self.last_polled_at,
            "next_due_at": self.next_due_at,
            "interval_hours": self.interval_hours,
            "consecutive_failures": self.consecutive_failures,
            "disabled_reason": self.disabled_reason,
            "seen_ids": list(self.seen_ids),
            "last_error": self.last_error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChannelState":
        known = set(cls.__dataclass_fields__)
        return cls(**{key: value for key, value in data.items() if key in known})


@dataclass(frozen=True)
class Candidate:
    """One URL a channel is offering, with whatever the channel knew about it."""

    url: str
    #: Identity as the *channel* names it: a feed guid, a sitemap loc, a raw
    #: file path. Deliberately not derived from the URL — a post that moves
    #: keeps its guid, and an id hashed from the URL would mint a second
    #: document for the same article and leave the first behind as a duplicate
    #: nobody notices.
    external_id: str
    channel_id: str = ""
    title: str | None = None
    published_at: str | None = None
    #: The source's own claim about when this last changed. A claim, not proof:
    #: the content hash is what decides whether anything is re-embedded.
    updated_at: str | None = None
    #: The full article body when the feed carried it.
    body_html: str | None = None
    #: Where a *reader* should be sent, when that is not where we fetched
    #: from. ``github_docs`` fetches raw Markdown because the rendered page
    #: extracts to navigation chrome, but a citation pointing at raw text is
    #: no use to anyone — so the repo page is carried alongside.
    display_url: str | None = None

    @property
    def reader_url(self) -> str:
        return self.display_url or self.url


@dataclass
class PollResult:
    """What one poll of one channel produced."""

    channel_id: str
    candidates: list[Candidate] = field(default_factory=list)
    #: The server said 304, or the change signal did not move.
    unchanged: bool = False
    etag: str | None = None
    last_modified: str | None = None
    #: Recorded, not raised. One dead channel must not abandon a poll run
    #: halfway through — the same reason the ingestion queue records a failed
    #: job instead of taking its worker down with it.
    error: str | None = None
    #: Candidates dropped by :meth:`ChannelConfig.accepts`, counted so a filter
    #: that silently matches nothing is visible rather than mysterious.
    filtered_out: int = 0

    @property
    def new_candidates(self) -> list[Candidate]:
        return list(self.candidates)

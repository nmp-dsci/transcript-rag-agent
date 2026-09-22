"""Whether a URL may be fetched, read conservatively.

The policy is a deliberate choice, not a default, and it is stricter than the
letter of the robots spec requires.

By the spec, a crawler obeys the rule group that names its own user-agent, or
the ``*`` group when nothing names it. Since this agent identifies itself
honestly as ``yt-agent-corpus`` and no site has a rule for that name, the
literal reading is that only ``*`` applies — and on at least one registered
source that reading would permit fetching article pages while the same file
carries ``Disallow: /`` for eleven named AI crawlers.

That file is legible about the site's intent, so this module treats a blanket
disallow aimed at AI crawlers as applying to us too. The cost is real and
measured: article *pages* on such a source are skipped. The consolation is
that the same sites publish full articles in their RSS feeds for readers, and
a feed the site offers is a feed the site meant to be read.

``robots.txt`` is not terms of service, and this module does not pretend to
resolve that. It resolves one narrow question — may this fetch be attempted —
in the direction of restraint.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from urllib.parse import urlparse, urlunparse
from urllib.robotparser import RobotFileParser

logger = logging.getLogger(__name__)

#: How this agent identifies itself. Honest, with a repo URL, and nothing
#: derived from any user's session.
USER_AGENT = "yt-agent-corpus/1.0 (+https://github.com/nmp-dsci/transcript-rag-agent)"

#: The token a robots group would use to name us.
USER_AGENT_TOKEN = "yt-agent-corpus"

#: Groups whose blanket disallow is treated as covering this agent. These are
#: the tokens sites use to say "not for AI ingestion"; since that is what this
#: corpus is, a rule aimed at them is taken as aimed at us.
AI_CRAWLER_TOKENS = (
    "GPTBot",
    "ChatGPT-User",
    "OAI-SearchBot",
    "ClaudeBot",
    "Claude-Web",
    "anthropic-ai",
    "CCBot",
    "Google-Extended",
    "Applebot-Extended",
    "Bytespider",
    "meta-externalagent",
    "FacebookBot",
    "cohere-ai",
    "cohere-training-data-crawler",
    "AI2Bot",
    "Ai2Bot-Dolma",
    "PerplexityBot",
    "Amazonbot",
    "Diffbot",
    "Omgilibot",
    "TimpiBot",
    "YouBot",
    "ImagesiftBot",
)

#: Seconds between requests to one host when the site names no crawl delay.
#: Polite by default; a poll run is never in a hurry.
DEFAULT_CRAWL_DELAY = 1.0

#: How long a fetched robots.txt is trusted. A poll run finishes well inside
#: this, so one host is read once per run.
CACHE_SECONDS = 3600.0


@dataclass
class RobotsVerdict:
    allowed: bool
    #: The group that refused, when one did. Named so a skipped source reads
    #: as a policy decision rather than a bug.
    refused_by: str | None = None
    crawl_delay: float = DEFAULT_CRAWL_DELAY

    @property
    def reason(self) -> str:
        if self.allowed:
            return "allowed"
        if self.refused_by == USER_AGENT_TOKEN:
            return "robots.txt disallows this path for all crawlers"
        return f"robots.txt disallows this path for {self.refused_by}, which covers this agent"


@dataclass
class _Entry:
    parser: RobotFileParser | None
    fetched_at: float
    crawl_delay: float = DEFAULT_CRAWL_DELAY


@dataclass
class RobotsPolicy:
    """Per-host ``robots.txt``, cached, with a token-bucket of one per host.

    One instance per poll run. It is shared across channels because politeness
    is a property of the host, not of the channel that happens to link there.
    """

    fetcher: object | None = None
    _cache: dict[str, _Entry] = field(default_factory=dict, repr=False)
    _last_request: dict[str, float] = field(default_factory=dict, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def _robots_url(self, url: str) -> tuple[str, str]:
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower()
        return host, urlunparse((parsed.scheme, parsed.netloc, "/robots.txt", "", "", ""))

    def _fetch_text(self, robots_url: str) -> str | None:
        """The robots.txt body, or ``None`` when it cannot be read.

        An unreachable or absent robots.txt is treated as permissive, which is
        the spec's own default. Only an explicit disallow blocks a fetch.
        """
        if self.fetcher is not None:
            return self.fetcher(robots_url)  # type: ignore[operator]
        import httpx

        try:
            response = httpx.get(
                robots_url,
                headers={"User-Agent": USER_AGENT},
                timeout=15.0,
                follow_redirects=True,
            )
        except httpx.HTTPError as exc:
            logger.debug("robots.txt unreachable at %s: %s", robots_url, exc)
            return None
        if response.status_code >= 400:
            return None
        return response.text

    def _entry(self, url: str) -> _Entry:
        host, robots_url = self._robots_url(url)
        with self._lock:
            cached = self._cache.get(host)
            if cached is not None and time.monotonic() - cached.fetched_at < CACHE_SECONDS:
                return cached
        text = self._fetch_text(robots_url)
        parser: RobotFileParser | None = None
        delay = DEFAULT_CRAWL_DELAY
        if text is not None:
            parser = RobotFileParser()
            parser.parse(text.splitlines())
            for token in (USER_AGENT_TOKEN, "*"):
                named = parser.crawl_delay(token)
                if named is not None:
                    delay = max(delay, float(named))
                    break
        entry = _Entry(parser=parser, fetched_at=time.monotonic(), crawl_delay=delay)
        with self._lock:
            self._cache[host] = entry
        return entry

    @staticmethod
    def _names_agent(parser: RobotFileParser, token: str) -> bool:
        """Whether the file declares a group naming ``token`` exactly.

        ``RobotFileParser.can_fetch`` falls back to substring containment when
        no group names the agent exactly (so a generic ``User-agent: Bot``
        group would match every token that merely contains "bot" — GPTBot,
        ClaudeBot, Amazonbot, ...). Checked here first so that fallback is
        never reached for :data:`AI_CRAWLER_TOKENS`: a group only "covers"
        this agent when the file actually names that crawler.
        """
        token_lower = token.lower()
        return any(
            agent.lower() == token_lower for entry in parser.entries for agent in entry.useragents
        )

    def check(self, url: str) -> RobotsVerdict:
        """Whether this URL may be fetched under the policy in the docstring."""
        entry = self._entry(url)
        if entry.parser is None:
            return RobotsVerdict(allowed=True, crawl_delay=entry.crawl_delay)
        # Our own token first: it resolves to the `*` group, and a refusal
        # there is the ordinary "this path is closed to everyone" case.
        if not entry.parser.can_fetch(USER_AGENT_TOKEN, url):
            return RobotsVerdict(False, USER_AGENT_TOKEN, entry.crawl_delay)
        for token in AI_CRAWLER_TOKENS:
            if self._names_agent(entry.parser, token) and not entry.parser.can_fetch(token, url):
                return RobotsVerdict(False, token, entry.crawl_delay)
        return RobotsVerdict(allowed=True, crawl_delay=entry.crawl_delay)

    def wait(self, url: str) -> None:
        """Sleep as long as this host's crawl delay requires, then mark now."""
        host, _ = self._robots_url(url)
        delay = self._entry(url).crawl_delay
        with self._lock:
            previous = self._last_request.get(host)
            now = time.monotonic()
            remaining = 0.0 if previous is None else delay - (now - previous)
            self._last_request[host] = now + max(remaining, 0.0)
        if remaining > 0:
            time.sleep(remaining)

"""Asking one channel what is there now.

A poller reads the network and returns candidates. It never writes a chunk,
never embeds anything, and never touches the corpus — which is what makes
``channels poll --dry-run`` the same code path as a real run minus its
consumer, rather than a second implementation that can drift.

Four kinds do real work:

* ``rss`` / ``atom`` — one conditional GET. Eight of the eleven registered
  sources answer ``304``, which is the cheapest possible "nothing new".
* ``sitemap`` — for the sources that publish no feed at all. Followed one level
  when the document is a ``<sitemapindex>``, because one registered source
  needs exactly that hop and none needs two.
* ``github_docs`` — the pinned file list, resolved to ``raw.githubusercontent``
  URLs. Never the rendered HTML page: measured, that page extracts to
  "Navigation Menu" and "Folders and files" before any content, while the raw
  Markdown extracts to fifteen properly headed sections.
* ``url_list`` — a hand-curated set. Polls nothing; it changes when someone
  edits the file, which makes it the safest place to start a corpus.

Discovery here means *unseen*, and nothing more. Whether a URL already stored
has since changed is a different question with a different mechanism, and it
lives in :mod:`src.channels.refresh`.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable
from urllib.parse import urlparse

from src.channels.feeds import FeedParseError, FeedItem, parse_feed, parse_sitemap
from src.channels.models import Candidate, ChannelConfig, ChannelState, PollResult
from src.channels.robots import USER_AGENT, RobotsPolicy
from src.documents.fetch import (
    FEED_CONTENT_TYPES,
    FEED_MAX_BYTES,
    GITHUB_API_ACCEPT,
    JSON_CONTENT_TYPES,
    DocumentFetchError,
    UnsafeUrlError,
    fetch_document,
)
from src.documents.models import FetchedPage

logger = logging.getLogger(__name__)

#: Child sitemaps followed from an index in one poll. Bounded because an index
#: can name dozens and a poll is not a site crawl.
MAX_SITEMAP_CHILDREN = 4

_GITHUB_REPO = re.compile(r"^/(?P<owner>[^/]+)/(?P<repo>[^/]+)/?$")

#: Markdown files taken from a single prefix in one poll. A prefix is a
#: directory, not a licence to mirror a repo: the largest in the register
#: offers 149 files and most of them are notes about other people's work.
#: Applied per prefix, not across a channel's whole ``path_prefixes``, so one
#: large directory cannot crowd out another; ``channel.max_items_per_poll``
#: is what bounds the size of a poll overall.
MAX_TREE_FILES = 120

#: The tree response for a large repo is JSON listing every blob, so it needs
#: the feed-sized cap rather than the document one.
TREE_MAX_BYTES = FEED_MAX_BYTES


def _default_fetch(url: str, **kwargs: Any) -> FetchedPage:
    return fetch_document(url, user_agent=USER_AGENT, **kwargs)


@dataclass
class PollContext:
    """What a poll run shares across channels.

    ``robots`` is shared because politeness is a property of the host, not of
    the channel that happens to link there — two channels on one host must
    queue behind each other.
    """

    robots: RobotsPolicy = field(default_factory=RobotsPolicy)
    fetch: Callable[..., FetchedPage] = _default_fetch


def raw_github_url(repo_url: str, path: str) -> str:
    """The ``raw.githubusercontent.com`` URL for one file in a repo.

    ``HEAD`` rather than a branch name: repos disagree about ``main`` versus
    ``master``, and resolving that would cost an API call per channel to learn
    something the raw host already knows.
    """
    parsed = urlparse(repo_url)
    match = _GITHUB_REPO.match(parsed.path)
    if parsed.hostname not in {"github.com", "www.github.com"} or match is None:
        raise ValueError(f"not a GitHub repository URL: {repo_url}")
    owner, repo = match.group("owner"), match.group("repo").removesuffix(".git")
    return f"https://raw.githubusercontent.com/{owner}/{repo}/HEAD/{path.lstrip('/')}"


def tree_api_url(repo_url: str) -> str:
    """The repository-tree API URL for a repo, recursive from ``HEAD``."""
    parsed = urlparse(repo_url)
    match = _GITHUB_REPO.match(parsed.path)
    if parsed.hostname not in {"github.com", "www.github.com"} or match is None:
        raise ValueError(f"not a GitHub repository URL: {repo_url}")
    owner, repo = match.group("owner"), match.group("repo").removesuffix(".git")
    return f"https://api.github.com/repos/{owner}/{repo}/git/trees/HEAD?recursive=1"


def blob_github_url(repo_url: str, path: str) -> str:
    """Where a reader should be sent for one file: the rendered page.

    Fetching is done against ``raw``; this is the citation target. GitHub
    slugifies Markdown headings the same way on both, so a section anchor
    taken from the raw text resolves on the rendered page.
    """
    return f"{repo_url.rstrip('/')}/blob/HEAD/{path.lstrip('/')}"


def parse_tree(body: str, truncated: bool = False) -> tuple[list[str], bool]:
    """Markdown paths in a repository-tree response, and whether it was cut.

    Returns ``(paths, complete)``. GitHub sets ``truncated`` on the response
    itself for very large repositories, and our own byte cap can cut it too;
    either way the caller is told, because a partial tree silently treated as
    a whole one is a channel that quietly stops seeing half its files.
    """
    if truncated:
        raise FeedParseError(
            f"repository tree exceeded the {TREE_MAX_BYTES:,}-byte cap and cannot be parsed"
        )
    try:
        payload = json.loads(body)
    except ValueError as exc:
        raise FeedParseError(f"repository tree is not JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise FeedParseError("repository tree is not a JSON object")
    entries = payload.get("tree")
    if not isinstance(entries, list):
        raise FeedParseError("repository tree has no tree array")
    paths = [
        str(entry["path"])
        for entry in entries
        if isinstance(entry, dict)
        and entry.get("type") == "blob"
        and isinstance(entry.get("path"), str)
        and str(entry["path"]).lower().endswith(".md")
    ]
    return sorted(paths), not bool(payload.get("truncated"))


def select_paths(
    paths: list[str],
    prefixes: tuple[str, ...],
    excludes: tuple[str, ...],
    channel_id: str = "",
) -> list[str]:
    """Paths under any prefix, minus anything matching an exclude fragment.

    Excludes are plain substrings rather than globs on purpose: what they have
    to express in the register is "anything under ``_internal/fetched/``", and
    a substring says that without inviting a pattern language into the file.

    ``MAX_TREE_FILES`` is applied per prefix, in the order the channel
    declares them, so a large early-alphabetical prefix cannot silently crowd
    out a later one; ``paths`` is expected pre-sorted, which keeps each
    prefix's matches in that same order. A prefix that gets capped is logged,
    the same way a GitHub-truncated tree is.
    """
    chosen: list[str] = []
    for prefix in prefixes:
        matches = [
            path
            for path in paths
            if path.startswith(prefix) and not any(fragment in path for fragment in excludes)
        ]
        if len(matches) > MAX_TREE_FILES:
            logger.warning(
                "channel %s: prefix %s has %d files, capped at %d",
                channel_id,
                prefix,
                len(matches),
                MAX_TREE_FILES,
            )
        for path in matches[:MAX_TREE_FILES]:
            if path not in chosen:
                chosen.append(path)
    return chosen


def _blocked(channel: ChannelConfig, url: str, reason: str) -> PollResult:
    logger.info("channel %s not polled: %s", channel.id, reason)
    return PollResult(channel_id=channel.id, error=f"{url}: {reason}")


def _to_candidates(
    channel: ChannelConfig,
    state: ChannelState,
    items: list[FeedItem],
) -> tuple[list[Candidate], int]:
    """Filter, deduplicate against what this channel has already offered, cap."""
    filtered = 0
    candidates: list[Candidate] = []
    seen = set(state.seen_ids)
    for item in items:
        if not channel.accepts(item.url, item.title):
            filtered += 1
            continue
        if item.external_id in seen:
            continue
        seen.add(item.external_id)
        candidates.append(
            Candidate(
                url=item.url,
                external_id=item.external_id,
                channel_id=channel.id,
                title=item.title,
                published_at=item.published_at,
                updated_at=item.updated_at,
                # Only trusted when the channel says this feed carries bodies.
                # A teaser stored as an article is worse than a page fetch.
                body_html=item.body_html if channel.body_in_feed else None,
            )
        )
        if len(candidates) >= channel.max_items_per_poll:
            break
    return candidates, filtered


def _fetch_xml(
    channel: ChannelConfig,
    url: str,
    context: PollContext,
    etag: str | None,
    last_modified: str | None,
) -> FetchedPage:
    context.robots.wait(url)
    return context.fetch(
        url,
        allowed_content_types=FEED_CONTENT_TYPES,
        max_bytes=FEED_MAX_BYTES,
        etag=etag,
        last_modified=last_modified,
    )


def poll_feed(channel: ChannelConfig, state: ChannelState, context: PollContext) -> PollResult:
    """One conditional GET of an RSS or Atom feed."""
    url = channel.url or ""
    verdict = context.robots.check(url)
    if not verdict.allowed:
        return _blocked(channel, url, verdict.reason)
    page = _fetch_xml(channel, url, context, state.etag, state.last_modified)
    if page.not_modified:
        return PollResult(
            channel_id=channel.id,
            unchanged=True,
            etag=page.etag or state.etag,
            last_modified=page.last_modified or state.last_modified,
        )
    items = parse_feed(page.body, page.truncated)
    candidates, filtered = _to_candidates(channel, state, items)
    return PollResult(
        channel_id=channel.id,
        candidates=candidates,
        unchanged=not candidates,
        etag=page.etag,
        last_modified=page.last_modified,
        filtered_out=filtered,
    )


def poll_sitemap(channel: ChannelConfig, state: ChannelState, context: PollContext) -> PollResult:
    """A sitemap, or one level of a sitemap index."""
    url = channel.url or ""
    verdict = context.robots.check(url)
    if not verdict.allowed:
        return _blocked(channel, url, verdict.reason)
    page = _fetch_xml(channel, url, context, state.etag, state.last_modified)
    if page.not_modified:
        return PollResult(
            channel_id=channel.id,
            unchanged=True,
            etag=page.etag or state.etag,
            last_modified=page.last_modified or state.last_modified,
        )
    sitemap = parse_sitemap(page.body, page.truncated)
    items = list(sitemap.urls)
    if sitemap.is_index:
        for child in sitemap.children[:MAX_SITEMAP_CHILDREN]:
            child_verdict = context.robots.check(child)
            if not child_verdict.allowed:
                logger.info(
                    "channel %s skipping child sitemap: %s", channel.id, child_verdict.reason
                )
                continue
            try:
                child_page = _fetch_xml(channel, child, context, None, None)
                items.extend(parse_sitemap(child_page.body, child_page.truncated).urls)
            except (DocumentFetchError, UnsafeUrlError, FeedParseError) as exc:
                # One unreadable child must not lose the children that parsed.
                logger.info("channel %s child sitemap %s failed: %s", channel.id, child, exc)
    candidates, filtered = _to_candidates(channel, state, items)
    return PollResult(
        channel_id=channel.id,
        candidates=candidates,
        unchanged=not candidates,
        etag=page.etag,
        last_modified=page.last_modified,
        filtered_out=filtered,
    )


def poll_github_docs(
    channel: ChannelConfig, state: ChannelState, context: PollContext
) -> PollResult:
    """The repo's Markdown, as raw-content URLs.

    ``paths`` names files literally and costs nothing. ``path_prefixes`` asks
    the repository-tree API once per poll and takes every ``.md`` beneath the
    named directories, because a README is usually an index: the twelve
    factors of ``12-factor-agents`` are twelve files under ``content/``, and
    pinning the README alone stores the table of contents rather than the
    book. It also buys what a pinned list cannot do — notice a file that did
    not exist at the last poll.

    Whether a file *already stored* has since changed remains loop B's
    question, answered there by a conditional GET per file.
    """
    url = channel.url or ""
    paths = list(channel.paths)
    if channel.path_prefixes:
        api = tree_api_url(url)
        verdict = context.robots.check(api)
        if not verdict.allowed:
            return _blocked(channel, api, verdict.reason)
        context.robots.wait(api)
        page = context.fetch(
            api,
            allowed_content_types=JSON_CONTENT_TYPES,
            max_bytes=TREE_MAX_BYTES,
            accept=GITHUB_API_ACCEPT,
        )
        found, complete = parse_tree(page.body, page.truncated)
        chosen = select_paths(found, channel.path_prefixes, channel.exclude_paths, channel.id)
        if not complete:
            # Recorded rather than swallowed: the candidate set is a subset of
            # the repo, and a later poll finding "nothing new" would otherwise
            # read as the repo being unchanged.
            logger.warning(
                "channel %s: repository tree was truncated by GitHub; %d paths seen",
                channel.id,
                len(found),
            )
        paths.extend(path for path in chosen if path not in paths)

    items = [FeedItem(url=raw_github_url(url, path), external_id=f"{url}#{path}") for path in paths]
    candidates, filtered = _to_candidates(channel, state, items)
    # Fetch from raw, cite the rendered page, per file rather than per repo:
    # with many files the repo root is the right page for none of them.
    candidates = [
        Candidate(
            **{
                **candidate.__dict__,
                "display_url": blob_github_url(url, candidate.external_id.split("#", 1)[-1]),
            }
        )
        for candidate in candidates
    ]
    return PollResult(
        channel_id=channel.id,
        candidates=candidates,
        unchanged=not candidates,
        filtered_out=filtered,
    )


def poll_url_list(channel: ChannelConfig, state: ChannelState, context: PollContext) -> PollResult:
    """A hand-curated set of URLs. Polls nothing."""
    items = [FeedItem(url=url, external_id=url) for url in channel.urls]
    candidates, filtered = _to_candidates(channel, state, items)
    return PollResult(
        channel_id=channel.id,
        candidates=candidates,
        unchanged=not candidates,
        filtered_out=filtered,
    )


POLLERS: dict[str, Callable[[ChannelConfig, ChannelState, PollContext], PollResult]] = {
    "rss": poll_feed,
    "atom": poll_feed,
    "sitemap": poll_sitemap,
    "github_docs": poll_github_docs,
    "url_list": poll_url_list,
}


def poll_channel(
    channel: ChannelConfig,
    state: ChannelState,
    context: PollContext | None = None,
) -> PollResult:
    """Poll one channel, returning its failure rather than raising it.

    Every expected failure — a dead host, a 500, XML that will not parse, a
    path robots closed — comes back as :attr:`PollResult.error`. A poll run
    covers many channels and must not abandon the rest because one is broken;
    the scheduler counts the failure and backs that channel off.
    """
    poller = POLLERS.get(channel.kind)
    if poller is None:  # pragma: no cover - ChannelConfig validates the kind
        return PollResult(channel_id=channel.id, error=f"no poller for kind {channel.kind!r}")
    try:
        return poller(channel, state, context or PollContext())
    except (DocumentFetchError, UnsafeUrlError, FeedParseError, ValueError) as exc:
        return PollResult(channel_id=channel.id, error=str(exc))

from __future__ import annotations

import pytest

from src.channels.models import ChannelConfig, ChannelState
import json

from src.channels.pollers import (
    MAX_TREE_FILES,
    MAX_SITEMAP_CHILDREN,
    PollContext,
    poll_channel,
    raw_github_url,
    select_paths,
    tree_api_url,
)
from src.documents.fetch import DocumentFetchError
from src.documents.models import FetchedPage

from tests.channels.test_feeds import ATOM, RSS, SITEMAP, SITEMAP_INDEX


def page(body: str, **kwargs) -> FetchedPage:
    defaults = dict(
        requested_url="https://a.example/f.xml",
        url="https://a.example/f.xml",
        status_code=200,
        content_type="application/xml",
        body=body,
    )
    return FetchedPage(**{**defaults, **kwargs})


def context(responses: dict[str, FetchedPage], robots, calls: list | None = None) -> PollContext:
    def fetch(url: str, **kwargs):
        if calls is not None:
            calls.append((url, kwargs.get("etag"), kwargs.get("last_modified")))
        if url not in responses:
            raise DocumentFetchError(f"unexpected request to {url}")
        return responses[url]

    return PollContext(robots=robots, fetch=fetch)


def test_an_rss_poll_offers_every_unseen_item(permissive_robots) -> None:
    channel = ChannelConfig(id="feed", kind="rss", url="https://a.example/f.xml")
    result = poll_channel(
        channel,
        ChannelState(channel_id="feed"),
        context({"https://a.example/f.xml": page(RSS)}, permissive_robots),
    )
    assert result.error is None
    assert [c.external_id for c in result.candidates] == [
        "tag:a.example,2026:newest",
        "tag:a.example,2024:older",
    ]
    assert not result.unchanged


def test_ids_already_seen_are_not_offered_again(permissive_robots) -> None:
    channel = ChannelConfig(id="feed", kind="rss", url="https://a.example/f.xml")
    state = ChannelState(channel_id="feed")
    state.remember(["tag:a.example,2026:newest"])
    result = poll_channel(
        channel, state, context({"https://a.example/f.xml": page(RSS)}, permissive_robots)
    )
    assert [c.external_id for c in result.candidates] == ["tag:a.example,2024:older"]


def test_a_feed_with_nothing_new_reports_unchanged(permissive_robots) -> None:
    channel = ChannelConfig(id="feed", kind="rss", url="https://a.example/f.xml")
    state = ChannelState(channel_id="feed")
    state.remember(["tag:a.example,2026:newest", "tag:a.example,2024:older"])
    result = poll_channel(
        channel, state, context({"https://a.example/f.xml": page(RSS)}, permissive_robots)
    )
    assert result.candidates == []
    assert result.unchanged


def test_the_stored_validators_are_sent_and_a_304_costs_nothing(permissive_robots) -> None:
    channel = ChannelConfig(id="feed", kind="rss", url="https://a.example/f.xml")
    state = ChannelState(
        channel_id="feed", etag='"v1"', last_modified="Mon, 01 Sep 2026 00:00:00 GMT"
    )
    calls: list = []
    result = poll_channel(
        channel,
        state,
        context(
            {"https://a.example/f.xml": page("", status_code=304, content_type="")},
            permissive_robots,
            calls,
        ),
    )
    assert calls == [("https://a.example/f.xml", '"v1"', "Mon, 01 Sep 2026 00:00:00 GMT")]
    assert result.unchanged
    assert result.candidates == []
    # The validators survive a 304 so the next poll is conditional too.
    assert result.etag == '"v1"'


def test_a_new_etag_is_carried_back_for_the_next_poll(permissive_robots) -> None:
    channel = ChannelConfig(id="feed", kind="rss", url="https://a.example/f.xml")
    result = poll_channel(
        channel,
        ChannelState(channel_id="feed"),
        context({"https://a.example/f.xml": page(RSS, etag='"v2"')}, permissive_robots),
    )
    assert result.etag == '"v2"'


def test_max_items_caps_a_first_poll_of_a_large_archive(permissive_robots) -> None:
    # One registered feed carries 1,215 items; a first poll is not a backfill.
    channel = ChannelConfig(
        id="feed", kind="rss", url="https://a.example/f.xml", max_items_per_poll=1
    )
    result = poll_channel(
        channel,
        ChannelState(channel_id="feed"),
        context({"https://a.example/f.xml": page(RSS)}, permissive_robots),
    )
    assert len(result.candidates) == 1
    # Newest first, so a capped poll takes the most recent rather than the oldest.
    assert result.candidates[0].external_id == "tag:a.example,2026:newest"


def test_filtered_out_candidates_are_counted_so_a_dead_filter_is_visible(permissive_robots) -> None:
    channel = ChannelConfig(
        id="feed", kind="rss", url="https://a.example/f.xml", include_title="(?i)nothing matches"
    )
    result = poll_channel(
        channel,
        ChannelState(channel_id="feed"),
        context({"https://a.example/f.xml": page(RSS)}, permissive_robots),
    )
    assert result.candidates == []
    assert result.filtered_out == 2


def test_an_atom_feed_uses_the_same_poller(permissive_robots) -> None:
    channel = ChannelConfig(id="feed", kind="atom", url="https://b.example/feed")
    result = poll_channel(
        channel,
        ChannelState(channel_id="feed"),
        context({"https://b.example/feed": page(ATOM)}, permissive_robots),
    )
    assert {c.url for c in result.candidates} == {"https://b.example/one", "https://b.example/two"}


def test_a_feed_body_is_only_trusted_when_the_channel_says_so(permissive_robots) -> None:
    # A teaser stored as an article is worse than a page fetch, so the
    # decision is per channel and measured, not guessed per item.
    responses = {"https://a.example/f.xml": page(RSS)}
    state = lambda: ChannelState(channel_id="feed")  # noqa: E731
    without = poll_channel(
        ChannelConfig(id="feed", kind="rss", url="https://a.example/f.xml", body_in_feed=False),
        state(),
        context(responses, permissive_robots),
    )
    with_body = poll_channel(
        ChannelConfig(id="feed", kind="rss", url="https://a.example/f.xml", body_in_feed=True),
        state(),
        context(responses, permissive_robots),
    )
    assert all(c.body_html is None for c in without.candidates)
    assert with_body.candidates[0].body_html is not None


def test_a_sitemap_poll_offers_its_urls(permissive_robots) -> None:
    channel = ChannelConfig(id="map", kind="sitemap", url="https://c.example/sitemap.xml")
    result = poll_channel(
        channel,
        ChannelState(channel_id="map"),
        context({"https://c.example/sitemap.xml": page(SITEMAP)}, permissive_robots),
    )
    assert [c.url for c in result.candidates][0] == "https://c.example/new"
    assert len(result.candidates) == 3


def test_a_sitemap_include_pattern_keeps_only_the_wanted_section(permissive_robots) -> None:
    # One registered sitemap has 535 locs of which 25 are wanted; without a
    # filter a sitemap channel is a whole-site crawl.
    channel = ChannelConfig(
        id="map",
        kind="sitemap",
        url="https://c.example/sitemap.xml",
        include=r"/new$",
    )
    result = poll_channel(
        channel,
        ChannelState(channel_id="map"),
        context({"https://c.example/sitemap.xml": page(SITEMAP)}, permissive_robots),
    )
    assert [c.url for c in result.candidates] == ["https://c.example/new"]
    assert result.filtered_out == 2


def test_a_sitemap_index_is_followed_one_level(permissive_robots) -> None:
    child = SITEMAP.replace("c.example/", "c.example/child-")
    result = poll_channel(
        ChannelConfig(id="map", kind="sitemap", url="https://c.example/sitemap.xml"),
        ChannelState(channel_id="map"),
        context(
            {
                "https://c.example/sitemap.xml": page(SITEMAP_INDEX),
                "https://c.example/sitemap-1.xml": page(child),
                "https://c.example/sitemap-2.xml": page(SITEMAP),
            },
            permissive_robots,
        ),
    )
    urls = {c.url for c in result.candidates}
    assert "https://c.example/child-new" in urls
    assert "https://c.example/new" in urls


def test_one_unreadable_child_sitemap_does_not_lose_the_others(permissive_robots) -> None:
    result = poll_channel(
        ChannelConfig(id="map", kind="sitemap", url="https://c.example/sitemap.xml"),
        ChannelState(channel_id="map"),
        context(
            {
                "https://c.example/sitemap.xml": page(SITEMAP_INDEX),
                "https://c.example/sitemap-2.xml": page(SITEMAP),
            },
            permissive_robots,
        ),
    )
    assert result.error is None
    assert [c.url for c in result.candidates][0] == "https://c.example/new"


def test_github_docs_resolves_raw_urls_and_makes_no_request(permissive_robots) -> None:
    calls: list = []
    channel = ChannelConfig(
        id="repo",
        kind="github_docs",
        url="https://github.com/owner/repo",
        paths=("README.md", "docs/guide.md"),
    )
    result = poll_channel(
        channel, ChannelState(channel_id="repo"), context({}, permissive_robots, calls)
    )
    assert calls == []
    assert [c.url for c in result.candidates] == [
        "https://raw.githubusercontent.com/owner/repo/HEAD/README.md",
        "https://raw.githubusercontent.com/owner/repo/HEAD/docs/guide.md",
    ]
    # Fetched from raw, cited at the rendered page — per file, because with
    # many files the repo root is the right page for none of them.
    assert [c.reader_url for c in result.candidates] == [
        "https://github.com/owner/repo/blob/HEAD/README.md",
        "https://github.com/owner/repo/blob/HEAD/docs/guide.md",
    ]


TREE_URL = "https://api.github.com/repos/owner/repo/git/trees/HEAD?recursive=1"


def tree_page(
    paths: list[str], truncated: bool = False, body_truncated: bool = False, **kwargs
) -> FetchedPage:
    if body_truncated:
        kwargs["truncated"] = True
    body = json.dumps(
        {
            "tree": [{"type": "blob", "path": path} for path in paths]
            + [{"type": "tree", "path": "content"}],
            "truncated": truncated,
        }
    )
    return page(
        body, requested_url=TREE_URL, url=TREE_URL, content_type="application/json", **kwargs
    )


def tree_channel(**kwargs) -> ChannelConfig:
    defaults = dict(
        id="repo",
        kind="github_docs",
        url="https://github.com/owner/repo",
        paths=("README.md",),
        path_prefixes=("content/",),
        max_items_per_poll=50,
    )
    return ChannelConfig(**{**defaults, **kwargs})


def test_a_prefix_takes_every_markdown_file_beneath_it(permissive_robots) -> None:
    # A README is an index; the documents are the files under content/. This
    # is the whole reason the tree call is worth one request.
    result = poll_channel(
        tree_channel(),
        ChannelState(channel_id="repo"),
        context(
            {
                TREE_URL: tree_page(
                    ["content/factor-01-a.md", "content/factor-02-b.md", "other/skip.md"]
                )
            },
            permissive_robots,
        ),
    )
    assert result.error is None
    assert [c.url for c in result.candidates] == [
        "https://raw.githubusercontent.com/owner/repo/HEAD/README.md",
        "https://raw.githubusercontent.com/owner/repo/HEAD/content/factor-01-a.md",
        "https://raw.githubusercontent.com/owner/repo/HEAD/content/factor-02-b.md",
    ]


def test_a_file_outside_every_prefix_is_left_alone(permissive_robots) -> None:
    result = poll_channel(
        tree_channel(),
        ChannelState(channel_id="repo"),
        context({TREE_URL: tree_page(["notes/third-party.md"])}, permissive_robots),
    )
    assert [c.external_id for c in result.candidates] == ["https://github.com/owner/repo#README.md"]


def test_excluded_fragments_drop_files_a_prefix_would_otherwise_take(permissive_robots) -> None:
    # Scraped threads and link dumps sit beside the author's own writing.
    result = poll_channel(
        tree_channel(exclude_paths=("_internal/",)),
        ChannelState(channel_id="repo"),
        context(
            {TREE_URL: tree_page(["content/real.md", "content/_internal/fetched/reddit.md"])},
            permissive_robots,
        ),
    )
    assert [c.url for c in result.candidates] == [
        "https://raw.githubusercontent.com/owner/repo/HEAD/README.md",
        "https://raw.githubusercontent.com/owner/repo/HEAD/content/real.md",
    ]


def test_only_markdown_blobs_are_taken(permissive_robots) -> None:
    # A tree lists images, code and directories too; none of them is a document.
    result = poll_channel(
        tree_channel(),
        ChannelState(channel_id="repo"),
        context(
            {TREE_URL: tree_page(["content/a.md", "content/diagram.png", "content/run.py"])},
            permissive_robots,
        ),
    )
    assert [c.external_id.split("#")[-1] for c in result.candidates] == [
        "README.md",
        "content/a.md",
    ]


def test_a_pinned_path_is_never_offered_twice_because_a_prefix_also_matched(
    permissive_robots,
) -> None:
    result = poll_channel(
        tree_channel(paths=("content/a.md",)),
        ChannelState(channel_id="repo"),
        context({TREE_URL: tree_page(["content/a.md", "content/b.md"])}, permissive_robots),
    )
    assert [c.external_id.split("#")[-1] for c in result.candidates] == [
        "content/a.md",
        "content/b.md",
    ]


def test_each_file_cites_its_own_rendered_page(permissive_robots) -> None:
    result = poll_channel(
        tree_channel(paths=()),
        ChannelState(channel_id="repo"),
        context({TREE_URL: tree_page(["content/factor-01-a.md"])}, permissive_robots),
    )
    assert result.candidates[0].reader_url == (
        "https://github.com/owner/repo/blob/HEAD/content/factor-01-a.md"
    )


def test_a_file_already_seen_is_not_offered_again(permissive_robots) -> None:
    # Discovery means unseen; whether a stored file changed is loop B's job.
    state = ChannelState(channel_id="repo")
    state.remember(["https://github.com/owner/repo#content/a.md"])
    result = poll_channel(
        tree_channel(paths=()),
        state,
        context({TREE_URL: tree_page(["content/a.md", "content/b.md"])}, permissive_robots),
    )
    assert [c.external_id.split("#")[-1] for c in result.candidates] == ["content/b.md"]


def test_a_repo_with_no_prefixes_never_calls_the_tree_api(permissive_robots) -> None:
    # The cheap path has to stay cheap: a pinned-only channel costs nothing.
    calls: list = []
    result = poll_channel(
        tree_channel(path_prefixes=()),
        ChannelState(channel_id="repo"),
        context({}, permissive_robots, calls),
    )
    assert calls == []
    assert len(result.candidates) == 1


def test_an_unparseable_tree_is_an_error_not_an_empty_repo(permissive_robots) -> None:
    # Reporting "nothing new" here would read as the repo being unchanged.
    result = poll_channel(
        tree_channel(),
        ChannelState(channel_id="repo"),
        context(
            {TREE_URL: page("not json", requested_url=TREE_URL, url=TREE_URL)}, permissive_robots
        ),
    )
    assert result.error is not None
    assert result.candidates == []


def test_githubs_own_truncation_flag_still_yields_the_files_it_did_return(
    permissive_robots, caplog
) -> None:
    # GitHub truncates very large trees but the entries it sent are complete
    # and usable, so the poll proceeds with the subset — and says so, because
    # a later "nothing new" would otherwise read as the repo being unchanged.
    result = poll_channel(
        tree_channel(),
        ChannelState(channel_id="repo"),
        context({TREE_URL: tree_page(["content/a.md"], truncated=True)}, permissive_robots),
    )
    assert result.error is None
    assert [c.external_id.split("#")[-1] for c in result.candidates] == [
        "README.md",
        "content/a.md",
    ]
    assert any("truncated" in record.message for record in caplog.records)


def test_a_tree_cut_by_our_own_byte_cap_is_an_error(permissive_robots) -> None:
    # Different case: our cap cuts mid-JSON, so what arrived cannot be trusted
    # to be a whole entry, let alone a whole tree.
    result = poll_channel(
        tree_channel(),
        ChannelState(channel_id="repo"),
        context(
            {TREE_URL: tree_page(["content/a.md"], body_truncated=True)},
            permissive_robots,
        ),
    )
    assert result.error is not None
    assert result.candidates == []


def test_the_tree_api_url_is_derived_from_the_repo_page() -> None:
    assert tree_api_url("https://github.com/o/r") == (
        "https://api.github.com/repos/o/r/git/trees/HEAD?recursive=1"
    )
    assert tree_api_url("https://github.com/o/r.git") == (
        "https://api.github.com/repos/o/r/git/trees/HEAD?recursive=1"
    )


def test_a_prefix_cannot_mirror_a_whole_repository() -> None:
    # A prefix is a directory, not a licence to take everything under it.
    many = [f"content/{index}.md" for index in range(MAX_TREE_FILES + 40)]
    assert len(select_paths(many, ("content/",), ())) == MAX_TREE_FILES


def test_a_url_list_offers_its_urls_and_makes_no_request(permissive_robots) -> None:
    calls: list = []
    channel = ChannelConfig(
        id="seeds", kind="url_list", urls=("https://a.example/one", "https://a.example/two")
    )
    result = poll_channel(
        channel, ChannelState(channel_id="seeds"), context({}, permissive_robots, calls)
    )
    assert calls == []
    assert [c.url for c in result.candidates] == ["https://a.example/one", "https://a.example/two"]
    assert all(c.reader_url == c.url for c in result.candidates)


def test_a_robots_refusal_becomes_an_error_not_an_exception(ai_blocking_robots) -> None:
    channel = ChannelConfig(id="feed", kind="rss", url="https://sub.example/feed")
    result = poll_channel(channel, ChannelState(channel_id="feed"), context({}, ai_blocking_robots))
    assert result.candidates == []
    assert "covers this agent" in (result.error or "")


def test_a_failed_fetch_is_reported_rather_than_raised(permissive_robots) -> None:
    # A poll run covers many channels and must not abandon the rest because
    # one host is down.
    channel = ChannelConfig(id="feed", kind="rss", url="https://a.example/f.xml")
    result = poll_channel(channel, ChannelState(channel_id="feed"), context({}, permissive_robots))
    assert result.candidates == []
    assert "unexpected request" in (result.error or "")


def test_unparseable_xml_is_reported_rather_than_raised(permissive_robots) -> None:
    channel = ChannelConfig(id="feed", kind="rss", url="https://a.example/f.xml")
    result = poll_channel(
        channel,
        ChannelState(channel_id="feed"),
        context({"https://a.example/f.xml": page("<html>nope</html>")}, permissive_robots),
    )
    assert "not a feed" in (result.error or "")


@pytest.mark.parametrize(
    "repo,path,expected",
    [
        (
            "https://github.com/o/r",
            "README.md",
            "https://raw.githubusercontent.com/o/r/HEAD/README.md",
        ),
        (
            "https://github.com/o/r/",
            "/README.md",
            "https://raw.githubusercontent.com/o/r/HEAD/README.md",
        ),
        (
            "https://github.com/o/r.git",
            "a/b.md",
            "https://raw.githubusercontent.com/o/r/HEAD/a/b.md",
        ),
    ],
)
def test_raw_github_urls(repo: str, path: str, expected: str) -> None:
    assert raw_github_url(repo, path) == expected


@pytest.mark.parametrize("bad", ["https://gitlab.com/o/r", "https://github.com/owner", "not a url"])
def test_non_repository_urls_are_refused(bad: str) -> None:
    with pytest.raises(ValueError, match="not a GitHub repository URL"):
        raw_github_url(bad, "README.md")


def test_the_number_of_child_sitemaps_followed_is_bounded() -> None:
    # A poll is not a site crawl; an index can name dozens.
    assert MAX_SITEMAP_CHILDREN <= 8

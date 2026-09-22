from __future__ import annotations

from src.channels.robots import (
    AI_CRAWLER_TOKENS,
    DEFAULT_CRAWL_DELAY,
    USER_AGENT,
    USER_AGENT_TOKEN,
    RobotsPolicy,
)

from tests.channels.conftest import AI_BLOCKING_ROBOTS, PERMISSIVE_ROBOTS


def policy(text: str | None) -> RobotsPolicy:
    return RobotsPolicy(fetcher=lambda url: text)


def test_an_ai_crawler_disallow_blocks_us_even_though_star_allows_the_path() -> None:
    # This is the whole chosen policy. By the letter of the spec only the `*`
    # group applies to us, and it permits article paths on this file. The
    # blanket disallow aimed at named AI crawlers is treated as aimed at us.
    verdict = policy(AI_BLOCKING_ROBOTS).check("https://sub.example/p/a-post")
    assert not verdict.allowed
    assert verdict.refused_by in AI_CRAWLER_TOKENS
    assert "covers this agent" in verdict.reason


def test_the_same_rule_closes_the_feed_not_only_the_article_paths() -> None:
    # A consequence worth pinning down: `Disallow: /` covers /feed too, so a
    # source refused this way is excluded entirely rather than read via RSS.
    assert not policy(AI_BLOCKING_ROBOTS).check("https://sub.example/feed").allowed


def test_a_path_closed_to_everyone_is_reported_as_such() -> None:
    verdict = policy(AI_BLOCKING_ROBOTS).check("https://sub.example/subscribe")
    assert not verdict.allowed
    assert verdict.refused_by == USER_AGENT_TOKEN
    assert "for all crawlers" in verdict.reason


def test_an_open_site_is_allowed() -> None:
    verdict = policy(PERMISSIVE_ROBOTS).check("https://open.example/blog/post")
    assert verdict.allowed
    assert verdict.reason == "allowed"
    assert verdict.crawl_delay == DEFAULT_CRAWL_DELAY


def test_a_missing_robots_file_is_permissive() -> None:
    # The spec's own default. Only an explicit disallow blocks a fetch.
    assert policy(None).check("https://nofile.example/x").allowed


def test_a_declared_crawl_delay_is_honoured_and_never_shortens_the_default() -> None:
    assert (
        policy("User-agent: *\nCrawl-delay: 5\n").check("https://slow.example/x").crawl_delay == 5.0
    )
    fast = policy("User-agent: *\nCrawl-delay: 0\n").check("https://fast.example/x")
    assert fast.crawl_delay == DEFAULT_CRAWL_DELAY


def test_robots_is_fetched_once_per_host() -> None:
    calls: list[str] = []

    def fetcher(url: str) -> str:
        calls.append(url)
        return PERMISSIVE_ROBOTS

    checker = RobotsPolicy(fetcher=fetcher)
    for path in ("/a", "/b", "/c"):
        checker.check(f"https://one.example{path}")
    checker.check("https://two.example/a")
    assert calls == ["https://one.example/robots.txt", "https://two.example/robots.txt"]


def test_the_user_agent_names_the_project_and_carries_a_contact_url() -> None:
    # An honest identity is half of the chosen policy: a site that wants to
    # exclude this agent has to be able to see who it is.
    assert USER_AGENT.startswith(USER_AGENT_TOKEN)
    assert "github.com/nmp-dsci/transcript-rag-agent" in USER_AGENT

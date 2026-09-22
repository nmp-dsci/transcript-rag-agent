from __future__ import annotations

import pytest

from src.channels.feeds import (
    BODY_CHAR_THRESHOLD,
    FeedParseError,
    looks_like_body,
    normalise_date,
    parse_feed,
    parse_sitemap,
)

LONG = "<p>" + ("article prose " * 200) + "</p>"

RSS = f"""<?xml version="1.0"?>
<rss version="2.0" xmlns:content="http://purl.org/rss/1.0/modules/content/">
  <channel>
    <title>A blog</title>
    <item>
      <title>Older post</title>
      <link>https://a.example/older</link>
      <guid>tag:a.example,2024:older</guid>
      <pubDate>Tue, 21 Jan 2025 19:00:10 +0000</pubDate>
      <description>a teaser</description>
    </item>
    <item>
      <title>Newest post</title>
      <link>https://a.example/newest</link>
      <guid>tag:a.example,2026:newest</guid>
      <pubDate>Mon, 21 Sep 2026 19:00:10 +0000</pubDate>
      <description>a teaser too</description>
      <content:encoded>{LONG}</content:encoded>
    </item>
  </channel>
</rss>
"""

ATOM = f"""<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <title>Second</title>
    <id>https://b.example/2</id>
    <link rel="self" href="https://b.example/feed"/>
    <link rel="alternate" href="https://b.example/two"/>
    <updated>2026-09-20T10:00:00Z</updated>
    <summary type="html">{LONG}</summary>
  </entry>
  <entry>
    <title>First</title>
    <id>https://b.example/1</id>
    <link href="https://b.example/one"/>
    <published>2026-09-21T10:00:00Z</published>
    <updated>2026-09-21T11:00:00Z</updated>
    <summary>short</summary>
  </entry>
</feed>
"""

SITEMAP = """<?xml version="1.0"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>https://c.example/old</loc><lastmod>2024-01-01T00:00:00Z</lastmod></url>
  <url><loc>https://c.example/new</loc><lastmod>2026-09-01T00:00:00Z</lastmod></url>
  <url><loc>https://c.example/undated</loc></url>
</urlset>
"""

SITEMAP_INDEX = """<?xml version="1.0"?>
<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <sitemap><loc>https://c.example/sitemap-1.xml</loc></sitemap>
  <sitemap><loc>https://c.example/sitemap-2.xml</loc></sitemap>
</sitemapindex>
"""


def test_rss_items_come_back_newest_first_regardless_of_document_order() -> None:
    # Feeds are not reliably sorted: one real source returned a three-year-old
    # entry first and yesterday's second.
    items = parse_feed(RSS)
    assert [item.title for item in items] == ["Newest post", "Older post"]


def test_identity_is_the_guid_not_the_link() -> None:
    # A post that moves keeps its guid; keying on the URL would fork it into
    # two documents and orphan the first.
    newest = parse_feed(RSS)[0]
    assert newest.external_id == "tag:a.example,2026:newest"
    assert newest.url == "https://a.example/newest"


def test_content_encoded_wins_over_description_and_a_teaser_is_dropped() -> None:
    newest, older = parse_feed(RSS)
    assert newest.body_html is not None and len(newest.body_html) >= BODY_CHAR_THRESHOLD
    # "a teaser" is not an article, so it is not offered as one.
    assert older.body_html is None


def test_atom_entries_prefer_the_alternate_link_over_self() -> None:
    items = parse_feed(ATOM)
    by_id = {item.external_id: item for item in items}
    assert by_id["https://b.example/2"].url == "https://b.example/two"
    # A link with no rel defaults to alternate.
    assert by_id["https://b.example/1"].url == "https://b.example/one"


def test_atom_ordering_uses_updated_in_preference_to_published() -> None:
    assert [item.title for item in parse_feed(ATOM)] == ["First", "Second"]


def test_rfc_822_and_iso_dates_both_normalise_to_utc_iso() -> None:
    assert normalise_date("Mon, 21 Sep 2026 19:00:10 +0000") == "2026-09-21T19:00:10+00:00"
    assert normalise_date("2026-09-21T19:00:10Z") == "2026-09-21T19:00:10+00:00"
    # A naive date is read as UTC rather than discarded.
    assert normalise_date("2026-09-21T19:00:10") == "2026-09-21T19:00:10+00:00"
    assert normalise_date("not a date") is None
    assert normalise_date(None) is None


def test_the_body_threshold_separates_an_article_from_a_teaser() -> None:
    assert looks_like_body("x" * BODY_CHAR_THRESHOLD)
    assert not looks_like_body("x" * (BODY_CHAR_THRESHOLD - 1))
    assert not looks_like_body(None)


def test_sitemap_urls_come_back_newest_first_with_undated_ones_last() -> None:
    sitemap = parse_sitemap(SITEMAP)
    assert not sitemap.is_index
    assert [item.url for item in sitemap.urls] == [
        "https://c.example/new",
        "https://c.example/old",
        "https://c.example/undated",
    ]
    assert sitemap.urls[-1].updated_at is None


def test_a_sitemap_index_reports_its_children_rather_than_urls() -> None:
    sitemap = parse_sitemap(SITEMAP_INDEX)
    assert sitemap.is_index
    assert sitemap.children == (
        "https://c.example/sitemap-1.xml",
        "https://c.example/sitemap-2.xml",
    )


def test_a_doctype_is_refused_before_the_parser_sees_it() -> None:
    # Entity expansion: a small document can expand to a large one, and no
    # legitimate feed carries a DOCTYPE.
    bomb = '<?xml version="1.0"?><!DOCTYPE lolz [<!ENTITY a "aaaa">]><rss><channel/></rss>'
    with pytest.raises(FeedParseError, match="DOCTYPE"):
        parse_feed(bomb)


def test_a_feed_cut_at_the_byte_cap_says_so_instead_of_blaming_the_xml() -> None:
    # Measured: one registered feed is 2.9 MB, and the default cap cut it
    # mid-CDATA. "unclosed CDATA section" sends the reader looking for a
    # malformed feed instead of a cap that needs raising.
    cut = RSS[: len(RSS) // 2]
    with pytest.raises(FeedParseError, match="byte cap"):
        parse_feed(cut, truncated=True)
    with pytest.raises(FeedParseError, match="not parseable as XML"):
        parse_feed(cut, truncated=False)


def test_xml_that_is_not_a_feed_is_rejected_by_name() -> None:
    with pytest.raises(FeedParseError, match="not a feed"):
        parse_feed("<html><body>hello</body></html>")
    with pytest.raises(FeedParseError, match="not a sitemap"):
        parse_sitemap("<rss><channel/></rss>")


def test_an_empty_feed_is_empty_rather_than_an_error() -> None:
    # A repo with no releases yet is an ordinary state, not a broken feed.
    assert parse_feed('<feed xmlns="http://www.w3.org/2005/Atom"><id>x</id></feed>') == []


def test_items_without_a_usable_link_are_skipped() -> None:
    feed = """<rss><channel>
      <item><title>no link</title><guid>not-a-url</guid></item>
      <item><title>fine</title><link>https://a.example/ok</link></item>
    </channel></rss>"""
    assert [item.url for item in parse_feed(feed)] == ["https://a.example/ok"]

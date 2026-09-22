"""The two extraction modes, and the Markdown path.

Every claim these tests pin down was measured against a real page first; the
numbers are recorded in ``.lavish/s26_new-channels-ingestion.html``.
"""

from __future__ import annotations

import pytest

from src.documents.extract import ARTICLE_MODE, RESUME_MODE, extract_document
from src.documents.models import FetchedPage

CHROME_PAGE = """<html><head><title>A post</title></head><body>
<nav>Navigation Menu Home Pricing Docs</nav>
<header><p>Site banner and a subscribe prompt</p></header>
<h2>Contents</h2><p>Section one. Section two.</p>
<article>
  <header><h1>A post about agents</h1></header>
  <h2>First section</h2><p>{body}</p>
</article>
<aside>Related posts you might like</aside>
<footer>Copyright someone, all rights reserved</footer>
</body></html>"""

BODY = "Prose that is long enough to survive the short-section merge. " * 4


def page(
    body: str, content_type: str = "text/html", url: str = "https://a.example/post"
) -> FetchedPage:
    return FetchedPage(
        requested_url=url, url=url, status_code=200, content_type=content_type, body=body
    )


def headings(document) -> list[str | None]:
    return [section.heading for section in document.sections]


def test_resume_mode_keeps_nav_header_and_footer() -> None:
    # This is the historical behaviour and must not change: on a resume or a
    # personal site the name and contact details are very often inside
    # <header>, and stripping them deletes the part a reviewer is asked about.
    document = extract_document(page(CHROME_PAGE.format(body=BODY)), mode=RESUME_MODE)
    assert "Navigation Menu" in document.text
    assert "Copyright someone" in document.text
    assert "Site banner" in document.text


def test_resume_mode_is_the_default() -> None:
    explicit = extract_document(page(CHROME_PAGE.format(body=BODY)), mode=RESUME_MODE)
    implicit = extract_document(page(CHROME_PAGE.format(body=BODY)))
    assert implicit.text == explicit.text


def test_article_mode_drops_navigation_asides_and_the_footer() -> None:
    document = extract_document(page(CHROME_PAGE.format(body=BODY)), mode=ARTICLE_MODE)
    assert "Navigation Menu" not in document.text
    assert "Copyright someone" not in document.text
    assert "Related posts" not in document.text
    assert "Prose that is long enough" in document.text


def test_article_mode_keeps_a_header_inside_the_article_but_drops_the_page_banner() -> None:
    # A <header> at the top of the page is site chrome; one inside <article>
    # is usually the article's own title block.
    document = extract_document(page(CHROME_PAGE.format(body=BODY)), mode=ARTICLE_MODE)
    assert "Site banner" not in document.text
    assert "A post about agents" in document.text


def test_article_mode_drops_a_leading_table_of_contents() -> None:
    # Indexing a contents list produces a chunk that matches every query about
    # the document and answers none of them.
    document = extract_document(page(CHROME_PAGE.format(body=BODY)), mode=ARTICLE_MODE)
    assert "Contents" not in headings(document)
    assert "First section" in headings(document)


@pytest.mark.parametrize(
    "label", ["Contents", "Table of contents", "On this page", "In this article"]
)
def test_the_toc_headings_recognised(label: str) -> None:
    body = (
        f"<html><body><h2>{label}</h2><p>one two three</p><h2>Real</h2><p>{BODY}</p></body></html>"
    )
    document = extract_document(page(body), mode=ARTICLE_MODE)
    assert label not in headings(document)


def test_a_document_that_is_only_a_contents_list_is_left_alone() -> None:
    # Deleting its one section would leave an empty document rather than a
    # cleaner one, and an index page is a legitimate thing to have fetched.
    body = f"<html><body><h2>Contents</h2><p>{BODY}</p></body></html>"
    document = extract_document(page(body), mode=ARTICLE_MODE)
    assert headings(document) == ["Contents"]


def test_an_unknown_mode_is_refused() -> None:
    with pytest.raises(ValueError, match="mode must be one of"):
        extract_document(page("<html></html>"), mode="readability")


# ── the Markdown path ────────────────────────────────────────────────────────

MARKDOWN = """# Twelve Factors

<div align="center">
<img src="https://img.shields.io/badge/build-passing.svg"
     alt="build badge">
</div>

Intro prose that is long enough to be a section on its own merits here.

## Factor 1: own your prompts

Body text for the first factor, also long enough to stand alone as prose.

```python
# this is a comment, not a heading
if a < b and c > d:
    print("<not a tag>")
```

## Factor 2: own your context window

![a diagram](https://example.com/diagram.png)

Body text for the second factor, long enough to be its own section too.
"""


def test_raw_markdown_yields_real_headings() -> None:
    # Measured: run through the HTML path a README produced twenty sections
    # and *zero* headings, which makes section-anchored citation impossible
    # while looking like a clean ingest.
    document = extract_document(
        page(MARKDOWN, url="https://raw.example/README.md"), mode=ARTICLE_MODE
    )
    assert document.title == "Twelve Factors"
    assert "Factor 1: own your prompts" in headings(document)
    assert "Factor 2: own your context window" in headings(document)


def test_hashes_inside_a_code_fence_are_not_headings() -> None:
    document = extract_document(
        page(MARKDOWN, url="https://raw.example/README.md"), mode=ARTICLE_MODE
    )
    assert "this is a comment, not a heading" not in headings(document)
    assert any("this is a comment" in section.text for section in document.sections)


def test_code_fences_keep_their_angle_brackets() -> None:
    # Inside a fence a `<` is far more likely to be an operator than a tag.
    document = extract_document(
        page(MARKDOWN, url="https://raw.example/README.md"), mode=ARTICLE_MODE
    )
    assert "if a < b and c > d:" in document.text
    assert "<not a tag>" in document.text


def test_inline_html_and_images_are_stripped_even_when_split_across_lines() -> None:
    # READMEs routinely open with a <div> of shield badges broken over several
    # lines; stored verbatim those tags embed as text and retrieve as nothing.
    document = extract_document(
        page(MARKDOWN, url="https://raw.example/README.md"), mode=ARTICLE_MODE
    )
    assert "<div" not in document.text
    assert "<img" not in document.text
    assert "img.shields.io" not in document.text
    assert "diagram.png" not in document.text


def test_markdown_links_keep_their_text_and_lose_their_target() -> None:
    # A URL embeds as a meaningless token and bloats the chunk it sits in,
    # while the link text is ordinary prose. Left as-is, a curated link
    # list's headings arrive carrying "[ ](https://awesome.re)".
    body = (
        "# Awesome Evals [ ](https://awesome.re)\n\n"
        "See [the eval guide](https://example.com/guide) for details on this, "
        "and also consider [another one](https://example.com/two) as well here.\n"
    )
    document = extract_document(page(body, url="https://raw.example/README.md"), mode=ARTICLE_MODE)
    assert document.title == "Awesome Evals"
    assert "the eval guide" in document.text
    assert "https://example.com/guide" not in document.text
    assert "](" not in document.text


def test_markdown_paragraph_structure_survives() -> None:
    # Without it a whole section collapses into one unbroken line and then
    # chunks as one oversized block instead of on paragraphs.
    body = "# T\n\nFirst paragraph of prose here.\n\nSecond paragraph of prose here.\n"
    document = extract_document(page(body, url="https://raw.example/README.md"), mode=ARTICLE_MODE)
    section = document.sections[0]
    assert section.text.splitlines() == [
        "First paragraph of prose here.",
        "Second paragraph of prose here.",
    ]


@pytest.mark.parametrize("url", ["https://raw.example/a.md", "https://raw.example/a.MARKDOWN"])
def test_markdown_is_recognised_by_path(url: str) -> None:
    document = extract_document(page("# Title\n\n" + BODY, url=url), mode=ARTICLE_MODE)
    assert document.title == "Title"


def test_markdown_is_recognised_by_content_type() -> None:
    document = extract_document(
        page("# Title\n\n" + BODY, content_type="text/markdown", url="https://a.example/x"),
        mode=ARTICLE_MODE,
    )
    assert document.title == "Title"


def test_resume_mode_keeps_markdown_link_targets() -> None:
    # A resume pasted as a raw .md URL must be untouched by the article-mode
    # Markdown path: the historical HTML parser keeps a link's target, while
    # _sections_from_markdown would replace it with the link text alone.
    body = "# Jane Doe\n\n[Email me](mailto:jane@example.com) or visit my site.\n"
    document = extract_document(
        page(body, url="https://raw.example/resume.md"),
        mode=RESUME_MODE,
    )
    assert "mailto:jane@example.com" in document.text


def test_plain_text_that_is_not_markdown_keeps_the_historical_behaviour() -> None:
    # A transcript pasted as plain text must behave exactly as it does today.
    body = "First block of prose here.\n\nSecond block of prose here."
    document = extract_document(page(body, content_type="text/plain", url="https://a.example/x"))
    assert headings(document) == [None]
    assert "First block" in document.text and "Second block" in document.text


def test_a_document_opening_below_h1_still_gets_a_title() -> None:
    # Chapter files in the registered repos open at "###" because the "#"
    # lives in the README. Without a fallback each one cites as a bare URL.
    document = extract_document(
        page("### 3. Own your context window\n\nBody text here.\n", url="https://raw.example/f.md"),
        mode=ARTICLE_MODE,
    )
    assert document.title == "3. Own your context window"


def test_a_level_one_heading_still_wins_over_a_later_one() -> None:
    document = extract_document(
        page(
            "### Subtitle\n\nIntro.\n\n# The Real Title\n\nBody.\n", url="https://raw.example/f.md"
        ),
        mode=ARTICLE_MODE,
    )
    assert document.title == "The Real Title"

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.guides.catalog import guide_detail, list_guides
from src.guides.importer import extract_sources, import_lavish_html, rewrite

SAMPLE = """<title>Tiny Guide</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Geist">
<style>
  :root { --paper: #fff; }
  .lede { font-size: 19px; }
</style>
<nav class="site"><a href="#thesis">Thesis</a></nav>
<header class="hero" id="top"><h1>Tiny &amp; mighty</h1><p class="sub">A test page.</p></header>
<main>
  <section id="thesis">
    <h2>Part one</h2>
    <p class="lede">Lede &mdash; here.</p>
    <p>Body with a <a href="https://youtu.be/abc123XYZ?t=42" target="_blank">quoted video</a>.</p>
    <ul><li>one</li><li id="keep-me">two</li></ul>
  </section>
  <section>
    <h3>Unnamed section</h3>
    <pre>code</pre>
  </section>
  <section id="sources">
    <h2>Sources</h2>
    <ol>
      <li><a href="https://www.youtube.com/watch?v=abc123XYZ" target="_blank">First Video</a> <span class="tk">— what it gave</span></li>
      <li><a href="https://www.youtube.com/watch?v=def456UVW" target="_blank">Second Video</a></li>
      <li><a href="https://www.youtube.com/watch?v=abc123XYZ">Duplicate</a></li>
    </ol>
  </section>
</main>
<script>console.log("inline ok")</script>
"""


def test_rewrite_adds_ids_cites_and_shared_stylesheet():
    page, css, title = rewrite(SAMPLE)
    assert title == "Tiny Guide"
    assert "--paper: #fff" in css and "<style>" not in page
    assert '<link rel="stylesheet" href="/guides/guide.css">' in page
    assert page.startswith("<!doctype html>") and page.rstrip().endswith("</html>")
    assert '<script src="/guides/guide-bridge.js"></script>' in page
    # Document-order, section-scoped ids; existing ids are kept.
    assert '<p class="lede" id="thesis-p1">' in page
    assert '<p id="thesis-p2">' in page
    assert '<li id="thesis-li1">one</li>' in page and '<li id="keep-me">two</li>' in page
    assert '<section id="top-section1">' in page  # the unnamed section
    assert '<h3 id="top-section1-h31">' in page and '<pre id="top-section1-pre1">' in page
    # YouTube links gain a cite around their text, with the timestamp kept.
    assert (
        '<a href="https://youtu.be/abc123XYZ?t=42" target="_blank">'
        '<cite data-video="abc123XYZ" data-t="42">quoted video</cite></a>'
    ) in page
    # Entities and the inline script survive untouched.
    assert "Tiny &amp; mighty" in page and "&mdash;" in page
    assert 'console.log("inline ok")' in page


def test_extract_sources_dedupes_and_reads_contribution():
    sources = extract_sources(SAMPLE)
    assert [item["video_id"] for item in sources] == ["abc123XYZ", "def456UVW"]
    assert sources[0]["title"] == "First Video"
    assert sources[0]["contributed"] == "what it gave"
    assert sources[1]["contributed"] == ""


def test_import_writes_the_guide_contract(tmp_path: Path):
    source = tmp_path / "page.html"
    source.write_text(SAMPLE, encoding="utf-8")
    guides_dir = tmp_path / "guides"
    manifest = import_lavish_html(
        source,
        slug="tiny-guide",
        guides_dir=guides_dir,
        known_videos={"abc123XYZ"},
        chunk_counts={"abc123XYZ": 12, "def456UVW": 30},
        compiled_at="2026-01-02",
    )
    assert manifest.title == "Tiny Guide"
    assert manifest.video_ids == ["abc123XYZ", "def456UVW"]
    assert manifest.chunk_count == 42
    # Four cites: body link + three appendix links; def456UVW is unknown.
    assert (manifest.cite_total, manifest.cite_valid) == (4, 3)
    root = guides_dir / "tiny-guide"
    assert (root / "guide.html").is_file()
    assert (root / "versions" / "v1.html").read_text() == (root / "guide.html").read_text()
    assert (guides_dir / "guide.css").read_text().startswith("  :root")
    assert ".guide-hl" in (guides_dir / "guide.css").read_text()
    claims = json.loads((root / "claims.json").read_text())
    assert claims["total"] == 4 and claims["structure_errors"] == []
    index = json.loads((guides_dir / "index.json").read_text())
    assert [item["slug"] for item in index["guides"]] == ["tiny-guide"]

    listing = list_guides(guides_dir)
    assert listing["guides"][0]["html_url"] == "/guides/tiny-guide/guide.html"
    assert listing["guides"][0]["versions"] == [1]
    detail = guide_detail("tiny-guide", guides_dir)
    assert detail is not None
    assert detail["claims"]["valid"] == 3 and detail["comments"] == []
    assert detail["version_urls"] == {"1": "/guides/tiny-guide/versions/v1.html"}
    assert guide_detail("missing", guides_dir) is None
    assert guide_detail("../etc", guides_dir) is None


def test_reimport_refuses_to_clobber_v1_without_force(tmp_path: Path):
    source = tmp_path / "page.html"
    source.write_text(SAMPLE, encoding="utf-8")
    guides_dir = tmp_path / "guides"
    import_lavish_html(source, slug="tiny-guide", guides_dir=guides_dir)
    v1 = (guides_dir / "tiny-guide" / "versions" / "v1.html").read_text()

    with pytest.raises(ValueError):
        import_lavish_html(source, slug="tiny-guide", guides_dir=guides_dir, title="Changed")

    # The published version is untouched by the refused re-import.
    assert (guides_dir / "tiny-guide" / "versions" / "v1.html").read_text() == v1

    manifest = import_lavish_html(
        source, slug="tiny-guide", guides_dir=guides_dir, title="Changed", force=True
    )
    assert manifest.title == "Changed"


def test_unreadable_manifest_is_skipped(tmp_path: Path):
    guides_dir = tmp_path / "guides"
    (guides_dir / "broken").mkdir(parents=True)
    (guides_dir / "broken" / "manifest.json").write_text("{not json", encoding="utf-8")
    (guides_dir / "not-a-guide.txt").write_text("x", encoding="utf-8")
    assert list_guides(guides_dir)["guides"] == []

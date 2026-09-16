from __future__ import annotations

from src.guides.verify import verify_html

PAGE = """
<section id="thesis">
  <p id="thesis-p1">Claim one <cite data-video="vid1" data-chunk="2" data-quote="the exact words">One</cite>.</p>
  <p id="thesis-p2">Claim two <cite data-video="vid1" data-chunk="9">Nine</cite>.</p>
  <p id="thesis-p3">Claim three <cite data-video="ghost">Ghost</cite>.</p>
  <p id="thesis-p4">Video-level <cite data-video="vid2">Two</cite>.</p>
  <p id="thesis-p5">Bad quote <cite data-video="vid1" data-chunk="2" data-quote="never said">One</cite>.</p>
</section>
"""

CHUNKS = {
    "vid1": [
        {"chunk_index": 1, "text": "first chunk"},
        {"chunk_index": 2, "text": "these are THE exact   words spoken"},
    ],
    "vid2": [{"chunk_index": 0, "text": "anything"}],
}


def lookup(video_id: str):
    return CHUNKS.get(video_id, [])


def test_resolves_video_chunk_and_quote():
    report = verify_html(PAGE, known_videos={"vid1", "vid2"}, chunk_lookup=lookup)
    assert report.total == 5
    by_index = {claim.cite_index: claim for claim in report.claims}
    assert by_index[0].valid and by_index[0].quote_ok is True
    assert by_index[1].valid is False and by_index[1].chunk_ok is False
    assert by_index[2].valid is False and by_index[2].video_ok is False
    assert by_index[3].valid and by_index[3].chunk_ok is None
    assert by_index[4].valid is False and by_index[4].quote_ok is False
    assert report.valid == 2
    assert report.pass_rate == 0.4
    assert report.sections == ["thesis"]
    assert all(claim.section_id == "thesis" for claim in report.claims)
    assert by_index[0].to_dict()["chunk_id"] == "chunk:vid1:2"


def test_video_level_only_when_no_chunk_lookup():
    report = verify_html(PAGE, known_videos={"vid1", "vid2"})
    # Chunk-bearing cites cannot be resolved without a lookup, so they fail
    # closed rather than pass silently.
    assert [claim.valid for claim in report.claims] == [False, False, False, True, False]


def test_structure_rules():
    page = (
        "<section><p>no id</p></section>"
        '<script src="https://evil.example/x.js"></script>'
        '<script src="/guides/guide-bridge.js"></script>'
        '<button onclick="alert(1)">x</button>'
    )
    report = verify_html(page, known_videos=set())
    assert "section without id" in report.structure_errors
    assert "external script https://evil.example/x.js" in report.structure_errors
    assert any(error.startswith("inline event handler") for error in report.structure_errors)
    assert len(report.structure_errors) == 3
    assert report.ok is False


def test_empty_page_is_ok():
    report = verify_html("<section id='a'></section>", known_videos=set())
    assert report.ok and report.pass_rate == 1.0 and report.total == 0


def test_unclosed_cite_is_a_structure_error_not_a_silent_pass():
    page = '<section id="a"><cite data-video="doesnotexist">quote</cite'
    report = verify_html(page, known_videos=set())
    assert report.ok is False
    assert any("never closed" in error for error in report.structure_errors)


def test_nested_cite_closes_the_previous_one_as_an_error():
    page = (
        '<section id="a">'
        '<cite data-video="vid1" data-chunk="2">first'
        '<cite data-video="vid2">second</cite>'
        "</section>"
    )
    report = verify_html(page, known_videos={"vid1", "vid2"}, chunk_lookup=lookup)
    assert report.ok is False
    assert any("still open" in error for error in report.structure_errors)
    # Both cites are recorded rather than the first being silently dropped.
    video_ids = [claim.video_id for claim in report.claims]
    assert "vid1" in video_ids
    assert "vid2" in video_ids

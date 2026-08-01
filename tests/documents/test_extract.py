"""Extracting a fetched page into headed, rankable sections."""

from __future__ import annotations

from src.documents.extract import document_id_for, extract_document
from src.documents.models import FetchedPage


def _page(body: str, content_type: str = "text/html", **overrides) -> FetchedPage:
    return FetchedPage(
        requested_url=overrides.pop("requested_url", "https://example.com/cv"),
        url=overrides.pop("url", "https://example.com/cv"),
        status_code=200,
        content_type=content_type,
        body=body,
        **overrides,
    )


RESUME = """
<html>
  <head><title>Nathan — Resume</title><style>.x{color:red}</style></head>
  <body>
    <header><p>Nathan Phillips — Sydney — nathan@example.com and other contact details</p></header>
    <h2>Experience</h2>
    <p>Led the retrieval platform team for four years, shipping a hybrid search stack.</p>
    <p>Owned the evaluation harness and the golden dataset used to gate releases.</p>
    <h2>Education</h2>
    <p>BSc in computer science, with a thesis on information retrieval evaluation.</p>
    <script>analytics('pageview')</script>
  </body>
</html>
"""


def test_headings_start_sections_and_text_lands_under_them() -> None:
    document = extract_document(_page(RESUME))

    headings = [section.heading for section in document.sections]
    assert "Experience" in headings
    assert "Education" in headings

    experience = next(s for s in document.sections if s.heading == "Experience")
    assert "retrieval platform team" in experience.text
    assert "thesis" not in experience.text


def test_the_title_comes_from_the_title_tag() -> None:
    assert extract_document(_page(RESUME)).title == "Nathan — Resume"


def test_script_and_style_contents_never_become_document_text() -> None:
    document = extract_document(_page(RESUME))

    assert "analytics" not in document.text
    assert "color:red" not in document.text


def test_header_content_is_kept_because_it_carries_the_contact_details() -> None:
    """A reader-mode heuristic that strips <header> would delete the name on
    exactly the documents this feature exists to review."""
    document = extract_document(_page(RESUME))

    assert "nathan@example.com" in document.text


def test_text_before_the_first_heading_is_its_own_section() -> None:
    document = extract_document(_page(RESUME))

    assert document.sections[0].heading is None
    assert "Nathan Phillips" in document.sections[0].text


def test_sections_are_indexed_in_document_order() -> None:
    document = extract_document(_page(RESUME))

    assert [section.index for section in document.sections] == list(range(len(document.sections)))


def test_a_section_labels_itself_by_heading_or_position() -> None:
    document = extract_document(_page(RESUME))

    assert document.sections[0].label == "section 1"
    assert next(s for s in document.sections if s.heading == "Experience").label == "Experience"


def test_block_tags_stop_words_running_together() -> None:
    document = extract_document(_page("<p>first thought here</p><p>second thought here</p>"))

    assert "herefirst" not in document.text
    assert "heresecond" not in document.text


def test_a_document_with_no_headings_is_one_section() -> None:
    body = "<p>" + ("a paragraph of prose that is long enough to stand alone. " * 3) + "</p>"

    document = extract_document(_page(body))

    assert len(document.sections) == 1
    assert document.sections[0].heading is None


def test_a_heading_with_too_little_under_it_folds_into_the_previous_section() -> None:
    """A heading with two words under it is a label, not a section — but its
    words are kept rather than dropped."""
    body = (
        "<h2>Summary</h2><p>" + ("a long opening paragraph with plenty of words. " * 3) + "</p>"
        "<h2>Skills</h2><p>Python.</p>"
    )

    document = extract_document(_page(body))

    assert [s.heading for s in document.sections] == ["Summary"]
    # Folded, not dropped: the heading joins the text it labelled.
    assert "Skills" in document.text
    assert "Python." in document.text


def test_the_first_heading_names_the_document_when_there_is_no_title_tag() -> None:
    body = "<h1>Team offsite invitation</h1><p>" + ("Details of the offsite. " * 5) + "</p>"

    assert extract_document(_page(body)).title == "Team offsite invitation"


def test_html_entities_are_decoded() -> None:
    body = "<p>" + ("Ben &amp; Jerry&#39;s tasting notes, at length. " * 3) + "</p>"

    assert "Ben & Jerry's" in extract_document(_page(body)).text


def test_malformed_markup_still_extracts_rather_than_raising() -> None:
    body = "<html><body><h2>Bio<p>unclosed everything, at some length here</body>"

    assert "unclosed everything" in extract_document(_page(body)).text


def test_a_stray_closing_tag_does_not_swallow_the_rest_of_the_document() -> None:
    body = "</script><p>" + ("real content that must survive. " * 3) + "</p>"

    assert "real content" in extract_document(_page(body)).text


# ── plain text ────────────────────────────────────────────────────────────────


def test_plain_text_is_split_on_blank_lines() -> None:
    body = (
        "First block of the note, easily long enough to stand on its own.\n\n"
        "Second block, also comfortably long enough to be its own section."
    )

    document = extract_document(_page(body, content_type="text/plain"))

    assert [s.heading for s in document.sections] == [None, None]
    assert "First block" in document.sections[0].text
    assert "Second block" in document.sections[1].text


def test_plain_text_has_no_title_to_take() -> None:
    assert extract_document(_page("just some words", content_type="text/plain")).title is None


# ── provenance ────────────────────────────────────────────────────────────────


def test_truncation_carries_through_so_a_cut_page_is_not_reviewed_as_whole() -> None:
    document = extract_document(_page(RESUME, truncated=True))

    assert document.truncated is True


def test_the_document_records_both_the_requested_and_the_final_url() -> None:
    page = _page(RESUME, requested_url="https://example.com/r", url="https://example.com/resume")

    document = extract_document(page)

    assert document.requested_url == "https://example.com/r"
    assert document.url == "https://example.com/resume"


def test_the_same_url_always_gets_the_same_id() -> None:
    assert document_id_for("https://example.com/cv") == document_id_for("https://example.com/cv")
    assert document_id_for("https://example.com/cv") != document_id_for("https://example.com/x")


def test_the_id_defaults_from_the_final_url_not_the_requested_one() -> None:
    """A redirect means the document *is* the page it landed on."""
    page = _page(RESUME, requested_url="https://example.com/r", url="https://example.com/resume")

    assert extract_document(page).id == document_id_for("https://example.com/resume")


def test_word_count_reports_the_extracted_text() -> None:
    document = extract_document(_page(RESUME))

    assert document.word_count > 20

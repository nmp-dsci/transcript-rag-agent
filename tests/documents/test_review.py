"""Choosing which sections of a document the answer call reads."""

from __future__ import annotations

from src.documents.models import Document, DocumentSection
from src.documents.review import (
    format_document_context,
    select_sections,
)


def _document(*sections: tuple[str | None, str], truncated: bool = False) -> Document:
    return Document(
        id="doc:abc",
        url="https://example.com/cv",
        requested_url="https://example.com/cv",
        title="A resume",
        truncated=truncated,
        sections=[
            DocumentSection(index=index, heading=heading, text=text)
            for index, (heading, text) in enumerate(sections)
        ],
    )


SMALL = _document(
    (None, "Nathan Phillips, Sydney."),
    ("Experience", "Led the retrieval platform team."),
    ("Education", "BSc computer science."),
)


# ── whole document when it fits ───────────────────────────────────────────────


def test_a_small_document_is_read_whole() -> None:
    """The parts a section is missing are as much the feedback as what it has."""
    selection = select_sections(SMALL, "is my experience section strong?")

    assert [s.index for s in selection.sections] == [0, 1, 2]
    assert selection.narrowed is False
    assert selection.covers_whole_document is True


def test_a_whole_document_selection_says_so_in_the_trace() -> None:
    detail = select_sections(SMALL, "any feedback?").detail()

    assert "whole document" in detail
    assert "all 3 sections" in detail


def test_an_empty_document_selects_nothing_without_erroring() -> None:
    selection = select_sections(_document(), "anything?")

    assert selection.sections == []
    assert selection.total_sections == 0


# ── ranked selection when it does not fit ─────────────────────────────────────


def _big_document() -> Document:
    return _document(
        ("Summary", "A summary paragraph. " * 40),
        ("Experience", "Retrieval platform and evaluation harness work. " * 40),
        ("Education", "Computer science degree and thesis. " * 40),
        ("Hobbies", "Cycling and bread baking on weekends. " * 40),
    )


def test_a_document_over_budget_is_narrowed_by_ranking() -> None:
    selection = select_sections(_big_document(), "hobbies", budget_chars=1200)

    assert selection.narrowed is True
    assert selection.total_sections == 4
    assert len(selection.sections) < 4


def test_narrowing_keeps_the_sections_the_question_is_about() -> None:
    selection = select_sections(_big_document(), "tell me about the hobbies", budget_chars=1200)

    assert any(section.heading == "Hobbies" for section in selection.sections)


def test_a_heading_is_matchable_because_questions_name_sections() -> None:
    """ "the education section" is a lexical hit on the heading, not the prose."""
    selection = select_sections(_big_document(), "the education section", budget_chars=1200)

    assert any(section.heading == "Education" for section in selection.sections)


def test_narrowed_sections_are_restored_to_document_order() -> None:
    """Feedback must describe the document the reader has, top to bottom."""
    selection = select_sections(_big_document(), "hobbies and summary", budget_chars=2500)

    indexes = [section.index for section in selection.sections]
    assert indexes == sorted(indexes)


def test_a_narrowed_selection_reports_what_was_left_out() -> None:
    """A review of 2 of 40 sections must not read as a review of the document."""
    detail = select_sections(_big_document(), "hobbies", budget_chars=1200).detail()

    assert "of 4 sections" in detail
    assert "did not fit" in detail


def test_at_least_one_section_survives_a_tiny_budget() -> None:
    """Answering from the best-matching section beats answering from nothing."""
    selection = select_sections(_big_document(), "hobbies", budget_chars=10)

    assert len(selection.sections) == 1


# ── the context the answer call sees ──────────────────────────────────────────


def test_the_context_labels_every_section_for_citation() -> None:
    context = format_document_context(SMALL, select_sections(SMALL, "feedback?"))

    assert "[§1]" in context
    assert "[§2] Experience" in context
    assert "[§3] Education" in context


def test_section_markers_use_the_documents_own_numbering() -> None:
    """Section 4 stays section 4 even when 2 and 3 were narrowed away."""
    document = _big_document()
    selection = select_sections(document, "hobbies", budget_chars=1200)

    context = format_document_context(document, selection)

    for section in selection.sections:
        assert f"[§{section.index + 1}]" in context


def test_the_context_names_the_document_and_its_url() -> None:
    context = format_document_context(SMALL, select_sections(SMALL, "feedback?"))

    assert "A resume" in context
    assert "https://example.com/cv" in context


def test_a_truncated_page_says_so_in_the_context() -> None:
    document = _document((None, "Some text."), truncated=True)

    context = format_document_context(document, select_sections(document, "x"))

    assert "cut short" in context


def test_a_narrowed_document_says_so_in_the_context() -> None:
    """The model must not describe a document it was only shown part of."""
    document = _big_document()
    selection = select_sections(document, "hobbies", budget_chars=1200)

    context = format_document_context(document, selection)

    assert "sections" in context and "not here" in context


def test_a_whole_document_context_carries_no_narrowing_note() -> None:
    context = format_document_context(SMALL, select_sections(SMALL, "feedback?"))

    assert "not here" not in context

"""Choosing which sections of a document the answer call reads."""

from __future__ import annotations

from src.documents.models import Document, DocumentSection
from src.documents.review import (
    MAX_REVIEW_QUERY_CHARS,
    REVIEW_INTENT_QUERIES,
    build_review_retrieval_query,
    classify_document,
    corpus_coverage_warning,
    document_topic,
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


def _page(url: str, *sections: tuple[str | None, str], title: str | None = None) -> Document:
    return Document(
        id="doc:x",
        url=url,
        requested_url=url,
        title=title,
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


# ── what the corpus is searched with ──────────────────────────────────────────


def test_the_url_is_not_what_gets_embedded() -> None:
    """A URL matches no transcript and dilutes the words that would."""
    query = build_review_retrieval_query(
        "Review https://example.com/cv against what recruiters look for", SMALL
    )

    assert "https://example.com/cv" not in query
    assert query == "Review against what recruiters look for"


def test_a_real_question_survives_the_url_coming_out() -> None:
    query = build_review_retrieval_query(
        "does my summary section match what hiring managers expect? https://example.com/cv",
        SMALL,
    )

    assert query.startswith("does my summary section match")


def test_a_bare_url_falls_back_to_the_criteria_for_its_kind() -> None:
    """ "review this: <url>" leaves nothing to retrieve on, so the document's
    kind supplies the criteria — SMALL calls itself "A resume"."""
    query = build_review_retrieval_query("review this: https://example.com/cv", SMALL)

    assert "https://" not in query
    assert query == REVIEW_INTENT_QUERIES["resume"]


def test_a_url_with_no_words_at_all_still_produces_a_query() -> None:
    query = build_review_retrieval_query("https://example.com/cv", SMALL)

    assert query == REVIEW_INTENT_QUERIES["resume"]
    assert "https://" not in query


def test_the_topic_tail_is_capped_so_it_stays_a_query() -> None:
    """Only the unrecognised-kind path carries a topic, and it stays bounded."""
    document = _page(
        "https://example.com/notes",
        *[(f"Heading number {index}", "body text") for index in range(200)],
    )

    assert classify_document(document) == "document"
    assert len(document_topic(document)) <= MAX_REVIEW_QUERY_CHARS
    assert len(build_review_retrieval_query("review this", document)) <= (
        MAX_REVIEW_QUERY_CHARS + len(REVIEW_INTENT_QUERIES["document"]) + 1
    )


def test_a_headingless_document_still_gets_review_intent() -> None:
    """No title and no headings: no topic to borrow, but "judge this" still has
    a generic answer, and it beats retrieving on two words of the question."""
    document = Document(
        id="doc:plain",
        url="https://example.com/note",
        requested_url="https://example.com/note",
        title=None,
        sections=[DocumentSection(index=0, heading=None, text="Some plain text.")],
    )

    query = build_review_retrieval_query("look at https://example.com/note", document)

    assert query == REVIEW_INTENT_QUERIES["document"]


# ── what kind of document is this ─────────────────────────────────────────────


def test_a_resume_is_recognised_by_the_sections_it_names_itself() -> None:
    document = _page(
        "https://example.com/cv",
        ("Professional summary", "Engineer with eight years of experience."),
        ("Work experience", "Led the retrieval platform team."),
        ("Education", "BSc computer science."),
    )

    assert classify_document(document) == "resume"


def test_one_resume_word_is_not_enough_to_call_something_a_resume() -> None:
    """A portfolio that mentions education once is still a portfolio."""
    document = _page(
        "https://nmp-dsci.github.io/",
        ("Selected systems", "Education tooling I built for a school."),
    )

    assert classify_document(document) == "portfolio"


def test_a_personal_site_host_is_a_portfolio_before_its_text_is_read() -> None:
    document = _page("https://someone.github.io/", (None, "Hello."))

    assert classify_document(document) == "portfolio"


def test_a_cover_letter_wins_over_the_resume_words_it_contains() -> None:
    document = _page(
        "https://example.com/letter",
        (None, "Dear hiring manager, my work experience and education are attached."),
    )

    assert classify_document(document) == "cover_letter"


def test_an_unrecognised_page_is_not_guessed_at() -> None:
    document = _page("https://example.com/notes", ("Meeting notes", "We agreed to ship on Friday."))

    assert classify_document(document) == "document"


# ── the fallback asks what to judge by, not what it is about ──────────────────


def test_a_bare_url_retrieves_the_criteria_for_the_documents_kind() -> None:
    """The failure this replaces: a portfolio's headings are its project names,
    so searching for them returned transcripts about building those projects
    and nothing about whether a portfolio like this gets you hired."""
    portfolio = _page(
        "https://nmp-dsci.github.io/",
        (None, "nathanphillips"),
        ("V2V Prod Agent", "A voice-to-voice banking agent."),
        ("ConvFinQA Agent", "Multi-turn financial Q&A."),
        title="Work · Nathan Phillips",
    )

    query = build_review_retrieval_query("review this: https://nmp-dsci.github.io/", portfolio)

    assert query == REVIEW_INTENT_QUERIES["portfolio"]
    assert "recruiters" in query and "hiring managers" in query
    assert "V2V" not in query and "ConvFinQA" not in query


def test_a_resume_and_a_portfolio_are_judged_by_different_criteria() -> None:
    resume = _page(
        "https://example.com/cv",
        ("Work experience", "Led a team."),
        ("Education", "BSc."),
    )
    portfolio = _page("https://someone.github.io/", (None, "Projects."))

    assert build_review_retrieval_query("review this", resume) == REVIEW_INTENT_QUERIES["resume"]
    assert (
        build_review_retrieval_query("review this", portfolio) == REVIEW_INTENT_QUERIES["portfolio"]
    )


def test_an_unrecognised_document_keeps_its_topic_because_nothing_better_exists() -> None:
    document = _page(
        "https://example.com/notes",
        ("Sharding strategy", "We partition by tenant id."),
        title="Design notes",
    )

    query = build_review_retrieval_query("review this", document)

    assert query.startswith(REVIEW_INTENT_QUERIES["document"])
    assert "Sharding strategy" in query


def test_a_real_question_still_beats_the_intent_fallback() -> None:
    """The user said what they wanted judged; do not overrule them."""
    portfolio = _page("https://someone.github.io/", (None, "Projects."))

    query = build_review_retrieval_query(
        "is my summary too long for a recruiter to scan? https://someone.github.io/", portfolio
    )

    assert query == "is my summary too long for a recruiter to scan?"


# ── a document is classified by what it calls itself ──────────────────────────


def test_a_portfolio_linking_to_a_resume_is_not_a_resume() -> None:
    """The guard that makes _RESUME_BY_NAME read labels and never body text.
    Swapping _labels(document) for the full haystack must fail here."""
    document = _page(
        "https://someone.example.com/",
        ("Selected work", "Three shipped systems."),
        (None, "Download my resume (PDF) · Contact me"),
        title="Work",
    )

    assert classify_document(document) == "portfolio"


def test_an_article_about_resumes_is_not_a_resume() -> None:
    """Resume section names are read from headings, so prose that discusses
    them is discussing them, not being them."""
    document = _page(
        "https://blog.example.com/how-to-write-a-resume",
        (
            "How to write a better one",
            "Your work experience section and your education section both matter, and "
            "your professional summary is what a recruiter reads first.",
        ),
        title="Advice for job seekers",
    )

    assert classify_document(document) == "document"


def test_a_testimonial_signed_sincerely_is_not_a_cover_letter() -> None:
    document = _page(
        "https://someone.github.io/",
        ("Selected systems", "Three shipped systems."),
        (None, "'Best engineer I have worked with.' — Sincerely, Priya, VP Eng"),
        title="Work",
    )

    assert classify_document(document) == "portfolio"


def test_a_cover_letter_is_recognised_by_how_it_opens() -> None:
    document = _page(
        "https://example.com/letter",
        (None, "Dear hiring manager, I am writing about the platform engineer role."),
    )

    assert classify_document(document) == "cover_letter"


def test_computer_vision_does_not_make_a_portfolio_a_resume() -> None:
    """ "CV" is two letters and this project's readers mean computer vision."""
    document = _page(
        "https://someone.example.com/",
        ("Computer Vision (CV) systems", "Detection and segmentation work."),
        ("Case studies", "Three shipped systems."),
        title="Work",
    )

    assert classify_document(document) == "portfolio"


def test_a_lookalike_host_is_not_a_personal_site() -> None:
    """endswith() would call evilnotgithub.io a portfolio."""
    document = _page("https://evilnotgithub.io/", (None, "Some words."))

    assert classify_document(document) == "document"


def test_a_real_subdomain_of_a_personal_site_host_still_matches() -> None:
    document = _page("https://someone.github.io/", (None, "Some words."))

    assert classify_document(document) == "portfolio"


# ── an unrecognised kind is reported, not hidden ──────────────────────────────


def test_an_unrecognised_document_warns_that_the_corpus_may_not_cover_it() -> None:
    """Every named kind is a career document. Retrieval will return its ten best
    chunks for a wedding invitation too, and they will be resume advice."""
    invitation = _page(
        "https://example.com/invite",
        (None, "Please join us to celebrate the marriage of Sam and Alex on 3 June."),
        title="Sam & Alex",
    )

    warning = corpus_coverage_warning(invitation)

    assert warning is not None
    assert "does not match a kind the corpus has criteria for" in warning


def test_a_recognised_kind_carries_no_warning() -> None:
    portfolio = _page("https://someone.github.io/", (None, "Projects."))

    assert corpus_coverage_warning(portfolio) is None


def test_the_warning_reaches_the_model_context() -> None:
    """A caveat the model was never told about is one it cannot make."""
    invitation = _page(
        "https://example.com/invite",
        (None, "Please join us to celebrate the marriage of Sam and Alex on 3 June."),
    )

    context = format_document_context(invitation, select_sections(invitation, "review this"))

    assert "does not match a kind the corpus has criteria for" in context


def test_a_recognised_kind_context_carries_no_coverage_note() -> None:
    context = format_document_context(SMALL, select_sections(SMALL, "review this"))

    assert "criteria for" not in context

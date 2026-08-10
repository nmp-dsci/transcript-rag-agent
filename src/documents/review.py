"""Choosing which of a document's sections the answer call actually reads.

Retrieval over a *document* is a different problem from retrieval over the
corpus, and treating them the same would be a mistake. The corpus is over a
thousand chunks across dozens of videos, so retrieval is mandatory. A resume, an invitation or a
landing page is a few thousand words: it fits, and a reviewer asked "is my
experience section strong?" needs to see the whole document to answer, because
the parts a section is *missing* are as much the feedback as the parts it has.

So the rule is **whole document when it fits, ranked selection when it does
not**, and the selection is reported either way — a review of 6 of 40 sections
must not read as a review of the document.

Ranking, when it is needed, is BM25 over the sections. Not embeddings: the
document was fetched a moment ago and has no index, embedding it per question
would cost more than the answer call, and a user's question about their own
document tends to share its vocabulary ("experience section", "the pricing
paragraph") — which is exactly the case lexical matching handles well.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal
from urllib.parse import urlparse

from src.documents.models import Document, DocumentSection

#: Characters of document text the answer call will accept before the document
#: has to be narrowed. Roughly a few thousand words — comfortably a whole
#: resume, a landing page, or an invitation, which is what this is for.
DEFAULT_SECTION_BUDGET_CHARS = 12_000

#: URLs in the *retrieval* query only. The answering prompt always keeps the
#: user's original wording; this is about what gets embedded.
_URL_IN_TEXT = re.compile(r"https?://\S+", re.IGNORECASE)

#: Below this many words, what is left of the question after the URL comes out
#: ("review this:", "thoughts?") carries no topic to search the corpus with, so
#: the review's *intent* has to supply the query instead.
MIN_REVIEW_QUERY_WORDS = 4

#: Cap on the topic tail, so a long document does not turn into a query longer
#: than the chunks it is meant to match.
MAX_REVIEW_QUERY_CHARS = 400

#: What sort of thing is being reviewed. Not what it is *about* — a portfolio
#: full of RAG projects is still a portfolio, and the criteria it should be
#: judged against are portfolio criteria, not RAG ones.
DocumentKind = Literal["resume", "portfolio", "profile", "cover_letter", "document"]

#: Hosts that host a personal site and essentially nothing else. A page here is
#: a portfolio even before its text is read. Matched as a whole host or a proper
#: parent domain, never as a substring — ``evilnotgithub.io`` is not GitHub.
_PORTFOLIO_HOSTS = ("github.io", "vercel.app", "netlify.app", "pages.dev", "surge.sh")

#: Section names a resume gives itself. Matched against **headings only**, for
#: the same reason ``_RESUME_BY_NAME`` is: a blog post *about* resumes says
#: "work experience" and "education" in its prose all day, and counting body
#: text would classify the article as the thing it is describing.
#:
#: Two independent hits are required, so a portfolio with one "Education"
#: heading is still a portfolio.
_RESUME_SIGNALS = (
    "work experience",
    "professional experience",
    "employment history",
    "work history",
    "education",
    "certifications",
    "professional summary",
    "career summary",
    "references available",
)
#: A document that calls *itself* a resume in its title or a heading is one, and
#: no counting of section names is needed. Matched against titles and headings
#: only, never body text: "download my resume" in a portfolio's footer is a link
#: to a resume, not a resume.
#:
#: ``cv`` is deliberately absent. It is two letters, and the readers of this
#: project write "CV" for computer vision far more often than for curriculum
#: vitae — a false resume on a computer-vision portfolio is a worse trade than
#: missing a page whose only self-description is "CV".
_RESUME_BY_NAME = re.compile(r"\b(resum[eé]s?|curriculum vitae)\b", re.IGNORECASE)

#: Cover-letter openings and sign-offs, matched against **headings and the first
#: characters of the document** rather than anywhere in it. A testimonial on a
#: portfolio signed "— Sincerely, Priya (VP Eng)" is not a cover letter, and at
#: one hit and highest precedence this would otherwise decide the whole
#: classification on someone else's signature.
_COVER_LETTER_SIGNALS = ("dear hiring", "dear sir", "dear ", "cover letter")
_COVER_LETTER_SIGNOFFS = ("yours sincerely", "yours faithfully", "kind regards")

#: How much of the start of a document counts as its opening, for the salutation
#: test. A cover letter opens with "Dear ..." — a page that says it 4,000
#: characters in is quoting one.
COVER_LETTER_OPENING_CHARS = 400

_PROFILE_SIGNALS = ("connections", "followers", "endorsements", "recommendations received")
_PORTFOLIO_SIGNALS = (
    "portfolio",
    "selected work",
    "selected systems",
    "selected projects",
    "case study",
    "case studies",
    "my projects",
    "featured project",
    "things i've built",
    "things i have built",
)

#: The query each kind is reviewed *against* — the criteria a reader of that
#: kind of document applies, written in the vocabulary the corpus uses.
#:
#: These are the load-bearing strings of the whole review path. A document with
#: no question attached is asking "judge this", and "judge this" only has an
#: answer once you know what sort of thing it is. Searching the corpus for what
#: the document is *about* answers a different question entirely: for a
#: portfolio of AI projects it returns transcripts about building AI, which
#: cannot tell you anything about whether the portfolio is any good.
REVIEW_INTENT_QUERIES: dict[DocumentKind, str] = {
    "resume": (
        "what makes a strong engineering resume — professional summary, experience bullets, "
        "quantified impact, skills section, keywords and ATS, formatting, and the mistakes "
        "recruiters and hiring managers say get a resume rejected"
    ),
    "portfolio": (
        "how to present engineering projects on a personal portfolio site so recruiters and "
        "hiring managers take them seriously — project write-ups, links to code and demos, "
        "quantified results, skills and experience, and the portfolio mistakes that get you "
        "passed over"
    ),
    "profile": (
        "how to optimise a professional profile so recruiters find and contact you — headline, "
        "summary, experience, skills, keywords, referrals and recommendations, and the profile "
        "mistakes that keep recruiters away"
    ),
    "cover_letter": (
        "what makes a strong cover letter or outreach message to a hiring manager — opening, "
        "specificity, evidence of impact, tailoring to the role, length, and the mistakes that "
        "get it ignored"
    ),
    "document": (
        "how to review this kind of document — structure, clarity, evidence for its claims, "
        "and what a reader looks for"
    ),
}


@dataclass
class SectionSelection:
    """Which sections were chosen, and whether anything was left out."""

    sections: list[DocumentSection] = field(default_factory=list)
    #: True when the document did not fit and was narrowed by ranking.
    narrowed: bool = False
    #: How many sections the document has in total, narrowed or not.
    total_sections: int = 0

    @property
    def covers_whole_document(self) -> bool:
        return not self.narrowed

    def detail(self) -> str:
        """A one-line description for the answer trace."""
        if not self.narrowed:
            return f"whole document — all {self.total_sections} sections in context"
        return (
            f"{len(self.sections)} of {self.total_sections} sections selected by "
            "BM25; the document did not fit in the context budget"
        )


def _records(document: Document) -> list[dict]:
    return [
        {
            "index": section.index,
            # The heading is part of what a question matches on: "the experience
            # section" is a lexical hit on the heading, not on the prose.
            "text": f"{section.heading or ''}\n{section.text}".strip(),
        }
        for section in document.sections
    ]


def select_sections(
    document: Document,
    question: str,
    *,
    budget_chars: int = DEFAULT_SECTION_BUDGET_CHARS,
) -> SectionSelection:
    """The sections the answer call should read, in document order.

    Document order is restored after ranking on purpose: a reviewer reads a
    document top to bottom, and presenting section 9 above section 2 because it
    scored higher would make the feedback describe a document nobody has.
    """
    total = len(document.sections)
    if total == 0:
        return SectionSelection(sections=[], narrowed=False, total_sections=0)

    if len(document.text) <= budget_chars:
        return SectionSelection(
            sections=list(document.sections), narrowed=False, total_sections=total
        )

    from src.rag import bm25

    by_index = {section.index: section for section in document.sections}
    # Ranked widest-first, then trimmed to the budget, so the selection is as
    # much of the document as fits rather than a fixed section count.
    ranked = bm25.search(_records(document), question, total, cache_key=f"doc:{document.id}")
    chosen: list[DocumentSection] = []
    used = 0
    for record in ranked:
        section = by_index.get(record["index"])
        if section is None:
            continue
        cost = len(section.text) + len(section.heading or "")
        if chosen and used + cost > budget_chars:
            continue
        chosen.append(section)
        used += cost

    chosen.sort(key=lambda section: section.index)
    return SectionSelection(sections=chosen, narrowed=len(chosen) < total, total_sections=total)


def document_topic(document: Document) -> str:
    """What the document is about, in the words the document itself uses.

    Title plus headings rather than body text: headings are the document's own
    summary of itself, and they are short enough to embed as a query without
    drowning the signal in prose.

    Note what this is *not* good for: deciding what a document should be judged
    against. See :func:`build_review_retrieval_query`.
    """
    parts = [document.title or ""]
    parts.extend(section.heading or "" for section in document.sections)
    topic = " ".join(part.strip() for part in parts if part and part.strip())
    return topic[:MAX_REVIEW_QUERY_CHARS].strip()


def _haystack(document: Document) -> str:
    """Title, headings and text, lowercased, for kind detection."""
    parts = [document.title or ""]
    for section in document.sections:
        parts.append(section.heading or "")
        parts.append(section.text)
    return " ".join(parts).lower()


def _labels(document: Document) -> str:
    """Only the names the document gives itself: its title and its headings."""
    parts = [document.title or ""]
    parts.extend(section.heading or "" for section in document.sections)
    return " ".join(parts)


def _hits(haystack: str, signals: tuple[str, ...]) -> int:
    return sum(1 for signal in signals if signal in haystack)


def _host_matches(host: str, domains: tuple[str, ...]) -> bool:
    """Whole-host or proper-subdomain match, never a substring one.

    ``host.endswith("github.io")`` is true of ``evilnotgithub.io``, which is a
    different site owned by someone else.
    """
    return any(host == domain or host.endswith("." + domain) for domain in domains)


def classify_document(document: Document) -> DocumentKind:
    """What sort of document this is, so it can be judged by the right criteria.

    Lexical and host-based, with no model call: this runs on every review, ahead
    of retrieval, and paying for an LLM round-trip to learn "this is a resume"
    would cost more than the answer it is setting up.

    One rule runs through all of it: **a document is classified by what it calls
    itself, not by what it mentions.** Resume section names and the resume/CV
    name are read from headings and the title; a cover letter has to open with
    its salutation or sign off with one, not merely contain the words somewhere.
    Body text is evidence only where a phrase is not something a document would
    say *about* another document — which is why portfolio and profile signals
    may match anywhere but resume ones may not. Without that rule a blog post
    about resumes classifies as a resume, and a portfolio quoting a testimonial
    signed "Sincerely, Priya" classifies as a cover letter.

    Order is the rest of the design. A resume lists projects and a portfolio
    lists experience, so the checks run most-distinctive first and the first
    match wins. Unrecognised documents get ``"document"`` rather than a guess,
    and that outcome is reported rather than hidden — see
    :func:`corpus_coverage_warning`.
    """
    host = (urlparse(document.url).hostname or "").lower()
    haystack = _haystack(document)
    labels = _labels(document).lower()
    opening = haystack[:COVER_LETTER_OPENING_CHARS]

    if (
        _hits(opening, _COVER_LETTER_SIGNALS)
        or _hits(labels, _COVER_LETTER_SIGNALS)
        or _hits(haystack, _COVER_LETTER_SIGNOFFS)
    ):
        return "cover_letter"
    if _RESUME_BY_NAME.search(_labels(document)):
        return "resume"
    if _hits(labels, _RESUME_SIGNALS) >= 2:
        return "resume"
    if _host_matches(host, ("linkedin.com",)) or _hits(haystack, _PROFILE_SIGNALS) >= 2:
        return "profile"
    if _host_matches(host, _PORTFOLIO_HOSTS) or _hits(haystack, _PORTFOLIO_SIGNALS):
        return "portfolio"
    return "document"


def corpus_coverage_warning(document: Document) -> str | None:
    """Said out loud when the corpus may hold no criteria for this document.

    Every named kind here is a career document, because that is what the corpus
    is about. When the kind cannot be named, retrieval still returns its ten
    best chunks and they are still career-advice chunks — the store has nothing
    else to offer. A wedding invitation gets reviewed half against resume
    guidance, and every part of the UI reports business as usual.

    That is the failure worth preventing, and the fix is not to suppress the
    answer — partial guidance is often still useful, and the user asked. The fix
    is to say which it is, in the trace and in the answer, so nobody mistakes
    "the best ten chunks in the corpus" for "criteria that apply to this".
    """
    if classify_document(document) != "document":
        return None
    return (
        "This document does not match a kind the corpus has criteria for "
        "(resume, portfolio, professional profile, cover letter). The retrieved "
        "chunks are the closest the corpus holds, which may be advice about a "
        "different kind of document entirely — say so rather than applying it."
    )


def build_review_retrieval_query(question: str, document: Document) -> str:
    """The query the *corpus* is searched with when a document is reviewed.

    The question a reviewer types usually contains the document's URL, and a URL
    is the worst possible thing to embed: it is a unique string that matches no
    transcript, and it dilutes the words that would have. So it comes out.

    What is left is often still a real question ("does my summary section match
    what recruiters look for") and is used as-is.

    But "review this: <url>" leaves nothing to search on, and this is the case
    the whole feature lives or dies on — it is the shortest thing a user can
    type. The obvious fallback, searching for what the document is *about*, is
    wrong and measurably so: a portfolio's headings are its project names, so it
    retrieves transcripts about building those projects and not one word about
    whether a portfolio like this gets you hired. Subject matter answers "what is
    this document"; a review needs "what should this document be judged
    against". Those are different questions with different answers.

    So the fallback is the document's *kind*, turned into the criteria query for
    that kind (:data:`REVIEW_INTENT_QUERIES`). A kind we cannot name keeps the
    topic as a tail, because for an unrecognised document its subject really is
    the best available guess at what would be worth retrieving — and the query
    says as much rather than pretending to criteria the corpus may not hold.
    """
    stripped = re.sub(r"\s+", " ", _URL_IN_TEXT.sub(" ", question)).strip()
    if len(stripped.split()) >= MIN_REVIEW_QUERY_WORDS:
        return stripped

    kind = classify_document(document)
    intent = REVIEW_INTENT_QUERIES[kind]
    if kind != "document":
        return intent
    topic = document_topic(document)
    return f"{intent} {topic}".strip() if topic else intent


def format_document_context(document: Document, selection: SectionSelection) -> str:
    """The document as the answer call sees it, with citable section markers.

    Every section is labelled ``[§N]`` using its own index, so a citation
    survives narrowing: section 7 is section 7 whether or not sections 3 to 6
    made it into the context.
    """
    lines = [f"DOCUMENT: {document.title or document.url}", f"URL: {document.url}"]
    coverage = corpus_coverage_warning(document)
    if coverage:
        # In the model's own context, not only in the trace: the answer is where
        # the caveat has to appear, and a caveat the model was never told about
        # is one it cannot make.
        lines.append(f"NOTE: {coverage}")
    if document.truncated:
        lines.append(
            "NOTE: this page was longer than the fetch limit and was cut short; "
            "the end of it is missing."
        )
    if selection.narrowed:
        lines.append(
            f"NOTE: {len(selection.sections)} of {selection.total_sections} sections "
            "are shown, chosen for this question; the rest of the document is not here."
        )
    lines.append("")
    for section in selection.sections:
        marker = f"[§{section.index + 1}]"
        heading = f" {section.heading}" if section.heading else ""
        lines.append(f"{marker}{heading}")
        lines.append(section.text)
        lines.append("")
    return "\n".join(lines).strip()

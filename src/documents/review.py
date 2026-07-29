"""Choosing which of a document's sections the answer call actually reads.

Retrieval over a *document* is a different problem from retrieval over the
corpus, and treating them the same would be a mistake. The corpus is 546 chunks
across 26 videos, so retrieval is mandatory. A resume, an invitation or a
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

from dataclasses import dataclass, field

from src.documents.models import Document, DocumentSection

#: Characters of document text the answer call will accept before the document
#: has to be narrowed. Roughly a few thousand words — comfortably a whole
#: resume, a landing page, or an invitation, which is what this is for.
DEFAULT_SECTION_BUDGET_CHARS = 12_000


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


def format_document_context(document: Document, selection: SectionSelection) -> str:
    """The document as the answer call sees it, with citable section markers.

    Every section is labelled ``[§N]`` using its own index, so a citation
    survives narrowing: section 7 is section 7 whether or not sections 3 to 6
    made it into the context.
    """
    lines = [f"DOCUMENT: {document.title or document.url}", f"URL: {document.url}"]
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

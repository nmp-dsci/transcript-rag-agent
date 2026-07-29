"""What a fetched document is, once it has been retrieved and extracted.

Two levels. :class:`FetchedPage` is the raw HTTP result — the bytes, where they
actually came from, and how they were bounded. :class:`Document` is what the
chat and the retrieval layer see: ordered :class:`DocumentSection` units with
headings, which is what lets feedback anchor to "the experience section"
instead of to a character offset.
"""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field


class FetchedPage(BaseModel):
    """One successful HTTP fetch, with the provenance a reviewer needs.

    ``url`` is where the bytes actually came from after redirects, which is not
    always ``requested_url`` — a review of a page has to name the page it
    reviewed, not the link someone pasted.
    """

    requested_url: str
    url: str
    status_code: int
    content_type: str
    body: str
    #: The body hit the byte cap and was cut short. Recorded rather than
    #: hidden: a review of half a page is not a review of the page.
    truncated: bool = False
    #: Every URL passed through, in order, starting with the requested one.
    redirect_chain: list[str] = Field(default_factory=list)
    fetched_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class DocumentSection(BaseModel):
    """One heading-delimited run of text.

    Sections are the unit of both retrieval and citation, so they carry an
    ``index`` that is stable for a given extraction — a feedback point that
    says "section 3" has to keep meaning the same section when the card
    re-renders.
    """

    index: int
    heading: str | None
    text: str

    @property
    def label(self) -> str:
        """How a citation names this section."""
        return self.heading or f"section {self.index + 1}"


class Document(BaseModel):
    """An extracted document, ready to be reviewed and retrieved over."""

    id: str
    url: str
    requested_url: str
    title: str | None = None
    sections: list[DocumentSection] = Field(default_factory=list)
    truncated: bool = False
    fetched_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def text(self) -> str:
        """The whole document as plain text, headings included."""
        parts: list[str] = []
        for section in self.sections:
            if section.heading:
                parts.append(section.heading)
            if section.text:
                parts.append(section.text)
        return "\n\n".join(parts)

    @property
    def word_count(self) -> int:
        return len(self.text.split())

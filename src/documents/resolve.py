"""Turning a chat message into the document it wants reviewed, if any.

The one place that decides between "ordinary question", "reuse the document
this thread is already about", and "fetch a new one" — so the rule lives once
rather than being re-derived at each call site.

Reuse comes first when a thread already has a document pinned. A follow-up
("now check the experience section") should read the same text the first answer
read, even if the page has changed underneath, and re-fetching would make a
conversation about one document quietly describe two.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

from src.documents.detect import find_document_url
from src.documents.extract import document_id_for, extract_document
from src.documents.fetch import DocumentFetchError, UnsafeUrlError, fetch_document
from src.documents.models import Document
from src.documents.review import (
    SectionSelection,
    format_document_context,
    select_sections,
)
from src.documents.store import DocumentStore

logger = logging.getLogger(__name__)

#: ``(url) -> FetchedPage``. Swapped in tests; the default is the guarded fetch.
FetchFn = Callable[[str], Any]


@dataclass
class ResolvedDocument:
    """The document a question is about, and how it was arrived at."""

    document: Document
    selection: SectionSelection
    #: Read from the store rather than fetched, so the page was not hit again.
    reused: bool = False

    @property
    def context(self) -> str:
        return format_document_context(self.document, self.selection)

    def detail(self) -> str:
        source = "reused from this thread" if self.reused else "fetched"
        return f"{source} — {self.selection.detail()}"


def resolve_document(
    question: str,
    *,
    store: DocumentStore,
    pinned_document_id: str | None = None,
    fetch: FetchFn | None = None,
) -> ResolvedDocument | None:
    """The document this question reviews, or ``None`` for an ordinary question.

    Raises :class:`~src.documents.fetch.UnsafeUrlError` or
    :class:`~src.documents.fetch.DocumentFetchError` when a URL *was* asked for
    and could not be retrieved — the caller has to tell the user, because
    silently answering from the corpus alone would look like a review of a page
    nobody read.
    """
    if pinned_document_id:
        pinned = store.get(pinned_document_id)
        if pinned is not None:
            return ResolvedDocument(
                document=pinned,
                selection=select_sections(pinned, question),
                reused=True,
            )
        logger.warning(
            "pinned document %s is gone; falling back to the message", pinned_document_id
        )

    url = find_document_url(question)
    if url is None:
        return None

    cached = store.get(document_id_for(url))
    if cached is not None:
        return ResolvedDocument(
            document=cached, selection=select_sections(cached, question), reused=True
        )

    page = (fetch or fetch_document)(url)
    document = extract_document(page)
    store.save(document)
    return ResolvedDocument(document=document, selection=select_sections(document, question))


def describe_failure(error: Exception) -> str:
    """A user-facing sentence for a fetch that did not happen.

    ``UnsafeUrlError`` messages are written to be shown verbatim; anything else
    is summarised, so an internal error string never reaches the chat.
    """
    if isinstance(error, UnsafeUrlError):
        return f"That link could not be reviewed: {error}"
    if isinstance(error, DocumentFetchError):
        return f"That link could not be fetched: {error}"
    return "That link could not be reviewed."

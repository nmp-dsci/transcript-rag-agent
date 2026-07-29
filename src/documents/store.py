"""Where a fetched document lives, and why it is not the chat history.

``dashboard/chat_history.json`` is committed to the repository. Somebody
reviewing their own resume, a private invitation, or an unpublished page would
otherwise have that text land in a tracked file and, sooner or later, in a
commit. So the history keeps only a reference — the document id — and the text
itself is written here, under ``.yt-agent/`` which is gitignored.

That split also gives follow-ups their pinned document for free: asking "now
check the experience section" in the same thread resolves the id and re-reads
the stored text rather than fetching the page again, which keeps a conversation
about one document consistent even if the page changes underneath it.

One JSON file per document, named by id: a kill mid-write corrupts at most the
document being written, and any single document can be inspected or deleted on
its own.
"""

from __future__ import annotations

import logging
from pathlib import Path

from pydantic import ValidationError

from src.documents.models import Document

logger = logging.getLogger(__name__)

DEFAULT_DOCUMENT_DIR = Path(".yt-agent/documents")


class DocumentStore:
    """Reads and writes extracted documents, keyed by id."""

    def __init__(self, directory: Path | str = DEFAULT_DOCUMENT_DIR) -> None:
        self.directory = Path(directory)

    def path_for(self, document_id: str) -> Path:
        """The file one document lives in.

        Ids are hashes produced by :func:`~src.documents.extract.document_id_for`,
        but this is the boundary where an id becomes a filesystem path, so the
        separator is rejected here rather than assumed away — an id carrying one
        would otherwise write outside the store.
        """
        if "/" in document_id or "\\" in document_id or document_id in {"", ".", ".."}:
            raise ValueError(f"unsafe document id: {document_id!r}")
        return self.directory / f"{document_id.replace(':', '_')}.json"

    def save(self, document: Document) -> Path:
        path = self.path_for(document.id)
        self.directory.mkdir(parents=True, exist_ok=True)
        path.write_text(document.model_dump_json(indent=2) + "\n", encoding="utf-8")
        return path

    def get(self, document_id: str) -> Document | None:
        """The stored document, or ``None`` if it is absent or unreadable.

        A missing document is an ordinary outcome, not an error: the store is
        derived state that can be cleared at any time, and the caller's job is
        then to fetch the page again rather than to fail.
        """
        try:
            path = self.path_for(document_id)
        except ValueError:
            return None
        if not path.exists():
            return None
        try:
            return Document.model_validate_json(path.read_text(encoding="utf-8"))
        except (ValidationError, ValueError, OSError):
            logger.warning("discarding unreadable document %s", document_id)
            return None

    def delete(self, document_id: str) -> bool:
        """Remove one document, reporting whether there was one to remove."""
        try:
            path = self.path_for(document_id)
        except ValueError:
            return False
        if not path.exists():
            return False
        path.unlink()
        return True

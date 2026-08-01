"""The document store: round-tripping, and staying out of committed history."""

from __future__ import annotations

import pytest

from src.documents.models import Document, DocumentSection
from src.documents.store import DEFAULT_DOCUMENT_DIR, DocumentStore


def _document(document_id: str = "doc:abc123") -> Document:
    return Document(
        id=document_id,
        url="https://example.com/cv",
        requested_url="https://example.com/cv",
        title="A resume",
        sections=[
            DocumentSection(index=0, heading=None, text="Nathan — Sydney"),
            DocumentSection(index=1, heading="Experience", text="Led a team."),
        ],
    )


def test_a_document_round_trips_through_the_store(tmp_path) -> None:
    store = DocumentStore(tmp_path)
    store.save(_document())

    loaded = store.get("doc:abc123")

    assert loaded is not None
    assert loaded.title == "A resume"
    assert [section.heading for section in loaded.sections] == [None, "Experience"]


def test_a_missing_document_is_none_rather_than_an_error(tmp_path) -> None:
    """The store is derived state; the caller's answer to absence is to fetch."""
    assert DocumentStore(tmp_path).get("doc:never-saved") is None


def test_an_unreadable_document_is_none_rather_than_a_crash(tmp_path) -> None:
    store = DocumentStore(tmp_path)
    store.save(_document())
    store.path_for("doc:abc123").write_text("{not json", encoding="utf-8")

    assert store.get("doc:abc123") is None


def test_saving_creates_the_directory(tmp_path) -> None:
    store = DocumentStore(tmp_path / "nested" / "documents")

    path = store.save(_document())

    assert path.exists()


def test_documents_are_one_file_each(tmp_path) -> None:
    """A kill mid-write can then corrupt at most the document being written."""
    store = DocumentStore(tmp_path)
    store.save(_document("doc:one"))
    store.save(_document("doc:two"))

    assert len(list(tmp_path.glob("*.json"))) == 2


def test_deleting_reports_whether_there_was_anything_to_delete(tmp_path) -> None:
    store = DocumentStore(tmp_path)
    store.save(_document())

    assert store.delete("doc:abc123") is True
    assert store.delete("doc:abc123") is False
    assert store.get("doc:abc123") is None


# ── the boundary where an id becomes a path ───────────────────────────────────


@pytest.mark.parametrize("document_id", ["../escape", "a/b", "a\\b", "", ".", ".."])
def test_an_id_that_would_escape_the_store_is_rejected(tmp_path, document_id: str) -> None:
    with pytest.raises(ValueError, match="unsafe document id"):
        DocumentStore(tmp_path).path_for(document_id)


@pytest.mark.parametrize("document_id", ["../escape", "a/b"])
def test_reads_and_deletes_of_an_unsafe_id_fail_closed(tmp_path, document_id: str) -> None:
    store = DocumentStore(tmp_path)

    assert store.get(document_id) is None
    assert store.delete(document_id) is False


# ── privacy ───────────────────────────────────────────────────────────────────


def test_the_default_location_is_inside_the_gitignored_directory() -> None:
    """Fetched text — a resume, a private invitation — must never reach the
    committed chat history, so the store defaults under .yt-agent/."""
    assert DEFAULT_DOCUMENT_DIR.parts[0] == ".yt-agent"


def test_the_gitignore_actually_covers_that_directory() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    ignored = (root / ".gitignore").read_text(encoding="utf-8").splitlines()

    assert ".yt-agent/" in [line.strip() for line in ignored]

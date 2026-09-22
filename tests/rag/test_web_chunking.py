from __future__ import annotations

from src.documents.models import Document, DocumentSection
from src.rag.web_chunking import build_web_chunks


def document(*sections: tuple[str | None, str], title: str = "A guide") -> Document:
    return Document(
        id="doc:1",
        url="https://a.example/post",
        requested_url="https://a.example/post",
        title=title,
        sections=[
            DocumentSection(index=index, heading=heading, text=text)
            for index, (heading, text) in enumerate(sections)
        ],
    )


def test_a_chunk_never_spans_two_headings() -> None:
    # The rule the whole design turns on: a chunk straddling two headings
    # knows which section it is in no better than a transcript chunk with no
    # timestamp knows when it was said.
    # Multi-paragraph sections, so each one is long enough to need splitting.
    def body(letter: str) -> str:
        return "\n".join(f"{letter} " * 30 for _ in range(6))

    doc = document(("First", body("a")), ("Second", body("b")), ("Third", body("c")))
    chunks = build_web_chunks(doc, external_id="guid", target_chars=200)
    assert len(chunks) > 3
    for chunk in chunks:
        assert chunk.heading == doc.sections[chunk.section_index].heading
        letters = set(chunk.text.replace(" ", "").replace("\n", ""))
        assert len(letters) == 1


def test_a_long_section_is_split_with_every_part_keeping_its_heading() -> None:
    # Measured: one registered source has an 11,855-character section.
    body = "\n".join(f"Paragraph number {index} with some prose in it." for index in range(40))
    chunks = build_web_chunks(document(("Long", body)), external_id="guid", target_chars=300)
    assert len(chunks) > 3
    assert {chunk.heading for chunk in chunks} == {"Long"}
    assert [chunk.part_index for chunk in chunks] == list(range(len(chunks)))
    assert all(chunk.section_index == 0 for chunk in chunks)


def test_chunk_indices_are_contiguous_across_sections() -> None:
    doc = document(("A", "one " * 100), ("B", "two " * 100))
    chunks = build_web_chunks(doc, external_id="guid", target_chars=200)
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))


def test_consecutive_parts_overlap_so_a_split_sentence_stays_with_its_lead_in() -> None:
    body = "\n".join(
        f"Sentence {index} carrying a distinct marker M{index}." for index in range(12)
    )
    chunks = build_web_chunks(
        document(("S", body)), external_id="guid", target_chars=160, overlap_chars=80
    )
    assert len(chunks) > 2
    overlaps = [bool(set(a.text.split()) & set(b.text.split())) for a, b in zip(chunks, chunks[1:])]
    assert all(overlaps)


def test_zero_overlap_produces_disjoint_parts() -> None:
    body = "\n".join(f"Unique line number {index} here." for index in range(12))
    chunks = build_web_chunks(
        document(("S", body)), external_id="guid", target_chars=120, overlap_chars=0
    )
    seen: set[str] = set()
    for chunk in chunks:
        lines = set(chunk.text.splitlines())
        assert not (lines & seen)
        seen |= lines


def test_a_paragraph_longer_than_the_target_is_left_whole() -> None:
    # Breaking prose at an arbitrary character is worse than one oversized
    # chunk, and the embedding model truncates gracefully.
    long_paragraph = "word " * 500
    chunks = build_web_chunks(document(("S", long_paragraph)), external_id="guid", target_chars=200)
    assert len(chunks) == 1
    assert chunks[0].text.strip() == long_paragraph.strip()


def test_empty_sections_produce_no_chunks() -> None:
    chunks = build_web_chunks(document(("Heading only", "   "), (None, "")), external_id="guid")
    assert chunks == []


def test_chunks_carry_an_anchor_and_a_citation_for_deep_linking() -> None:
    chunks = build_web_chunks(document(("How to use this FAQ", "prose " * 40)), external_id="guid")
    assert chunks[0].anchor == "how-to-use-this-faq"
    assert chunks[0].citation == "A guide · § How to use this FAQ"


def test_a_headingless_section_still_chunks_and_cites_the_document() -> None:
    chunks = build_web_chunks(document((None, "prose " * 40)), external_id="guid")
    assert chunks
    assert chunks[0].heading is None
    assert chunks[0].anchor is None
    assert chunks[0].citation == "A guide"


def test_each_chunk_hashes_its_own_text_so_a_small_edit_is_cheap() -> None:
    first = build_web_chunks(document(("A", "one " * 60), ("B", "two " * 60)), external_id="guid")
    edited = build_web_chunks(
        document(("A", "one " * 60), ("B", "three " * 60)), external_id="guid"
    )
    unchanged = {chunk.content_hash for chunk in first} & {chunk.content_hash for chunk in edited}
    assert unchanged
    assert {c.content_hash for c in first} != {c.content_hash for c in edited}


def test_the_reader_url_overrides_the_document_url() -> None:
    # github_docs fetches raw Markdown but a citation must point at the
    # rendered page.
    chunks = build_web_chunks(
        document(("A", "prose " * 40)),
        external_id="guid",
        url="https://github.com/o/r",
    )
    assert chunks[0].url == "https://github.com/o/r"

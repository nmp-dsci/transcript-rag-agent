"""Reading a rubric reviewer's reply back into verdicts nobody can forge.

The rules under test are the ones the whole slice rests on: the model supplies
identifiers and a judgement and nothing else, a failure that names no section of
the document does not count as a finding, and every rubric in a pack gets a row
whether or not the model produced one for it.
"""

from __future__ import annotations

from types import SimpleNamespace

from src.documents.models import Document, DocumentSection
from src.documents.rubric_review import (
    RubricReview,
    build_answer,
    build_references,
    format_rubrics,
    parse_pack_verdicts,
    sort_verdicts,
    verdicts_as_findings,
)


def _evidence(video_id: str = "vid1", start: float = 120.0, quote: str = "say the number"):
    return SimpleNamespace(
        video_id=video_id,
        chunk_id=f"chunk:{video_id}:3",
        quote=quote,
        model_quote=quote,
        start_seconds=start,
        quote_start_seconds=start + 4,
        channel_name="A Recruiter",
        title="How to write it",
        ratio=1.0,
        youtube_url=lambda: f"https://www.youtube.com/watch?v={video_id}&t={int(start + 2)}s",
    )


def _rubric(rubric_id: str, criterion: str = "Quantify the outcome.", evidence=None):
    return SimpleNamespace(
        rubric_id=rubric_id,
        criterion=criterion,
        check="If a bullet has no number, fail.",
        why="Recruiters skim for numbers.",
        contested=False,
        unit_id="raptor:theme:1",
        unit_kind="raptor",
        unit_title="Quantified impact",
        creators=["A Recruiter"],
        evidence=[_evidence()] if evidence is None else evidence,
    )


def _pack(*rubrics, topic: str = "resume-design"):
    return SimpleNamespace(
        topic=topic,
        name="Resume design",
        artifact="resume",
        rubrics=list(rubrics),
    )


def _document() -> Document:
    return Document(
        id="doc:1",
        url="https://example.com",
        requested_url="https://example.com",
        title="Portfolio",
        sections=[
            DocumentSection(index=index, heading=f"H{index}", text="body") for index in range(3)
        ],
    )


def test_every_rubric_gets_a_row_even_when_the_model_skips_it() -> None:
    pack = _pack(_rubric("r0101"), _rubric("r0102"), _rubric("r0103"))
    payload = {"verdicts": [{"rubric_id": "r0102", "verdict": "pass"}]}

    verdicts, outcome = parse_pack_verdicts(payload, pack, [0, 1, 2])

    assert [v.rubric_id for v in verdicts] == ["r0101", "r0102", "r0103"]
    assert [v.verdict for v in verdicts] == ["unjudged", "pass", "unjudged"]
    assert outcome.missing_rubric_ids == ["r0101", "r0103"]
    assert verdicts[0].note


def test_citation_metadata_comes_from_the_pack_not_the_model() -> None:
    """The model names a rubric; the timestamp and link are read off the pack.

    The reply below supplies a video id, a timestamp and a url of its own, all
    wrong. None of them may appear on the verdict.
    """
    pack = _pack(_rubric("r0101"))
    payload = {
        "verdicts": [
            {
                "rubric_id": "r0101",
                "verdict": "fail",
                "severity": "major",
                "sections": [2],
                "finding": "no number in the second card",
                "video_id": "WRONG",
                "start_seconds": 9999,
                "url": "https://example.com/not-a-source",
            }
        ]
    }

    verdicts, _ = parse_pack_verdicts(payload, pack, [0, 1, 2])

    assert verdicts[0].evidence[0].video_id == "vid1"
    assert verdicts[0].evidence[0].start_seconds == 120.0
    assert verdicts[0].evidence[0].url == "https://www.youtube.com/watch?v=vid1&t=122s"


def test_a_failure_naming_no_section_is_not_counted() -> None:
    """The anti-recitation rule: advice about documents in general does not score."""
    pack = _pack(_rubric("r0101"))
    payload = {
        "verdicts": [
            {"rubric_id": "r0101", "verdict": "fail", "severity": "blocker", "finding": "generic"}
        ]
    }

    verdicts, outcome = parse_pack_verdicts(payload, pack, [0, 1, 2])

    assert verdicts[0].verdict == "unjudged"
    assert verdicts[0].severity == "none"
    assert outcome.unanchored_failures == ["r0101"]
    assert "named no section" in verdicts[0].note


def test_sections_are_validated_against_the_ones_the_model_was_shown() -> None:
    """`[§4]` on a three-section selection is a section nobody was given."""
    pack = _pack(_rubric("r0101"))
    payload = {
        "verdicts": [
            {"rubric_id": "r0101", "verdict": "fail", "sections": [1, 4, "§3"], "finding": "x"}
        ]
    }

    verdicts, _ = parse_pack_verdicts(payload, pack, [0, 1, 2])

    assert verdicts[0].sections == [0, 2]


def test_unknown_and_duplicate_rubric_ids_are_dropped_and_counted() -> None:
    pack = _pack(_rubric("r0101"))
    payload = {
        "verdicts": [
            {"rubric_id": "r0101", "verdict": "pass"},
            {"rubric_id": "r0101", "verdict": "fail", "sections": [1]},
            {"rubric_id": "r9999", "verdict": "fail", "sections": [1]},
        ]
    }

    verdicts, outcome = parse_pack_verdicts(payload, pack, [0, 1, 2])

    assert [v.verdict for v in verdicts] == ["pass"]
    assert outcome.duplicate_rubric_ids == ["r0101"]
    assert outcome.unknown_rubric_ids == ["r9999"]


def test_severity_is_only_kept_on_a_failure_and_is_validated() -> None:
    pack = _pack(_rubric("r0101"), _rubric("r0102"), _rubric("r0103"))
    payload = {
        "verdicts": [
            {"rubric_id": "r0101", "verdict": "pass", "severity": "blocker"},
            {"rubric_id": "r0102", "verdict": "fail", "severity": "APOCALYPTIC", "sections": [1]},
            {"rubric_id": "r0103", "verdict": "N/A", "severity": "major"},
        ]
    }

    verdicts, _ = parse_pack_verdicts(payload, pack, [0, 1, 2])

    assert [(v.verdict, v.severity) for v in verdicts] == [
        ("pass", "none"),
        ("fail", "minor"),
        ("n-a", "none"),
    ]


def test_only_failures_become_findings_for_scoring() -> None:
    """Passes and n-a are most of a review and must score nothing.

    Otherwise the highest-recall strategy is to emit the whole pack, which is
    the attack ``KNOWN_GAP_attack2.md`` documents.
    """
    pack = _pack(_rubric("r0101"), _rubric("r0102"), _rubric("r0103"))
    payload = {
        "verdicts": [
            {"rubric_id": "r0101", "verdict": "fail", "sections": [1], "finding": "no number"},
            {"rubric_id": "r0102", "verdict": "pass"},
            {"rubric_id": "r0103", "verdict": "n-a"},
        ]
    }
    verdicts, _ = parse_pack_verdicts(payload, pack, [0, 1, 2])
    review = RubricReview(
        document_id="doc:1",
        document_url="https://example.com",
        document_kind="portfolio",
        verdicts=verdicts,
    )

    findings = verdicts_as_findings(review)

    # Qualified by pack: `r0101` exists in all four packs, and an unqualified
    # id would hand the scorer two different rules under one name.
    assert [f.id for f in findings] == ["resume-design:r0101"]
    assert findings[0].detail == "no number"
    assert findings[0].citations[0].video_id == "vid1"


def test_stats_report_the_provenance_share_deterministically() -> None:
    pack = _pack(_rubric("r0101"), _rubric("r0102"))
    verdicts, _ = parse_pack_verdicts(
        {"verdicts": [{"rubric_id": "r0101", "verdict": "fail", "sections": [1], "finding": "x"}]},
        pack,
        [0, 1, 2],
    )
    review = RubricReview(
        document_id="doc:1",
        document_url="https://example.com",
        document_kind="portfolio",
        verdicts=verdicts,
    )

    stats = review.stats

    assert stats["rubrics_total"] == 2
    assert stats["with_id_and_timestamp"] == 2
    assert stats["id_and_timestamp_share"] == 1.0
    assert stats["verdicts"]["fail"] == 1
    assert stats["verdicts"]["unjudged"] == 1


def test_references_number_only_the_sources_the_answer_cites() -> None:
    pack = _pack(
        _rubric("r0101"),
        _rubric("r0102", evidence=[_evidence("vid2", 300.0, "put a link on it")]),
    )
    verdicts, _ = parse_pack_verdicts(
        {
            "verdicts": [
                {"rubric_id": "r0101", "verdict": "fail", "sections": [1], "finding": "x"},
                {"rubric_id": "r0102", "verdict": "pass"},
            ]
        },
        pack,
        [0, 1, 2],
    )

    references = build_references(sort_verdicts(verdicts))

    assert [r["label"] for r in references] == ["[1]"]
    assert references[0]["video_id"] == "vid1"
    assert references[0]["timestamp_url"] == references[0]["source_url"]


def test_answer_cites_sections_and_sources_it_actually_has() -> None:
    pack = _pack(_rubric("r0101"))
    verdicts, outcome = parse_pack_verdicts(
        {
            "verdicts": [
                {
                    "rubric_id": "r0101",
                    "verdict": "fail",
                    "severity": "blocker",
                    "sections": [3],
                    "finding": "the third card has no number",
                }
            ]
        },
        pack,
        [0, 1, 2],
    )
    review = RubricReview(
        document_id="doc:1",
        document_url="https://example.com",
        document_kind="portfolio",
        verdicts=verdicts,
        packs=[outcome],
    )

    answer = build_answer(review)

    assert "## Blocker (1)" in answer
    assert "**r0101**" in answer
    assert "[§3]" in answer
    assert "[1]" in answer
    assert "1 fail" in answer


def test_failures_sort_worst_first() -> None:
    pack = _pack(_rubric("r0101"), _rubric("r0102"), _rubric("r0103"))
    verdicts, _ = parse_pack_verdicts(
        {
            "verdicts": [
                {"rubric_id": "r0101", "verdict": "pass"},
                {"rubric_id": "r0102", "verdict": "fail", "severity": "minor", "sections": [1]},
                {"rubric_id": "r0103", "verdict": "fail", "severity": "blocker", "sections": [1]},
            ]
        },
        pack,
        [0, 1, 2],
    )

    assert [v.rubric_id for v in sort_verdicts(verdicts)] == ["r0103", "r0102", "r0101"]


def test_prompt_lists_the_id_first_and_never_the_rationale() -> None:
    """`why` is the argument for a rule; a reviewer given it argues with the rule."""
    block = format_rubrics(_pack(_rubric("r0101")))

    assert block.startswith("r0101: Quantify the outcome.")
    assert "CHECK:" in block
    assert "Recruiters skim" not in block


def test_document_sections_are_numbered_from_one_in_the_prompt() -> None:
    """`sections: [3]` from the model means `DocumentSection.index == 2`."""
    document = _document()
    pack = _pack(_rubric("r0101"))
    verdicts, _ = parse_pack_verdicts(
        {"verdicts": [{"rubric_id": "r0101", "verdict": "fail", "sections": [3], "finding": "x"}]},
        pack,
        [section.index for section in document.sections],
    )

    assert verdicts[0].sections == [2]

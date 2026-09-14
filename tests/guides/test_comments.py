from __future__ import annotations

from pathlib import Path

import pytest

from src.guides.catalog import GuidePaths, guide_detail
from src.guides.comments import (
    Receipt,
    ReceiptError,
    add_comment,
    latest_comments,
    open_comments,
    parse_receipt,
    resolve_comments,
    write_receipt,
)


def test_comments_append_and_resolve(tmp_path: Path):
    paths = GuidePaths(tmp_path, "g")
    first = add_comment(
        paths,
        body="  Cite the numbers  per creator ",
        anchor="thesis-p2",
        section_id="thesis",
        quote="iteration budgets",
    )
    second = add_comment(paths, body="Group sources by channel", anchor="sources")
    assert (
        first["id"].startswith("c-")
        and first["status"] == "open"
        and first["body"] == "Cite the numbers per creator"
    )
    assert [c["id"] for c in open_comments(paths)] == [first["id"], second["id"]]
    with pytest.raises(ValueError):
        add_comment(paths, body="   ")

    receipt = parse_receipt(
        {
            "summary": "Two edits.",
            "items": [
                {
                    "id": first["id"],
                    "outcome": "addressed",
                    "reason": "added per-creator figures",
                    "sections": ["thesis-p2"],
                },
                {"id": second["id"], "outcome": "Rejected", "reason": "the corpus groups by theme"},
            ],
            "changed_sections": ["thesis-p2"],
        },
        expected_ids=[first["id"], second["id"]],
        version=2,
    )
    resolve_comments(paths, receipt)
    latest = latest_comments(paths)
    assert [c["status"] for c in latest] == ["addressed", "rejected"]
    assert (
        latest[0]["resolved_in_version"] == 2
        and latest[1]["reason"] == "the corpus groups by theme"
    )
    assert open_comments(paths) == []
    # The file keeps every line; the reader sees one record per id.
    assert paths.comments.read_text().count("\n") == 4
    target = write_receipt(paths, receipt)
    assert target.name == "v2.json"
    detail = guide_detail("g", tmp_path)
    assert detail is None  # no manifest yet — comments alone do not make a guide


def test_receipt_validation():
    ids = ["c-1", "c-2"]
    with pytest.raises(ReceiptError, match="missing comment ids: c-2"):
        parse_receipt(
            {"items": [{"id": "c-1", "outcome": "addressed"}]}, expected_ids=ids, version=2
        )
    with pytest.raises(ReceiptError, match="unknown comment ids: c-9"):
        parse_receipt(
            {
                "items": [
                    {"id": "c-1", "outcome": "addressed"},
                    {"id": "c-2", "outcome": "deferred"},
                    {"id": "c-9", "outcome": "rejected"},
                ]
            },
            expected_ids=ids,
            version=2,
        )
    with pytest.raises(ReceiptError, match="more than once"):
        parse_receipt(
            {
                "items": [
                    {"id": "c-1", "outcome": "addressed"},
                    {"id": "c-1", "outcome": "deferred"},
                ]
            },
            expected_ids=["c-1"],
            version=2,
        )
    with pytest.raises(ReceiptError, match="outcome must be one of"):
        parse_receipt(
            {"items": [{"id": "c-1", "outcome": "done"}]}, expected_ids=["c-1"], version=2
        )
    with pytest.raises(ReceiptError):
        parse_receipt("nope", expected_ids=[], version=1)
    ok = parse_receipt({"items": []}, expected_ids=[], version=3)
    assert isinstance(ok, Receipt) and ok.version == 3 and ok.to_dict()["items"] == []

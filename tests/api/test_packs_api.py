"""The pack reader the Experiments tab renders, over a temporary ``experts/``.

Everything served is a file the build wrote, so these tests write the files and
read them back through the same functions the routes call. The properties worth
protecting are the ones a reviewer would be misled by if they broke: a deep link
that silently points at the wrong second, a header claim recomputed from
something other than the rubrics under it, and an override that quietly re-admits
a video the pack is supposed to have excluded.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.api.packs import list_packs, pack_checks, pack_detail, set_member_override
from src.rag.packs import ExpertPack, PackStore

CATALOG: dict[str, Any] = {
    "version": 1,
    "excluded_video_ids": ["property1"],
    "exclusion_reason": "property videos are visible leakage",
    "held_out_video_ids": ["heldout1"],
    "packs": [
        {
            "topic": "resume-design",
            "name": "Resume design",
            "artifact": "resume",
            "blurb": "What a resume has to do on the page.",
            "routing_text": "how to write a technical resume",
        },
        {
            "topic": "job-search",
            "name": "Job search",
            "artifact": "job search",
            "blurb": "Everything around the resume.",
            "routing_text": "how to get interviews",
        },
    ],
}


def evidence(video: str, channel: str, *, quote: str = "the quote", ratio: float = 1.0):
    return {
        "video_id": video,
        "chunk_id": f"chunk:{video}:2",
        "chunk_index": 2,
        "start_seconds": 100.0,
        "end_seconds": 170.0,
        "quote": quote,
        "model_quote": quote,
        "quote_start_seconds": 142.0,
        "channel_name": channel,
        "title": f"{channel} video",
        # Already carries a `t=`. A deep link built by appending to this would
        # produce two, and YouTube honours the first.
        "source_url": f"https://www.youtube.com/watch?v={video}&t=51s",
        "ratio": ratio,
    }


def pack_payload(topic: str, arm: str, rubrics: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "version": 1,
        "topic": topic,
        "name": topic,
        "arm": arm,
        "artifact": "resume",
        "routing_text": "how to write a technical resume",
        "blurb": "b",
        "generated_at": "2026-08-10T00:00:00+00:00",
        "provenance": {
            "corpus_digest": "deadbeefdeadbeef",
            "chunk_count": 1372,
            "video_count": 56,
            "excluded_video_ids": ["property1"],
            "held_out_video_ids": ["heldout1"],
            "rubric_model": "deepseek-v4-flash",
        },
        "members": [
            {
                "video_id": "v1",
                "title": "One",
                "channel_name": "Channel One",
                "score": 0.73,
                "routed": True,
                "chunk_count": 27,
            }
        ],
        "units": [
            {
                "unit_id": "raptor:theme:4",
                "kind": "raptor",
                "title": "A theme",
                "summary": "s",
                "chunk_ids": ["chunk:v1:2"],
                "video_ids": ["v1"],
                "retained": 12,
                "creator_count": 3,
                "top_creator": "Channel One",
                "top_creator_share": 0.4,
            }
        ],
        "rubrics": rubrics,
        "gaps": [
            {
                "unit_id": "raptor:theme:9",
                "unit_kind": "raptor",
                "unit_title": "Unused",
                "videos": 1,
                "chunks": 8,
                "reason": "every chunk comes from one creator (Channel One)",
            }
        ],
        "stats": {"unit_budget": 6, "gaps": 1, "members_included": 1},
    }


def rubric(rubric_id: str, creators: list[tuple[str, str]], **kwargs: Any) -> dict[str, Any]:
    return {
        "rubric_id": rubric_id,
        "criterion": f"criterion {rubric_id}",
        "check": "check it",
        "why": "because",
        "contested": kwargs.get("contested", False),
        "unit_id": "raptor:theme:4",
        "unit_kind": "raptor",
        "unit_title": "A theme",
        "evidence": [
            evidence(video, channel, ratio=kwargs.get("ratio", 1.0)) for video, channel in creators
        ],
    }


@pytest.fixture
def packs_dir(tmp_path: Path) -> Path:
    root = tmp_path / "experts"
    root.mkdir()
    (root / "packs.json").write_text(json.dumps(CATALOG), encoding="utf-8")
    topic = root / "resume-design"
    topic.mkdir()
    shipped = pack_payload(
        "resume-design",
        "merged",
        [
            rubric("r0101", [("v1", "Channel One"), ("v2", "Channel Two")]),
            rubric("r0102", [("v1", "Channel One")], ratio=0.85),
            rubric("r0103", [("v1", "Channel One"), ("v3", "Channel Three")], contested=True),
        ],
    )
    (topic / "pack.json").write_text(json.dumps(shipped), encoding="utf-8")
    (topic / "merged.json").write_text(json.dumps(shipped), encoding="utf-8")
    (topic / "raptor.json").write_text(
        json.dumps(pack_payload("resume-design", "raptor", [rubric("r0101", [("v1", "A")])])),
        encoding="utf-8",
    )
    return root


def test_list_packs_shows_a_declared_pack_that_was_never_built(packs_dir: Path):
    """Hiding it would let the tab look complete while a pack was missing."""
    rows = {row["topic"]: row for row in list_packs(packs_dir)["packs"]}
    assert rows["resume-design"]["built"] is True
    assert rows["job-search"]["built"] is False
    assert "checks" not in rows["job-search"]


def test_list_packs_carries_the_exclusion_the_router_applies(packs_dir: Path):
    listed = list_packs(packs_dir)
    assert listed["excluded_video_ids"] == ["property1"]
    assert listed["held_out_video_ids"] == ["heldout1"]


def test_list_packs_of_a_missing_directory_is_empty_not_an_error(tmp_path: Path):
    listed = list_packs(tmp_path / "nope")
    assert listed["packs"] == []
    assert "build-packs" in listed["build_command"]


def test_checks_count_creators_not_videos(packs_dir: Path):
    """Two of three rules cite two creators; one cites one channel only."""
    detail = pack_detail("resume-design", packs_dir)
    assert detail is not None
    assert detail["checks"]["multi_creator_rubrics"] == 2
    assert detail["checks"]["multi_creator_share"] == pytest.approx(2 / 3, abs=1e-4)


def test_checks_are_recomputed_from_the_rubrics_not_read_from_stats(packs_dir: Path):
    """A header that copies stored stats can disagree with the rows under it."""
    path = packs_dir / "resume-design" / "pack.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["stats"]["rubrics"] = 99
    payload["stats"]["multi_creator_share"] = 1.0
    path.write_text(json.dumps(payload), encoding="utf-8")
    detail = pack_detail("resume-design", packs_dir)
    assert detail is not None
    assert detail["checks"]["rubrics"] == 3


def test_quote_resolution_counts_the_evidence_that_snapped_cleanly(packs_dir: Path):
    detail = pack_detail("resume-design", packs_dir)
    assert detail is not None
    checks = detail["checks"]
    # Five quotes; the one rubric built at ratio 0.85 contributes the only miss.
    assert checks["evidence_total"] == 5
    assert checks["evidence_resolved"] == 4
    assert checks["quote_resolution"] == pytest.approx(0.8)


def test_every_evidence_row_carries_a_single_timestamp_deep_link(packs_dir: Path):
    """The stored source_url already has a `t=`; the link must not inherit it."""
    detail = pack_detail("resume-design", packs_dir)
    assert detail is not None
    for row in detail["rubrics"]:
        for item in row["evidence"]:
            assert item["url"].count("t=") == 1
            assert item["url"] == (f"https://www.youtube.com/watch?v={item['video_id']}&t=140s")


def test_detail_exposes_the_contested_badge_and_the_unit_it_came_from(packs_dir: Path):
    detail = pack_detail("resume-design", packs_dir)
    assert detail is not None
    contested = [row for row in detail["rubrics"] if row["contested"]]
    assert [row["rubric_id"] for row in contested] == ["r0103"]
    assert detail["rubrics"][0]["unit_id"] == "raptor:theme:4"


def test_detail_lists_every_built_arm_for_the_comparison(packs_dir: Path):
    detail = pack_detail("resume-design", packs_dir)
    assert detail is not None
    assert sorted(detail["arms"]) == ["merged", "raptor"]
    assert detail["arms"]["merged"]["checks"]["rubrics"] == 3


def test_detail_of_a_declared_but_unbuilt_pack_still_answers(packs_dir: Path):
    detail = pack_detail("job-search", packs_dir)
    assert detail is not None
    assert detail["built"] is False
    assert "rubrics" not in detail


def test_detail_of_an_undeclared_topic_is_none(packs_dir: Path):
    assert pack_detail("tax-advice", packs_dir) is None


def test_ablation_is_only_attached_to_the_topic_it_scored(packs_dir: Path):
    PackStore(packs_dir).save_ablation(
        {
            "kind": "pack-ablation",
            "topic": "resume-design",
            "metrics": ["criteria_recall"],
            "baseline": "raptor",
            "cells": [
                {
                    "setup": "merged",
                    "scores": {"criteria_recall": 0.21},
                    "score_spread": {"criteria_recall_min": 0.15, "criteria_recall_max": 0.26},
                    "rubrics": 17,
                }
            ],
            "verdicts": {"criteria_recall": {"leader": "merged", "decisive": False}},
        }
    )
    resume = pack_detail("resume-design", packs_dir)
    other = pack_detail("job-search", packs_dir)
    assert resume is not None and other is not None
    assert resume["ablation"]["cells"][0]["arm"] == "merged"
    assert resume["ablation"]["cells"][0]["score_spread"]["criteria_recall_max"] == 0.26
    assert other["ablation"] is None


def test_override_is_recorded_in_the_manifest_and_applies_at_the_next_build(
    packs_dir: Path,
):
    result = set_member_override("resume-design", "v9", True, packs_dir)
    assert result["overrides"] == {"v9": True}
    assert result["applies"] == "at the next build"
    # Recorded where a rebuild reads it back, not in the rendered pack.
    manifest = json.loads(
        (packs_dir / "resume-design" / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["overrides"] == {"v9": True}
    assert pack_detail("resume-design", packs_dir)["overrides"] == {"v9": True}


def test_override_clears_back_to_the_router(packs_dir: Path):
    set_member_override("resume-design", "v9", True, packs_dir)
    result = set_member_override("resume-design", "v9", None, packs_dir)
    assert result["overrides"] == {}


def test_an_excluded_video_cannot_be_pinned_back_into_a_pack(packs_dir: Path):
    """Otherwise the panel stores a decision the build will silently refuse."""
    with pytest.raises(ValueError, match="excluded from every pack"):
        set_member_override("resume-design", "property1", True, packs_dir)


def test_a_held_out_video_cannot_be_pinned_in_either(packs_dir: Path):
    with pytest.raises(ValueError, match="excluded from every pack"):
        set_member_override("resume-design", "heldout1", True, packs_dir)


def test_pack_checks_flag_a_citation_from_an_excluded_video(packs_dir: Path):
    """The leak check is arithmetic over ids, independent of the router."""
    leaked = ExpertPack.model_validate(
        pack_payload(
            "resume-design",
            "merged",
            [rubric("r0101", [("property1", "Property Channel"), ("v2", "Other")])],
        )
    )
    assert pack_checks(leaked)["excluded_video_citations"] == 1


def test_membership_marks_a_video_that_reached_no_source_unit(packs_dir: Path):
    """Routed in is not the same as having a say, and the table must say so.

    Three job-search videos were ingested after the last ``index-themes``: they
    route into the pack and are cited by nothing, because the theme layer and
    the entity graph are derived state built at an earlier chunk count. Shown as
    ordinary members, the membership count would stand in for coverage.
    """
    path = packs_dir / "resume-design" / "pack.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["members"].append(
        {
            "video_id": "late1",
            "title": "Ingested after the theme layer was built",
            "channel_name": "Later Channel",
            "score": 0.51,
            "routed": True,
            "chunk_count": 16,
        }
    )
    path.write_text(json.dumps(payload), encoding="utf-8")
    detail = pack_detail("resume-design", packs_dir)
    assert detail is not None
    rows = {row["video_id"]: row for row in detail["members"]}
    assert rows["late1"]["in_units"] is False
    assert rows["late1"]["cited"] is False
    # v1 is in the stored unit and quoted by every rubric.
    assert rows["v1"]["in_units"] is True
    assert rows["v1"]["cited"] is True


def test_a_corrupt_pack_file_reads_as_unbuilt_rather_than_500ing(packs_dir: Path):
    """A half-written pack must not take the whole tab down."""
    (packs_dir / "resume-design" / "pack.json").write_text("{not json", encoding="utf-8")
    rows = {row["topic"]: row for row in list_packs(packs_dir)["packs"]}
    assert rows["resume-design"]["built"] is False
    detail = pack_detail("resume-design", packs_dir)
    assert detail is not None and detail["built"] is False
    # The arms are separate files and are unaffected.
    assert "merged" in detail["arms"]


# -- staleness ------------------------------------------------------------
#
# V5's acceptance gate failed on one check: "a reader is told when the pack is
# behind the corpus they are browsing". The card printed the build's own counts
# in the present tense while the header above it showed a corpus 11 videos
# larger. These pin the comparison that replaced it.


def _live(*videos: tuple[str, int]) -> list[dict[str, Any]]:
    return [{"video_id": vid, "chunk_count": n} for vid, n in videos]


def test_a_pack_behind_the_live_corpus_reports_how_far_behind(packs_dir: Path):
    detail = pack_detail(
        "resume-design", packs_dir, live_videos=_live(*[(f"v{i}", 10) for i in range(67)])
    )
    assert detail is not None
    stale = detail["staleness"]
    assert stale["current"] is False
    assert stale["build_videos"] == 56 and stale["build_chunks"] == 1372
    assert stale["live_videos"] == 67 and stale["live_chunks"] == 670
    assert stale["behind_videos"] == 11
    assert stale["build_digest"] == "deadbeefdeadbeef"
    assert stale["live_digest"] != stale["build_digest"]


def test_a_rechunk_that_shrinks_the_corpus_reports_a_negative_gap(packs_dir: Path):
    """Signed, because "behind by -4 chunks" is a lie in the other direction."""
    detail = pack_detail(
        "resume-design", packs_dir, live_videos=_live(*[(f"v{i}", 10) for i in range(50)])
    )
    assert detail is not None
    stale = detail["staleness"]
    assert stale["behind_videos"] == -6
    assert stale["behind_chunks"] == 500 - 1372
    assert stale["current"] is False


def test_staleness_is_absent_rather_than_guessed_when_no_corpus_is_passed(packs_dir: Path):
    """Absent and up-to-date must not render the same."""
    detail = pack_detail("resume-design", packs_dir)
    assert detail is not None
    assert "staleness" not in detail


def test_a_pack_matching_the_live_corpus_reports_current(packs_dir: Path):
    """The digest must agree with the one the build recorded, not merely the counts."""
    from src.evals.matrix_cache import corpus_digest

    live = _live(("a", 3), ("b", 4))
    store = PackStore(packs_dir)
    pack = store.load_pack("resume-design")
    assert pack is not None
    pack.provenance.corpus_digest = corpus_digest(["a", "b"], 7)
    pack.provenance.video_count = 2
    pack.provenance.chunk_count = 7
    store.save_pack(pack, shipped=True)

    detail = pack_detail("resume-design", packs_dir, live_videos=live)
    assert detail is not None
    assert detail["staleness"]["current"] is True
    assert detail["staleness"]["behind_videos"] == 0

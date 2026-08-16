"""Unit assembly, quote reconciliation and the pack's own gate arithmetic.

The bugs these lock down are the ones that shipped: three declared packs built
themselves empty because the creator-diversity floor was handed no chunk
records and profiled every unit as zero creators, and the three arms of the
comparison were sized from the raw pools rather than the eligible ones, so one
arm would have spent twice the LLM calls of another and the difference would
have been reported as a result about abstraction.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Sequence

import pytest

from src.rag.pack_build import shared_budget
from src.rag.packs import (
    QUOTE_MATCH_RATIO,
    PackGap,
    PackMember,
    Rubric,
    RubricEvidence,
    SourceUnit,
    apply_overrides,
    creator_profile,
    dedupe_rubrics,
    merge_units,
    pack_statistics,
    raptor_units,
    reconcile_evidence,
    select_units,
    snap_quote,
    unit_excerpts,
)


def record(
    chunk_id: str,
    video_id: str,
    channel: str,
    *,
    text: str = "some transcript words",
    index: int = 0,
    start: float = 0.0,
    end: float = 10.0,
) -> dict[str, Any]:
    return {
        "chunk_id": chunk_id,
        "video_id": video_id,
        "channel_name": channel,
        "title": f"{channel} video",
        "source_url": f"https://www.youtube.com/watch?v={video_id}",
        "text": text,
        "chunk_index": index,
        "start_seconds": start,
        "end_seconds": end,
    }


def theme(theme_id: str, members: Sequence[tuple[str, str]]) -> Any:
    """A stand-in for a ThemeStore theme: ids and member (chunk, video) pairs."""
    return SimpleNamespace(
        theme_id=theme_id,
        title=f"title {theme_id}",
        summary=f"summary {theme_id}",
        members=[SimpleNamespace(chunk_id=chunk, video_id=video) for chunk, video in members],
    )


def unit(unit_id: str, kind: str, *, retained: int = 10, eligible: bool = True) -> SourceUnit:
    return SourceUnit(
        unit_id=unit_id,
        kind=kind,
        title=unit_id,
        summary="",
        chunk_ids=[f"{unit_id}:{n}" for n in range(retained)],
        video_ids=["v1", "v2"],
        retained=retained,
        creator_count=2 if eligible else 1,
        eligible=eligible,
        reject_reason="" if eligible else "one creator",
    )


# ─── creator diversity ───────────────────────────────────────────────────────


def test_creator_profile_counts_chunks_not_videos():
    """One channel with three videos in a unit is still one voice."""
    records = {
        "a": record("a", "v1", "Loud Channel"),
        "b": record("b", "v2", "Loud Channel"),
        "c": record("c", "v3", "Loud Channel"),
        "d": record("d", "v4", "Quiet Channel"),
    }
    creators, top, share = creator_profile(["a", "b", "c", "d"], records)
    assert creators == 2
    assert top == "Loud Channel"
    assert share == 0.75


def test_raptor_units_without_records_is_a_type_error_not_an_empty_pack():
    """The failure that built three packs empty must now be loud.

    With no lookup every unit profiles as zero creators, trips the floor, and
    the pack comes out empty with a gap log blaming the corpus. A caller that
    forgets the records has to find out immediately.
    """
    with pytest.raises(TypeError):
        raptor_units([theme("theme:0", [("a", "v1")])], ["v1"])  # type: ignore[call-arg]


def test_raptor_units_are_eligible_when_records_are_supplied():
    records = {
        f"c{n}": record(f"c{n}", "v1" if n % 2 else "v2", "A" if n % 2 else "B") for n in range(6)
    }
    units = raptor_units(
        [theme("theme:0", [(chunk, records[chunk]["video_id"]) for chunk in records])],
        ["v1", "v2"],
        records,
    )
    assert len(units) == 1
    assert units[0].eligible
    assert units[0].creator_count == 2
    assert units[0].reject_reason == ""


def test_raptor_unit_dominated_by_one_channel_is_kept_but_marked():
    """A theme that is one channel with visitors is logged, not deleted.

    The corpus's most contrarian résumé voice sits in a theme like this. It
    cannot honestly become a shared rule and it must not silently vanish
    either, so the unit survives carrying its reason.
    """
    records = {f"c{n}": record(f"c{n}", "v1", "Loud") for n in range(9)}
    records["c9"] = record("c9", "v2", "Quiet")
    units = raptor_units(
        [theme("theme:0", [(chunk, records[chunk]["video_id"]) for chunk in records])],
        ["v1", "v2"],
        records,
    )
    assert len(units) == 1
    assert not units[0].eligible
    assert "Loud" in units[0].reject_reason


def test_raptor_unit_from_a_single_video_is_rejected_by_the_video_floor():
    records = {f"c{n}": record(f"c{n}", "v1", f"Creator {n}") for n in range(6)}
    units = raptor_units([theme("theme:0", [(chunk, "v1") for chunk in records])], ["v1"], records)
    assert not units[0].eligible
    assert "one creator's position" in units[0].reject_reason


# ─── arm sizing ──────────────────────────────────────────────────────────────


def test_shared_budget_counts_eligible_units_only():
    """Otherwise one arm quietly spends twice the LLM calls of another.

    app-architecture has fourteen themes and four that clear the creator floor;
    counting the raw pool would hand the communities arm eight calls against
    the raptor arm's four and then report the difference as an abstraction
    result.
    """
    raptor = [unit(f"raptor:{n}", "raptor", eligible=n < 4) for n in range(14)]
    communities = [unit(f"community:{n}", "communities") for n in range(20)]
    assert shared_budget(8, raptor, communities) == 4


def test_shared_budget_caps_at_the_requested_budget():
    raptor = [unit(f"raptor:{n}", "raptor") for n in range(12)]
    communities = [unit(f"community:{n}", "communities") for n in range(12)]
    assert shared_budget(6, raptor, communities) == 6


def test_shared_budget_survives_a_pack_with_nothing_to_say():
    assert shared_budget(8, [], []) == 8


def test_merge_alternates_and_never_exceeds_the_shared_budget():
    raptor = [unit(f"raptor:{n}", "raptor") for n in range(4)]
    communities = [unit(f"community:{n}", "communities") for n in range(4)]
    merged = merge_units(raptor, communities, 4)
    assert [item.kind for item in merged] == [
        "raptor",
        "communities",
        "raptor",
        "communities",
    ]
    assert len(merged) == 4


def test_select_units_never_returns_an_ineligible_unit():
    raptor = [unit("raptor:0", "raptor", eligible=False), unit("raptor:1", "raptor")]
    communities = [unit("community:0", "communities")]
    for arm in ("raptor", "communities", "merged"):
        assert all(item.eligible for item in select_units(arm, raptor, communities, 4))


def test_select_units_rejects_an_unknown_arm():
    with pytest.raises(ValueError, match="Unknown pack arm"):
        select_units("bm25", [], [], 4)


# ─── excerpts ────────────────────────────────────────────────────────────────


def test_unit_excerpts_round_robin_across_videos_before_going_deep():
    """One creator's long stretch must not fill the whole prompt."""
    records = {}
    members: list[tuple[str, str]] = []
    for n in range(10):
        records[f"a{n}"] = record(f"a{n}", "v1", "Loud", index=n)
        members.append((f"a{n}", "v1"))
    for n in range(2):
        records[f"b{n}"] = record(f"b{n}", "v2", "Quiet", index=n)
        members.append((f"b{n}", "v2"))
    source = SourceUnit(
        unit_id="raptor:0",
        kind="raptor",
        title="t",
        summary="",
        chunk_ids=[chunk for chunk, _ in members],
        video_ids=["v1", "v2"],
        retained=12,
    )
    picked = unit_excerpts(source, records, 4)
    assert [item["video_id"] for item in picked] == ["v1", "v2", "v1", "v2"]


# ─── quotes ──────────────────────────────────────────────────────────────────


def test_snap_quote_returns_the_transcript_span_not_the_model_wording():
    """A model normalises "gonna"; rendering its version misquotes the creator."""
    text = "You are gonna want to quantify the impact of every bullet point."
    span, ratio, offset = snap_quote(
        "You are going to want to quantify the impact of every bullet point", text
    )
    assert "gonna" in span and "going to" not in span
    assert ratio > QUOTE_MATCH_RATIO
    assert offset == 0


def test_snap_quote_paraphrase_falls_under_the_acceptance_floor():
    text = "You are gonna want to quantify the impact of every bullet point."
    _, ratio, _ = snap_quote("You are going to want to quantify the impact", text)
    assert ratio < QUOTE_MATCH_RATIO


def test_snap_quote_rejects_a_quote_stitched_from_two_places():
    text = "Alpha beta gamma. " + "filler " * 40 + "delta epsilon zeta."
    _, ratio, _ = snap_quote("Alpha beta gamma delta epsilon zeta", text)
    assert ratio == 0.0


def test_reconcile_evidence_drops_a_quote_that_is_not_in_the_excerpt_it_names():
    excerpts = [record("chunk:v1:0", "v1", "A", text="Tailor the resume to the posting.")]
    rows = [
        {"excerpt": 1, "quote": "Tailor the resume to the posting"},
        {"excerpt": 1, "quote": "Always include a photograph of yourself"},
    ]
    evidence = reconcile_evidence(rows, excerpts)
    assert len(evidence) == 1
    assert evidence[0].quote == "Tailor the resume to the posting"


def test_reconcile_evidence_drops_an_excerpt_number_that_was_never_shown():
    excerpts = [record("chunk:v1:0", "v1", "A", text="Tailor the resume.")]
    assert reconcile_evidence([{"excerpt": 7, "quote": "Tailor the resume"}], excerpts) == []


def test_reconcile_evidence_interpolates_the_quote_offset_within_the_chunk():
    text = "First sentence here. " * 4 + "The quantified bullet is what matters."
    excerpts = [record("chunk:v1:0", "v1", "A", text=text, start=100.0, end=200.0)]
    evidence = reconcile_evidence(
        [{"excerpt": 1, "quote": "The quantified bullet is what matters"}], excerpts
    )
    assert evidence[0].start_seconds == 100.0
    assert 160.0 < evidence[0].quote_start_seconds < 200.0


def test_youtube_url_is_built_from_the_video_id_not_the_stored_source_url():
    """Eight corpus urls already carry a ``t=``; appending a second one loses.

    YouTube honours the first ``t`` in a url, so a link built by appending
    lands wherever the stored url already pointed — about twenty seconds early
    — while looking exactly like a working deep link.
    """
    evidence = RubricEvidence(
        video_id="abc123",
        chunk_id="chunk:abc123:4",
        chunk_index=4,
        start_seconds=100.0,
        end_seconds=170.0,
        quote="q",
        model_quote="q",
        quote_start_seconds=142.0,
        source_url="https://www.youtube.com/watch?v=abc123&t=51s&pp=ygUS",
    )
    url = evidence.youtube_url()
    assert url == "https://www.youtube.com/watch?v=abc123&t=140s"
    assert url.count("t=") == 1


# ─── statistics ──────────────────────────────────────────────────────────────


def rubric(rubric_id: str, creators: Sequence[tuple[str, str]], *, contested: bool = False):
    return Rubric(
        rubric_id=rubric_id,
        criterion=f"criterion {rubric_id}",
        check="check it",
        contested=contested,
        unit_id="raptor:0",
        unit_kind="raptor",
        unit_title="t",
        evidence=[
            RubricEvidence(
                video_id=video,
                chunk_id=f"chunk:{video}:0",
                chunk_index=0,
                start_seconds=0.0,
                end_seconds=10.0,
                quote="q",
                model_quote="q",
                quote_start_seconds=0.0,
                channel_name=channel,
            )
            for video, channel in creators
        ],
    )


def test_multi_creator_share_counts_creators_not_videos():
    """Two videos from one channel is one voice, and must not count as two.

    The corpus audit found a theme spanning four videos where 145 of 146 chunks
    came from a single podcast. A share computed over video ids would have
    called that corroborated.
    """
    one_channel_two_videos = rubric("r1", [("v1", "Same Channel"), ("v2", "Same Channel")])
    two_channels = rubric("r2", [("v3", "One"), ("v4", "Another")])
    stats = pack_statistics([one_channel_two_videos, two_channels], [], [], [])
    assert stats["multi_creator_rubrics"] == 1
    assert stats["multi_creator_share"] == 0.5


def test_statistics_count_citations_from_an_excluded_video():
    """The leak check is independent of the mechanism that prevents the leak."""
    leaked = rubric("r1", [("blocked", "Property Channel"), ("v2", "Other")])
    stats = pack_statistics([leaked], [], [], [], ["blocked"])
    assert stats["excluded_video_citations"] == 1


def test_statistics_of_an_empty_pack_do_not_divide_by_zero():
    stats = pack_statistics([], [], [], [])
    assert stats["multi_creator_share"] == 0.0
    assert stats["rubrics"] == 0


# ─── membership ──────────────────────────────────────────────────────────────


def test_overrides_can_pin_a_video_out_and_hand_one_in():
    routed = [
        PackMember(video_id="v1", score=0.7, routed=True, chunk_count=10),
        PackMember(video_id="v2", score=0.4, routed=True, chunk_count=5),
    ]
    catalogue = {"v3": {"title": "T", "channel_name": "C", "chunk_count": 3}}
    members = apply_overrides(routed, {"v2": False, "v3": True}, catalogue)
    by_id = {member.video_id: member for member in members}
    assert by_id["v1"].included
    assert not by_id["v2"].included
    assert by_id["v3"].included and not by_id["v3"].routed


# ─── dedupe ──────────────────────────────────────────────────────────────────


def test_dedupe_keeps_the_first_statement_of_a_repeated_rule():
    first = rubric("r1", [("v1", "A")])
    restated = rubric("r2", [("v2", "B")])
    distinct = rubric("r3", [("v3", "C")])

    def embed(texts: Sequence[str]) -> list[list[float]]:
        vectors = {
            first.criterion: [1.0, 0.0],
            restated.criterion: [1.0, 0.0],
            distinct.criterion: [0.0, 1.0],
        }
        return [vectors[text] for text in texts]

    kept, dropped = dedupe_rubrics([first, restated, distinct], embed)
    assert [item.rubric_id for item in kept] == ["r1", "r3"]
    assert [item.rubric_id for item in dropped] == ["r2"]


# ─── gaps ────────────────────────────────────────────────────────────────────


def test_gap_rows_carry_the_creator_columns_the_builder_passes():
    """They were passed and silently dropped for want of the fields.

    ``videos`` is the number a coverage claim overstates with, so a rejected
    unit has to say how many *creators* were behind it and how loud the loudest
    was — in data, not only in prose inside ``reason``.
    """
    gap = PackGap(
        unit_id="raptor:theme:5",
        unit_kind="raptor",
        unit_title="Modular Monoliths Beat the Microservices Mess",
        videos=4,
        chunks=76,
        creator_count=3,
        top_creator="NDC Conferences",
        top_creator_share=0.87,
        reason="87% of the theme's chunks are NDC Conferences",
    )
    assert gap.creator_count == 3
    assert gap.top_creator == "NDC Conferences"
    assert gap.top_creator_share == 0.87

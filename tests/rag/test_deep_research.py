"""The deep-research loop's arithmetic, its causal chain, and its dedupe guard.

Three things are worth locking down here, and they are the three a reader would
otherwise have to take on trust.

**Round two is caused by the critic.** Every probe carries the gap id that
produced it and every rubric carries its probe's unit id, so a round-two rule
can be traced back to a named gap. A test that only checked "the pack grew"
would pass for a loop that ran a second round of the planner's own leftovers.

**Round two cannot bank round one's coverage.** The publisher is handed the
previous round's surviving rules first and dedupes the whole list, so a gap
answered with a restatement drops out. That is the structural refusal of the
padding attack ``criteria_recall`` is open to (``src/evals/KNOWN_GAP_attack2.md``).

**The one-shot control is nested and equally funded.** ``deep-oneshot`` and
``deep-r2`` must spend the same executor budget on the same opening probes, or
the comparison between them is about spend.
"""

from __future__ import annotations

from typing import Any, Sequence

import pytest

from src.rag.deep_research import (
    GapCritic,
    ResearchGap,
    ResearchPlanner,
    ResearchProbe,
    _scoring_corpus,
    gap_closure,
    gap_probes,
    parse_gaps,
    parse_probes,
    probe_unit,
    publish,
    restatement_audit,
    rubric_diff,
    score_rows,
)
from src.rag.packs import (
    ExpertPack,
    PackCatalog,
    PackDeclaration,
    PackMember,
    Rubric,
    RubricEvidence,
    SourceUnit,
)


# ─── Fixtures ────────────────────────────────────────────────────────────────


def record(chunk_id: str, video_id: str, channel: str) -> dict[str, Any]:
    return {
        "chunk_id": chunk_id,
        "video_id": video_id,
        "channel_name": channel,
        "title": f"{channel} video",
        "text": "some transcript words about resumes",
        "chunk_index": 0,
        "start_seconds": 0.0,
        "end_seconds": 10.0,
    }


def probe(probe_id: str, *, origin: str = "plan", rank: int = 1) -> ResearchProbe:
    return ResearchProbe(
        probe_id=probe_id,
        facet=f"facet {probe_id}",
        question=f"question {probe_id}",
        origin=origin,
        rank=rank,
    )


def rubric(rubric_id: str, criterion: str, unit_id: str) -> Rubric:
    return Rubric(
        rubric_id=rubric_id,
        criterion=criterion,
        check="count them",
        unit_id=unit_id,
        unit_kind="probe",
        unit_title=unit_id,
        evidence=[
            RubricEvidence(
                video_id="v1",
                chunk_id=f"{unit_id}:c1",
                chunk_index=0,
                start_seconds=0.0,
                end_seconds=10.0,
                quote="some transcript words",
                model_quote="some transcript words",
                quote_start_seconds=0.0,
                channel_name="Alpha",
                ratio=1.0,
            )
        ],
    )


class FakeWorkspace:
    """Only the three things :func:`publish` touches: embed, catalog, provenance."""

    def __init__(self, vectors: dict[str, list[float]] | None = None) -> None:
        self.vectors = vectors or {}
        self.catalog = PackCatalog(excluded_video_ids=["blocked"])

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        # One-hot per distinct string: identical criteria embed identically
        # (cosine 1.0) and different ones are orthogonal (cosine 0.0), so dedupe
        # fires on exact repeats and on nothing else. A fake that let two
        # unrelated criteria score above the threshold would make every test
        # below pass for the wrong reason.
        distinct = sorted(set(texts))
        return [[1.0 if name == text else 0.0 for name in distinct] for text in texts]

    def provenance(self, notes: Sequence[str] = ()) -> Any:
        from src.rag.packs import PackProvenance

        return PackProvenance(notes=list(notes))


DECLARATION = PackDeclaration(
    topic="resume-design", name="Resume design", routing_text="resumes", artifact="resume"
)


# ─── Planner and critic parsing ──────────────────────────────────────────────


def test_parse_probes_numbers_in_plan_order_and_truncates() -> None:
    """The planner's order is the priority order, so ids follow it exactly."""
    probes = parse_probes(
        [
            {"facet": "length", "question": "how long should a resume be", "why": "a"},
            {"facet": "bullets", "question": "what goes in a bullet", "why": "b"},
            {"facet": "third", "question": "third question", "why": "c"},
        ],
        2,
    )
    assert [p.probe_id for p in probes] == ["p01", "p02"]
    assert [p.rank for p in probes] == [1, 2]
    assert probes[0].facet == "length"
    assert all(p.origin == "plan" for p in probes)


def test_parse_probes_drops_rows_with_no_question() -> None:
    """The question is the executor's input; an empty one would retrieve nothing."""
    probes = parse_probes(
        [{"facet": "length", "question": "   "}, {"facet": "bullets", "question": "ok"}], 5
    )
    assert [p.question for p in probes] == ["ok"]


def test_parse_gaps_requires_both_the_hole_and_where_to_look() -> None:
    """A gap with no probe cannot cause a round; it is an opinion, not a task."""
    gaps = parse_gaps(
        [
            {"missing": "nothing says what the top section holds", "probe": "resume header"},
            {"missing": "vibes", "probe": ""},
            {"missing": "", "probe": "orphan"},
        ],
        5,
    )
    assert [g.gap_id for g in gaps] == ["g01"]
    assert gaps[0].probe == "resume header"


def test_gap_probes_carry_their_cause_and_cannot_collide_with_round_one() -> None:
    """The causal chain the report renders: probe → gap, rubric → probe."""
    probes = gap_probes(
        [
            ResearchGap(gap_id="g01", missing="no rule about the header", probe="resume header"),
            ResearchGap(gap_id="g02", missing="no rule about dates", probe="employment dates"),
        ]
    )
    assert [p.origin for p in probes] == ["gap:g01", "gap:g02"]
    assert [p.probe_id for p in probes] == ["p51", "p52"]
    # Ranks feed the rubric id, so they must not overlap the plan's 1..N.
    assert min(p.rank for p in probes) > 50
    assert [p.question for p in probes] == ["resume header", "employment dates"]


def test_planner_and_critic_send_the_artifact_into_their_prompts() -> None:
    """A planner told to decompose "document" would plan for the wrong thing."""
    seen: list[str] = []

    class Recorder:
        def invoke(self, messages: list) -> object:
            seen.append(str(messages[0].content))
            return type("R", (), {"content": '{"probes":[],"gaps":[]}'})()

    ResearchPlanner(Recorder(), "m").plan(DECLARATION, ["a title"], count=4)
    GapCritic(Recorder(), "m").critique(DECLARATION, [], [], count=3)
    assert all("resume" in prompt for prompt in seen)
    assert "4" in seen[0] and "3" in seen[1]


# ─── Executor retrieval ──────────────────────────────────────────────────────


def _vectors() -> tuple[dict[str, list[float]], dict[str, dict[str, Any]]]:
    records = {
        "c1": record("c1", "v1", "Alpha"),
        "c2": record("c2", "v2", "Beta"),
        "c3": record("c3", "v3", "Gamma"),
        "c4": record("c4", "blocked", "Property"),
    }
    vectors = {
        "c1": [1.0, 0.0],
        "c2": [0.9, 0.1],
        "c3": [0.1, 1.0],
        "c4": [1.0, 0.0],
    }
    return vectors, records


def test_probe_unit_retrieves_only_the_packs_member_videos() -> None:
    """Membership is the exclusion mechanism; a blocked video must be unreachable
    even when its chunk is the single best match for the probe."""
    vectors, records = _vectors()
    unit = probe_unit(probe("p01"), [1.0, 0.0], vectors, records, ["v1", "v2", "v3"], pool=10)
    assert "c4" not in unit.chunk_ids
    assert unit.video_ids == ["v1", "v2", "v3"]


def test_probe_unit_ranks_by_similarity_and_breaks_ties_on_chunk_id() -> None:
    """The same probe over the same corpus must retrieve the same passages in the
    same order, or two runs of the loop are not comparable artifacts."""
    vectors, records = _vectors()
    unit = probe_unit(probe("p01"), [1.0, 0.0], vectors, records, ["v1", "v2", "v3"], pool=2)
    assert unit.chunk_ids == ["c1", "c2"]


def test_probe_unit_applies_the_same_creator_floor_as_the_cluster_arms() -> None:
    """One creator's passages cannot become a shared rule — the same rule
    ``raptor_units`` applies, so an executor call is not a way around it."""
    records = {"c1": record("c1", "v1", "Alpha"), "c2": record("c2", "v1", "Alpha")}
    vectors = {"c1": [1.0, 0.0], "c2": [0.9, 0.0]}
    unit = probe_unit(probe("p01"), [1.0, 0.0], vectors, records, ["v1"], pool=10)
    assert unit.eligible is False
    assert "one video" in unit.reject_reason


def test_probe_unit_with_nothing_retrievable_says_so_rather_than_looking_empty() -> None:
    vectors, records = _vectors()
    unit = probe_unit(probe("p01"), [1.0, 0.0], vectors, records, [], pool=10)
    assert unit.chunk_ids == []
    assert unit.eligible is False
    assert "no chunk" in unit.reject_reason


# ─── Publisher ───────────────────────────────────────────────────────────────


def _outcome(unit: SourceUnit, probe_id: str) -> Any:
    from src.rag.deep_research import ProbeOutcome

    return ProbeOutcome(
        probe=probe(probe_id),
        unit_id=unit.unit_id,
        chunks=len(unit.chunk_ids),
        videos=len(unit.video_ids),
        eligible=unit.eligible,
        reject_reason=unit.reject_reason,
        # What run_probes records: an ineligible probe's reason is why the call
        # was never spent, and publish must repeat that rather than blame dedupe.
        reason="" if unit.eligible else unit.reject_reason,
        spent_call=unit.eligible,
    )


def _unit(unit_id: str, *, eligible: bool = True) -> SourceUnit:
    return SourceUnit(
        unit_id=unit_id,
        kind="probe",
        title=unit_id,
        chunk_ids=[f"{unit_id}:c1"],
        video_ids=["v1", "v2"],
        retained=1,
        creator_count=2,
        eligible=eligible,
        reject_reason="" if eligible else "one creator",
    )


def test_round_two_cannot_bank_round_ones_rules_as_new_coverage() -> None:
    """The padding refusal. A gap answered with a restatement of a rule the pack
    already had adds nothing, and the drop is what says the critic found no new
    ground there."""
    workspace = FakeWorkspace()
    unit_one, unit_two = _unit("probe:p01"), _unit("probe:p51")
    kept_r1 = [rubric("r0101", "Every bullet carries a number", "probe:p01")]
    candidates = [
        *kept_r1,
        rubric("r5101", "Every bullet carries a number", "probe:p51"),
        rubric("r5102", "The header names one target role", "probe:p51"),
    ]
    pack, dropped = publish(
        workspace,
        DECLARATION,
        "deep-r2",
        members=[PackMember(video_id="v1")],
        units=[unit_one, unit_two],
        candidates=candidates,
        outcomes=[_outcome(unit_one, "p01"), _outcome(unit_two, "p51")],
    )
    assert [r.rubric_id for r in pack.rubrics] == ["r0101", "r5102"]
    assert [r.rubric_id for r in dropped] == ["r5101"]
    # The inherited rule is never the one dropped: it is passed first and dedupe
    # keeps the earlier statement.
    assert "r0101" in {r.rubric_id for r in pack.rubrics}


def test_publish_logs_a_probe_that_was_never_spent_rather_than_dropping_it() -> None:
    """ "We chose not to look" and "we looked and found nothing" are different
    admissions, and a coverage claim that cannot tell them apart is not one."""
    workspace = FakeWorkspace()
    good, skipped = _unit("probe:p01"), _unit("probe:p02", eligible=False)
    pack, _ = publish(
        workspace,
        DECLARATION,
        "deep-r1",
        members=[PackMember(video_id="v1")],
        units=[good, skipped],
        candidates=[rubric("r0101", "a rule", "probe:p01")],
        outcomes=[_outcome(good, "p01"), _outcome(skipped, "p02")],
    )
    assert [gap.unit_id for gap in pack.gaps] == ["probe:p02"]
    assert pack.gaps[0].reason == "one creator"
    assert pack.stats["executor_calls"] == 1
    assert pack.stats["unit_budget"] == 2


def test_publish_records_the_notes_that_say_which_round_this_is() -> None:
    workspace = FakeWorkspace()
    unit = _unit("probe:p01")
    pack, _ = publish(
        workspace,
        DECLARATION,
        "deep-r1",
        members=[],
        units=[unit],
        candidates=[rubric("r0101", "a rule", "probe:p01")],
        outcomes=[_outcome(unit, "p01")],
        notes=["deep-research round 1"],
    )
    assert pack.provenance.notes == ["deep-research round 1"]
    assert pack.arm == "deep-r1"


# ─── Diff ────────────────────────────────────────────────────────────────────


def _pack(arm: str, rubrics: Sequence[Rubric]) -> ExpertPack:
    return ExpertPack(topic="resume-design", name="Resume design", arm=arm, rubrics=list(rubrics))


def test_rubric_diff_separates_what_survived_from_what_the_round_added() -> None:
    before = _pack("deep-r1", [rubric("r0101", "a rule", "probe:p01")])
    after = _pack(
        "deep-r2",
        [rubric("r0101", "a rule", "probe:p01"), rubric("r5101", "a new rule", "probe:p51")],
    )
    diff = rubric_diff(before, after)
    assert [row["rubric_id"] for row in diff["kept"]] == ["r0101"]
    assert [row["rubric_id"] for row in diff["added"]] == ["r5101"]
    assert diff["removed"] == []
    # The unit id is what carries a rule back to the gap that caused it.
    assert diff["added"][0]["unit_id"] == "probe:p51"


# ─── Diagnostics ─────────────────────────────────────────────────────────────


def _embed_by_table(table: dict[str, list[float]]):
    def embed(texts: Sequence[str]) -> list[list[float]]:
        return [table[text] for text in texts]

    return embed


def test_restatement_audit_names_the_rule_a_new_one_nearly_repeats() -> None:
    """The dedupe threshold is a cliff and this is its shadow: two statements of
    the same rule landing just under it both survive, and the report has to show
    that rather than count the second one as new ground."""
    prior = [rubric("r0601", "Format the resume for ATS parsing", "probe:p06")]
    added = [rubric("r5301", "Keep the resume formatting ATS-safe", "probe:p53")]
    rows = restatement_audit(
        prior,
        added,
        _embed_by_table(
            {
                "Format the resume for ATS parsing": [1.0, 0.0],
                "Keep the resume formatting ATS-safe": [0.83, 0.5577],
            }
        ),
    )
    assert rows[0]["rubric_id"] == "r5301"
    assert rows[0]["nearest_prior_id"] == "r0601"
    assert rows[0]["nearest_prior_cosine"] == pytest.approx(0.83, abs=1e-3)
    # Reported, never filtered — a threshold tuned here would make the loop's
    # own numbers a function of a knob the loop owns.
    assert rows[0]["dedupe_threshold"] == 0.86


def test_restatement_audit_of_a_round_that_added_nothing_is_empty() -> None:
    assert restatement_audit([rubric("r0101", "a", "u")], [], lambda texts: []) == []


def test_gap_closure_refuses_to_call_a_gap_closed_by_the_same_old_ground() -> None:
    """A gap is asked about by construction. Whether the round got *nearer* to it
    than round one already was is the separate question, and a probe for a facet
    this corpus barely discusses retrieves the nearest thing it does discuss."""
    gaps = [
        ResearchGap(gap_id="g01", missing="nothing about contact details", probe="contact"),
        ResearchGap(gap_id="g04", missing="nothing about github links", probe="github"),
    ]
    prior = [rubric("r0101", "keep it to one page", "probe:p01")]
    added = [
        rubric("r5101", "use ATS-safe formatting", "probe:p51"),
        rubric("r5401", "link only to live github pages", "probe:p54"),
    ]
    rows = gap_closure(
        gaps,
        prior,
        added,
        {"probe:p51": "gap:g01", "probe:p54": "gap:g04"},
        _embed_by_table(
            {
                "nothing about contact details": [1.0, 0.0, 0.0],
                "nothing about github links": [0.0, 1.0, 0.0],
                "keep it to one page": [0.5, 0.0, 0.866],
                # g01's own probe came back with something further from g01 than
                # a rule round one already had: not closed.
                "use ATS-safe formatting": [0.2, 0.0, 0.9798],
                "link only to live github pages": [0.0, 0.9, 0.4359],
            }
        ),
    )
    first, second = rows
    assert first["gap_id"] == "g01" and first["closed"] is False
    assert first["round_one_best_cosine"] == pytest.approx(0.5, abs=1e-3)
    assert second["gap_id"] == "g04" and second["closed"] is True
    assert second["best_new_rubric_id"] == "r5401"


def test_gap_closure_only_considers_rules_from_that_gaps_own_probe() -> None:
    """A rule from another gap's probe cannot close this one — the causal chain
    is the point, and crediting any nearby rule would erase it."""
    gaps = [ResearchGap(gap_id="g01", missing="nothing about contact", probe="contact")]
    rows = gap_closure(
        gaps,
        [rubric("r0101", "keep it short", "probe:p01")],
        [rubric("r5401", "put your email at the top", "probe:p54")],
        {"probe:p54": "gap:g04"},
        _embed_by_table(
            {
                "nothing about contact": [1.0, 0.0],
                "keep it short": [0.0, 1.0],
                "put your email at the top": [1.0, 0.0],
            }
        ),
    )
    assert rows[0]["rules_from_this_probe"] == 0
    assert rows[0]["best_new_rubric_id"] is None
    assert rows[0]["closed"] is False


# ─── Score rows ──────────────────────────────────────────────────────────────


def test_score_rows_carry_the_counts_that_expose_padding() -> None:
    """``criteria_recall`` rises with finding count, so a row without the counts
    beside it cannot be read for the shape of the arm that produced it."""
    run = {
        "cells": [
            {
                "setup": "deep-r2",
                "scores": {"criteria_recall": 0.4},
                "score_spread": {"criteria_recall_min": 0.3},
                "findings_total": 21,
                "citations_total": 36,
                "citations_resolved": 36,
                "findings_grounded": 18,
                "executor_calls": 10,
            }
        ]
    }
    rows = score_rows(run)
    assert rows[0]["arm"] == "deep-r2"
    assert rows[0]["findings_total"] == 21
    assert rows[0]["citations_total"] == 36
    assert rows[0]["score_spread"]["criteria_recall_min"] == 0.3


def test_score_rows_of_an_empty_run_is_empty_rather_than_an_exception() -> None:
    assert score_rows({}) == []


def test_the_run_records_the_corpus_it_was_scored_on_not_only_the_one_built_from() -> None:
    """Ingestion does not stop for an experiment.

    The arms are built at one digest and their citations are resolved against
    whatever the store holds when somebody runs the scorer. Both have to be in
    the file, and the mismatch has to be a value rather than something a reader
    is expected to notice — this is the same rule the matrix cache follows by
    putting the corpus digest in its cell fingerprint.
    """
    records = [
        {"chunk_id": "chunk:v1:0", "video_id": "v1", "text": "a"},
        {"chunk_id": "chunk:v2:0", "video_id": "v2", "text": "b"},
    ]
    moved = _scoring_corpus(records, "builtdigest", lambda rows: "livedigest")
    assert moved == {
        "scoring_corpus_digest": "livedigest",
        "scoring_chunk_count": 2,
        "scoring_video_count": 2,
        "scored_on_build_corpus": False,
    }
    same = _scoring_corpus(records, "livedigest", lambda rows: "livedigest")
    assert same["scored_on_build_corpus"] is True


# ─── Report storage ──────────────────────────────────────────────────────────


def test_a_corrupt_report_reads_as_no_loop_has_been_run(tmp_path) -> None:
    """The same rule the pack reader follows: a half-written file must not take
    the tab down with a 500."""
    from src.rag.deep_research import load_report, save_report
    from src.rag.packs import PackStore

    store = PackStore(tmp_path)
    assert load_report(store, "resume-design") is None
    save_report(store, "resume-design", {"kind": "deep-research"})
    assert load_report(store, "resume-design") == {"kind": "deep-research"}
    (tmp_path / "resume-design" / "research.json").write_text("{not json", encoding="utf-8")
    assert load_report(store, "resume-design") is None


def test_run_deep_research_refuses_an_undeclared_topic(tmp_path) -> None:
    from src.rag.deep_research import run_deep_research

    (tmp_path / "packs.json").write_text('{"version":1,"packs":[]}', encoding="utf-8")
    with pytest.raises(ValueError, match="No declared pack"):
        run_deep_research(
            None,  # type: ignore[arg-type]
            topic="nope",
            packs_dir=tmp_path,
            records=[],
        )


def test_the_build_report_recomputes_its_verdicts_rather_than_serving_stored_ones(tmp_path):
    """A report saved before the verdict fix must not keep serving the old reading.

    ``evidence_precision`` is scored once per arm, so it has no ``*_max`` to
    clear. The old ``winner()`` read that missing key as "nothing to beat" and
    called the lead decisive, printing "leader clears the runner-up's own
    spread" — a comparison that was never performed. Every committed report
    carries that stored verdict, so the reader has to re-derive it.
    """
    import json

    from src.rag.deep_research import research_report

    topic = "resume-design"
    (tmp_path / topic).mkdir(parents=True)
    (tmp_path / "packs.json").write_text(
        json.dumps(
            {
                "version": 1,
                "packs": [
                    {
                        "topic": topic,
                        "name": "R",
                        "artifact": "resume",
                        "blurb": "b",
                        "routing_text": "r",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / topic / "research.json").write_text(
        json.dumps(
            {
                "kind": "deep-research",
                "topic": topic,
                "scores": {
                    "metrics": ["criteria_recall", "evidence_precision"],
                    # The lie, as committed reports carry it.
                    "verdicts": {
                        "evidence_precision": {
                            "metric": "evidence_precision",
                            "leader": "merged",
                            "decisive": True,
                            "reason": "leader clears the runner-up's own spread",
                        }
                    },
                    "rows": [
                        {
                            "arm": "merged",
                            "scores": {"evidence_precision": 0.824},
                            "score_spread": {},
                        },
                        {
                            "arm": "deep-r2",
                            "scores": {"evidence_precision": 0.769},
                            "score_spread": {},
                        },
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    report = research_report(topic, tmp_path)
    assert report is not None
    verdict = report["scores"]["verdicts"]["evidence_precision"]
    assert verdict["leader"] == "merged"
    assert verdict["decisive"] is False, "a single draw per arm cannot clear a spread"
    assert verdict["basis"] == "unrepeated"
    assert "clears" not in verdict["reason"]


def test_an_added_rule_the_shipped_pack_already_covers_is_marked_rediscovered():
    """V8's gate failed because the loop rediscovered ground the hand build held.

    Its evaluator found r5403's first citation is the same chunk the shipped
    r0403 cites, and noted the page never showed it — the diff read as ten
    discoveries. Overlap is measured on cited chunk ids, not wording: two rules
    phrased differently off the same transcript second are the same finding.
    """
    from src.rag.deep_research import research_report

    report = research_report("resume-design")
    if report is None or not (report.get("diff") or {}).get("added"):
        import pytest

        pytest.skip("no committed deep-research diff in this checkout")

    added = report["diff"]["added"]
    marked = [row for row in added if "already_in_shipped" in row]
    assert marked, "every added row should carry the comparison"

    by_id = {row["rubric_id"]: row for row in added}
    # The rule the evaluator traced by hand.
    assert "r0403" in by_id["r5403"]["already_in_shipped"]
    # And at least one addition really is new ground, or the check is vacuous.
    assert any(not row["already_in_shipped"] for row in marked)


# ─── V8: the frontier round ──────────────────────────────────────────────────
#
# The three changes the frontier arm makes, each tested for what it *forbids*
# rather than for what it happened to produce on one corpus. A test that pinned
# the rules the loop wrote would pass for a loop that had stopped iterating.


def test_a_frontier_probe_cannot_retrieve_into_ground_an_earlier_round_read() -> None:
    """The mechanism V8's failure was traced to.

    Every round-two probe drew between five and twelve of its eighteen passages
    out of the 72 round one had already had in front of it, and the one-shot
    control rediscovered at the same rate — the loop was searching more, not
    better. Excluding what an earlier round read is free, deterministic and
    computed from the loop's own output.
    """
    vectors, records = _vectors()
    unrestricted = probe_unit(
        probe("p61"), [1.0, 0.0], vectors, records, ["v1", "v2", "v3"], pool=2
    )
    assert unrestricted.chunk_ids == ["c1", "c2"]
    frontier = probe_unit(
        probe("p61"),
        [1.0, 0.0],
        vectors,
        records,
        ["v1", "v2", "v3"],
        pool=2,
        exclude_chunk_ids=["c1", "c2"],
    )
    assert frontier.chunk_ids == ["c3"]


def test_excluding_everything_reads_as_an_unspent_call_not_an_empty_answer() -> None:
    """A probe with nowhere new to look has to say so. "We found nothing" and
    "there was nothing left to look at" are different admissions."""
    vectors, records = _vectors()
    unit = probe_unit(
        probe("p61"),
        [1.0, 0.0],
        vectors,
        records,
        ["v1", "v2", "v3"],
        pool=5,
        exclude_chunk_ids=["c1", "c2", "c3"],
    )
    assert unit.chunk_ids == []
    assert unit.eligible is False
    assert "an earlier round already read" in unit.reject_reason


def test_the_original_arms_retrieve_exactly_as_they_did_before() -> None:
    """The frontier lever is opt-in. An arm built without it must be unchanged,
    or the committed arms stop being the thing the new one is compared against."""
    vectors, records = _vectors()
    before = probe_unit(probe("p01"), [1.0, 0.0], vectors, records, ["v1", "v2", "v3"], pool=3)
    after = probe_unit(
        probe("p01"), [1.0, 0.0], vectors, records, ["v1", "v2", "v3"], pool=3, exclude_chunk_ids=[]
    )
    assert before.chunk_ids == after.chunk_ids == ["c1", "c2", "c3"]


def _cited(rubric_id: str, unit_id: str, chunk_ids: Sequence[str]) -> Rubric:
    return Rubric(
        rubric_id=rubric_id,
        criterion=f"criterion {rubric_id}",
        unit_id=unit_id,
        unit_kind="probe",
        unit_title=unit_id,
        evidence=[
            RubricEvidence(
                video_id="v1",
                chunk_id=chunk_id,
                chunk_index=0,
                start_seconds=0.0,
                end_seconds=10.0,
                quote="q",
                model_quote="q",
                quote_start_seconds=0.0,
                channel_name="Alpha",
                ratio=1.0,
            )
            for chunk_id in chunk_ids
        ],
    )


def test_a_rule_resting_on_no_passage_of_its_own_is_refused() -> None:
    """The complement of the cosine dedupe, on evidence identity rather than
    wording. The cliff at 0.86 has a shadow — six of round two's ten additions
    sat between 0.744 and 0.859 and were kept — and an identity check has none."""
    from src.rag.deep_research import admit_novel_evidence

    prior = [_cited("r0101", "probe:p01", ["c1", "c2"])]
    candidates = [
        _cited("r6101", "probe:p61", ["c2"]),
        _cited("r6102", "probe:p61", ["c2", "c9"]),
        _cited("r6201", "probe:p62", ["c9"]),
    ]
    admitted, refused = admit_novel_evidence(prior, candidates)
    # r6101 rests only on a passage the prior rule rests on.
    assert [r.rubric_id for r in admitted] == ["r6102"]
    assert [row["rubric_id"] for row in refused] == ["r6101", "r6201"]
    assert refused[0]["already_rested_on_by"] == ["r0101"]
    # r6201's only chunk was claimed by r6102 earlier in this same round, so the
    # rule is transitive across the round rather than only against the last one.
    assert refused[1]["already_rested_on_by"] == ["r6102"]


def test_the_admission_rule_does_not_simulate_the_scorers_exclusivity_gate() -> None:
    """Deliberately weaker than the gate it resembles.

    A candidate that brings one new passage is admitted even though it also
    lands on another rule's only evidence, which the scorer would punish.
    Mirroring the gate any closer would make this a scoring strategy rather than
    a build rule, and the ``deep-r2-admit`` ablation exists so a reader can
    subtract whatever precision this buys.
    """
    from src.rag.deep_research import admit_novel_evidence

    prior = [_cited("r0101", "probe:p01", ["c1"])]
    admitted, refused = admit_novel_evidence(prior, [_cited("r6101", "probe:p61", ["c1", "c2"])])
    assert [r.rubric_id for r in admitted] == ["r6101"]
    assert refused == []


def test_nothing_is_refused_when_a_round_brings_all_new_ground() -> None:
    from src.rag.deep_research import admit_novel_evidence

    admitted, refused = admit_novel_evidence(
        [_cited("r0101", "probe:p01", ["c1"])], [_cited("r6101", "probe:p61", ["c2"])]
    )
    assert [r.rubric_id for r in admitted] == ["r6101"]
    assert refused == []


def test_coverage_profile_lists_every_member_video_unread_first() -> None:
    """The only ordering decision in the whole signal, and no cut-off.

    A filter chosen here would be this module steering the critic; the critic is
    handed the whole table and the counts, and the video titles the planner was
    already given.
    """
    from src.rag.deep_research import coverage_profile

    records = [
        record("chunk:v1:0", "v1", "Alpha"),
        record("chunk:v1:1", "v1", "Alpha"),
        record("chunk:v2:0", "v2", "Beta"),
        record("chunk:v2:1", "v2", "Beta"),
        record("chunk:v2:2", "v2", "Beta"),
        record("chunk:blocked:0", "blocked", "Property"),
    ]
    rows = coverage_profile(records, ["v1", "v2"], ["chunk:v1:0", "chunk:v1:1"])
    assert [row.video_id for row in rows] == ["v2", "v1"]
    assert (rows[0].chunks, rows[0].read, rows[0].unread) == (3, 0, 3)
    assert (rows[1].chunks, rows[1].read, rows[1].unread) == (2, 2, 0)
    # A video outside the pack's membership is not on the menu at all — which is
    # how the held-out and blocked videos stay invisible to the critic.
    assert "blocked" not in {row.video_id for row in rows}


def test_the_coverage_critic_is_told_where_it_has_not_read_and_nothing_else() -> None:
    """The one constraint the novelty claim rests on: no transcript, no evidence,
    nothing from the held-out expert. Counts and the titles the planner already
    had, and that is the whole of the new input."""
    from src.rag.deep_research import CoverageGapCritic, CoverageProfile

    seen: list[str] = []

    class Recorder:
        def invoke(self, messages: list) -> object:
            seen.append("\n".join(str(m.content) for m in messages))
            return type("R", (), {"content": '{"gaps":[]}'})()

    CoverageGapCritic(Recorder(), "m").critique(
        DECLARATION,
        [rubric("r0101", "keep it to one page", "probe:p01")],
        [probe("p01")],
        [CoverageProfile(video_id="v1", title="A resume video", chunks=25, read=4)],
        count=4,
    )
    body = seen[0]
    assert "21 of 25 passages still unread" in body
    assert "A resume video" in body
    assert "keep it to one page" in body
    # The transcript itself never appears — the record fixture's words are the
    # thing that must not be in the prompt.
    assert "some transcript words about resumes" not in body


def test_the_frontier_rounds_rubric_ids_cannot_collide_with_the_first_loops() -> None:
    """Two critics answering the same round one produce two packs a reader puts
    side by side. ``r5101`` meaning two different rules in them is a trap."""
    gaps = [ResearchGap(gap_id="g01", missing="m", probe="p")]
    assert [p.probe_id for p in gap_probes(gaps)] == ["p51"]
    assert [p.probe_id for p in gap_probes(gaps, start=60)] == ["p61"]
    assert gap_probes(gaps, start=60)[0].rank == 61


def test_rediscovery_rate_counts_additions_the_shipped_pack_already_cites() -> None:
    """The number V8 turns on. Four of ten is what ``deep-r2`` scored and what
    the one-shot control scored; an arm that does not move it has not moved the
    finding, whatever its scores do."""
    from src.rag.deep_research import rediscovery_rate

    shipped = _pack("merged", [_cited("r0403", "u", ["c7"])])
    row = rediscovery_rate(
        shipped,
        [_cited("r6101", "probe:p61", ["c7"]), _cited("r6201", "probe:p62", ["c8"])],
    )
    assert row["added"] == 2
    assert row["rediscovered"] == 1
    assert row["rate"] == 0.5
    assert row["rows"][0]["already_in_shipped"] == ["r0403"]
    assert row["rows"][1]["already_in_shipped"] == []


def test_rediscovery_against_no_shipped_pack_is_unmeasured_not_zero() -> None:
    from src.rag.deep_research import rediscovery_rate

    row = rediscovery_rate(None, [_cited("r6101", "probe:p61", ["c1"])])
    assert row["rediscovered"] is None and row["rate"] is None


def test_the_new_arm_is_built_on_the_corpus_the_old_ones_were_or_not_at_all() -> None:
    """Ingestion does not stop for an experiment, and an arm built on a bigger
    corpus is not comparable with the pack it has to beat. The check is a digest
    match, not a heuristic, and there is no "close enough" branch."""
    from src.rag.deep_research import corpus_as_built
    from src.rag.packs import corpus_digest

    old = [
        {"chunk_id": "chunk:v1:0", "video_id": "v1", "text": "a"},
        {"chunk_id": "chunk:v1:1", "video_id": "v1", "text": "b"},
    ]
    live = [*old, {"chunk_id": "chunk:v2:0", "video_id": "v2", "text": "c"}]
    assert corpus_as_built(live, corpus_digest(old)) == old
    assert corpus_as_built(live, corpus_digest(live)) == live
    with pytest.raises(ValueError, match="no prefix of the live corpus"):
        corpus_as_built(live, "0000000000000000")


def test_publish_can_be_told_why_a_unit_produced_nothing() -> None:
    """A refusal that was not a cosine dedupe must not be logged as one."""
    workspace = FakeWorkspace()
    kept, refused_unit = _unit("probe:p01"), _unit("probe:p61")
    pack, _ = publish(
        workspace,
        DECLARATION,
        "deep-frontier",
        members=[],
        units=[kept, refused_unit],
        candidates=[rubric("r0101", "a rule", "probe:p01")],
        outcomes=[_outcome(kept, "p01"), _outcome(refused_unit, "p61")],
        reason_by_unit={"probe:p61": "it rested only on passages an admitted rule rested on"},
    )
    gap = next(row for row in pack.gaps if row.unit_id == "probe:p61")
    assert gap.reason == "it rested only on passages an admitted rule rested on"


def test_the_frontier_arm_gets_a_row_in_the_rounds_table_like_every_other_round() -> None:
    """The arm with the strongest claim was the one missing from the table where
    spend is read against what a round added, which reads as a claim made
    somewhere the comparison is not.

    Derived rather than stored, so a report written before the field existed
    gains the row without the packs being rebuilt — every input is already in
    the frontier block and none of it is a judgement.
    """
    from src.rag.deep_research import _fill_frontier_round
    from src.rag.packs import PackStore

    report = {
        "frontier": {
            "arm": "deep-frontier",
            "probes": [
                {"unit_id": "probe:p61", "spent_call": True},
                {"unit_id": "probe:p62", "spent_call": True},
            ],
            "diff": {"added": [{"rubric_id": "r6101"}, {"rubric_id": "r6402"}]},
            "budget": [{"arm": "deep-frontier", "rubrics": 24, "citations": 48}],
            "deduped_against_previous": ["r6202"],
            "refused": [{"rubric_id": "r6401"}, {"rubric_id": "r6302"}],
        }
    }
    _fill_frontier_round(PackStore("experts"), "nonexistent-topic", report)
    row = report["frontier"]["round"]
    assert row["arm"] == "deep-frontier"
    assert row["executor_calls"] == 2
    assert row["critic_calls"] == 1 and row["planner_calls"] == 0
    assert row["rubrics"] == 24 and row["citations"] == 48
    assert row["added_rubric_ids"] == ["r6101", "r6402"]
    # Both refusals, not just the cosine one: the wording dedupe and the
    # evidence-novelty rule are two ways of saying "this round added nothing
    # here", and counting one would understate what the round threw away.
    assert row["deduped_against_previous"] == 3


def test_a_frontier_block_that_already_records_its_round_is_left_alone() -> None:
    """A future build's own record must not be overwritten by a reconstruction."""
    from src.rag.deep_research import _fill_frontier_round
    from src.rag.packs import PackStore

    report = {"frontier": {"arm": "deep-frontier", "round": {"arm": "written-by-the-build"}}}
    _fill_frontier_round(PackStore("experts"), "resume-design", report)
    assert report["frontier"]["round"] == {"arm": "written-by-the-build"}


def test_a_report_with_no_frontier_block_gains_nothing(tmp_path) -> None:
    from src.rag.deep_research import _fill_frontier_round
    from src.rag.packs import PackStore

    report: dict = {"kind": "deep-research"}
    _fill_frontier_round(PackStore(tmp_path), "resume-design", report)
    assert report == {"kind": "deep-research"}


def test_dropping_rules_is_not_a_reliable_way_to_buy_recall_but_it_is_a_way() -> None:
    """The claim this module used to make, corrected.

    It said ``criteria_recall`` could not be bought by dropping rules, because
    dropping rules only removes chances to match. V8's evaluator disproved it
    from the committed run: ``deep-r2-admit`` is a strict subset of ``deep-r2``
    — the same rules minus two — and scores *higher*. The mechanism is
    :func:`~src.evals.critique.enforce_one_to_one`, not grounding: a criterion
    may be paired with at most one finding, so thinning the pool removes
    competitors for it.

    Pinned against the committed artifacts rather than argued, so the wrong
    claim cannot quietly come back.
    """
    import json
    from pathlib import Path

    from src.rag.packs import PackStore

    root = Path(__file__).resolve().parents[2]
    run_path = root / "evals" / "runs" / "critique-15rTnqKBlO8-20260811-025343.json"
    store = PackStore(root / "experts")
    full = store.load_pack("resume-design", "deep-r2")
    thinned = store.load_pack("resume-design", "deep-r2-admit")
    if not run_path.is_file() or full is None or thinned is None:
        pytest.skip("committed deep-research arms are not in this checkout")

    ids_full = {rubric.rubric_id for rubric in full.rubrics}
    ids_thin = {rubric.rubric_id for rubric in thinned.rubrics}
    assert ids_thin < ids_full, "the ablation must be a strict subset to make the point"

    cells = {
        cell["setup"]: cell for cell in json.loads(run_path.read_text(encoding="utf-8"))["cells"]
    }
    assert (
        cells["deep-r2-admit"]["scores"]["criteria_recall"]
        > cells["deep-r2"]["scores"]["criteria_recall"]
    ), "a strict subset scoring higher is exactly what the old claim ruled out"

    # And the criterion that moved was already available, grounded, in both.
    for setup in ("deep-r2", "deep-r2-admit"):
        rubric = next(row for row in cells[setup]["findings"] if row["id"] == "r0401")
        assert rubric["grounded"] is True
    assert not any(row["id"] == "c22" and row["counted"] for row in cells["deep-r2"]["matches"])
    assert any(row["id"] == "c22" and row["counted"] for row in cells["deep-r2-admit"]["matches"])

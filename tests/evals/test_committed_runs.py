"""The deterministic CI eval gate over committed snapshots (``evals/runs/``).

This is what the CI ``eval-gate`` job runs. It needs no corpus and no API key: it
re-scores the committed runs' deterministic metrics from their stored
``retrieved_chunk_ids`` against the *current* golden labels and enforces floors on
the headline retrieval claims. So it catches three kinds of regression without
re-running retrieval — a snapshot whose stored numbers no longer reconcile with its
retrieved ids (tampering), a golden-set edit that silently invalidates a committed
run (drift), and a real drop below a claimed floor.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.evals.golden import evaluate_entry, load_golden
from src.evals.ir_metrics import IR_METRIC_NAMES

RUNS_DIR = Path(__file__).resolve().parents[2] / "evals" / "runs"

#: The metrics a committed run must reproduce exactly from its retrieved ids.
DETERMINISTIC_METRICS = ["context_recall", "video_recall", *IR_METRIC_NAMES]


def _load(pattern: str) -> list[dict]:
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(RUNS_DIR.glob(pattern))]


def _golden_by_id() -> dict:
    return {entry.id: entry for entry in load_golden()}


def _assert_reproducible(entry: dict, golden: dict) -> None:
    """The entry's stored deterministic scores must recompute from its ids."""
    reference = golden.get(entry["id"])
    assert reference is not None, f"committed run references unknown golden id {entry['id']!r}"
    recomputed = evaluate_entry(reference, "", entry["retrieved_chunk_ids"])
    for metric in DETERMINISTIC_METRICS:
        if metric in entry["scores"]:
            assert entry["scores"][metric] == pytest.approx(recomputed[metric], abs=1e-4), (
                f"{entry['id']} {metric}: stored {entry['scores'][metric]} "
                f"!= recomputed {recomputed[metric]} — golden labels changed, re-baseline the run"
            )


class TestAblationRuns:
    def test_at_least_one_ablation_is_committed(self) -> None:
        assert _load("ablation-*.json"), (
            "commit an ablation run: uv run python -m src.cli eval-ablation"
        )

    def test_ablation_scores_reproduce_from_retrieved_ids(self) -> None:
        golden = _golden_by_id()
        for run in _load("ablation-*.json"):
            for cell in run["cells"]:
                for entry in cell["entries"]:
                    _assert_reproducible(entry, golden)

    def test_headline_retrieval_claims_hold(self) -> None:
        runs = _load("ablation-*.json")
        latest = max(runs, key=lambda run: run["run_id"])
        cells = {cell["label"]: cell["averages"] for cell in latest["cells"]}

        assert latest["baseline"] == "semantic"
        # Retrieval almost always surfaces the right source video, so the open
        # problem is chunk-level ranking rather than finding the source. This
        # was an exact 1.0 on the 9-question set; at 20 it is high but no longer
        # perfect, which is the claim the README now makes.
        for label, averages in cells.items():
            assert averages["video_recall"] >= 0.9, f"{label} video_recall"
        # Sanity floors: no configuration collapses.
        for label, averages in cells.items():
            assert averages["context_recall"] >= 0.45, f"{label} context_recall"
            assert averages["ndcg@10"] >= 0.45, f"{label} ndcg@10"
        # The defensible headline: hybrid fusion improves early-rank recall over
        # plain semantic. If this stops being true, the claim in the README is stale.
        assert "hybrid" in cells and "semantic" in cells
        assert cells["hybrid"]["recall@3"] > cells["semantic"]["recall@3"]

    def test_an_extended_sweep_records_the_variant_axes_it_measured(self) -> None:
        """A row labelled "hyde" has to be traceable to the config that produced
        it — otherwise the committed snapshot is a name, not a measurement."""
        extended = [run for run in _load("ablation-*.json") if run.get("sweep") == "extended"]
        if not extended:
            pytest.skip("no extended sweep committed yet")
        for run in extended:
            configs = {cell["label"]: cell["config"] for cell in run["cells"]}
            assert configs["hyde"]["query_transform"] == "hyde"
            assert configs["multi-query"]["query_transform"] == "multi_query"
            assert configs["contextual"]["contextual"] is True
            assert configs["semantic"]["query_transform"] is None
            assert configs["semantic"]["contextual"] is False

    def test_the_extended_sweep_shares_the_default_sweeps_baseline(self) -> None:
        """Deltas from the two sweeps are read side by side in the README, which
        only holds while both measure against the same first configuration."""
        for run in _load("ablation-*.json"):
            assert run["cells"][0]["label"] == run["baseline"] == "semantic"


class TestGoldenRuns:
    def test_at_least_one_golden_run_is_committed(self) -> None:
        assert _load("eval-*.json"), (
            "commit a golden run: uv run python -m src.cli eval-golden --setup rag_llm --retrieval hybrid"
        )

    def test_golden_runs_carry_full_provenance(self) -> None:
        for run in _load("eval-*.json"):
            config = run["config"]
            for field in (
                "answer_model",
                "embedding_model",
                "retrieval_mode",
                "top_k",
                "judge_model",
            ):
                assert config.get(field) not in (None, ""), (
                    f"{run['run_id']} missing config.{field}"
                )
            assert run["summary"]["scored"] >= 1

    def test_golden_deterministic_scores_reproduce_from_retrieved_ids(self) -> None:
        golden = _golden_by_id()
        for run in _load("eval-*.json"):
            for entry in run["entries"]:
                if entry.get("error"):
                    continue
                _assert_reproducible(entry, golden)


class TestCritiqueRuns:
    """The held-out claim, re-checked from the committed file with no API key.

    Every assertion here is arithmetic over what the run stored, so a snapshot
    whose numbers stopped matching its own evidence fails in CI — which matters
    more for this run than for the others, because its whole value is the claim
    that one video was absent, and a leak looks like nothing at all until
    somebody counts.
    """

    def test_at_least_one_critique_run_is_committed(self) -> None:
        assert _load("critique-*.json"), (
            "commit a held-out critique run: see src.evals.critique_run"
        )

    def test_the_held_out_video_appears_nowhere_in_any_run(self) -> None:
        for run in _load("critique-*.json"):
            held_out = run["held_out_video_id"]
            for cell in run["cells"]:
                leaked = [
                    chunk_id
                    for chunk_id in cell["retrieved_chunk_ids"]
                    if chunk_id.startswith(f"chunk:{held_out}:")
                ]
                assert leaked == [], f"{run['run_id']} {cell['setup']} retrieved {leaked}"
                assert held_out not in cell["retrieved_video_ids"]
                for finding in cell["findings"]:
                    for citation in finding["citations"]:
                        assert citation["video_id"] != held_out, (
                            f"{run['run_id']} {finding['id']} cites the held-out video"
                        )

    def test_the_stored_leak_count_matches_a_fresh_scan(self) -> None:
        """The run's own ``held_out_leaks`` is re-derived, not trusted."""
        from src.evals.critique import Citation, held_out_leaks

        for run in _load("critique-*.json"):
            for cell in run["cells"]:
                recomputed = held_out_leaks(
                    run["held_out_video_id"],
                    chunk_ids=cell["retrieved_chunk_ids"],
                    video_ids=cell["retrieved_video_ids"],
                    citations=[
                        Citation(
                            video_id=citation["video_id"],
                            start_seconds=citation["start_seconds"],
                            quote=citation["quote"],
                        )
                        for finding in cell["findings"]
                        for citation in finding["citations"]
                    ],
                )
                assert len(recomputed) == cell["held_out_leaks"]

    def test_provenance_holds_at_ninety_five_percent(self) -> None:
        """The gate: cited timestamps must resolve to text containing the quote.

        Read off the stored per-citation checks rather than recomputed against
        the corpus, so this runs in CI where there is no Chroma — the
        recomputation against the real transcript is
        ``test_critique.py``'s job, and it skips without a corpus.
        """
        for run in _load("critique-*.json"):
            for cell in run["cells"]:
                checks = [
                    check for finding in cell["findings"] for check in finding["citation_checks"]
                ]
                if not checks:
                    continue
                resolved = sum(1 for check in checks if check["resolved"])
                assert resolved / len(checks) >= 0.95, (
                    f"{run['run_id']} {cell['setup']}: only {resolved}/{len(checks)} "
                    "cited timestamps resolve to the quote"
                )

    def test_every_score_reproduces_from_the_stored_detail(self) -> None:
        """Scores are re-derived from the matches and findings beside them.

        A run whose headline number no longer follows from its own evidence is
        the failure this catches — including the one where somebody edits a
        score by hand.

        ``applicable`` comes from the committed dataset, **not** from the run's
        own ``applies_to`` copy. Deriving it from the run (as an earlier version
        did) let a hand-edited file define the denominator it was checked
        against, so every run passed.

        A cell the grounding gate cannot grade publishes ``None`` rather than a
        number, so the figure this arithmetic reconciles with is the *ungated*
        pair — which is the same pair, computed under the same old rule, kept on
        the cell precisely so nothing is lost. Both are checked: the gated score
        must be absent and the ungated one must still follow from the detail.
        """
        from src.evals.critique import load_critique_dataset

        applicable = {c.id for c in load_critique_dataset().applicable()}
        for run in _load("critique-*.json"):
            for cell in run["cells"]:
                # ``counted``, not ``matched``: a criterion paired with a finding
                # that rests on no exclusive evidence does not score.
                counted = sum(
                    1 for match in cell["matches"] if match["counted"] and match["id"] in applicable
                )
                grounded = sum(1 for finding in cell["findings"] if finding["grounded"])
                recall = pytest.approx(counted / len(applicable), abs=1e-4)
                precision = pytest.approx(grounded / len(cell["findings"]), abs=1e-4)
                where = f"{run['run_id']} {cell['setup']}"
                if cell.get("gradable") is False:
                    assert cell["scores"]["criteria_recall"] is None, where
                    assert cell["scores"]["evidence_precision"] is None, where
                    assert cell["criteria_recall_ungated"] == recall, where
                    assert cell["evidence_precision_ungated"] == precision, where
                else:
                    assert cell["scores"]["criteria_recall"] == recall, where
                    assert cell["scores"]["evidence_precision"] == precision, where

    def test_every_run_says_which_gate_produced_its_scores(self) -> None:
        """A number lifted out of a run file has to carry the rule that made it.

        Two runs of this harness scored under different grounding gates are not
        comparable, and the run that does not say which one it used is the one a
        later slice quotes as if it were the current series.
        """
        from src.evals.critique import GROUNDING_GATES

        for run in _load("critique-*.json"):
            assert run.get("grounding_gate") in GROUNDING_GATES, run["run_id"]
            ungraded = sorted(
                cell["setup"] for cell in run["cells"] if cell.get("gradable") is False
            )
            assert run.get("ungraded_cells") == ungraded, run["run_id"]
            for cell in run["cells"]:
                assert cell.get("grounding_gate") == run["grounding_gate"]
                assert isinstance(cell.get("gradable"), bool)

    def test_an_ungraded_cell_publishes_no_number_the_gate_did_not_certify(self) -> None:
        """Every derived recall on an ungraded row is absent, not zero, not stale.

        The gate refuses to grade an arm that cannot say what each finding's own
        reasoning retrieved. That verdict is worthless if the row still carries
        the old point estimate in one of the *other* recall columns — a reader
        comparing arms would simply read the column that still has a number. So
        ``criteria_recall_all``, ``criteria_recall_grouped`` and the whole
        matcher spread go with it, and the reason travels with the cell.
        """
        for run in _load("critique-*.json"):
            for cell in run["cells"]:
                if cell.get("gradable") is not False:
                    continue
                where = f"{run['run_id']} {cell['setup']}"
                assert cell["criteria_recall_all"] is None, where
                assert cell["criteria_recall_grouped"] is None, where
                assert set(cell["score_spread"].values()) == {None}, where
                assert cell["ungradable_reason"], where
                # And the published figures survive, or the verdict cost data.
                assert isinstance(cell["criteria_recall_ungated"], float), where
                assert isinstance(cell["evidence_precision_ungated"], float), where

    def test_a_graded_cell_only_grounds_on_chunks_its_own_finding_retrieved(self) -> None:
        """The gate itself, re-derived from the committed file.

        ``citations_off_retrieval`` is the scorer's own count; this recomputes it
        from the per-finding record beside it, so a cell that claims to have
        passed the provenance gate has to demonstrate it rather than assert it.
        """
        for run in _load("critique-*.json"):
            if run.get("grounding_gate") != "retrieval_provenance":
                continue
            for cell in run["cells"]:
                if cell.get("gradable") is False:
                    continue
                off = 0
                for finding in cell["findings"]:
                    own = set(finding.get("retrieved_chunk_ids") or [])
                    assert own, f"{run['run_id']} {finding['id']} records no own retrieval"
                    off += sum(
                        1
                        for check in finding["citation_checks"]
                        if check["resolved"] and check["chunk_id"] not in own
                    )
                assert off == cell["citations_off_retrieval"] == 0, (
                    f"{run['run_id']} {cell['setup']}: {off} citations land outside "
                    "the chunks their own finding retrieved"
                )

    def test_the_run_matches_the_committed_criteria_dataset(self) -> None:
        """Every match row must be the criterion the dataset says it is.

        The run carries a copy of each criterion's text, timestamp and
        applicability for the UI. That copy is not a second source of truth — if
        it drifts from ``critique_dataset.json``, the scores were computed
        against something no longer in the repo.
        """
        from src.evals.critique import load_critique_dataset

        dataset = load_critique_dataset()
        by_id = {c.id: c for c in dataset.criteria}
        for run in _load("critique-*.json"):
            assert run["criteria_total"] == len(dataset.criteria)
            assert run["criteria_applicable"] == len(dataset.applicable())
            for cell in run["cells"]:
                assert [m["id"] for m in cell["matches"]] == list(by_id)
                for match in cell["matches"]:
                    criterion = by_id[match["id"]]
                    assert match["criterion"] == criterion.criterion
                    assert match["quote"] == criterion.quote
                    assert match["start_seconds"] == criterion.start_seconds
                    assert match["applies_to"] == list(criterion.applies_to)

    def test_a_finding_never_claims_evidence_another_finding_also_claims(self) -> None:
        """Grounding rests on exclusive evidence — re-derived, not trusted.

        This is the rule that closes the padding attack (18 findings reciting the
        criteria on one shared quote used to score a perfect sweep), so the
        committed run has to demonstrate it rather than assert it.
        """
        for run in _load("critique-*.json"):
            for cell in run["cells"]:
                owners: dict[str, set[str]] = {}
                for finding in cell["findings"]:
                    for check in finding["citation_checks"]:
                        if check["resolved"] and check["chunk_id"]:
                            owners.setdefault(check["chunk_id"], set()).add(finding["id"])
                for finding in cell["findings"]:
                    exclusive = [
                        chunk_id
                        for chunk_id in finding["exclusive_chunk_ids"]
                        if owners.get(chunk_id) == {finding["id"]}
                    ]
                    assert bool(exclusive) == finding["grounded"], finding["id"]

    def test_the_reported_spread_brackets_the_reported_score(self) -> None:
        """A score outside its own spread means the two were computed apart."""
        for run in _load("critique-*.json"):
            for cell in run["cells"]:
                # Outside the spread check, which an ungraded cell skips: the
                # ballots are stored whether or not the cell was graded, and a
                # run missing them cannot be re-scored from its own votes.
                assert len(cell["match_runs"]) == cell["match_repeats"]
                spread = cell["score_spread"]
                if spread["criteria_recall_min"] is None:
                    continue
                assert (
                    spread["criteria_recall_min"]
                    <= cell["scores"]["criteria_recall"]
                    <= spread["criteria_recall_max"]
                )

    def test_a_finding_may_not_be_spent_on_two_criteria(self) -> None:
        for run in _load("critique-*.json"):
            for cell in run["cells"]:
                used = [m["finding_id"] for m in cell["matches"] if m["matched"]]
                assert len(used) == len(set(used)), f"{run['run_id']} reused a finding"

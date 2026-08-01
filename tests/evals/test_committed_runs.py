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

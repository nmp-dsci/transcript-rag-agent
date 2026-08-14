"""The ablation verdict must not claim a comparison it did not perform.

``winner()`` decides which arm leads a metric and how much weight that lead can
carry. Only ``criteria_recall`` is scored repeatedly, so only it has a
``*_max`` for the leader to clear; ``evidence_precision``, ``provenance`` and
``contested_coverage`` are single draws. The original guard read the missing
key as ``None`` and fell straight through to ``decisive``, so those metrics
always rendered "leader clears the runner-up's own spread" — describing a
comparison nobody had run. These tests pin the three-way outcome that replaced
it, and the last one fails if the fallback ever comes back.
"""

from __future__ import annotations

from typing import Any

from src.evals.pack_ablation import winner


def _run(*cells: dict[str, Any]) -> dict[str, Any]:
    return {"kind": "pack-ablation", "cells": list(cells)}


def _cell(setup: str, scores: dict[str, float], spread: dict[str, float] | None = None):
    return {"setup": setup, "scores": scores, "score_spread": spread or {}}


def test_a_lead_that_clears_the_runner_ups_spread_is_decisive() -> None:
    run = _run(
        _cell(
            "merged",
            {"criteria_recall": 0.60},
            {"criteria_recall_min": 0.55, "criteria_recall_max": 0.65},
        ),
        _cell(
            "raptor",
            {"criteria_recall": 0.30},
            {"criteria_recall_min": 0.25, "criteria_recall_max": 0.40},
        ),
    )
    verdict = winner(run, "criteria_recall")
    assert verdict["leader"] == "merged"
    assert verdict["decisive"] is True
    assert verdict["basis"] == "cleared-spread"


def test_a_lead_inside_the_runner_ups_spread_is_not_decisive() -> None:
    run = _run(
        _cell(
            "merged",
            {"criteria_recall": 0.368},
            {"criteria_recall_min": 0.263, "criteria_recall_max": 0.474},
        ),
        _cell(
            "raptor",
            {"criteria_recall": 0.263},
            {"criteria_recall_min": 0.211, "criteria_recall_max": 0.368},
        ),
    )
    verdict = winner(run, "criteria_recall")
    assert verdict["decisive"] is False
    assert verdict["basis"] == "inside-spread"
    assert "inside" in verdict["reason"]


def test_two_arms_on_the_same_score_are_reported_as_tied() -> None:
    run = _run(
        _cell("merged", {"criteria_recall": 0.263}, {"criteria_recall_max": 0.30}),
        _cell("raptor", {"criteria_recall": 0.263}, {"criteria_recall_max": 0.30}),
    )
    verdict = winner(run, "criteria_recall")
    assert verdict["decisive"] is False
    assert verdict["basis"] == "tied"
    assert set(verdict["tied"]) == {"merged", "raptor"}


def test_one_scored_arm_cannot_be_decisive_against_nothing() -> None:
    verdict = winner(_run(_cell("merged", {"criteria_recall": 0.263})), "criteria_recall")
    assert verdict["leader"] == "merged"
    assert verdict["decisive"] is False
    assert verdict["basis"] == "single-arm"


def test_a_metric_scored_once_per_arm_is_unrepeated_not_decisive() -> None:
    """The regression guard: this is the exact shape that used to lie.

    ``evidence_precision`` carries no ``*_max`` because it is not repeated. The
    old code called that decisive and printed a spread comparison; the honest
    answer is that the gap's reliability was never measured.
    """
    run = _run(
        _cell(
            "merged",
            {"criteria_recall": 0.263, "evidence_precision": 0.824},
            {"criteria_recall_min": 0.211, "criteria_recall_max": 0.368},
        ),
        _cell(
            "deep-r2",
            {"criteria_recall": 0.368, "evidence_precision": 0.769},
            {"criteria_recall_min": 0.263, "criteria_recall_max": 0.474},
        ),
    )
    verdict = winner(run, "evidence_precision")

    assert verdict["leader"] == "merged"
    assert verdict["runner_up_max"] is None
    assert verdict["decisive"] is False, "a single draw per arm cannot clear a spread"
    assert verdict["basis"] == "unrepeated"
    # The reason must not describe a comparison that was never performed.
    assert "clears" not in verdict["reason"]
    assert "inside" not in verdict["reason"]


def test_provenance_ties_at_one_rather_than_declaring_a_winner() -> None:
    run = _run(
        _cell("merged", {"provenance": 1.0}),
        _cell("deep-r2", {"provenance": 1.0}),
    )
    verdict = winner(run, "provenance")
    assert verdict["decisive"] is False
    assert verdict["basis"] == "tied"


def test_a_metric_no_arm_scored_has_no_leader() -> None:
    verdict = winner(_run(_cell("merged", {"criteria_recall": 0.263})), "contested_coverage")
    assert verdict["leader"] is None
    assert verdict["decisive"] is False
    assert verdict["basis"] == "no-scored-arm"


# ─── Against the baseline, not the runner-up ─────────────────────────────────
#
# ``winner()`` answers "which arm leads and does the lead survive the noise".
# The V8 gate asks something narrower — does the loop-built pack reach the
# hand-built one — and ``winner`` cannot answer it: when two loop arms tie for
# the lead it reports ``tied`` and says nothing whatever about the baseline.
# That is exactly what happened on the committed run, where ``deep-frontier``
# and ``deep-r2-admit`` scored identically.


def _baselined(*cells: dict[str, Any], baseline: str = "merged") -> dict[str, Any]:
    return {"kind": "pack-ablation", "baseline": baseline, "cells": list(cells)}


def test_an_arm_whose_worst_repeat_beats_the_baselines_best_is_credited() -> None:
    """The committed shape: 0.4211 floor against a 0.3684 ceiling."""
    from src.evals.pack_ablation import against_baseline

    run = _baselined(
        _cell(
            "merged",
            {"criteria_recall": 0.2632},
            {"criteria_recall_min": 0.2105, "criteria_recall_max": 0.3684},
        ),
        _cell(
            "deep-frontier",
            {"criteria_recall": 0.4211},
            {"criteria_recall_min": 0.4211, "criteria_recall_max": 0.4737},
        ),
    )
    row = against_baseline(run)[0]
    assert row["arm"] == "deep-frontier" and row["baseline"] == "merged"
    assert row["beats_baseline"] is True
    assert row["basis"] == "ranges-disjoint"
    assert row["delta"] == 0.1579


def test_the_same_point_estimate_on_a_wider_range_is_not_credited() -> None:
    """``deep-r2-admit`` scores what ``deep-frontier`` scores and does not clear.

    This is the pair the whole slice turns on, and the reason the comparison
    cannot be read off the point estimates: identical scores, one range disjoint
    from the baseline's and one not.
    """
    from src.evals.pack_ablation import against_baseline

    run = _baselined(
        _cell(
            "merged",
            {"criteria_recall": 0.2632},
            {"criteria_recall_min": 0.2105, "criteria_recall_max": 0.3684},
        ),
        _cell(
            "deep-r2-admit",
            {"criteria_recall": 0.4211},
            {"criteria_recall_min": 0.3158, "criteria_recall_max": 0.4737},
        ),
    )
    row = against_baseline(run)[0]
    assert row["value"] == 0.4211, "same score as the arm that is credited"
    assert row["beats_baseline"] is False
    assert row["basis"] == "ranges-overlap"


def test_ranges_that_merely_touch_count_as_overlapping() -> None:
    """The conservative form: strictly greater, so a tie at the boundary loses."""
    from src.evals.pack_ablation import against_baseline

    run = _baselined(
        _cell("merged", {"criteria_recall": 0.2632}, {"criteria_recall_max": 0.3684}),
        _cell("deep-x", {"criteria_recall": 0.4211}, {"criteria_recall_min": 0.3684}),
    )
    assert against_baseline(run)[0]["basis"] == "ranges-overlap"


def test_a_metric_with_no_model_in_its_path_is_not_filed_as_unrepeated() -> None:
    """``evidence_precision`` is quote resolution and set arithmetic.

    It is scored once per arm because nothing in it can disagree, not because
    nobody bothered to repeat it. Calling that "unrepeated" would put a
    deterministic number under the same doubt as a judged one.
    """
    from src.evals.pack_ablation import against_baseline

    run = _baselined(
        _cell("merged", {"evidence_precision": 0.8235}),
        _cell("deep-frontier", {"evidence_precision": 0.875}),
    )
    row = against_baseline(run, "evidence_precision")[0]
    assert row["basis"] == "deterministic"
    assert row["beats_baseline"] is True
    assert "no model" in row["reason"]


def test_an_ungraded_cell_yields_no_comparison_rather_than_a_zero() -> None:
    from src.evals.pack_ablation import against_baseline

    run = _baselined(
        _cell("merged", {"contested_coverage": None}),  # type: ignore[arg-type]
        _cell("deep-frontier", {"contested_coverage": None}),  # type: ignore[arg-type]
    )
    row = against_baseline(run, "contested_coverage")[0]
    assert row["basis"] == "ungraded"
    assert row["delta"] is None and row["beats_baseline"] is None


def test_a_run_with_no_named_baseline_has_nothing_to_compare_against() -> None:
    from src.evals.pack_ablation import against_baseline

    run = {"cells": [_cell("deep-frontier", {"criteria_recall": 0.4})]}
    assert against_baseline(run) == []


def test_the_baseline_table_covers_every_metric_the_harness_scores() -> None:
    from src.evals.critique import CRITIQUE_METRICS
    from src.evals.pack_ablation import baseline_table

    run = _baselined(
        _cell("merged", {"criteria_recall": 0.2632}),
        _cell("deep-frontier", {"criteria_recall": 0.4211}),
    )
    table = baseline_table(run)
    assert set(table) == set(CRITIQUE_METRICS)
    assert all(row["baseline"] == "merged" for rows in table.values() for row in rows)


def test_the_committed_run_carries_the_comparison_it_is_quoted_for() -> None:
    """The point of moving this out of the panel: it survives outside a browser.

    A reader of ``evals/runs/*.json`` used to see only ``verdicts``, which says
    "tied" on the metric the slice was gated on.
    """
    import json
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[2]
        / "evals"
        / "runs"
        / "critique-15rTnqKBlO8-20260811-025343.json"
    )
    if not path.is_file():
        import pytest

        pytest.skip("committed deep-research run is not in this checkout")
    run = json.loads(path.read_text(encoding="utf-8"))
    rows = {row["arm"]: row for row in run["against_baseline"]["criteria_recall"]}
    assert rows["deep-frontier"]["basis"] == "ranges-disjoint"
    # And the free ablation, which ties it on the point estimate, does not.
    assert rows["deep-r2-admit"]["basis"] == "ranges-overlap"
    # The verdict row alone would have told a reader neither of those things.
    assert run["verdicts"]["criteria_recall"]["basis"] == "tied"

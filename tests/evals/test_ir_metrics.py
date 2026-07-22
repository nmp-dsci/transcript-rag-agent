from __future__ import annotations

from math import log2

import pytest

from src.evals.ir_metrics import (
    DEFAULT_KS,
    IR_METRIC_NAMES,
    entry_ir_metrics,
    mean_metrics,
    ndcg_at_k,
    recall_at_k,
    recall_curve,
    reciprocal_rank,
)

# Relevant set is {A, C}; retrieved order puts A first, C fourth.
RETRIEVED = ["A", "B", "C", "D", "E"]
RELEVANT = ["A", "C"]


class TestRecallAtK:
    def test_counts_only_within_the_cutoff(self) -> None:
        # A is in the top 1, C is not until rank 3.
        assert recall_at_k(RETRIEVED, RELEVANT, 1) == 0.5
        assert recall_at_k(RETRIEVED, RELEVANT, 2) == 0.5
        assert recall_at_k(RETRIEVED, RELEVANT, 3) == 1.0

    def test_k_beyond_retrieved_is_recall_over_all(self) -> None:
        assert recall_at_k(RETRIEVED, RELEVANT, 10) == 1.0

    def test_non_positive_k_finds_nothing(self) -> None:
        assert recall_at_k(RETRIEVED, RELEVANT, 0) == 0.0
        assert recall_at_k(RETRIEVED, RELEVANT, -3) == 0.0

    def test_empty_expected_is_perfect_by_convention(self) -> None:
        # Matches golden.context_recall: nothing to find cannot be missed.
        assert recall_at_k(RETRIEVED, [], 5) == 1.0
        assert recall_at_k([], [], 0) == 1.0

    def test_duplicates_do_not_inflate_the_cutoff(self) -> None:
        # A repeated at ranks 1 and 2 must not push C out of a top-3 window: after
        # de-duplication the window is [A, C, B], so both relevant chunks count.
        retrieved = ["A", "A", "C", "B"]
        assert recall_at_k(retrieved, RELEVANT, 3) == 1.0


class TestRecallCurve:
    def test_curve_covers_every_cutoff(self) -> None:
        curve = recall_curve(RETRIEVED, RELEVANT)
        assert set(curve) == set(DEFAULT_KS)
        assert curve == {1: 0.5, 3: 1.0, 5: 1.0, 10: 1.0}

    def test_curve_is_monotonic_non_decreasing(self) -> None:
        curve = recall_curve(RETRIEVED, RELEVANT, ks=(1, 2, 3, 4, 5))
        values = [curve[k] for k in sorted(curve)]
        assert values == sorted(values)


class TestReciprocalRank:
    def test_first_relevant_at_rank_one(self) -> None:
        assert reciprocal_rank(["A", "B"], ["A"]) == 1.0

    def test_first_relevant_at_rank_three(self) -> None:
        assert reciprocal_rank(["X", "Y", "A", "C"], RELEVANT) == pytest.approx(1 / 3)

    def test_no_relevant_retrieved_scores_zero(self) -> None:
        assert reciprocal_rank(["X", "Y", "Z"], RELEVANT) == 0.0

    def test_duplicates_before_first_hit_do_not_shift_the_rank(self) -> None:
        # X repeated must not count as two positions before the first real hit.
        assert reciprocal_rank(["X", "X", "A"], ["A"]) == 0.5

    def test_empty_expected_is_perfect_by_convention(self) -> None:
        assert reciprocal_rank(RETRIEVED, []) == 1.0


class TestNdcgAtK:
    def test_ideal_ranking_scores_one(self) -> None:
        # Both relevant chunks first: DCG already equals IDCG.
        assert ndcg_at_k(["A", "C", "B"], RELEVANT, 10) == pytest.approx(1.0)

    def test_later_placement_is_penalised(self) -> None:
        # A at 1, C at 3. DCG = 1/log2(2) + 1/log2(4); IDCG = 1/log2(2) + 1/log2(3).
        dcg = 1 / log2(2) + 1 / log2(4)
        idcg = 1 / log2(2) + 1 / log2(3)
        assert ndcg_at_k(RETRIEVED, RELEVANT, 10) == pytest.approx(dcg / idcg)
        assert ndcg_at_k(RETRIEVED, RELEVANT, 10) < 1.0

    def test_ordering_matters_even_at_equal_recall(self) -> None:
        early = ndcg_at_k(["A", "C", "X", "Y"], RELEVANT, 4)
        late = ndcg_at_k(["X", "Y", "A", "C"], RELEVANT, 4)
        assert early > late

    def test_cutoff_excludes_relevant_below_k(self) -> None:
        # C sits at rank 3, outside k=2: only A contributes to DCG.
        dcg = 1 / log2(2)
        idcg = 1 / log2(2) + 1 / log2(3)
        assert ndcg_at_k(RETRIEVED, RELEVANT, 2) == pytest.approx(dcg / idcg)

    def test_non_positive_k_and_empty_expected(self) -> None:
        assert ndcg_at_k(RETRIEVED, RELEVANT, 0) == 0.0
        assert ndcg_at_k(RETRIEVED, [], 5) == 1.0


class TestEntryIrMetrics:
    def test_emits_every_named_metric_rounded(self) -> None:
        scores = entry_ir_metrics(RETRIEVED, RELEVANT)
        assert set(scores) == set(IR_METRIC_NAMES)
        assert scores["recall@1"] == 0.5
        assert scores["recall@3"] == 1.0
        assert scores["mrr"] == 1.0  # A is first
        assert 0.0 < scores["ndcg@10"] < 1.0
        assert all(round(v, 4) == v for v in scores.values())


class TestMeanMetrics:
    def test_average_of_reciprocal_rank_is_mrr(self) -> None:
        per_entry: list[dict[str, float | None]] = [{"mrr": 1.0}, {"mrr": 0.5}, {"mrr": 0.0}]
        assert mean_metrics(per_entry, ["mrr"]) == {"mrr": 0.5}

    def test_missing_metric_is_skipped_not_zeroed(self) -> None:
        per_entry: list[dict[str, float | None]] = [
            {"recall@5": 1.0},
            {"recall@5": None},
            {},
        ]
        # Only the one real number counts; the None and the absent key are ignored.
        assert mean_metrics(per_entry, ["recall@5"]) == {"recall@5": 1.0}

    def test_metric_with_no_values_is_absent(self) -> None:
        assert mean_metrics([{"mrr": None}], ["mrr"]) == {}

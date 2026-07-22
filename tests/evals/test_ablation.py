from __future__ import annotations

import pytest

from src.evals.ablation import (
    ABLATION_METRICS,
    AblationConfig,
    default_configs,
    format_table,
    run_ablation,
)
from src.evals.golden import GoldenEntry


def _entry(entry_id: str, domain: str, video: str, chunks: list[int]) -> GoldenEntry:
    return GoldenEntry(
        id=entry_id,
        question=f"question {entry_id}",
        reference_answer="a reference answer",
        expected_video_ids=[video],
        expected_chunk_ids=[f"chunk:{video}:{i}" for i in chunks],
        domain=domain,
    )


ENTRIES = [
    _entry("g1", "property", "v1", [0, 1]),
    _entry("g2", "ai-coding", "v2", [0]),
]

# A retriever that ranks well only under "hybrid+rerank": semantic half-misses.
RETRIEVED = {
    ("semantic", "question g1"): ["chunk:v1:0", "chunk:noise:9"],
    ("semantic", "question g2"): ["chunk:noise:1"],
    ("hybrid+rerank", "question g1"): ["chunk:v1:0", "chunk:v1:1"],
    ("hybrid+rerank", "question g2"): ["chunk:v2:0"],
}


def _fake_retrieve(question: str, config: AblationConfig) -> list[str]:
    return RETRIEVED[(config.label, question)]


CONFIGS = [
    AblationConfig(label="semantic", retrieval_mode="semantic"),
    AblationConfig(label="hybrid+rerank", retrieval_mode="hybrid", rerank=True),
]


class TestRunAblation:
    def test_reports_a_cell_per_config_baseline_first(self) -> None:
        result = run_ablation(ENTRIES, CONFIGS, _fake_retrieve)
        assert [cell["label"] for cell in result["cells"]] == ["semantic", "hybrid+rerank"]
        assert result["baseline"] == "semantic"
        assert result["entries"] == 2
        assert result["metrics"] == ABLATION_METRICS

    def test_averages_capture_the_retrieval_quality_gap(self) -> None:
        result = run_ablation(ENTRIES, CONFIGS, _fake_retrieve)
        semantic, reranked = result["cells"]
        # hybrid+rerank retrieves every expected chunk; semantic misses g2 entirely
        # and only half of g1.
        assert reranked["averages"]["context_recall"] == 1.0
        assert semantic["averages"]["context_recall"] < 1.0
        assert reranked["averages"]["mrr"] >= semantic["averages"]["mrr"]

    def test_deltas_are_measured_against_the_baseline(self) -> None:
        result = run_ablation(ENTRIES, CONFIGS, _fake_retrieve)
        assert [d["label"] for d in result["deltas"]] == ["hybrid+rerank"]
        delta = result["deltas"][0]["vs_baseline"]
        assert set(delta) == set(ABLATION_METRICS)
        # Reranking strictly improves recall here, so the delta is positive.
        assert delta["recall@5"] > 0
        assert delta["ndcg@10"] > 0

    def test_per_domain_breakdown_is_reported(self) -> None:
        result = run_ablation(ENTRIES, CONFIGS, _fake_retrieve)
        by_domain = result["cells"][0]["by_domain"]
        assert set(by_domain) == {"property", "ai-coding"}
        # ai-coding (g2) is missed entirely by semantic retrieval.
        assert by_domain["ai-coding"]["context_recall"] == 0.0
        assert by_domain["property"]["context_recall"] == 0.5

    def test_entries_retain_their_retrieved_ids(self) -> None:
        result = run_ablation(ENTRIES, CONFIGS, _fake_retrieve)
        first = result["cells"][0]["entries"][0]
        assert first["id"] == "g1"
        assert first["retrieved_chunk_ids"] == ["chunk:v1:0", "chunk:noise:9"]

    def test_empty_configs_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one configuration"):
            run_ablation(ENTRIES, [], _fake_retrieve)

    def test_run_id_is_derived_from_the_supplied_time(self) -> None:
        from datetime import datetime, timezone

        moment = datetime(2026, 7, 22, 9, 30, 0, tzinfo=timezone.utc)
        result = run_ablation(ENTRIES, CONFIGS, _fake_retrieve, now=moment)
        assert result["run_id"] == "ablation-20260722-093000"


class TestDefaultConfigs:
    def test_sweeps_semantic_hybrid_and_hybrid_rerank(self) -> None:
        labels = [c.label for c in default_configs()]
        assert labels == ["semantic", "hybrid", "hybrid+rerank"]
        assert default_configs()[0].retrieval_mode == "semantic"
        assert default_configs()[-1].rerank is True

    def test_top_k_is_threaded_through(self) -> None:
        assert all(c.top_k == 7 for c in default_configs(top_k=7))


class TestFormatTable:
    def test_table_lists_every_config_and_the_deltas(self) -> None:
        table = format_table(run_ablation(ENTRIES, CONFIGS, _fake_retrieve))
        assert "semantic" in table
        assert "hybrid+rerank" in table
        assert "deltas vs semantic" in table
        assert "recall@10" in table

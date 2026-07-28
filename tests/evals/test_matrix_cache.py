"""Cell fingerprinting and the load/save cache, isolated from live settings."""

from __future__ import annotations

import dataclasses

from src.config import Settings
from src.evals.golden import GoldenEntry
from src.evals.matrix_cache import (
    _ENGINE_SETTINGS,
    _SETTINGS_DEFAULTS,
    cell_fingerprint,
    load_cell,
    save_cell,
)


def entry(**overrides) -> GoldenEntry:
    base = dict(
        id="g001",
        question="What changed in the budget?",
        reference_answer="Negative gearing was grandfathered.",
        expected_video_ids=["v1"],
        expected_chunk_ids=["chunk:v1:0"],
        domain="property",
    )
    base.update(overrides)
    return GoldenEntry.model_validate(base)


class FakeSettings:
    deepseek_model = "deepseek-v4-flash"
    embedding_model = "all-MiniLM-L6-v2"
    retrieval_mode = "hybrid"
    rag_top_k = 10
    rerank_enabled = True
    rerank_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    neighbor_span = 0
    neo4j_uri = "bolt://localhost:7687"
    # Engine-specific settings, at the defaults a shipped Settings carries.
    retrieval_candidates = _SETTINGS_DEFAULTS["retrieval_candidates"]
    transcript_filter_top_k = _SETTINGS_DEFAULTS["transcript_filter_top_k"]
    transcript_filter_min_score = _SETTINGS_DEFAULTS["transcript_filter_min_score"]
    rag_max_depth = _SETTINGS_DEFAULTS["rag_max_depth"]
    rag_max_followups = _SETTINGS_DEFAULTS["rag_max_followups"]
    rag_followup_top_k = _SETTINGS_DEFAULTS["rag_followup_top_k"]
    rag_novelty_min_chunks = _SETTINGS_DEFAULTS["rag_novelty_min_chunks"]
    rag_max_total_followups = _SETTINGS_DEFAULTS["rag_max_total_followups"]
    rag_agent_max_iterations = _SETTINGS_DEFAULTS["rag_agent_max_iterations"]


def test_same_inputs_produce_the_same_fingerprint() -> None:
    settings = FakeSettings()
    e = entry()
    first = cell_fingerprint("rag_llm", e, settings, judge_model="deepseek-v4-flash")
    second = cell_fingerprint("rag_llm", e, settings, judge_model="deepseek-v4-flash")
    assert first == second


def test_changing_the_answer_model_changes_the_fingerprint() -> None:
    settings = FakeSettings()
    e = entry()
    before = cell_fingerprint("rag_llm", e, settings, judge_model="j")
    settings.deepseek_model = "deepseek-v5"
    after = cell_fingerprint("rag_llm", e, settings, judge_model="j")
    assert before != after


def test_changing_the_judge_model_changes_the_fingerprint() -> None:
    settings = FakeSettings()
    e = entry()
    before = cell_fingerprint("rag_llm", e, settings, judge_model="judge-a")
    after = cell_fingerprint("rag_llm", e, settings, judge_model="judge-b")
    assert before != after


def test_editing_the_golden_question_changes_the_fingerprint() -> None:
    settings = FakeSettings()
    before = cell_fingerprint("rag_llm", entry(), settings, judge_model="j")
    after = cell_fingerprint(
        "rag_llm", entry(question="A different question?"), settings, judge_model="j"
    )
    assert before != after


def test_different_setups_get_different_fingerprints_for_the_same_question() -> None:
    settings = FakeSettings()
    e = entry()
    rag_llm_fp = cell_fingerprint("rag_llm", e, settings, judge_model="j")
    graph_rag_fp = cell_fingerprint("graph_rag", e, settings, judge_model="j")
    assert rag_llm_fp != graph_rag_fp


def test_neo4j_uri_only_affects_graph_rags_fingerprint() -> None:
    settings = FakeSettings()
    e = entry()
    rag_llm_before = cell_fingerprint("rag_llm", e, settings, judge_model="j")
    settings.neo4j_uri = "bolt://otherhost:7687"
    rag_llm_after = cell_fingerprint("rag_llm", e, settings, judge_model="j")
    assert rag_llm_before == rag_llm_after  # rag_llm never touches Neo4j

    settings.neo4j_uri = "bolt://localhost:7687"
    graph_rag_before = cell_fingerprint("graph_rag", e, settings, judge_model="j")
    settings.neo4j_uri = "bolt://otherhost:7687"
    graph_rag_after = cell_fingerprint("graph_rag", e, settings, judge_model="j")
    assert graph_rag_before != graph_rag_after


def test_engine_settings_name_real_settings_fields() -> None:
    """A typo'd field name would silently stop tracking that setting."""
    field_names = {field.name for field in dataclasses.fields(Settings)}
    assert set(_ENGINE_SETTINGS) <= field_names


def test_changing_a_setting_an_engine_ignores_leaves_its_fingerprint_alone() -> None:
    settings = FakeSettings()
    e = entry()
    before = {
        setup: cell_fingerprint(setup, e, settings, judge_model="j")
        for setup in ("rag_llm", "rag_llm_recursive", "rag_agent", "graph_rag")
    }

    # The agent's iteration cap: only rag_agent runs a ReAct loop.
    settings.rag_agent_max_iterations = 20
    assert cell_fingerprint("rag_agent", e, settings, judge_model="j") != before["rag_agent"]
    for untouched in ("rag_llm", "rag_llm_recursive", "graph_rag"):
        assert cell_fingerprint(untouched, e, settings, judge_model="j") == before[untouched]

    # The recursion budget: only rag_llm_recursive fans out follow-ups.
    settings.rag_agent_max_iterations = _SETTINGS_DEFAULTS["rag_agent_max_iterations"]
    settings.rag_max_depth = 3
    assert (
        cell_fingerprint("rag_llm_recursive", e, settings, judge_model="j")
        != before["rag_llm_recursive"]
    )
    for untouched in ("rag_llm", "rag_agent", "graph_rag"):
        assert cell_fingerprint(untouched, e, settings, judge_model="j") == before[untouched]


def test_changing_retrieval_breadth_changes_every_engines_fingerprint() -> None:
    """Every engine retrieves through the shared context provider, so widening
    the candidate pool can change any of their answers."""
    settings = FakeSettings()
    e = entry()
    before = {
        setup: cell_fingerprint(setup, e, settings, judge_model="j")
        for setup in ("rag_llm", "rag_llm_recursive", "rag_agent", "graph_rag")
    }

    settings.retrieval_candidates = 60

    for setup, fingerprint in before.items():
        assert cell_fingerprint(setup, e, settings, judge_model="j") != fingerprint


def test_settings_left_at_their_defaults_do_not_enter_the_fingerprint() -> None:
    """Cells scored before an engine setting was tracked stay valid as long as
    that setting is still at its default — only a deviation from it, the case
    the cache would otherwise answer with a stale score, forces a rescore."""

    class Minimal:
        """A settings stand-in that predates the engine-specific fields."""

        deepseek_model = FakeSettings.deepseek_model
        embedding_model = FakeSettings.embedding_model
        retrieval_mode = FakeSettings.retrieval_mode
        rag_top_k = FakeSettings.rag_top_k
        rerank_enabled = FakeSettings.rerank_enabled
        rerank_model = FakeSettings.rerank_model
        neighbor_span = FakeSettings.neighbor_span
        neo4j_uri = FakeSettings.neo4j_uri

    e = entry()
    for setup in ("rag_llm", "rag_llm_recursive", "rag_agent", "graph_rag"):
        assert cell_fingerprint(setup, e, Minimal(), judge_model="j") == cell_fingerprint(
            setup, e, FakeSettings(), judge_model="j"
        )


def test_save_then_load_round_trips(tmp_path) -> None:
    result = {"id": "g001", "scores": {"answer_correctness": 0.5}, "error": None}
    save_cell("abc123", result, cache_dir=tmp_path)
    loaded = load_cell("abc123", cache_dir=tmp_path)
    assert loaded == result


def test_load_returns_none_on_a_cache_miss(tmp_path) -> None:
    assert load_cell("nonexistent", cache_dir=tmp_path) is None


def test_a_cached_error_is_treated_as_a_miss_so_it_retries(tmp_path) -> None:
    # save_cell refuses to persist an error; simulate one written some other
    # way (or by an older version) to confirm load_cell still guards it.
    path = tmp_path / "abc123.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"id": "g001", "error": "boom"}', encoding="utf-8")
    assert load_cell("abc123", cache_dir=tmp_path) is None


def test_save_cell_does_not_persist_an_error_result(tmp_path) -> None:
    save_cell("abc123", {"id": "g001", "error": "boom"}, cache_dir=tmp_path)
    assert not (tmp_path / "abc123.json").exists()


def test_load_returns_none_for_a_corrupt_cache_file(tmp_path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{not valid json", encoding="utf-8")
    assert load_cell("bad", cache_dir=tmp_path) is None

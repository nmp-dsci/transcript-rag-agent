"""Cell fingerprinting and the load/save cache, isolated from live settings."""

from __future__ import annotations

from src.evals.golden import GoldenEntry
from src.evals.matrix_cache import cell_fingerprint, load_cell, save_cell


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

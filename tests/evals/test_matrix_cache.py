"""Cell fingerprinting and the load/save cache, isolated from live settings."""

from __future__ import annotations

import dataclasses

import pytest

from src.config import Settings
from src.evals.golden import GoldenEntry
from src.evals.matrix_cache import (
    _ENGINE_SETTINGS,
    _FILTERED_SETUPS,
    _SETTINGS_DEFAULTS,
    behavior_material,
    cell_fingerprint,
    corpus_digest,
    corpus_digest_for,
    corpus_stats_for,
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


# ── retrieval variants ────────────────────────────────────────────────────────


class VariantSettings(FakeSettings):
    contextual_chunk_collection = _SETTINGS_DEFAULTS["contextual_chunk_collection"]


def test_the_retrieval_variants_are_separate_cells_from_the_baseline() -> None:
    """rag_llm_hyde answers the same question a different way; reusing rag_llm's
    cached cell for it would report the baseline's score as the variant's."""
    settings = VariantSettings()
    e = entry()

    fingerprints = {
        cell_fingerprint(setup, e, settings)
        for setup in ("rag_llm", "rag_llm_hyde", "rag_llm_contextual")
    }

    assert len(fingerprints) == 3


def test_repointing_the_contextual_index_invalidates_only_that_variant() -> None:
    e = entry()
    settings = VariantSettings()
    moved = VariantSettings()
    moved.contextual_chunk_collection = "some_other_collection"

    assert cell_fingerprint("rag_llm_contextual", e, settings) != cell_fingerprint(
        "rag_llm_contextual", e, moved
    )
    # No other engine reads that collection, so their cells stay valid.
    for setup in ("rag_llm", "rag_llm_hyde", "graph_rag"):
        assert cell_fingerprint(setup, e, settings) == cell_fingerprint(setup, e, moved)


def test_retrieval_breadth_still_reaches_the_new_vector_variants() -> None:
    e = entry()
    settings = VariantSettings()
    wider = VariantSettings()
    wider.retrieval_candidates = settings.retrieval_candidates + 10

    for setup in ("rag_llm_hyde", "rag_llm_contextual"):
        assert cell_fingerprint(setup, e, settings) != cell_fingerprint(setup, e, wider)


def test_holding_out_a_video_changes_the_fingerprint():
    """A full-corpus cell must never answer for a held-out one.

    Same question, same engine, same judge — the only difference is that one
    run could see the video and the other could not, which is precisely the
    difference the held-out experiment measures. Sharing a cache entry between
    them produces a number that looks fine and means nothing.
    """
    settings = FakeSettings()
    e = entry()
    full = cell_fingerprint("rag_llm_filtered", e, settings)
    held_out = cell_fingerprint("rag_llm_filtered", e, settings, exclude_video_ids=["15rTnqKBlO8"])
    other = cell_fingerprint("rag_llm_filtered", e, settings, exclude_video_ids=["3pFRqPqzBCM"])

    assert full != held_out
    assert held_out != other


def test_an_empty_exclusion_leaves_the_fingerprint_untouched():
    """Cells scored before held-out runs existed must stay valid."""
    settings = FakeSettings()
    e = entry()

    assert cell_fingerprint("rag_llm", e, settings) == cell_fingerprint(
        "rag_llm", e, settings, exclude_video_ids=[]
    )


# --- corpus identity -----------------------------------------------------


def test_ingesting_videos_changes_the_fingerprint():
    """The defect this exists to prevent.

    A ``rag_llm`` baseline scored over 38 videos and a ``rag_llm_filtered`` arm
    scored over 53 differ in the corpus and in nothing else this material
    tracked, so the cache handed back the stale arm and the comparison read as
    filtered-vs-unfiltered when it was actually old-corpus-vs-new.
    """
    settings = FakeSettings()
    e = entry()
    before = cell_fingerprint("rag_llm", e, settings, corpus=corpus_digest(["a", "b"], 40))
    after = cell_fingerprint("rag_llm", e, settings, corpus=corpus_digest(["a", "b", "c"], 60))

    assert before != after


def test_rechunking_the_same_videos_changes_the_fingerprint():
    """Same videos, different chunking: the retrieved units are not the same."""
    same_videos = ["a", "b"]
    settings = FakeSettings()
    e = entry()

    assert corpus_digest(same_videos, 40) != corpus_digest(same_videos, 80)
    assert cell_fingerprint(
        "rag_llm", e, settings, corpus=corpus_digest(same_videos, 40)
    ) != cell_fingerprint("rag_llm", e, settings, corpus=corpus_digest(same_videos, 80))


def test_corpus_digest_ignores_video_order_and_duplicates():
    """It identifies the corpus, not the order one bulk read happened to return."""
    assert corpus_digest(["b", "a", "a"], 10) == corpus_digest(["a", "b"], 10)


def test_an_unknown_corpus_is_omitted_so_unit_tests_need_no_store():
    """``None`` means "not supplied", and only tests supply nothing."""
    settings = FakeSettings()
    e = entry()

    assert cell_fingerprint("rag_llm", e, settings) == cell_fingerprint(
        "rag_llm", e, settings, corpus=None
    )
    assert cell_fingerprint("rag_llm", e, settings) != cell_fingerprint(
        "rag_llm", e, settings, corpus=corpus_digest(["a"], 1)
    )


def test_corpus_digest_for_reads_a_store_in_one_call():
    class FakeCollection:
        def __init__(self):
            self.calls = 0

        def get(self, include):
            self.calls += 1
            return {"metadatas": [{"video_id": "a"}, {"video_id": "b"}, {"video_id": "a"}]}

    class FakeStore:
        def __init__(self):
            self.collection = FakeCollection()

    store = FakeStore()
    assert corpus_digest_for(store) == corpus_digest(["a", "b"], 3)
    assert store.collection.calls == 1


def test_corpus_stats_for_counts_videos_and_chunks_in_the_same_read():
    """The sizes a reader needs, from the read the digest already pays for.

    Distinct videos, not rows: the store holds one row per chunk, so counting
    rows would report the chunk count twice and call one of them a video count.
    """

    class FakeCollection:
        def __init__(self):
            self.calls = 0

        def get(self, include):
            self.calls += 1
            return {"metadatas": [{"video_id": "a"}, {"video_id": "b"}, {"video_id": "a"}]}

    class FakeStore:
        def __init__(self):
            self.collection = FakeCollection()

    store = FakeStore()
    digest, videos, chunks = corpus_stats_for(store)
    assert (digest, videos, chunks) == (corpus_digest(["a", "b"], 3), 2, 3)
    assert store.collection.calls == 1


# --- retrieval behaviour that moves no config field --------------------------


def corpus_wide_entry(**overrides) -> GoldenEntry:
    """A golden entry the corpus-wide detector fires on."""
    base = dict(
        id="g010",
        question="What are the main themes across this corpus of videos?",
        reference_answer="Career preparation, property, and system design.",
        expected_video_ids=[],
        expected_chunk_ids=[],
        domain="career",
        question_type="global",
    )
    base.update(overrides)
    return entry(**base)


def test_changing_the_corpus_wide_behaviour_invalidates_the_cells_it_moves():
    """The defect this exists to prevent, one level up from task #58.

    ``top_k`` became a budget rather than a cap, so a corpus-wide question now
    routes to every video above ``min_score``. No configuration field moves —
    the decision is derived from the question at query time — so without this
    the six cells scored under the capped behaviour come back from the cache and
    are averaged into one column with fourteen post-change locals.
    """
    settings = FakeSettings()
    e = corpus_wide_entry()

    capped = cell_fingerprint("rag_llm_filtered", e, settings, retrieval_behavior="capped")
    budget = cell_fingerprint("rag_llm_filtered", e, settings, retrieval_behavior="budget")

    assert capped != budget


def test_the_diversity_switch_is_a_different_behaviour_again():
    """Turning it on changes what a corpus-wide answer is built from, and it is
    not a ``Settings`` field either — so the cache has to hear about it."""
    settings = FakeSettings()
    e = corpus_wide_entry()

    assert cell_fingerprint(
        "rag_llm_filtered", e, settings, retrieval_behavior="budget"
    ) != cell_fingerprint("rag_llm_filtered", e, settings, retrieval_behavior="budget+diversity")


def test_only_the_cells_the_behaviour_can_reach_are_invalidated():
    """Adding a behaviour version must be a decision, not a demolition.

    Six cells of a forty-cell run, not forty: a setup that never runs the
    summary filter cannot be moved by a change to what the filter does.
    """
    settings = FakeSettings()
    e = corpus_wide_entry()

    for setup in ("rag_llm", "rag_llm_hyde", "rag_llm_contextual", "graph_rag", "rag_agent"):
        assert cell_fingerprint(setup, e, settings, retrieval_behavior="budget") == (
            cell_fingerprint(setup, e, settings, retrieval_behavior="capped")
        )


def test_a_question_the_detector_does_not_fire_on_keeps_its_cached_cell():
    """Fourteen of the twenty golden questions are untouched by this change, and
    re-scoring them would cost tokens to reproduce identical numbers."""
    settings = FakeSettings()
    local = entry(question="How do I make my resume ATS-friendly?")

    assert cell_fingerprint(
        "rag_llm_filtered", local, settings, retrieval_behavior="budget"
    ) == cell_fingerprint("rag_llm_filtered", local, settings, retrieval_behavior="capped")


def test_absence_has_exactly_one_meaning():
    """It is omitted from the material in three cases, and all three mean the
    same thing: *this cell behaves as the cache's untracked history did*.

    That is what makes ``capped`` reuse a pre-existing cell rather than
    re-score it — the pre-change arm of an A/B costs nothing — while leaving no
    reading under which a *changed* behaviour is mistaken for an absent one.
    """
    corpus_wide = corpus_wide_entry().question
    local = "How do I make my resume ATS-friendly?"

    assert behavior_material("rag_llm_filtered", corpus_wide, "capped") is None
    assert behavior_material("rag_llm", corpus_wide, "budget") is None
    assert behavior_material("rag_llm_filtered", local, "budget") is None
    assert behavior_material("rag_llm_filtered", corpus_wide, "budget") == "budget"


def test_the_shipped_default_is_what_a_default_provider_implements():
    """The default must track the code, not a literal somebody typed once."""
    from src.rag.context import MultiTranscriptRagContextProvider

    provider = MultiTranscriptRagContextProvider(raw_store=object(), chunk_store=object())
    settings = FakeSettings()
    e = corpus_wide_entry()

    assert cell_fingerprint("rag_llm_filtered", e, settings) == cell_fingerprint(
        "rag_llm_filtered", e, settings, retrieval_behavior=provider.retrieval_behavior
    )


def test_an_unrecognised_behaviour_raises_rather_than_hashing():
    """Failing closed. A name nobody registered is a behaviour nobody wrote
    down, and hashing it would mint a fingerprint that quietly never hits."""
    settings = FakeSettings()

    with pytest.raises(ValueError, match="unknown retrieval behavior"):
        cell_fingerprint("rag_llm_filtered", corpus_wide_entry(), settings, retrieval_behavior="v9")

    with pytest.raises(ValueError, match="unknown retrieval behavior"):
        cell_fingerprint("rag_llm", entry(), settings, retrieval_behavior="")


def test_every_setup_that_runs_the_summary_filter_is_covered():
    """Scoping the behaviour to one setup is only safe while one setup filters.

    Derived from the setup registry rather than restated, so adding a second
    filtered setup fails here instead of silently reusing its stale cells.
    """
    from src.chat.setups import SETUP_SPECS

    filtered = {spec.key for spec in SETUP_SPECS if spec.filter_transcripts}

    assert filtered == set(_FILTERED_SETUPS)


def test_every_derivable_behaviour_name_is_accepted():
    """A provider must never be able to report a name the fingerprint rejects."""
    from src.rag.question_scope import corpus_wide_behavior

    settings = FakeSettings()
    for lifted in (True, False):
        for cap in (0, 2):
            name = corpus_wide_behavior(cap_lifted=lifted, max_chunks_per_video=cap)
            cell_fingerprint(
                "rag_llm_filtered", corpus_wide_entry(), settings, retrieval_behavior=name
            )

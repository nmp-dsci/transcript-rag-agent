from __future__ import annotations

import json

import pytest

from src.rag.query_transform import (
    HydeTransform,
    MultiQueryTransform,
    QueryPlan,
    build_query_transform,
)


class FakeLlm:
    """Returns canned contents in order, recording every prompt it was sent."""

    def __init__(self, *contents: str) -> None:
        self.contents = list(contents)
        self.calls: list[list] = []

    def invoke(self, messages: list):
        self.calls.append(messages)

        class Response:
            content = self.contents.pop(0) if self.contents else ""

        return Response()


class BrokenLlm:
    def __init__(self) -> None:
        self.calls = 0

    def invoke(self, messages: list):
        self.calls += 1
        raise RuntimeError("upstream is down")


# ── HyDE ──────────────────────────────────────────────────────────────────────


def test_hyde_embeds_the_hypothetical_passage_instead_of_the_question() -> None:
    llm = FakeLlm("From July the deduction is capped at fifty thousand dollars.")

    plan = HydeTransform(llm, model="deepseek-v4").expand("does negative gearing survive?")

    assert plan.kind == "hyde"
    assert plan.queries == ["From July the deduction is capped at fifty thousand dollars."]
    assert plan.degraded is False
    # The question itself is deliberately not retrieved alongside it — HyDE is
    # the claim that the passage retrieves better, not a blend of the two.
    assert "does negative gearing survive?" not in plan.queries


def test_hyde_degrades_to_the_raw_question_when_the_model_fails() -> None:
    llm = BrokenLlm()

    plan = HydeTransform(llm, model="deepseek-v4").expand("what changed?")

    assert plan.queries == ["what changed?"]
    assert plan.degraded is True


def test_hyde_degrades_when_the_model_returns_an_empty_passage() -> None:
    plan = HydeTransform(FakeLlm("   "), model="deepseek-v4").expand("what changed?")

    assert plan.queries == ["what changed?"]
    assert plan.degraded is True


# ── multi-query ───────────────────────────────────────────────────────────────


def test_multi_query_keeps_the_original_question_first() -> None:
    llm = FakeLlm(json.dumps({"queries": ["negative gearing changes", "gearing deduction cap"]}))

    plan = MultiQueryTransform(llm, model="deepseek-v4", variants=3).expand("what changed?")

    assert plan.queries[0] == "what changed?"
    assert plan.queries == ["what changed?", "negative gearing changes", "gearing deduction cap"]


def test_multi_query_drops_a_variant_that_echoes_the_question() -> None:
    llm = FakeLlm(json.dumps({"queries": ["What changed?", "gearing deduction cap"]}))

    plan = MultiQueryTransform(llm, model="deepseek-v4", variants=3).expand("what changed?")

    assert plan.queries == ["what changed?", "gearing deduction cap"]


def test_multi_query_caps_the_expansion_at_the_configured_variant_count() -> None:
    """``variants`` counts paraphrases; the original question is always extra."""
    llm = FakeLlm(json.dumps({"queries": ["one", "two", "three", "four", "five"]}))

    plan = MultiQueryTransform(llm, model="deepseek-v4", variants=2).expand("q")

    assert plan.queries == ["q", "one", "two"]


def test_multi_query_accepts_a_fenced_json_response() -> None:
    llm = FakeLlm('```json\n{"queries": ["gearing cap"]}\n```')

    plan = MultiQueryTransform(llm, model="deepseek-v4").expand("q")

    assert plan.queries == ["q", "gearing cap"]


@pytest.mark.parametrize(
    "content",
    ["not json at all", '["a", "b"]', '{"other": ["a"]}'],
)
def test_multi_query_degrades_on_an_unparseable_response(content: str) -> None:
    plan = MultiQueryTransform(FakeLlm(content), model="deepseek-v4").expand("q")

    assert plan.queries == ["q"]
    assert plan.degraded is True


# ── cache ─────────────────────────────────────────────────────────────────────


def test_a_cached_expansion_costs_no_second_call(tmp_path) -> None:
    llm = FakeLlm("hypothetical passage")
    transform = HydeTransform(llm, model="deepseek-v4", cache_dir=tmp_path)

    first = transform.expand("q")
    second = transform.expand("q")

    assert len(llm.calls) == 1
    assert second.queries == first.queries
    assert second.cached is True and first.cached is False


def test_the_cache_key_separates_models_and_variant_counts(tmp_path) -> None:
    """A different model writes a different passage, so it must not be reused."""
    one = HydeTransform(FakeLlm(), model="model-a", cache_dir=tmp_path)
    two = HydeTransform(FakeLlm(), model="model-b", cache_dir=tmp_path)
    assert one.cache_key("q") != two.cache_key("q")

    three = MultiQueryTransform(FakeLlm(), model="model-a", cache_dir=tmp_path, variants=2)
    four = MultiQueryTransform(FakeLlm(), model="model-a", cache_dir=tmp_path, variants=5)
    assert three.cache_key("q") != four.cache_key("q")


def test_a_degraded_expansion_is_never_cached(tmp_path) -> None:
    """Otherwise one outage would pin the raw question for that query forever."""
    broken = HydeTransform(BrokenLlm(), model="deepseek-v4", cache_dir=tmp_path)
    assert broken.expand("q").degraded is True

    working = HydeTransform(FakeLlm("passage"), model="deepseek-v4", cache_dir=tmp_path)
    plan = working.expand("q")

    assert plan.queries == ["passage"]
    assert plan.degraded is False


def test_a_corrupt_cache_file_is_a_miss_not_a_crash(tmp_path) -> None:
    transform = HydeTransform(FakeLlm("passage"), model="deepseek-v4", cache_dir=tmp_path)
    (tmp_path / f"{transform.cache_key('q')}.json").write_text("{not json", encoding="utf-8")

    assert transform.expand("q").queries == ["passage"]


# ── trace detail ──────────────────────────────────────────────────────────────


def test_a_degraded_plan_reports_the_failure_rather_than_a_transform() -> None:
    detail = QueryPlan(kind="hyde", queries=["q"], degraded=True).detail()

    assert "failed" in detail
    assert "question as asked" in detail


def test_a_cached_plan_says_so_rather_than_claiming_a_call() -> None:
    assert "cached" in QueryPlan(kind="hyde", queries=["p"], cached=True).detail()
    assert "generated" in QueryPlan(kind="hyde", queries=["p"]).detail()


# ── factory ───────────────────────────────────────────────────────────────────


class _Settings:
    deepseek_api_key = "key"
    deepseek_model = "deepseek-v4"
    deepseek_base_url = None
    llm_timeout_seconds = 120.0
    query_cache_dir = None
    multi_query_variants = 3


def test_no_transform_configured_means_retrieve_on_the_question() -> None:
    assert build_query_transform(None, _Settings()) is None
    assert build_query_transform("", _Settings()) is None


def test_an_unknown_transform_name_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown query transform"):
        build_query_transform("magic", _Settings())


def test_a_transform_without_an_api_key_says_which_key_it_needs() -> None:
    settings = _Settings()
    settings.deepseek_api_key = ""

    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
        build_query_transform("hyde", settings)

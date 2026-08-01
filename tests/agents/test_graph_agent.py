"""GraphRagAgent routing, evidence assembly, and citation-derived references.

The store and LLM are fakes: these tests pin the agent's contract — route
dispatch, label numbering, reference construction from cited labels, and the
pseudo-chunk context the eval runner scores against — not Cypher.
"""

from __future__ import annotations

import json

from src.agents.graph_agent import GraphRagAgent, question_terms
from src.agents.models import RagQuestionRequest
from src.rag.graph_models import GraphClaim, GraphCommunity, GraphEntity


CLAIM = GraphClaim(
    id="claim:1",
    text="Negative gearing is capped from July 2027.",
    entities=["negative gearing"],
    chunk_id="chunk:vid1:3",
    video_id="vid1",
    source_url="https://www.youtube.com/watch?v=vid1",
    video_title="Budget special",
    upload_date="2026-06-10",
    start_seconds=120.0,
    end_seconds=180.0,
)

LATER_CLAIM = CLAIM.model_copy(
    update={
        "id": "claim:2",
        "text": "Investors have adapted to the new rules.",
        "chunk_id": "chunk:vid2:7",
        "video_id": "vid2",
        "source_url": "https://www.youtube.com/watch?v=vid2",
        "upload_date": "2026-06-16",
    }
)


class FakeStore:
    def __init__(self) -> None:
        self.claims = [CLAIM, LATER_CLAIM]
        self.communities_list = [
            GraphCommunity(
                id=0,
                entity_ids=["negative-gearing"],
                entity_names=["negative gearing"],
                summary="The channel argues the budget reshapes investor strategy.",
                claim_count=2,
            )
        ]

    def resolve_entities(self, terms, limit=6):
        return [GraphEntity(id="negative-gearing", name="negative gearing", mentions=5)]

    def claims_about(self, entity_ids, limit=40, hops=1):
        return self.claims if entity_ids else []

    def communities(self):
        return self.communities_list

    def top_claims_for_community(self, community_id, limit=12):
        return self.claims[:limit]


class RoutedLLM:
    """First call answers the router, second call answers the question."""

    def __init__(self, route: str, answer: str) -> None:
        self.route = route
        self.answer = answer
        self.prompts: list[str] = []

    def invoke(self, messages):
        self.prompts.append("\n".join(str(m.content) for m in messages))
        if len(self.prompts) == 1:
            content = json.dumps({"route": self.route, "entities": ["negative gearing"]})
        else:
            content = self.answer
        return type("R", (), {"content": content})()


def make_agent(route: str, answer: str) -> tuple[GraphRagAgent, RoutedLLM]:
    llm = RoutedLLM(route, answer)
    return GraphRagAgent(llm, FakeStore(), context_provider=None), llm


def test_question_terms_drops_stopwords() -> None:
    terms = question_terms("How did the channel's view on rate cuts change?")
    assert "channel" not in terms
    assert "cuts" in terms


def test_local_route_builds_claim_references() -> None:
    agent, llm = make_agent("local", "Capped from July 2027 [g1].")
    answer = agent.answer(RagQuestionRequest(question="Is negative gearing capped?"))

    assert agent.last_route == "local"
    assert agent.last_llm_calls == 2
    assert [ref.label for ref in answer.references] == ["[g1]"]
    ref = answer.references[0]
    assert ref.video_id == "vid1"
    assert ref.chunk_index == 3
    assert "t=120s" in str(ref.timestamp_url)
    # The claims went into the evidence block the answer was grounded in.
    assert "[g1] (2026-06-10 — Budget special)" in llm.prompts[1]


def test_uncited_labels_produce_no_references() -> None:
    agent, _llm = make_agent("local", "The corpus does not say.")
    answer = agent.answer(RagQuestionRequest(question="Is X true?"))
    assert answer.references == []


def test_temporal_route_orders_timeline_and_context() -> None:
    agent, llm = make_agent("temporal", "Shifted by June [g1][g2].")
    answer = agent.answer(RagQuestionRequest(question="How did the stance evolve?"))

    assert agent.last_route == "temporal"
    assert "Claim timeline (oldest first):" in llm.prompts[1]
    assert {ref.video_id for ref in answer.references} == {"vid1", "vid2"}
    # last_context carries dated claim texts as the judge's contexts.
    texts = [chunk.text for chunk in agent.last_context.retrieved_chunks]
    assert any(text.startswith("[2026-06-10]") for text in texts)


def test_global_route_uses_community_summaries() -> None:
    agent, llm = make_agent("global", "One main theme [c1], e.g. [g1].")
    answer = agent.answer(RagQuestionRequest(question="What themes recur?"))

    assert agent.last_route == "global"
    assert "Community (negative gearing)" in llm.prompts[1]
    # [c1] is not a citable video reference; [g1] is.
    assert [ref.label for ref in answer.references] == ["[g1]"]
    texts = [chunk.text for chunk in agent.last_context.retrieved_chunks]
    assert any("Community summary [c1]" in text for text in texts)


def test_router_failure_degrades_to_local() -> None:
    class BrokenRouterLLM(RoutedLLM):
        def invoke(self, messages):
            self.prompts.append("x")
            if len(self.prompts) == 1:
                return type("R", (), {"content": "not json at all"})()
            return type("R", (), {"content": "answer [g1]"})()

    llm = BrokenRouterLLM("ignored", "ignored")
    agent = GraphRagAgent(llm, FakeStore(), context_provider=None)
    agent.answer(RagQuestionRequest(question="Anything?"))
    assert agent.last_route == "local"


def test_resolution_falls_back_to_content_words() -> None:
    class PhraseBlindStore(FakeStore):
        """Matches single words only, like CONTAINS against short entity names."""

        def resolve_entities(self, terms, limit=6):
            if any(" " in term for term in terms):
                return []
            return super().resolve_entities(terms, limit)

    llm = RoutedLLM("temporal", "Evolved [g1].")
    agent = GraphRagAgent(llm, PhraseBlindStore(), context_provider=None)

    # The router names a phrase no entity name contains; the word-level
    # fallback still anchors the subgraph and produces a timeline.
    import json as _json

    def routed_invoke(messages):
        llm.prompts.append("\n".join(str(m.content) for m in messages))
        if len(llm.prompts) == 1:
            content = _json.dumps({"route": "temporal", "entities": ["budget tax changes"]})
        else:
            content = "Evolved [g1]."
        return type("R", (), {"content": content})()

    llm.invoke = routed_invoke
    answer = agent.answer(RagQuestionRequest(question="How did gearing evolve?"))
    assert answer.references, "fallback resolution should surface timeline claims"


def test_answer_records_a_route_and_llm_trace() -> None:
    agent = GraphRagAgent(RoutedLLM("temporal", "Timeline [g1]."), FakeStore())
    agent.answer(RagQuestionRequest(question="how did views change over time?"))

    phases = [step.phase for step in agent.last_trace]
    assert phases == ["route", "retrieve", "llm"]
    assert agent.last_trace[0].label == "Route → temporal"
    assert "negative gearing" in agent.last_trace[0].detail
    assert "dated claims" in agent.last_trace[1].detail
    assert all(isinstance(step.elapsed_ms, int) for step in agent.last_trace)


def test_global_route_trace_counts_communities() -> None:
    agent = GraphRagAgent(RoutedLLM("global", "Themes [c1]."), FakeStore())
    agent.answer(RagQuestionRequest(question="what are the main themes?"))

    reduce_step = agent.last_trace[1]
    assert reduce_step.label == "Community summaries"
    assert "1 community summaries" in reduce_step.detail


def test_global_route_detail_does_not_claim_entity_anchoring() -> None:
    """``_global_evidence`` takes no terms, so nothing may anchor the graph."""

    class NoEntityRouter(RoutedLLM):
        def invoke(self, messages):
            self.prompts.append("router" if not self.prompts else "answer")
            if len(self.prompts) == 1:
                return type("R", (), {"content": json.dumps({"route": "global", "entities": []})})()
            return type("R", (), {"content": "Themes [c1]."})()

    agent = GraphRagAgent(NoEntityRouter("global", "unused"), FakeStore())
    agent.answer(RagQuestionRequest(question="what are the main themes?"))

    route_step = agent.last_trace[0]
    assert route_step.label == "Route → global"
    assert "content words" not in route_step.detail
    assert "community summaries" in route_step.detail
    assert agent.last_trace[1].label == "Community summaries"


def test_local_route_detail_still_names_the_anchoring_fallback() -> None:
    class NoEntityRouter(RoutedLLM):
        def invoke(self, messages):
            self.prompts.append("router" if not self.prompts else "answer")
            if len(self.prompts) == 1:
                return type("R", (), {"content": json.dumps({"route": "local", "entities": []})})()
            return type("R", (), {"content": "Answer [g1]."})()

    agent = GraphRagAgent(NoEntityRouter("local", "unused"), FakeStore())
    agent.answer(RagQuestionRequest(question="what did they say about gearing?"))

    assert "content words will anchor the graph" in agent.last_trace[0].detail


def test_router_failure_is_distinguishable_from_a_local_classification() -> None:
    """A degraded route must not read like a genuine "local" answer."""

    class FailingRouterLLM(RoutedLLM):
        def invoke(self, messages):
            self.prompts.append("router" if not self.prompts else "answer")
            if len(self.prompts) == 1:
                raise RuntimeError("router timed out")
            return type("R", (), {"content": "Answer [g1]."})()

    agent = GraphRagAgent(FailingRouterLLM("local", "unused"), FakeStore())
    agent.answer(RagQuestionRequest(question="what did they say about gearing?"))

    route_step = agent.last_trace[0]
    assert route_step.label == "Route → local (router failed)"
    assert "router timed out" in route_step.detail

    genuine = GraphRagAgent(RoutedLLM("local", "Answer [g1]."), FakeStore())
    genuine.answer(RagQuestionRequest(question="what did they say about gearing?"))
    assert genuine.last_trace[0].label == "Route → local"
    assert "failed" not in genuine.last_trace[0].detail


def test_route_error_clears_on_the_next_successful_route() -> None:
    agent = GraphRagAgent(RoutedLLM("temporal", "Timeline [g1]."), FakeStore())
    agent.last_route_error = "stale failure"
    agent.answer(RagQuestionRequest(question="how did views change?"))

    assert agent.last_route_error is None
    assert agent.last_trace[0].label == "Route → temporal"


def test_trace_resets_between_answers() -> None:
    agent = GraphRagAgent(RoutedLLM("temporal", "Timeline [g1]."), FakeStore())
    agent.answer(RagQuestionRequest(question="first?"))
    first_len = len(agent.last_trace)

    agent.llm = RoutedLLM("temporal", "Timeline [g1].")
    agent.answer(RagQuestionRequest(question="second?"))

    assert len(agent.last_trace) == first_len

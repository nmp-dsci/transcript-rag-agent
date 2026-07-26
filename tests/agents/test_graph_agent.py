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

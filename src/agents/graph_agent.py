"""GraphRAG answer agent (P4): route → graph evidence → cited answer.

The fourth answer path beside ``rag_llm``, recursive, and ``rag_agent``. One
router call classifies the question (per the s05 review decision the graph is
built over the whole corpus and *scoped per question at query time*):

* ``local`` — entity-anchored: claims from the knowledge graph subgraph plus
  the existing vector retrieval, answered together. Graph claims carry their
  source chunk's video id, timestamps and ``upload_date``, so citations stay
  deep-linkable exactly like vector answers.
* ``global`` — corpus-level: reduce over the pre-built community summaries
  (Full GraphRAG summarizes at index time) plus representative dated claims.
* ``temporal`` — trend: the entity's claim timeline ordered by
  ``upload_date``, narrated as a dated story.

Evidence is labelled — ``[gN]`` graph claims, ``[N]`` transcript chunks,
``[cN]`` community summaries — and references are built from the labels the
answer actually cites, mirroring the fallback-reference behaviour of the
vector agent rather than trusting the model to emit reference JSON.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

import time

from src.agents.models import (
    RagAnswerReference,
    RagQuestionRequest,
    RagTranscriptAnswer,
    TraceStep,
)
from src.agents.prompts import (
    GRAPH_ANSWER_SYSTEM_PROMPT,
    GRAPH_GLOBAL_SYSTEM_PROMPT,
    GRAPH_ROUTER_SYSTEM_PROMPT,
    GRAPH_TEMPORAL_SYSTEM_PROMPT,
    build_graph_router_prompt,
)
from src.config import Settings
from src.rag.context import MultiTranscriptRagContextProvider
from src.rag.graph_models import GraphClaim, GraphCommunity, GraphEntity
from src.rag.graph_store import GraphStore
from src.rag.models import RetrievedChunk
from src.rag.references import youtube_timestamp_url

logger = logging.getLogger(__name__)

ROUTES = ("local", "global", "temporal")

#: Evidence caps, as __init__ defaults below. Named so the System Design tab's
#: flow diagram can cite the exact numbers each route runs with, rather than
#: a description that silently drifts if these ever change.
DEFAULT_MAX_CLAIMS = 30
DEFAULT_TIMELINE_CLAIMS = 60
DEFAULT_MAX_COMMUNITIES = 12

_STOPWORDS = frozenset(
    "a an and are as at be but by did do does for from has have how in is it of on "
    "or that the their this to was we what when where which who why will with you "
    "your channel video videos corpus say says said stance view views".split()
)

#: A placeholder URL for evidence with no single video behind it (community
#: summaries). Keeps the pseudo-chunk contract satisfied without inventing a
#: citable link — summaries are never emitted as references.
_CORPUS_URL = "https://www.youtube.com/"


@dataclass
class GraphContext:
    """Duck-typed stand-in for TranscriptContext: what runners and logs read."""

    context_text: str
    retrieved_chunks: list[RetrievedChunk] = field(default_factory=list)
    selected_transcripts: list = field(default_factory=list)
    context_mode: str = "graph"
    cache_status: str = "graph"
    top_k: int | None = None


@dataclass
class _Evidence:
    """Labelled evidence for one answer call."""

    block: str
    claims: dict[str, GraphClaim] = field(default_factory=dict)  # label -> claim
    chunks: dict[str, RetrievedChunk] = field(default_factory=dict)  # label -> chunk
    communities: dict[str, GraphCommunity] = field(default_factory=dict)


class GraphRagAgent:
    def __init__(
        self,
        llm,
        store: GraphStore,
        context_provider: MultiTranscriptRagContextProvider | None = None,
        max_claims: int = DEFAULT_MAX_CLAIMS,
        timeline_claims: int = DEFAULT_TIMELINE_CLAIMS,
        max_communities: int = DEFAULT_MAX_COMMUNITIES,
    ) -> None:
        self.llm = llm
        self.store = store
        self.context_provider = context_provider
        self.max_claims = max_claims
        self.timeline_claims = timeline_claims
        self.max_communities = max_communities
        self.last_context: GraphContext | None = None
        self.last_route: str | None = None
        # Why the most recent route is what it is: a router exception degrades
        # to "local", and the trace has to say so rather than read like a
        # genuine local classification.
        self.last_route_error: str | None = None
        self.last_llm_calls: int = 0
        # Ordered TraceSteps for the most recent answer() call, so the runner
        # can persist how the route decision and evidence assembly actually went.
        self.last_trace: list[TraceStep] = []

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        context_provider: MultiTranscriptRagContextProvider | None = None,
        store: GraphStore | None = None,
    ) -> "GraphRagAgent":
        from langchain_openai import ChatOpenAI

        kwargs: dict[str, object] = {
            "api_key": settings.deepseek_api_key,
            "model": settings.deepseek_model,
        }
        if settings.deepseek_base_url:
            kwargs["base_url"] = settings.deepseek_base_url
        return cls(
            ChatOpenAI(**kwargs),
            store or GraphStore.from_settings(settings),
            context_provider,
        )

    # ── routing ───────────────────────────────────────────────────────────

    def route(self, question: str) -> tuple[str, list[str]]:
        """Classify the question and name its entities; degrade to local."""
        self.last_llm_calls += 1
        self.last_route_error = None
        try:
            response = self.llm.invoke(
                [_system(GRAPH_ROUTER_SYSTEM_PROMPT), _human(build_graph_router_prompt(question))]
            )
            data = _json_object(str(getattr(response, "content", response) or ""))
            route = str(data.get("route", "local")).strip().lower()
            entities = [
                str(term).strip() for term in (data.get("entities") or []) if str(term).strip()
            ]
            if route not in ROUTES:
                route = "local"
            return route, entities
        except Exception as exc:
            logger.warning("Graph router failed (%s); defaulting to local", exc)
            self.last_route_error = str(exc)
            return "local", []

    # ── answering ─────────────────────────────────────────────────────────

    def answer(self, request: RagQuestionRequest) -> RagTranscriptAnswer:
        self.last_llm_calls = 0
        self.last_trace = []
        route_started = time.monotonic()
        route, entity_terms = self.route(request.question)
        self.last_route = route
        route_error = self.last_route_error
        self.last_trace.append(
            TraceStep(
                phase="route",
                label=f"Route → {route}" + (" (router failed)" if route_error else ""),
                detail=(
                    f"router call failed ({route_error}); defaulted to local with no "
                    "named entities — content words will anchor the graph"
                    if route_error
                    else f"router named entities: {', '.join(entity_terms)}"
                    if entity_terms
                    else "router named no entities; content words will anchor the graph"
                ),
                model=getattr(self.llm, "model_name", None),
                elapsed_ms=int((time.monotonic() - route_started) * 1000),
            )
        )
        if route == "global":
            evidence = self._global_evidence()
            system_prompt = GRAPH_GLOBAL_SYSTEM_PROMPT
        elif route == "temporal":
            evidence = self._temporal_evidence(request.question, entity_terms)
            system_prompt = GRAPH_TEMPORAL_SYSTEM_PROMPT
        else:
            evidence = self._local_evidence(request, entity_terms)
            system_prompt = GRAPH_ANSWER_SYSTEM_PROMPT

        self.last_llm_calls += 1
        answer_started = time.monotonic()
        response = self.llm.invoke(
            [
                _system(system_prompt),
                _human(f"Question: {request.question}\n\nEvidence:\n{evidence.block}"),
            ]
        )
        answer_text = str(getattr(response, "content", response) or "").strip()
        self.last_trace.append(
            TraceStep(
                phase="llm",
                label="Answer",
                detail=f"one narration call over the {route} route's evidence",
                model=getattr(self.llm, "model_name", None),
                elapsed_ms=int((time.monotonic() - answer_started) * 1000),
            )
        )
        self.last_context = GraphContext(
            context_text=evidence.block,
            retrieved_chunks=_evidence_chunks(evidence),
            top_k=request.top_k,
        )
        return RagTranscriptAnswer(
            question=request.question,
            answer=answer_text,
            references=_references_for(answer_text, evidence),
        )

    # ── evidence assembly ─────────────────────────────────────────────────

    def _resolve(self, question: str, entity_terms: list[str]) -> list[str]:
        terms = entity_terms or question_terms(question)
        entities = self.store.resolve_entities(terms)
        if not entities:
            # Router phrases ("budget tax changes") rarely match entity names
            # verbatim; retry on the content words of the terms and question so
            # "budget" still anchors the subgraph. A single short content word
            # can coincidentally substring-match an unrelated entity's name, so
            # the fallback only keeps entities with a strong enough match
            # (one long word or two distinct words) instead of any match.
            words = question_terms(" ".join([*terms, question]))
            candidates = self.store.resolve_entities(words)
            entities = [entity for entity in candidates if _strong_fallback_match(entity, words)]
        return [entity.id for entity in entities]

    def _local_evidence(self, request: RagQuestionRequest, entity_terms: list[str]) -> _Evidence:
        resolve_started = time.monotonic()
        entity_ids = self._resolve(request.question, entity_terms)
        claims = self.store.claims_about(entity_ids, limit=self.max_claims)
        self.last_trace.append(
            TraceStep(
                phase="retrieve",
                label="Graph claims",
                detail=(
                    f"{len(entity_ids)} entities resolved → {len(claims)} claims "
                    f"(cap {self.max_claims}, 1-hop neighbours included)"
                ),
                elapsed_ms=int((time.monotonic() - resolve_started) * 1000),
            )
        )
        chunks: list[RetrievedChunk] = []
        if self.context_provider is not None:
            context = self.context_provider.get_context(
                question=request.question,
                source_url=str(request.source_url) if request.source_url else None,
                top_k=request.top_k,
                channel_id=request.channel_id,
                retrieval_mode=request.retrieval_mode,
            )
            chunks = list(context.retrieved_chunks or [])
            # The provider recorded its own filter/retrieve/rerank steps.
            self.last_trace.extend(getattr(context, "trace", None) or [])
        evidence = _Evidence(block="")
        lines: list[str] = []
        if claims:
            lines.append("Knowledge-graph claims:")
            for index, claim in enumerate(claims, 1):
                label = f"[g{index}]"
                evidence.claims[label] = claim
                lines.append(f"{label} {_claim_line(claim)}")
        if chunks:
            lines.append("")
            lines.append("Transcript chunks:")
            for index, chunk in enumerate(chunks, 1):
                label = f"[{index}]"
                evidence.chunks[label] = chunk
                lines.append(f"{label} ({chunk.video_id}) {chunk.text}")
        if not lines:
            lines.append("No graph or transcript evidence was found.")
        evidence.block = "\n".join(lines)
        return evidence

    def _global_evidence(self) -> _Evidence:
        reduce_started = time.monotonic()
        evidence = _Evidence(block="")
        lines: list[str] = []
        claim_index = 1
        # Leiden over a sparse entity graph produces a long tail of tiny
        # communities (a real corpus built 282 of them). The reduce call reads
        # only the biggest themes: communities() orders by claim_count, so the
        # cap keeps the strongest evidence and the context bounded.
        communities = [
            community
            for community in self.store.communities()
            if community.summary and community.claim_count
        ][: self.max_communities]
        for community_number, community in enumerate(communities, 1):
            community_label = f"[c{community_number}]"
            evidence.communities[community_label] = community
            names = ", ".join(community.entity_names[:8]) or "unnamed"
            lines.append(f"{community_label} Community ({names}):")
            lines.append(community.summary or "No summary available.")
            for claim in self.store.top_claims_for_community(community.id, limit=2):
                label = f"[g{claim_index}]"
                evidence.claims[label] = claim
                lines.append(f"  {label} {_claim_line(claim)}")
                claim_index += 1
            lines.append("")
        if not lines:
            lines.append("No communities are built yet. Run index-graph to build the graph.")
        evidence.block = "\n".join(lines).strip()
        self.last_trace.append(
            TraceStep(
                phase="retrieve",
                label="Community summaries",
                detail=(
                    f"{len(evidence.communities)} community summaries "
                    f"(cap {self.max_communities}) with "
                    f"{len(evidence.claims)} representative claims"
                ),
                elapsed_ms=int((time.monotonic() - reduce_started) * 1000),
            )
        )
        return evidence

    def _temporal_evidence(self, question: str, entity_terms: list[str]) -> _Evidence:
        timeline_started = time.monotonic()
        entity_ids = self._resolve(question, entity_terms)
        claims = self.store.claims_about(entity_ids, limit=self.timeline_claims, hops=0)
        self.last_trace.append(
            TraceStep(
                phase="retrieve",
                label="Claim timeline",
                detail=(
                    f"{len(entity_ids)} entities resolved → {len(claims)} dated claims "
                    f"(cap {self.timeline_claims}), ordered oldest first"
                ),
                elapsed_ms=int((time.monotonic() - timeline_started) * 1000),
            )
        )
        evidence = _Evidence(block="")
        lines = ["Claim timeline (oldest first):"]
        for index, claim in enumerate(claims, 1):
            label = f"[g{index}]"
            evidence.claims[label] = claim
            lines.append(f"{label} {_claim_line(claim)}")
        if not claims:
            lines.append("No dated claims found for the question's entities.")
        evidence.block = "\n".join(lines)
        return evidence


# ── helpers ──────────────────────────────────────────────────────────────────


def question_terms(question: str) -> list[str]:
    """Fallback entity terms when the router names none: content words."""
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9'-]{3,}", question.lower())
    return [word for word in words if word not in _STOPWORDS][:8]


#: A single content word this long or longer is specific enough on its own to
#: anchor the fallback; shorter words need a second, independent match so a
#: common-word coincidence can't anchor to an unrelated entity.
_STRONG_FALLBACK_TOKEN_LEN = 6


def _strong_fallback_match(entity: GraphEntity, words: list[str]) -> bool:
    """Whether the content-word fallback's match on ``entity`` is strong enough.

    Requires either one long word or two distinct words to appear in the
    entity's name/aliases, rather than any single short word coincidentally
    matching a substring of an unrelated entity's name.
    """
    names = [entity.name.lower(), *(alias.lower() for alias in entity.aliases)]
    matched = {word for word in words if any(word in name for name in names)}
    if any(len(word) >= _STRONG_FALLBACK_TOKEN_LEN for word in matched):
        return True
    return len(matched) >= 2


def _claim_line(claim: GraphClaim) -> str:
    date = claim.upload_date or "undated"
    title = f" — {claim.video_title}" if claim.video_title else ""
    return f"({date}{title}) {claim.text}"


def _evidence_chunks(evidence: _Evidence) -> list[RetrievedChunk]:
    """Evidence as pseudo-chunks: what the judge scores faithfulness against.

    Graph claims and community summaries become the RAGAS contexts (the s05
    design: measure the answer against what the graph actually retrieved), and
    claim-backed chunk ids keep the deterministic IR metrics computable.
    """
    chunks: list[RetrievedChunk] = []
    for claim in evidence.claims.values():
        chunks.append(
            RetrievedChunk(
                transcript_id=f"transcript:{claim.video_id}",
                video_id=claim.video_id,
                source_url=claim.source_url or _CORPUS_URL,
                chunk_index=claim.chunk_index,
                text=f"[{claim.upload_date or 'undated'}] {claim.text}",
                start_seconds=claim.start_seconds,
                end_seconds=claim.end_seconds,
                title=claim.video_title,
                upload_date=claim.upload_date,
            )
        )
    for label, community in evidence.communities.items():
        if not community.summary:
            continue
        chunks.append(
            RetrievedChunk(
                transcript_id="transcript:corpus",
                video_id="corpus",
                source_url=_CORPUS_URL,
                chunk_index=community.id,
                text=f"Community summary {label}: {community.summary}",
            )
        )
    chunks.extend(evidence.chunks.values())
    return chunks


def _references_for(answer_text: str, evidence: _Evidence) -> list[RagAnswerReference]:
    """References for the labels the answer cites, graph claims first.

    Community summaries have no single video behind them, so ``[cN]`` labels
    never become references; the claims beneath them do.
    """
    cited = set(re.findall(r"\[(?:g|c)?\d+\]", answer_text))
    references: list[RagAnswerReference] = []
    for label, claim in evidence.claims.items():
        if label not in cited or not claim.source_url:
            continue
        references.append(
            RagAnswerReference(
                label=label,
                source_url=claim.source_url,
                timestamp_url=youtube_timestamp_url(claim.source_url, claim.start_seconds),
                start_seconds=claim.start_seconds,
                end_seconds=claim.end_seconds,
                chunk_index=claim.chunk_index,
                video_id=claim.video_id,
            )
        )
    for label, chunk in evidence.chunks.items():
        if label not in cited:
            continue
        references.append(
            RagAnswerReference(
                label=label,
                source_url=chunk.source_url,
                timestamp_url=youtube_timestamp_url(str(chunk.source_url), chunk.start_seconds),
                start_seconds=chunk.start_seconds,
                end_seconds=chunk.end_seconds,
                chunk_index=chunk.chunk_index,
                video_id=chunk.video_id,
            )
        )
    return references


def _system(content: str):
    from langchain_core.messages import SystemMessage

    return SystemMessage(content=content)


def _human(content: str):
    from langchain_core.messages import HumanMessage

    return HumanMessage(content=content)


def _json_object(content: str) -> dict:
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("router response must be a JSON object")
    return value

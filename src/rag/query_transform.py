"""Query-side retrieval transforms: HyDE and multi-query fan-out.

Both attack the same weakness from the same side. A user's question and the
transcript passage that answers it are written in different registers — the
question is short and abstract ("does negative gearing survive the changes?"),
the passage is long and concrete ("...so from July they're capping the
deduction at...") — and a single embedding of the question has to bridge that
gap on its own.

* **HyDE** (Gao et al. 2022) closes the gap by moving the query: the model
  writes the passage that *would* answer the question, and retrieval embeds
  that instead. The hypothetical passage may be factually wrong — it is never
  shown to anyone and never enters the answer — but it is written in the
  register of the corpus, which is what the vector search matches on.
* **Multi-query** closes it by widening: the model paraphrases the question
  several ways, each phrasing retrieves independently, and the ranked lists are
  RRF-fused. A chunk that several phrasings agree on rises; one that only the
  original wording found still gets a chance.

Two properties make these usable as *measured* variants rather than demos:

* **Cache by (kind, model, question)** — the expansion for a question is
  written to ``.yt-agent/query_cache`` and reused, so sweeping the golden set
  again costs nothing and produces the identical retrieval. Without that, an
  ablation column that calls an LLM per question would be neither cheap nor
  reproducible, and re-running it would silently change the numbers.
* **Degrade to the raw question** — an LLM failure returns the question as
  asked with ``degraded`` set, so retrieval always runs. The flag travels into
  the answer trace: a degraded expansion reads as the fallback it was, never as
  a successful transform.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from src.agents.prompts import (
    HYDE_SYSTEM_PROMPT,
    MULTI_QUERY_PROMPT,
    build_hyde_prompt,
    build_multi_query_prompt,
)

logger = logging.getLogger(__name__)

#: Transform names accepted by ``--query-transform`` and ``build_query_transform``.
QUERY_TRANSFORMS = ("hyde", "multi_query")


@dataclass(frozen=True)
class QueryPlan:
    """What retrieval should actually search for, and how it was arrived at.

    ``queries`` is the ordered list of strings to embed — one for HyDE (the
    hypothetical passage), several for multi-query (the original first, then
    its paraphrases). It is never empty: a failed expansion falls back to the
    question as asked and says so through ``degraded``.
    """

    kind: str
    queries: list[str] = field(default_factory=list)
    elapsed_ms: int | None = None
    #: The LLM call failed and ``queries`` is the untransformed question.
    degraded: bool = False
    #: Served from the disk cache, so no LLM call was made for this question.
    cached: bool = False

    @property
    def label(self) -> str:
        return {"hyde": "HyDE hypothetical passage", "multi_query": "Multi-query expansion"}.get(
            self.kind, self.kind
        )

    def detail(self) -> str:
        """A one-line description of what this expansion did, for a trace step.

        Reports only what happened: a degraded plan says the transform failed,
        a cached one says no call was made, and the queries quoted are the ones
        retrieval is about to embed.
        """
        if self.degraded:
            return f"{self.kind} failed; retrieval ran on the question as asked"
        source = "cached" if self.cached else "generated"
        if self.kind == "hyde":
            probe = self.queries[0] if self.queries else ""
            return (
                f'{source} hypothetical passage embedded instead of the question: "{_clip(probe)}"'
            )
        listed = " | ".join(f'"{_clip(query, 60)}"' for query in self.queries)
        return f"{source} {len(self.queries)} query variants retrieved separately: {listed}"


def _clip(text: str, limit: int = 160) -> str:
    collapsed = " ".join(text.split())
    return collapsed if len(collapsed) <= limit else f"{collapsed[: limit - 1]}…"


class QueryTransform(Protocol):
    """Expands one question into the queries retrieval should embed."""

    kind: str

    def expand(self, question: str) -> QueryPlan: ...


class ChatModel(Protocol):
    def invoke(self, messages: list) -> object: ...


class _CachedQueryTransform:
    """Shared cache and failure handling for the LLM-backed transforms."""

    kind = "none"

    def __init__(
        self,
        llm: ChatModel,
        model: str | None = None,
        cache_dir: Path | None = None,
    ) -> None:
        self.llm = llm
        self.model = model
        self.cache_dir = cache_dir
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)

    # ── subclass contract ────────────────────────────────────────────────────

    def _cache_salt(self) -> str:
        """Anything besides kind/model/question that changes the expansion."""
        return ""

    def _generate(self, question: str) -> list[str]:
        """The expanded queries, or raise for the caller to degrade."""
        raise NotImplementedError

    # ── public ───────────────────────────────────────────────────────────────

    def expand(self, question: str) -> QueryPlan:
        started = time.monotonic()
        cached = self._read_cache(question)
        if cached is not None:
            return QueryPlan(
                kind=self.kind,
                queries=cached,
                elapsed_ms=int((time.monotonic() - started) * 1000),
                cached=True,
            )
        try:
            queries = [query for query in self._generate(question) if query.strip()]
        except Exception as exc:  # noqa: BLE001 - a failed expansion must still retrieve
            logger.warning(
                "%s expansion failed; retrieving on the raw question: %s", self.kind, exc
            )
            return QueryPlan(
                kind=self.kind,
                queries=[question],
                elapsed_ms=int((time.monotonic() - started) * 1000),
                degraded=True,
            )
        if not queries:
            logger.warning(
                "%s expansion returned nothing; retrieving on the raw question", self.kind
            )
            return QueryPlan(
                kind=self.kind,
                queries=[question],
                elapsed_ms=int((time.monotonic() - started) * 1000),
                degraded=True,
            )
        self._write_cache(question, queries)
        return QueryPlan(
            kind=self.kind,
            queries=queries,
            elapsed_ms=int((time.monotonic() - started) * 1000),
        )

    # ── cache ────────────────────────────────────────────────────────────────

    def cache_key(self, question: str) -> str:
        """Cache identity: the transform, the model, its options, the question.

        The model is part of the key because a different model writes a
        different hypothetical passage, and reusing one across models would
        attribute one model's retrieval to another.
        """
        material = f"{self.kind}|{self.model or ''}|{self._cache_salt()}|{question}"
        return hashlib.sha256(material.encode("utf-8")).hexdigest()[:32]

    def _cache_path(self, question: str) -> Path | None:
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"{self.cache_key(question)}.json"

    def _read_cache(self, question: str) -> list[str] | None:
        path = self._cache_path(question)
        if path is None or not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        queries = data.get("queries") if isinstance(data, dict) else None
        if not isinstance(queries, list):
            return None
        usable = [str(query) for query in queries if str(query).strip()]
        return usable or None

    def _write_cache(self, question: str, queries: list[str]) -> None:
        path = self._cache_path(question)
        if path is None:
            return
        payload = {
            "kind": self.kind,
            "model": self.model,
            "question": question,
            "queries": queries,
        }
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    # ── llm ──────────────────────────────────────────────────────────────────

    def _invoke(self, system_prompt: str, user_prompt: str) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage

        response = self.llm.invoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        )
        return str(getattr(response, "content", response) or "")


class HydeTransform(_CachedQueryTransform):
    """Embed a hypothetical answer passage instead of the question.

    Returns exactly one query — the generated passage. The original question is
    deliberately *not* retrieved alongside it: HyDE is the claim that the
    passage retrieves better than the question, and fusing the two would
    measure a blend rather than the technique.
    """

    kind = "hyde"

    def _generate(self, question: str) -> list[str]:
        passage = self._invoke(HYDE_SYSTEM_PROMPT, build_hyde_prompt(question)).strip()
        if not passage:
            raise ValueError("HyDE returned an empty passage")
        return [passage]


class MultiQueryTransform(_CachedQueryTransform):
    """Retrieve for several phrasings of the question and fuse the rankings.

    The original question is always the first query, so multi-query can only
    add recall to what the plain search already found — the paraphrases widen
    the net rather than replacing it. ``variants`` counts the paraphrases, so
    the expansion is at most ``1 + variants`` queries.
    """

    kind = "multi_query"

    def __init__(
        self,
        llm: ChatModel,
        model: str | None = None,
        cache_dir: Path | None = None,
        variants: int = 3,
    ) -> None:
        super().__init__(llm, model=model, cache_dir=cache_dir)
        self.variants = max(1, variants)

    def _cache_salt(self) -> str:
        return f"variants={self.variants}"

    def _generate(self, question: str) -> list[str]:
        content = self._invoke(
            MULTI_QUERY_PROMPT, build_multi_query_prompt(question, self.variants)
        )
        variants = _parse_queries(content)
        # The question itself is never a "variant" the model has to spend a slot
        # on, and a model that echoes it must not make it appear twice.
        queries = [question]
        seen = {question.strip().lower()}
        for variant in variants:
            key = variant.strip().lower()
            if key in seen:
                continue
            seen.add(key)
            queries.append(variant.strip())
            if len(queries) >= 1 + self.variants:
                break
        return queries


def _parse_queries(content: str) -> list[str]:
    """The ``queries`` list out of a multi-query response.

    Accepts the JSON object the prompt asks for, with or without a markdown
    fence. Raises ``ValueError`` on anything else so the caller degrades to the
    raw question rather than retrieving on a parse artefact.
    """
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"multi-query response was not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("multi-query response must be a JSON object")
    queries = data.get("queries")
    if not isinstance(queries, list):
        raise ValueError("multi-query response has no 'queries' list")
    return [str(query) for query in queries if str(query).strip()]


def build_query_transform(name: str | None, settings: Any) -> QueryTransform | None:
    """The named transform wired to the configured chat model, or ``None``.

    ``None``/``""`` means "retrieve on the question as asked", which is the
    behaviour every setup had before these existed — so a caller can pass a
    configured value straight through without branching.
    """
    if not name:
        return None
    if name not in QUERY_TRANSFORMS:
        raise ValueError(
            f"Unknown query transform: {name}. Choose one of: {', '.join(QUERY_TRANSFORMS)}"
        )
    if not getattr(settings, "deepseek_api_key", ""):
        # Worth naming here rather than letting an empty key surface as an
        # authentication error per question: these transforms are the only
        # part of a retrieval sweep that calls a model at all.
        raise ValueError(
            f"The {name} query transform needs DEEPSEEK_API_KEY — it rewrites each "
            "question with an LLM before retrieval."
        )
    from langchain_openai import ChatOpenAI

    kwargs: dict[str, Any] = {
        "api_key": settings.deepseek_api_key,
        "model": settings.deepseek_model,
        "timeout": settings.llm_timeout_seconds,
    }
    if settings.deepseek_base_url:
        kwargs["base_url"] = settings.deepseek_base_url
    llm = ChatOpenAI(**kwargs)
    cache_dir = settings.query_cache_dir
    if name == "hyde":
        return HydeTransform(llm, model=settings.deepseek_model, cache_dir=cache_dir)
    return MultiQueryTransform(
        llm,
        model=settings.deepseek_model,
        cache_dir=cache_dir,
        variants=settings.multi_query_variants,
    )

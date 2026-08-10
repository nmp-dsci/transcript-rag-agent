"""The live half of the held-out critique eval — stores, retrieval, one call.

:mod:`src.evals.critique` is pure arithmetic over findings. This module is what
produces those findings: it opens the stores with the held-out video excluded,
retrieves the criteria the artifact should be judged against, asks the answering
model for structured findings, scores them and writes the run.

Two things here are worth reading rather than skimming.

**Exclusion is set at construction, not per call.** Both stores are built with
``exclude_video_ids``, so every read path they own — the vector query, the BM25
``collection.get`` the context provider issues directly, the summary router that
decides which videos get searched at all, and neighbour expansion — is filtered
without any caller having to remember. The alternative, an argument threaded
through four call chains, is the version where one path forgets and the
experiment quietly measures nothing.

**The proof does not trust the filter.** After the run,
:func:`~src.evals.critique.held_out_leaks` re-scans every retrieved chunk id,
every video id and every citation for the held-out video by prefix, and the count
goes in the run file. A filter that stopped applying looks exactly like a filter
that is working, so the number a reviewer reads is produced by a check that would
still fail if every ``$nin`` in the stack were deleted.

Cost: **two LLM calls per setup** — one to produce the critique, one to pair its
findings against the held-out criteria. No RAGAS judge runs at all: three of the
four metrics are pure string and id arithmetic, and the fourth is the single
matching call. That is what makes it affordable for V5, V6 and V8 to re-run this
harness rather than treat it as a one-off number.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Callable, Sequence

from src.config import Settings
from src.evals.critique import (
    DEFAULT_MATCH_REPEATS,
    EXCLUSION_VERSION,
    TIMESTAMP_TOLERANCE_SECONDS,
    ChunkTextFn,
    Criterion,
    CriterionMatch,
    CritiqueDataset,
    EmbedFn,
    Finding,
    MatchFn,
    MatchResult,
    SetupCritique,
    build_run,
    load_critique_dataset,
    parse_findings,
    repeated_matcher,
    score_critique,
    verify_dataset,
)

#: The setup the baseline row measures: the summary-filtered chunk dump. Named
#: here rather than passed in, because "better than chunk-dumping" is the claim
#: this whole harness exists to make falsifiable — the baseline is part of the
#: experiment's definition, not a parameter of it.
BASELINE_SETUP = "rag_llm_filtered"

CRITIQUE_SYSTEM_PROMPT = """You review a document against criteria drawn from \
expert video transcripts.

You are given (a) the document under review and (b) numbered excerpts from expert \
transcripts. Work out, from the excerpts alone, the general rules those experts \
apply to this kind of document, then judge the document against them.

Rules:
- Base every finding on the excerpts. Do not use criteria you happen to know but \
that no excerpt states.
- State each finding's `criterion` as a general, checkable rule that would apply \
to any document of this kind — not as a comment about this document. Put the \
comment about this document in `detail`.
- Cite the excerpt(s) the rule came from. `quote` must be words copied verbatim \
from the excerpt, `video_id` and `start_seconds` must be that excerpt's own.
- Set `contested` to true only when two or more excerpts from DIFFERENT videos \
disagree about the rule, and cite both sides. Do not average conflicting advice \
into one finding.
- Do not repeat the same rule as two findings.

Return only JSON in this shape:
{"findings": [{"id": "f01", "criterion": "...", "detail": "...", \
"contested": false, "citations": [{"video_id": "...", "start_seconds": 0.0, \
"quote": "..."}]}]}"""


# ── resolvers ─────────────────────────────────────────────────────────────


def chunk_text_lookup(
    settings: Settings,
    *,
    tolerance: float = TIMESTAMP_TOLERANCE_SECONDS,
) -> ChunkTextFn:
    """``(video_id, seconds) -> the (chunk_id, text) pairs spoken around then``.

    Every chunk whose span covers the timestamp is a candidate, because chunks
    overlap by design and a quote near a boundary lives in both. Deliberately
    built on a store with **no** exclusion: this resolver is the provenance
    check, and it has to be able to see the held-out video in order to notice
    that something cited it.
    """
    from src.rag.embeddings import HuggingFaceEmbeddingModel
    from src.rag.storage import TranscriptChunkStore

    store = TranscriptChunkStore(
        settings.chroma_path,
        embedding_model=HuggingFaceEmbeddingModel(settings.embedding_model),
        collection_name=settings.chunk_collection,
    )
    cache: dict[str, list[tuple[float, float, str, str]]] = {}

    def spans(video_id: str) -> list[tuple[float, float, str, str]]:
        if video_id not in cache:
            cache[video_id] = [
                (
                    float(chunk.start_seconds or 0.0),
                    float(chunk.end_seconds or 0.0),
                    chunk.chunk_id,
                    chunk.text,
                )
                for chunk in store.chunks_for_videos([video_id])
            ]
        return cache[video_id]

    def lookup(video_id: str, seconds: float) -> list[tuple[str, str]]:
        return [
            (chunk_id, text)
            for start, end, chunk_id, text in spans(video_id)
            if start - tolerance <= seconds <= end + tolerance
        ]

    return lookup


def embedder(settings: Settings) -> EmbedFn:
    """The local sentence-transformers model, as a batch embed callable."""
    from src.rag.embeddings import HuggingFaceEmbeddingModel

    model = HuggingFaceEmbeddingModel(settings.embedding_model)

    def embed(texts: Sequence[str]) -> list[list[float]]:
        return model.embed_documents(list(texts))

    return embed


#: The prompt is deliberately left as it was when the committed run was scored.
#: An independent review put this matcher against an adversarial set — polarity
#: inversions ("always list every certification" vs "drop the ones that add
#: nothing"), homonyms ("number your sections" vs "put numbers on your claims"),
#: pairs sharing only a rationale ("recruiters skim"), and half-rules — and it
#: refused 9 of 9 without any of them being named here. Spelling them out would
#: be a change to the instrument that produced the committed score, and this
#: harness exists to make such changes visible: if it is ever edited, bump
#: :data:`MATCHER_VERSION` so cached pairings from the old wording are not
#: reused, and re-run rather than leaving the number and the prompt out of step.
MATCH_SYSTEM_PROMPT = """You decide which review criteria a critique applied.

You are given a numbered list of CRITERIA (rules an expert applies to a kind of
document) and a numbered list of FINDINGS (points a reviewer made about one
document). For each criterion, name the finding that applies that same underlying
rule, or null if none does.

A finding matches a criterion only when it asserts the same rule. Being about the
same subject is not enough: "link to the source code of each project" and "list
your profile links so people can contact you" are both about links and are
different rules. Judge the rule, not the wording, and not whether the finding is
correct about the document.

Each finding may be used for at most one criterion. If a finding is plausible for
two criteria, give it to the closer one and leave the other null.

Return only JSON in this shape:
{"matches": [{"criterion_id": "c01", "finding_id": "f03", "why": "both say ..."},
             {"criterion_id": "c02", "finding_id": null, "why": "not raised"}]}"""


#: Where resolved pairings are remembered. Beside the matrix cache and for the
#: same reason: gitignored derived state, one file per entry, safe to delete.
DEFAULT_MATCH_CACHE_DIR = Path(".yt-agent/critique_cache")

#: Bumped when the matcher prompt or the voting rule changes, so a stored
#: pairing produced under the old rules is not silently reused under the new.
MATCHER_VERSION = "v2-vote5"


def cached_matcher(
    inner: MatchFn,
    *,
    cache_dir: Path = DEFAULT_MATCH_CACHE_DIR,
    repeats: int = DEFAULT_MATCH_REPEATS,
) -> MatchFn:
    """Remember a resolved pairing, keyed by the exact matrix it resolved.

    The fingerprint is every criterion and every finding as text, plus the
    matcher version and the repeat count — so re-scoring an unchanged run is
    free *and* returns the identical number, while any change to what is being
    matched (or to how) recomputes. That is the second half of the fix for a
    non-deterministic scorer: voting narrows the noise, caching means a rerun
    does not re-roll it at all, and a score that moves is a score somebody
    changed something to move.
    """

    def match(criteria: Sequence[Criterion], findings: Sequence[Finding]) -> MatchResult:
        material = json.dumps(
            {
                "version": MATCHER_VERSION,
                "repeats": repeats,
                "criteria": [[c.id, c.criterion] for c in criteria],
                "findings": [[f.id, f.criterion, f.detail] for f in findings],
            },
            sort_keys=True,
        )
        key = hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]
        path = cache_dir / f"{key}.json"
        if path.exists():
            try:
                stored = json.loads(path.read_text(encoding="utf-8"))
                return _result_from_cache(stored, criteria, findings)
            except (json.JSONDecodeError, OSError, KeyError):
                pass  # A corrupt entry is a miss, never a wrong answer.
        result = inner(criteria, findings)
        cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "runs": result.runs,
                    "why": {m.criterion.id: m.why for m in result.matches},
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return result

    return match


def _result_from_cache(
    stored: dict[str, Any], criteria: Sequence[Criterion], findings: Sequence[Finding]
) -> MatchResult:
    """Rebuild a pairing from the stored per-run votes.

    The *votes* are cached, not the resolved consensus, so a change to the
    voting rule re-resolves from the same evidence instead of needing the calls
    to be paid for again.
    """
    from src.evals.critique import consensus, enforce_one_to_one

    runs = [dict(run) for run in stored["runs"]]
    matches = enforce_one_to_one(consensus(criteria, findings, runs))
    why = stored.get("why") or {}
    for row in matches:
        row.why = why.get(row.criterion.id)
    return MatchResult(matches=matches, runs=runs)


def llm_matcher(settings: Settings) -> MatchFn:
    """Pair criteria to findings with one LLM call, then commit the pairing.

    One call per setup, no per-pair scoring: the whole matrix is decided in a
    single judgement, which is both cheaper than a cross-encoder sweep and better
    at the thing being judged (see :data:`~src.evals.critique.MATCH_THRESHOLD`
    for the calibration that ruled the local models out).

    The model's own one-to-one instruction is not trusted —
    :func:`~src.evals.critique.enforce_one_to_one` re-applies it downstream —
    and the ``why`` it gives for every pairing is carried into the run file, so a
    reviewer reads the reason a criterion counted rather than a bare tick.
    """
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_openai import ChatOpenAI

    from src.agents.llm import chat_model_kwargs

    llm = ChatOpenAI(**chat_model_kwargs(settings))

    def match(criteria: Sequence[Criterion], findings: Sequence[Finding]) -> MatchResult:
        rows = [CriterionMatch(criterion=c) for c in criteria]
        if not criteria or not findings:
            return MatchResult(matches=rows, runs=[{c.id: None for c in criteria}])
        prompt = (
            "CRITERIA\n"
            + "\n".join(f"{c.id}: {c.criterion}" for c in criteria)
            + "\n\nFINDINGS\n"
            + "\n".join(f"{f.id}: {f.criterion} — {f.detail}" for f in findings)
        )
        response = llm.invoke(
            [SystemMessage(content=MATCH_SYSTEM_PROMPT), HumanMessage(content=prompt)]
        )
        payload = _json_object(str(getattr(response, "content", response)))
        by_id = {row.criterion.id: row for row in rows}
        finding_ids = {f.id for f in findings}
        by_finding = {f.id: f for f in findings}
        for entry in payload.get("matches") or []:
            if not isinstance(entry, dict):
                continue
            row = by_id.get(str(entry.get("criterion_id") or ""))
            if row is None:
                continue
            # Recorded before the null check: "not raised" is the reason a
            # reader most wants, and discarding it left every unmatched row in
            # the UI with no explanation at all.
            row.why = str(entry.get("why") or "").strip() or None
            finding_id = entry.get("finding_id")
            if not finding_id or str(finding_id) not in finding_ids:
                continue
            row.finding_id = str(finding_id)
            row.finding_criterion = by_finding[str(finding_id)].criterion
            # The matcher gives a verdict, not a similarity. 1.0 records "this
            # pairing was asserted"; the reason is what a reviewer actually
            # reads, and it is what ``enforce_one_to_one`` breaks ties on when
            # the model reuses a finding despite being told not to.
            row.score = 1.0
        return MatchResult(matches=rows, runs=[{row.criterion.id: row.finding_id for row in rows}])

    return match


# ── the critique itself ───────────────────────────────────────────────────

#: ``(dataset, setup) -> one critique``. The live implementation is
#: :func:`build_critique_fn`; the tests supply a fake so the run assembly and
#: scoring can be exercised with no models, no Chroma and no API key.
CritiqueFn = Callable[[CritiqueDataset, str], SetupCritique]


def load_artifact(dataset: CritiqueDataset) -> Any:
    """The stored document under review, or a clear failure.

    Read from the document store rather than re-fetched: the run has to be
    reproducible, and a page that changed between two runs would move the scores
    for reasons that are not about retrieval.
    """
    from src.documents.store import DocumentStore

    document = DocumentStore().get(dataset.artifact_id)
    if document is None:
        raise ValueError(
            f"No stored document {dataset.artifact_id!r}. Review "
            f"{dataset.artifact_url} once in the chat so it is fetched and stored."
        )
    return document


def build_critique_fn(settings: Settings, dataset: CritiqueDataset) -> CritiqueFn:
    """A critique callable over the real stack, held-out video excluded.

    The stores are built once and shared across setups, exactly as
    :func:`src.evals.ablation.build_retrieve` does — so a multi-setup run loads
    the embedding model and the cross-encoder once between them.
    """
    from src.agents.llm import chat_model_kwargs
    from src.documents.review import REVIEW_INTENT_QUERIES, select_sections
    from src.rag.context import MultiTranscriptRagContextProvider
    from src.rag.embeddings import HuggingFaceEmbeddingModel
    from src.rag.eval import estimate_tokens
    from src.rag.storage import RawTranscriptStore, TranscriptChunkStore
    from src.rag.summaries import TranscriptSummaryStore

    excluded = [dataset.held_out_video_id]
    embedding_model = HuggingFaceEmbeddingModel(settings.embedding_model)
    # The raw store has no exclusion of its own — it lives in src/rag, which this
    # module does not own — and it can read a full transcript by video id. Nothing
    # reaches it on the corpus-wide path this harness uses, but "unreachable
    # today" is how a leak gets built in, so it is wrapped rather than trusted.
    raw_store = _HeldOutBlocked(
        RawTranscriptStore(
            settings.chroma_path, collection_name=settings.raw_transcript_collection
        ),
        excluded,
    )
    chunk_store = TranscriptChunkStore(
        settings.chroma_path,
        embedding_model=embedding_model,
        collection_name=settings.chunk_collection,
        exclude_video_ids=excluded,
    )
    summary_store = TranscriptSummaryStore(
        settings.chroma_path,
        embedding_model=embedding_model,
        embedding_model_name=settings.embedding_model,
        collection_name=settings.transcript_summary_collection,
        exclude_video_ids=excluded,
    )
    reranker = None
    if settings.rerank_enabled:
        from src.rag.rerank import CrossEncoderReranker

        reranker = CrossEncoderReranker.from_model_name(settings.rerank_model)

    provider = MultiTranscriptRagContextProvider(
        raw_store=raw_store,
        chunk_store=chunk_store,
        # No indexer. A held-out run must never be able to write to the corpus,
        # and an auto-index of the held-out video is the one write that would
        # destroy the experiment rather than merely dirty it.
        indexer=None,
        summary_store=summary_store,
        retrieval_mode=settings.retrieval_mode,
        retrieval_candidates=settings.retrieval_candidates,
        reranker=reranker,
        neighbor_span=settings.neighbor_span,
    )

    document = load_artifact(dataset)
    query = REVIEW_INTENT_QUERIES.get(dataset.artifact_kind, REVIEW_INTENT_QUERIES["document"])

    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(**chat_model_kwargs(settings))

    def critique(_: CritiqueDataset, setup: str) -> SetupCritique:
        started = time.monotonic()
        result = SetupCritique(setup=setup)
        try:
            context = provider.get_context(
                query,
                top_k=settings.rag_top_k,
                filter_transcripts=setup == BASELINE_SETUP,
                transcript_filter_top_k=settings.transcript_filter_top_k,
                transcript_filter_min_score=settings.transcript_filter_min_score,
                retrieval_mode=settings.retrieval_mode,
            )
        except Exception as exc:  # noqa: BLE001 - a failed setup is a reported cell
            result.error = f"{type(exc).__name__}: {exc}"
            result.elapsed_seconds = time.monotonic() - started
            return result

        chunks = list(context.retrieved_chunks or [])
        result.retrieved_chunk_ids = [
            f"chunk:{chunk.video_id}:{chunk.chunk_index}" for chunk in chunks
        ]
        result.retrieved_video_ids = sorted({chunk.video_id for chunk in chunks})
        excerpts = _format_excerpts(chunks)
        selection = select_sections(document, query)
        prompt = (
            f"DOCUMENT UNDER REVIEW ({dataset.artifact_kind}) — {dataset.artifact_url}\n"
            f"{_format_document(selection.sections)}\n\n"
            f"EXPERT TRANSCRIPT EXCERPTS\n{excerpts}"
        )
        result.token_estimate = estimate_tokens(prompt)
        result.trace = [
            {
                "phase": "filter" if setup == BASELINE_SETUP else "retrieve",
                "label": "Retrieve review criteria",
                "detail": (
                    f"{len(chunks)} chunks from {len(result.retrieved_video_ids)} videos "
                    f"— {dataset.held_out_video_id} excluded from every store"
                ),
            },
            {
                "phase": "read",
                "label": "Read the document",
                "detail": selection.detail(),
            },
        ]
        try:
            from langchain_core.messages import HumanMessage, SystemMessage

            response = llm.invoke(
                [
                    SystemMessage(content=CRITIQUE_SYSTEM_PROMPT),
                    HumanMessage(content=prompt),
                ]
            )
            answer = str(getattr(response, "content", response))
            result.answer = answer
            result.findings = parse_findings(_json_object(answer))
        except Exception as exc:  # noqa: BLE001 - a failed setup is a reported cell
            result.error = f"{type(exc).__name__}: {exc}"
        result.elapsed_seconds = time.monotonic() - started
        return result

    return critique


class _HeldOutBlocked:
    """A raw store that refuses the held-out video instead of returning it.

    Fails loud. A held-out run that silently reads the transcript it is meant to
    be blind to produces a number that looks perfect and means nothing, so the
    right behaviour on that path is to stop the run.
    """

    def __init__(self, inner: Any, excluded: Sequence[str]) -> None:
        self._inner = inner
        self._excluded = set(excluded)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def get_raw_document(self, video_id: str) -> Any:
        self._check(video_id)
        return self._inner.get_raw_document(video_id)

    def join_raw_text(self, video_id: str) -> str:
        self._check(video_id)
        return str(self._inner.join_raw_text(video_id))

    def ensure_raw_document(self, source_url: str, refresh: bool = False) -> Any:
        for video_id in self._excluded:
            if video_id in str(source_url):
                self._check(video_id)
        return self._inner.ensure_raw_document(source_url, refresh)

    def _check(self, video_id: str) -> None:
        if video_id in self._excluded:
            raise ValueError(
                f"{video_id} is held out of this run and must not be read. "
                "Reaching this means an exclusion was bypassed."
            )


def _format_excerpts(chunks: Sequence[Any]) -> str:
    parts: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        start = float(getattr(chunk, "start_seconds", 0.0) or 0.0)
        end = float(getattr(chunk, "end_seconds", 0.0) or 0.0)
        title = getattr(chunk, "title", None) or ""
        parts.append(
            f"[{index}] video_id={chunk.video_id} start_seconds={start:.1f} "
            f"end_seconds={end:.1f} title={title!r}\n{chunk.text}"
        )
    return "\n\n".join(parts)


def _format_document(sections: Sequence[Any]) -> str:
    return "\n\n".join(
        f"## {section.heading}\n{section.text}" if section.heading else section.text
        for section in sections
    )


_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def _json_object(text: str) -> dict[str, Any]:
    """The JSON object in a model reply, fenced or not. ``{}`` if there is none."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z]*\n?", "", stripped)
        stripped = re.sub(r"\n?```$", "", stripped).strip()
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        match = _JSON_BLOCK.search(stripped)
        if match is None:
            return {}
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
    return parsed if isinstance(parsed, dict) else {}


# ── the run ───────────────────────────────────────────────────────────────


def run_critique_eval(
    dataset: CritiqueDataset,
    setups: Sequence[str],
    critique: CritiqueFn,
    match: MatchFn,
    chunk_text: ChunkTextFn,
    *,
    config: dict[str, Any] | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Critique the artifact with each setup, score them, assemble the run.

    The dataset's own quotes are verified first. If the ground truth cannot
    resolve against the transcript it claims to come from, nothing measured
    against it means anything, so this refuses to run rather than producing a
    number that looks fine.
    """
    bad = [check for check in verify_dataset(dataset, chunk_text) if not check.resolved]
    if bad:
        raise ValueError(
            "The held-out criteria do not all resolve against the transcript: "
            + "; ".join(f"{check.citation.start_seconds}s {check.reason}" for check in bad[:5])
        )

    cells: list[dict[str, Any]] = []
    for setup in setups:
        if on_progress is not None:
            on_progress(f"[{setup}] critiquing {dataset.artifact_url}")
        result = critique(dataset, setup)
        cells.append(score_critique(result, dataset, match, chunk_text))
    return build_run(
        dataset,
        cells,
        config={
            "exclusion_version": EXCLUSION_VERSION,
            **(config or {}),
        },
        baseline=setups[0] if setups else BASELINE_SETUP,
    )


def rescore_committed_run(
    run: dict[str, Any],
    dataset: CritiqueDataset,
    match: MatchFn,
    chunk_text: ChunkTextFn,
    *,
    config: dict[str, Any] | None = None,
    now: Any = None,
) -> dict[str, Any]:
    """Re-score a committed run's stored findings under the current scorer.

    The same move :mod:`src.evals.rejudge` makes for the matrix, and needed for
    the same reason: when the *scoring rule* changes, the old number is not
    comparable and the old answers are still perfectly good. Re-running
    retrieval and the critique call to re-apply an arithmetic change would also
    change the findings, so the two effects could not be told apart.

    Retrieval is not repeated, so the held-out guarantee is inherited from the
    source run — and re-checked here, against the ids that run stored, rather
    than assumed.
    """
    cells: list[dict[str, Any]] = []
    for stored in run.get("cells", []):
        critique = SetupCritique(
            setup=str(stored.get("setup") or ""),
            findings=parse_findings({"findings": stored.get("findings", [])}),
            retrieved_chunk_ids=list(stored.get("retrieved_chunk_ids") or []),
            retrieved_video_ids=list(stored.get("retrieved_video_ids") or []),
            elapsed_seconds=float(stored.get("elapsed_seconds") or 0.0),
            token_estimate=int(stored.get("token_estimate") or 0),
            answer=str(stored.get("answer") or ""),
            error=stored.get("error"),
            trace=list(stored.get("trace") or []),
        )
        cells.append(score_critique(critique, dataset, match, chunk_text))
    rescored = build_run(
        dataset,
        cells,
        config={
            **(run.get("config") or {}),
            **(config or {}),
            "rescored_from": run.get("run_id"),
        },
        baseline=str(run.get("baseline") or BASELINE_SETUP),
        now=now,
    )
    rescored["rescored_from"] = run.get("run_id")
    return rescored


def run_default_critique_eval(
    settings: Settings,
    *,
    setups: Sequence[str] | None = None,
    dataset_path: Path | None = None,
    repeats: int = DEFAULT_MATCH_REPEATS,
    on_progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Load the held-out dataset and measure the named setups over the real stack."""
    dataset = load_critique_dataset(dataset_path)
    chosen = list(setups or [BASELINE_SETUP])
    return run_critique_eval(
        dataset,
        chosen,
        build_critique_fn(settings, dataset),
        cached_matcher(
            repeated_matcher(llm_matcher(settings), repeats=repeats),
            repeats=repeats,
        ),
        chunk_text_lookup(settings),
        config={
            "answer_model": settings.deepseek_model,
            "matcher": "llm",
            "matcher_model": settings.deepseek_model,
            "matcher_version": MATCHER_VERSION,
            "match_repeats": repeats,
            "embedding_model": settings.embedding_model,
            "retrieval_mode": settings.retrieval_mode,
            "top_k": settings.rag_top_k,
            "rerank_enabled": settings.rerank_enabled,
            "transcript_filter_top_k": settings.transcript_filter_top_k,
            "transcript_filter_min_score": settings.transcript_filter_min_score,
            "corpus_note": (
                "3 of 53 videos have no stored summary, so the summary-filtered "
                "setup routes over 50 videos before the held-out one is removed"
            ),
        },
        on_progress=on_progress,
    )

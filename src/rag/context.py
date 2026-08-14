from __future__ import annotations

import time
from typing import TYPE_CHECKING, Callable, Protocol

from src.agents.context import TranscriptContext
from src.agents.models import TraceStep, chunk_id, chunk_ids_for
from src.rag.chunking import format_timestamp
from src.rag.indexing import RagIndexer
from src.rag.question_scope import corpus_wide_behavior, corpus_wide_signal
from src.rag.references import format_chunk_reference
from src.rag.storage import RawTranscriptStore, TranscriptChunkStore, transcript_from_raw_document
from src.rag.summaries import TranscriptSummaryStore

if TYPE_CHECKING:  # pragma: no cover - typing only
    from src.rag.query_transform import QueryTransform

#: A scoped store search: ``(query, n) -> chunks``. Each branch of
#: ``get_context`` binds its own scope into one of these so the query-transform
#: fan-out can re-run *that* search per expanded query without knowing whether
#: it is corpus-wide, channel-scoped or restricted to one video.
SearchFn = Callable[[str, int], list]


class RetrievalError(ValueError):
    """A retrieval failure carrying the stages measured before it.

    Several of ``get_context``'s failure modes only surface after real work has
    run and been timed — a summary filter that matched nothing, a channel query
    that came back empty. Raising a bare error would drop exactly the steps that
    explain the failure, so they travel on the exception and the caller can
    persist what actually ran. Subclasses ``ValueError`` because that is what
    ``get_context`` has always raised.
    """

    def __init__(self, message: str, trace: list[TraceStep] | None = None) -> None:
        super().__init__(message)
        self.trace: list[TraceStep] = list(trace or [])


class _Reranker(Protocol):
    """The shape ``_refine`` needs from a reranker — see :mod:`src.rag.rerank`."""

    def rerank(self, query: str, chunks: list, top_k: int) -> list: ...


class RagTranscriptContextProvider:
    def __init__(
        self,
        raw_store: RawTranscriptStore,
        chunk_store: TranscriptChunkStore,
        indexer: RagIndexer | None = None,
        top_k: int = 10,
    ) -> None:
        self.raw_store = raw_store
        self.chunk_store = chunk_store
        self.indexer = indexer
        self.top_k = top_k

    def get_transcript(
        self, video_id: str, source_url: str, query: str | None = None
    ) -> TranscriptContext:
        cache_status = "hit"
        if not self.chunk_store.has_chunks(video_id):
            if self.indexer is None:
                raise ValueError(
                    "No RAG chunks found. Run index-rag first or configure auto-indexing."
                )
            result = self.indexer.index(source_url, refresh=False)
            cache_status = result.cache_status

        raw_document, raw_cache_status = self.raw_store.ensure_raw_document(
            source_url, refresh=False
        )
        if cache_status == "hit":
            cache_status = raw_cache_status
        retrieved = self.chunk_store.query(video_id, query or "", self.top_k)
        transcript = transcript_from_raw_document(raw_document)
        return TranscriptContext(
            transcript=transcript,
            cache_status=cache_status,
            context_text=format_retrieved_chunks(retrieved),
            context_mode="rag",
            retrieved_chunks=retrieved,
            top_k=self.top_k,
        )


def format_retrieved_chunks(chunks) -> str:
    parts: list[str] = []
    for index, chunk in enumerate(chunks, 1):
        start = format_timestamp(chunk.start_seconds)
        end = format_timestamp(chunk.end_seconds)
        parts.append(f"[{index}] {start}-{end}\n{chunk.text}")
    return "\n\n".join(parts)


class MultiTranscriptRagContextProvider:
    """Retrieval for the multi-transcript agents.

    Scope narrows in three steps, most specific first: a single video
    (``source_url``), a whole channel (``channel_id``), or the entire corpus.
    Within the chosen scope, retrieval can run semantically or as a hybrid of
    semantic and BM25 rankings, optionally reranked and widened to neighbouring
    chunks before the answer call sees it.

    The optional summary pre-filter narrows the corpus to whole videos before
    any chunk is searched. Its ``top_k`` is a budget rather than a relevance
    test, so it is lifted for questions that are about the corpus as a whole —
    see :meth:`_filter_budget`. Lifting that cap makes breadth *possible*
    without making it happen, because the answer still sees only ``top_k``
    chunks; :func:`select_diverse_sources`, switched on with
    ``corpus_wide_max_per_video``, is the half that spends that budget across
    videos rather than within one.

    An optional ``query_transform`` sits in front of all of that, rewriting
    *what* is embedded (see :mod:`src.rag.query_transform`) before any of the
    scoping applies. It only ever changes the vector search: BM25 fusion below
    keeps matching the question as the user asked it, because keyword-matching
    a hypothetical passage would search for words nobody typed.
    """

    def __init__(
        self,
        raw_store: RawTranscriptStore,
        chunk_store: TranscriptChunkStore,
        indexer: RagIndexer | None = None,
        summary_store: TranscriptSummaryStore | None = None,
        retrieval_mode: str = "semantic",
        retrieval_candidates: int = 30,
        reranker: _Reranker | None = None,
        neighbor_span: int = 0,
        query_transform: "QueryTransform | None" = None,
        corpus_wide_filter: bool = True,
        corpus_wide_max_per_video: int = 0,
    ) -> None:
        self.raw_store = raw_store
        self.chunk_store = chunk_store
        self.indexer = indexer
        self.summary_store = summary_store
        self.retrieval_mode = retrieval_mode
        self.retrieval_candidates = retrieval_candidates
        self.reranker = reranker
        self.neighbor_span = neighbor_span
        self.query_transform = query_transform
        #: Whether a corpus-wide question lifts the summary filter's ``top_k``
        #: cap (see :meth:`_filter_budget`). A switch rather than a constant so
        #: the change can be A/B'd against the old behaviour when the matrix is
        #: re-run: the eval harness has to be able to produce both arms.
        self.corpus_wide_filter = corpus_wide_filter
        #: How many chunks one video may contribute to the final ``top_k`` on a
        #: corpus-wide question. ``0`` is off, and off is the shipped default:
        #: this changes what every corpus-wide answer is built from, and no
        #: shipped behaviour should move before the matrix has measured it. See
        #: :func:`select_diverse_sources` for what "diversity-constrained" means
        #: here and what it costs.
        self.corpus_wide_max_per_video = corpus_wide_max_per_video

    @property
    def retrieval_behavior(self) -> str:
        """The corpus-wide behaviour this provider implements, as a name.

        Derived from the switches, so it cannot drift from them. It exists
        because neither switch is a ``Settings`` field and neither is therefore
        visible to :func:`src.evals.matrix_cache.cell_fingerprint`: a harness
        running a non-default provider passes this value so the cache can tell
        two behaviours apart instead of averaging them into one column.
        """
        return corpus_wide_behavior(
            cap_lifted=self.corpus_wide_filter,
            max_chunks_per_video=self.corpus_wide_max_per_video,
        )

    def get_context(
        self,
        question: str,
        source_url: str | None = None,
        top_k: int = 10,
        filter_transcripts: bool = False,
        transcript_filter_top_k: int = 5,
        transcript_filter_min_score: float = 0.25,
        channel_id: str | None = None,
        retrieval_mode: str | None = None,
    ) -> TranscriptContext:
        cache_status = "hit"
        selected_transcripts = []
        trace: list[TraceStep] = []
        mode = retrieval_mode or self.retrieval_mode
        # Only a corpus-wide question routed through the summary filter can be
        # narrowed by the defect this cap addresses, so only that path arms it.
        # Every other branch keeps the shipped selection byte for byte.
        max_per_video = 0
        # Retrieve wide, then let fusion/reranking narrow to top_k. With neither
        # enabled this collapses to the original single top_k query.
        candidates = (
            max(top_k, self.retrieval_candidates)
            if (mode == "hybrid" or self.reranker is not None)
            else top_k
        )
        if source_url is None:
            if not self.chunk_store.has_any_chunks():
                raise RetrievalError(
                    "No indexed transcript chunks found. Run index-rag for one or more "
                    "YouTube URLs first.",
                    trace,
                )
            if filter_transcripts:
                if self.summary_store is None:
                    raise RetrievalError("Transcript filtering requires a summary store", trace)
                filter_top_k, scope_signal = self._filter_budget(
                    question, transcript_filter_top_k, trace
                )
                if scope_signal is not None:
                    max_per_video = self.corpus_wide_max_per_video
                filter_started = time.monotonic()
                selected_transcripts = self.summary_store.query_relevant_transcripts(
                    question,
                    top_k=filter_top_k,
                    min_score=transcript_filter_min_score,
                )
                trace.append(
                    TraceStep(
                        phase="filter",
                        label="Summary filter",
                        detail=(
                            f"{len(selected_transcripts)} videos matched by per-video "
                            f"summary (top {filter_top_k}"
                            f"{' — cap lifted' if scope_signal else ''}, min score "
                            f"{transcript_filter_min_score})"
                        ),
                        # The count is the measurement; *which* videos is the
                        # check. A filter that kept five videos tells you
                        # nothing — a filter that kept five career videos and no
                        # property ones is the whole claim, so the list goes on
                        # a line that wraps instead of into the clipped detail.
                        note=_matched_videos_note(selected_transcripts),
                        elapsed_ms=int((time.monotonic() - filter_started) * 1000),
                    )
                )
                if not selected_transcripts:
                    raise RetrievalError(
                        "No transcript summaries matched the question. Try lowering "
                        "--transcript-filter-min-score or run without "
                        "--filter-transcripts.",
                        trace,
                    )
                filtered_video_ids = [summary.video_id for summary in selected_transcripts]
                if channel_id:
                    channel_video_ids = set(self.chunk_store.channel_video_ids(channel_id))
                    filtered_video_ids = [
                        vid for vid in filtered_video_ids if vid in channel_video_ids
                    ]
                    if not filtered_video_ids:
                        raise RetrievalError(
                            "No transcript summaries matched the question within "
                            f"channel {channel_id!r}. Try lowering "
                            "--transcript-filter-min-score, running without "
                            "--filter-transcripts, or checking the channel_id.",
                            trace,
                        )
                retrieved = self._search(
                    lambda query, count: self.chunk_store.query_by_video_ids(
                        filtered_video_ids, query, count
                    ),
                    question,
                    candidates,
                    mode,
                    f"{len(filtered_video_ids)} filtered videos",
                    trace,
                )
            elif channel_id:
                # The searches run and are timed inside ``_search``, which
                # records them before returning: "the channel returned nothing"
                # is the diagnostic, and it would otherwise have to be inferred
                # from the error text.
                retrieved = self._search(
                    lambda query, count: self.chunk_store.query_by_channel(
                        channel_id, query, count
                    ),
                    question,
                    candidates,
                    mode,
                    f"channel {channel_id}",
                    trace,
                )
                if not retrieved:
                    raise RetrievalError(
                        f"No indexed chunks found for channel {channel_id!r}. The "
                        "channel may not be indexed, or its chunks predate the "
                        "channel metadata backfill.",
                        trace,
                    )
            else:
                retrieved = self._search(
                    self.chunk_store.query_all,
                    question,
                    candidates,
                    mode,
                    "the whole corpus",
                    trace,
                )
            retrieved = self._refine(
                question,
                retrieved,
                top_k,
                mode,
                channel_id,
                trace=trace,
                max_per_video=max_per_video,
            )
            transcript = _context_transcript_from_chunks(retrieved)
        else:
            video_id = _extract_video_id(source_url)
            if not self.chunk_store.has_chunks(video_id):
                if self.indexer is None:
                    raise RetrievalError(
                        f"No RAG chunks found for {source_url}. Run index-rag first.", trace
                    )
                result = self.indexer.index(source_url, refresh=False)
                cache_status = result.cache_status
            raw_document, raw_cache_status = self.raw_store.ensure_raw_document(
                source_url, refresh=False
            )
            if cache_status == "hit":
                cache_status = raw_cache_status
            retrieved = self._search(
                lambda query, count: self.chunk_store.query_by_url(source_url, query, count),
                question,
                candidates,
                mode,
                f"video {video_id}",
                trace,
            )
            retrieved = self._refine(question, retrieved, top_k, mode, None, video_id, trace=trace)
            transcript = transcript_from_raw_document(raw_document)

        return TranscriptContext(
            transcript=transcript,
            cache_status=cache_status,
            context_text=format_retrieved_chunks_with_references(retrieved),
            context_mode="rag",
            retrieved_chunks=retrieved,
            selected_transcripts=selected_transcripts,
            top_k=top_k,
            trace=trace,
        )

    def _filter_budget(
        self,
        question: str,
        transcript_filter_top_k: int,
        trace: list[TraceStep],
    ) -> tuple[int, str | None]:
        """How many videos the summary filter may keep for this question.

        ``transcript_filter_top_k`` is a *budget*, not a relevance criterion —
        ``transcript_filter_min_score`` is the criterion. For a question whose
        answer lives in a few videos the budget is the point: it is what stops
        a resume question dragging in three system-design talks. For a question
        about the corpus as a whole ("what are the main themes across this
        corpus?") the budget is simply wrong. There is no top five; the answer
        is the spread, and capping at five produces a confident answer about a
        fifth of the evidence with nothing in the output to say so.

        So a corpus-wide question keeps the criterion and loses the budget: the
        cap rises to the number of summarised videos, which is "no cap" stated
        as a number. Every video that clears ``min_score`` is routed to.

        Two alternatives were rejected.

        *Bypass the filter entirely* — the obvious reading of "top-k narrows a
        global question" — throws away the part that works. Without the filter,
        corpus-wide retrieval is a flat top_k over every chunk, which is not
        broader in any way that matters: ten chunks still come from at most ten
        videos, and now nothing has checked that those videos are relevant at
        all. Bypassing removes the relevance test and keeps the narrowing, which
        is the wrong half.

        *Raise the default ``top_k``* would change every question to fix some,
        and the live V2 numbers say the filter's tight routing is what local
        questions are winning on. A per-question decision is the only one that
        does not trade one regression for another.

        The failure mode is deliberately asymmetric. A false positive lifts a
        cap: ``min_score`` still rejects irrelevant videos, so a local question
        misread as global degrades *towards* the unfiltered baseline, which is
        the arm that already scores better on local questions. A false negative
        is the behaviour that shipped for months. Detection erring generous
        therefore costs less than detection erring strict, and the patterns in
        :mod:`src.rag.question_scope` are written accordingly.
        """
        if not self.corpus_wide_filter:
            return transcript_filter_top_k, None
        signal = corpus_wide_signal(question)
        if signal is None:
            return transcript_filter_top_k, None
        assert self.summary_store is not None  # caller checked
        widened = max(transcript_filter_top_k, self.summary_store.count())
        trace.append(
            TraceStep(
                phase="route",
                label="Corpus-wide question",
                detail=(
                    f"matched {signal!r} — summary filter cap raised from "
                    f"{transcript_filter_top_k} to {widened} (every video above "
                    f"the score threshold)"
                ),
                # No ``query``: this step did not search, and the question it
                # classified is already on screen above the trace. Naming the
                # signal in ``detail`` is what makes the decision checkable —
                # a reader can hold "matched 'corpus'" against the question
                # they can see.
            )
        )
        return widened, signal

    def _search(
        self,
        search: SearchFn,
        question: str,
        candidates: int,
        mode: str,
        scope_desc: str,
        trace: list[TraceStep],
    ) -> list:
        """Retrieve candidates for one question within one scope, and record it.

        With no ``query_transform`` this is a single scoped search on the
        question as asked. With one, the question is first expanded (a step of
        its own, so the trace shows what was actually embedded), each expanded
        query searches the same scope independently, and the rankings are
        RRF-fused — which is a second step, because the fused order is not any
        one search's order.
        """
        if self.query_transform is None:
            started = time.monotonic()
            retrieved = search(question, candidates)
            self._record_retrieve(
                trace, retrieved, mode, scope_desc, candidates, started, 1, query=question
            )
            return retrieved

        plan = self.query_transform.expand(question)
        trace.append(
            TraceStep(
                phase="llm",
                label=plan.label,
                detail=plan.detail(),
                model=getattr(self.query_transform, "model", None),
                elapsed_ms=plan.elapsed_ms,
            )
        )
        queries = plan.queries or [question]
        started = time.monotonic()
        rankings = []
        for query in queries:
            try:
                rankings.append(search(query, candidates))
            except Exception:  # noqa: BLE001 - one bad variant must not sink retrieval
                continue
        if not rankings:
            # Every expanded variant failed independently of the LLM
            # expansion step itself; falling back to the raw question keeps
            # this failure mode as degrade-gracefully as query expansion's.
            rankings = [search(question, candidates)]
        # Report the candidate *pool* the searches found between them, before
        # fusion reorders it — a chunk two phrasings both found is one
        # candidate, not two.
        self._record_retrieve(
            trace,
            _unique_chunks(rankings),
            mode,
            scope_desc,
            candidates,
            started,
            len(rankings),
            # The expansion step above lists the variants; this one names the
            # query they were all expanded from.
            query=question,
        )
        if len(rankings) == 1:
            return rankings[0]

        from src.rag.fusion import fuse_rankings

        fuse_started = time.monotonic()
        fused = fuse_rankings(rankings, top_k=candidates)
        trace.append(
            TraceStep(
                phase="merge",
                label="Fuse query variants",
                detail=(
                    f"RRF-fused {len(rankings)} per-query rankings — {len(fused)} kept "
                    f"(asked for {candidates})"
                ),
                chunk_ids=chunk_ids_for(fused),
                elapsed_ms=int((time.monotonic() - fuse_started) * 1000),
            )
        )
        return fused

    @staticmethod
    def _record_retrieve(
        trace: list[TraceStep],
        retrieved: list,
        mode: str,
        scope_desc: str,
        candidates: int,
        started: float,
        searches: int = 1,
        query: str | None = None,
    ) -> None:
        """One step for the scoped search (or searches) that just ran.

        ``searches`` is the number of queries that searched this scope — more
        than one when a query transform fanned the question out — and
        ``retrieved`` is the deduplicated pool they found between them.

        ``query`` is what was actually embedded, and it is in the detail because
        it is not always the user's question: a follow-up is rewritten to stand
        alone, and a document review searches for the criteria the document
        should be judged against rather than for the words the user typed. A
        trace that reports only "semantic search over the whole corpus" cannot
        show you that the corpus was searched for the wrong thing — which is
        exactly the failure most worth catching here.
        """
        scope = scope_desc if searches == 1 else f"{scope_desc} × {searches} queries"
        asked = "asked for" if searches == 1 else "each asked for"
        trace.append(
            TraceStep(
                phase="retrieve",
                label="Retrieve candidates",
                detail=(
                    f"{mode} search over {scope} — "
                    f"{len(retrieved)} candidates ({asked} {candidates})"
                ),
                query=_shorten(query) if query else None,
                chunk_ids=chunk_ids_for(retrieved),
                elapsed_ms=int((time.monotonic() - started) * 1000),
            )
        )

    def _refine(
        self,
        question: str,
        retrieved: list,
        top_k: int,
        mode: str,
        channel_id: str | None,
        video_id: str | None = None,
        trace: list[TraceStep] | None = None,
        max_per_video: int = 0,
    ) -> list:
        """Fuse, rerank, and widen a candidate set down to the final top_k.

        When ``trace`` is given, each stage that actually ran appends a
        :class:`TraceStep` describing what it did to the candidate set.

        ``max_per_video`` arms :func:`select_diverse_sources` for the final cut
        to ``top_k``. When it is set, the ranking stages are asked to *order*
        the candidate pool rather than to cut it, because a per-video cap can
        only promote a chunk that something still ranked: cutting to ``top_k``
        first and diversifying afterwards would have nothing left to promote
        from. When it is ``0`` — the default, and every path but a corpus-wide
        filtered one — this method behaves exactly as it did before.
        """
        if mode == "hybrid":
            fuse_width = (
                max(top_k, self.retrieval_candidates)
                if (self.reranker is not None or max_per_video > 0)
                else top_k
            )
            fuse_started = time.monotonic()
            before_fuse = len(retrieved)
            retrieved = self._fuse_with_bm25(question, retrieved, fuse_width, channel_id, video_id)
            if trace is not None:
                trace.append(
                    TraceStep(
                        phase="merge",
                        label="BM25 fusion",
                        detail=(
                            f"RRF-fused {before_fuse} semantic candidates with keyword "
                            f"hits — {len(retrieved)} kept"
                        ),
                        chunk_ids=chunk_ids_for(retrieved),
                        elapsed_ms=int((time.monotonic() - fuse_started) * 1000),
                    )
                )
        # The width the ranking stages cut to. Diversity does the cutting when
        # it is armed, so they only sort — computed here rather than on entry
        # because BM25 fusion can widen the pool it applies to.
        select_width = len(retrieved) if max_per_video > 0 else top_k
        if self.reranker is not None and retrieved:
            rerank_started = time.monotonic()
            before_rerank = len(retrieved)
            degraded = False
            try:
                retrieved = self.reranker.rerank(question, retrieved, select_width)
            except Exception:
                # A reranker failure must degrade to the underlying ranking
                # rather than lose the answer entirely.
                retrieved = retrieved[:select_width]
                degraded = True
            if trace is not None:
                trace.append(
                    TraceStep(
                        phase="rerank",
                        label="Cross-encoder rerank",
                        detail=(
                            f"reranker failed; kept the top {len(retrieved)} of "
                            f"{before_rerank} by prior ranking"
                            if degraded
                            else f"reordered {before_rerank} candidates, kept top {len(retrieved)}"
                        ),
                        chunk_ids=chunk_ids_for(retrieved),
                        elapsed_ms=int((time.monotonic() - rerank_started) * 1000),
                    )
                )
        else:
            before_trim = len(retrieved)
            retrieved = retrieved[:select_width]
            if trace is not None and before_trim > len(retrieved):
                # Without this the preceding stage's step is the last word on
                # what the answer call saw, and it would overstate the count.
                trace.append(
                    TraceStep(
                        phase="merge",
                        label="Trim to top_k",
                        detail=(
                            f"no reranker; kept the first {len(retrieved)} of "
                            f"{before_trim} in ranking order"
                        ),
                        chunk_ids=chunk_ids_for(retrieved),
                    )
                )
        if max_per_video > 0:
            diversify_started = time.monotonic()
            pool = retrieved
            retrieved = select_diverse_sources(pool, top_k, max_per_video)
            if trace is not None:
                trace.append(
                    TraceStep(
                        phase="merge",
                        label="Source diversity",
                        detail=(
                            f"at most {max_per_video} chunk(s) per video — kept "
                            f"{len(retrieved)} of {len(pool)} ranked candidates, "
                            f"from {source_breadth(retrieved)} videos "
                            f"(the pool held {source_breadth(pool)})"
                        ),
                        # The chunk ids are the check on the count: a reader can
                        # see *which* chunk the cap displaced and from where,
                        # which is the only way to tell breadth bought by
                        # promoting a ranked chunk from breadth bought by
                        # retrieving a worse one.
                        chunk_ids=chunk_ids_for(retrieved),
                        elapsed_ms=int((time.monotonic() - diversify_started) * 1000),
                    )
                )
        if self.neighbor_span > 0:
            expand_started = time.monotonic()
            before_expand = len(retrieved)
            retrieved = self._expand_neighbors(retrieved)
            if trace is not None:
                trace.append(
                    TraceStep(
                        phase="merge",
                        label="Neighbor expansion",
                        detail=(
                            f"±{self.neighbor_span} adjacent chunks pasted around "
                            f"{before_expand} hits — {len(retrieved)} total"
                        ),
                        chunk_ids=chunk_ids_for(retrieved),
                        elapsed_ms=int((time.monotonic() - expand_started) * 1000),
                    )
                )
        return retrieved

    def _fuse_with_bm25(
        self,
        question: str,
        semantic: list,
        fuse_width: int,
        channel_id: str | None,
        video_id: str | None,
    ) -> list:
        from src.rag import bm25
        from src.rag.fusion import fuse_chunks

        records = self._bm25_records(channel_id, video_id)
        if not records:
            return semantic
        keyword = bm25.search(
            records,
            question,
            max(fuse_width, self.retrieval_candidates),
            # The exclusion belongs in the key, not just in the records. The
            # index cache is keyed by (key, record count), and two different
            # held-out videos with the same chunk count produce record lists of
            # identical length — the second would silently reuse the first's
            # index and search a corpus it was supposed to have never seen.
            cache_key=(f"hybrid:{video_id or channel_id or 'all'}{self.chunk_store.exclusion_key}"),
        )
        # Widen recall, don't merely re-rank: a resolver rebuilds a real
        # RetrievedChunk for any keyword hit the semantic pass missed, so fusion
        # can surface a BM25-only chunk instead of dropping it. Records that lack
        # the identity a citation needs are skipped, never fabricated.
        return fuse_chunks(semantic, keyword, top_k=fuse_width, resolver=_record_resolver(records))

    def _bm25_records(self, channel_id: str | None, video_id: str | None) -> list[dict]:
        where: dict[str, object] | None = None
        if video_id:
            where = {"video_id": video_id}
        elif channel_id:
            where = {"channel_id": channel_id}
        result = self.chunk_store.collection.get(
            # The lexical half builds its own scope, so it needs the store's
            # held-out filter applied here too — the vector path's ``$nin`` does
            # nothing for a ``collection.get`` issued from this module.
            where=self.chunk_store.scoped_where(where),
            include=["documents", "metadatas"],
        )
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []
        records: list[dict] = []
        for index, meta in enumerate(metadatas):
            meta = meta or {}
            text = documents[index] if index < len(documents) else ""
            if not text:
                continue
            records.append(
                {
                    # transcript_id / source_url are what let a keyword-only hit be
                    # rebuilt into a citable RetrievedChunk in _chunk_from_record.
                    "transcript_id": _meta_str(meta.get("transcript_id")),
                    "video_id": _meta_str(meta.get("video_id")),
                    "chunk_index": _meta_int(meta.get("chunk_index"), index),
                    "text": text,
                    "start_seconds": meta.get("start_seconds"),
                    "end_seconds": meta.get("end_seconds"),
                    "source_url": _meta_str(meta.get("source_url")) or None,
                    "channel_id": _meta_str(meta.get("channel_id")) or None,
                    "channel_name": _meta_str(meta.get("channel_name")) or None,
                    "title": _meta_str(meta.get("title")) or None,
                    "upload_date": _meta_str(meta.get("upload_date")) or None,
                }
            )
        return records

    def _expand_neighbors(self, retrieved: list) -> list:
        """Paste adjacent chunks around each hit, keeping retrieval order.

        Neighbours inherit no score — they are context, not retrieval results,
        and are dropped if already present so a chunk never appears twice.
        """
        seen = {(chunk.video_id, chunk.chunk_index) for chunk in retrieved}
        widened: list = []
        for chunk in retrieved:
            neighbors = self.chunk_store.neighbors(
                chunk.video_id, chunk.chunk_index, self.neighbor_span
            )
            for neighbor in neighbors:
                key = (neighbor.video_id, neighbor.chunk_index)
                if key in seen:
                    continue
                seen.add(key)
                widened.append(_as_retrieved(neighbor))
            widened.append(chunk)
        return widened


def select_diverse_sources(chunks: list, top_k: int, max_per_video: int) -> list:
    """The final ``top_k``, with no video contributing more than ``max_per_video``.

    **The defect.** Lifting the summary filter's *video* cap for a corpus-wide
    question does not lift the *chunk* budget, and the chunk budget is what
    actually bounds breadth: retrieval returns ``top_k`` chunks, so an answer
    draws on at most ``top_k`` videos however many the filter admits. Measured
    on ``matrix-20260811-023531`` (71 videos, 1792 chunks), the six corpus-wide
    golden questions moved from 3.33 to 4.50 distinct videos per answer when the
    cap was lifted — still below the 5.00 of the *unfiltered* baseline the
    filtered arm was meant to catch up with, and three of the six (g010, g011,
    g013) did not move at all. Admitting seventy-one videos to a ten-chunk
    budget changes nothing if the top ten chunks all come from three of them.

    **The rule.** Walk the ranked candidates in order and take a chunk unless
    its video already holds ``max_per_video`` places. Chunks passed over are
    *parked*, not dropped: if the cap leaves the selection short of ``top_k``,
    they are taken back in rank order until it is full. The result is then
    restored to the candidates' own ranking order, so the cap changes *which*
    chunks reach the answer and never what order they arrive in.

    Three properties follow, and they are the reason this design was chosen over
    the two alternatives:

    * Every selected chunk was ranked by the retriever (and by the reranker,
      where one ran). The relevance test is preserved in full — the cap can only
      promote a chunk relevance had already ranked below the cut, never admit
      one it never saw.
    * The selection is exactly as large as it would have been. Backfill
      guarantees ``min(top_k, len(chunks))`` chunks, so a corpus with fewer
      videos than ``top_k // max_per_video`` loses no context.
    * With ``max_per_video`` at or above ``top_k`` the result is the unchanged
      ranking, so the switch degrades to a no-op rather than to a surprise.

    *Round-robin over the admitted videos* was rejected: taking each video's
    best chunk in turn hands a place to every video the summary filter admitted,
    whatever that video's best chunk scores. Breadth then measures the filter's
    admissions rather than the corpus's evidence, and a video that scraped past
    ``min_score`` contributes a chunk that nothing tested at the chunk level. It
    manufactures the number this change is measured by, which is the one failure
    mode that would make the measurement worthless.

    *An MMR-style re-rank* trading relevance against coverage was rejected for a
    different reason: it works, but the trade is a continuous knob with no
    natural setting and the result is unreadable in a trace — you cannot say
    which chunk was displaced or why, only that a blended score moved. A hard
    per-video cap is one integer, and the trace can name the chunk it cost.

    **The failure mode, stated plainly.** On a question whose evidence really is
    concentrated in one or two videos, the cap displaces that video's third and
    fourth chunks in favour of another video's weaker ones. Breadth rises and
    the answer is built on worse evidence. **A breadth gain bought by retrieving
    worse chunks is not a gain**, and breadth alone cannot tell the two apart:
    it has to be read next to ``faithfulness`` and ``context_precision`` on the
    same cells. That is why this is gated to questions the corpus-wide detector
    fires on — where the spread *is* the answer — and why the switch ships off.
    """
    if max_per_video <= 0 or top_k <= 0:
        return chunks[:top_k] if top_k > 0 else []
    counts: dict[str, int] = {}
    # Rank travels with the chunk rather than being looked up afterwards: a pool
    # that happens to contain the same object twice must not have both copies
    # collapse onto one rank.
    kept: list[tuple[int, object]] = []
    parked: list[tuple[int, object]] = []
    for index, chunk in enumerate(chunks):
        video_id = str(getattr(chunk, "video_id", "") or "")
        # A chunk with no video identity is never grouped with another: it has
        # no source to be over-represented by, and bucketing them all under ""
        # would cap unrelated chunks against each other.
        capped = bool(video_id) and counts.get(video_id, 0) >= max_per_video
        if len(kept) < top_k and not capped:
            if video_id:
                counts[video_id] = counts.get(video_id, 0) + 1
            kept.append((index, chunk))
        else:
            parked.append((index, chunk))
    if len(kept) < top_k:
        kept.extend(parked[: top_k - len(kept)])
    return [chunk for _, chunk in sorted(kept, key=lambda pair: pair[0])]


def source_breadth(chunks_or_ids) -> int:
    """How many distinct videos a retrieval (or an answer's chunk ids) drew on.

    The instrument for the change above, and deliberately a pure function of
    ids: it costs no model call, so it can be computed over every cell of an
    already-committed matrix run without re-scoring anything. Accepts chunk
    objects or the ``chunk:<video_id>:<index>`` ids a run records, because the
    live retrieval has the former and a committed run only has the latter.

    Faithfulness cannot adjudicate a breadth change on its own — an answer built
    from three chunks of one video can be perfectly faithful to them — so this is
    the number that says whether a corpus-wide answer actually spanned the
    corpus.
    """
    videos: set[str] = set()
    for item in chunks_or_ids or []:
        if isinstance(item, str):
            parts = item.split(":")
            video_id = parts[1] if len(parts) == 3 else ""
        else:
            video_id = str(getattr(item, "video_id", "") or "")
        if video_id:
            videos.add(video_id)
    return len(videos)


#: How much of the embedded query a trace step shows. Set above the longest
#: query this system generates — the ``portfolio`` review-criteria query is 259
#: characters — because clipping the query is exactly the failure this field
#: exists to prevent. A truncated query looks like a query, so a reader would
#: not know the part that mattered was the part cut off.
MAX_TRACE_QUERY_CHARS = 400


def _shorten(text: str) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= MAX_TRACE_QUERY_CHARS:
        return collapsed
    return collapsed[: MAX_TRACE_QUERY_CHARS - 1].rstrip() + "…"


def _matched_videos_note(summaries: list) -> str | None:
    """The videos a summary filter routed to, as one readable line.

    Titled rather than id-only: ``chunk:5kxPMauR4fs`` does not tell a reader
    that an interview question routed to *Job Interview Simulation* and not to
    *Sharding in System Design Interviews*, which is the only thing this step
    is here to let them check. The id follows the title because it is what the
    chunk ids below are keyed on, and the score follows both because a match
    scraping the threshold is a different claim from a confident one.
    """
    if not summaries:
        return None
    parts: list[str] = []
    for summary in summaries:
        title = (getattr(summary, "title", None) or "").strip()
        video_id = getattr(summary, "video_id", "")
        score = getattr(summary, "score", None)
        label = f"{title} ({video_id})" if title else str(video_id)
        parts.append(f"{label} {score:.2f}" if isinstance(score, (int, float)) else label)
    return " · ".join(parts)


def _unique_chunks(rankings: list[list]) -> list:
    """One list of the chunks several rankings found, first occurrence kept.

    Only for reporting: the fused *order* comes from RRF, not from this
    concatenation, so this exists purely so a trace step counts and lists each
    candidate once no matter how many phrasings retrieved it.
    """
    seen: set[str] = set()
    unique: list = []
    for ranking in rankings:
        for chunk in ranking:
            key = chunk_id(chunk)
            if key is None or key in seen:
                continue
            seen.add(key)
            unique.append(chunk)
    return unique


def _meta_str(value: object) -> str:
    """A Chroma metadata value as a plain string (``""`` for ``None``/missing)."""
    return "" if value is None else str(value)


def _meta_int(value: object, default: int) -> int:
    """A Chroma metadata value as an int, falling back when it is not numeric."""
    if isinstance(value, (int, float, str)):
        try:
            return int(value)
        except (TypeError, ValueError):
            return default
    return default


def _record_resolver(records: list[dict]):
    """A ``fuse_chunks`` resolver over a set of BM25 records, keyed like fusion.

    Only keyword-only keys reach it — fusion resolves the semantic objects first —
    so a chunk is rebuilt at most once, and only if it actually enters the result.
    """
    from src.rag.fusion import chunk_key

    by_key = {
        chunk_key(record.get("video_id", ""), record.get("chunk_index", 0)): record
        for record in records
    }

    def resolve(key: str):
        record = by_key.get(key)
        return _chunk_from_record(record) if record is not None else None

    return resolve


def _chunk_from_record(record: dict):
    """Rebuild a ``RetrievedChunk`` from a BM25 record, or ``None`` if it can't be.

    A citation must point at a real chunk, so a record missing the identity
    ``RetrievedChunk`` requires — ``transcript_id`` or ``source_url`` — is dropped
    rather than reconstructed with invented values. The chunk carries no score:
    it was found only by keyword, so it has no semantic distance of its own.
    """
    from pydantic import HttpUrl, ValidationError

    from src.rag.models import RetrievedChunk

    transcript_id = record.get("transcript_id")
    source_url = record.get("source_url")
    if not transcript_id or not source_url:
        return None
    try:
        return RetrievedChunk(
            transcript_id=str(transcript_id),
            video_id=str(record.get("video_id", "")),
            source_url=HttpUrl(str(source_url)),
            chunk_index=int(record.get("chunk_index", 0) or 0),
            text=str(record.get("text", "")),
            start_seconds=record.get("start_seconds"),
            end_seconds=record.get("end_seconds"),
            channel_id=record.get("channel_id"),
            channel_name=record.get("channel_name"),
            title=record.get("title"),
            upload_date=record.get("upload_date"),
            score=None,
        )
    except (ValidationError, ValueError):
        return None


def _as_retrieved(chunk):
    from src.rag.models import RetrievedChunk

    if isinstance(chunk, RetrievedChunk):
        return chunk
    return RetrievedChunk(**chunk.model_dump(), score=None)


class ChunkLabeller:
    """Hands out citation numbers that stay unique for the life of one answer.

    A single retrieval formats its chunks ``[1]``, ``[2]`` … and that is
    unambiguous, because there is one list. The agentic path retrieves several
    times, and each call used to start from ``[1]`` again — so an answer built
    from three tool calls contained three different chunks all labelled ``[1]``,
    and a reader following a citation had no way to tell which one was meant.
    The numbering was the ambiguity, not the model.

    One labeller per answer fixes that: the first time a chunk is seen it is
    given the next number, and every later sighting gets the *same* number back.
    So a label names exactly one chunk, and one chunk carries exactly one label
    even when two tool calls both retrieve it — which is the common case, since
    successive queries in a research loop overlap heavily.

    Identity is ``(video_id, chunk_index)``, the same key the agent's context
    merge deduplicates on. That is deliberate: it makes labels agree with
    positions in the merged chunk list, so ``[n]`` is the *n*-th merged chunk,
    and the label-based fallbacks stay correct on this path instead of pointing
    into whichever tool call happened to run last.
    """

    def __init__(self) -> None:
        self._labels: dict[tuple[str, int], int] = {}

    def label_for(self, chunk) -> int:
        key = (str(chunk.video_id), int(chunk.chunk_index))
        if key not in self._labels:
            self._labels[key] = len(self._labels) + 1
        return self._labels[key]

    def __len__(self) -> int:
        return len(self._labels)


def format_retrieved_chunks_with_references(chunks, labeller: ChunkLabeller | None = None) -> str:
    """Render chunks as numbered, citable context blocks.

    Without a ``labeller`` the numbering is positional and restarts at ``[1]``,
    which is right for a single retrieval whose whole context is this one list.
    Pass a :class:`ChunkLabeller` when several retrievals feed one answer, so the
    numbering runs on across them.
    """
    parts: list[str] = []
    for index, chunk in enumerate(chunks, 1):
        label = labeller.label_for(chunk) if labeller is not None else index
        parts.append(f"{format_chunk_reference(label, chunk)}\n{chunk.text}")
    return "\n\n".join(parts)


def _context_transcript_from_chunks(chunks):
    from datetime import datetime, timezone

    from src.transcripts.models import Transcript

    if chunks:
        first = chunks[0]
        return Transcript(
            video_id="all",
            url=first.source_url,
            provider="rag",
            raw_text=" ".join(chunk.text for chunk in chunks),
            fetched_at=datetime.now(timezone.utc),
        )
    return Transcript(
        video_id="all",
        url="https://www.youtube.com/watch?v=unknown",
        provider="rag",
        raw_text="",
        fetched_at=datetime.now(timezone.utc),
    )


def _extract_video_id(source_url: str) -> str:
    from src.transcripts.youtube import extract_video_id

    return extract_video_id(source_url)

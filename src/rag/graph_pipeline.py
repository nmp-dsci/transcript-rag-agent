"""The GraphRAG build pipeline: extraction (cached) -> Leiden -> summaries.

One code path shared by the `index-graph` CLI command (whole corpus, run
manually) and the ingestion queue's automatic post-ingest hook (scoped to
just-added videos) — extraction and community-rebuild logic never drifts
between the manual and automatic triggers.
"""

from __future__ import annotations

from typing import Callable

from src.config import Settings
from src.rag.communities import CommunitySummarizer, build_communities
from src.rag.graph_extract import GraphExtractor
from src.rag.graph_store import GraphStore
from src.rag.storage import TranscriptChunk


def build_llm(settings: Settings) -> object:
    from langchain_openai import ChatOpenAI

    from src.agents.llm import chat_model_kwargs

    return ChatOpenAI(**chat_model_kwargs(settings))


def build_graph(
    settings: Settings,
    chunks: list[TranscriptChunk],
    *,
    store: GraphStore | None = None,
    refresh: bool = False,
    skip_communities: bool = False,
    on_progress: Callable[[str], None] | None = None,
) -> dict[str, object]:
    """Extract entities/claims for ``chunks`` (cached by chunk hash) and
    upsert them, then rebuild Leiden communities over the whole graph unless
    ``skip_communities``.

    Community rebuild re-summarizes *every* community via the LLM — its cost
    scales with total graph size, not with ``chunks`` — so the automatic
    post-ingest hook always passes ``skip_communities=True``; only the manual
    `index-graph` CLI run pays that cost, on request.

    ``store`` lets a caller that already holds a live connection (the API's
    lazy singleton) reuse it; passed ``None``, one is opened and closed here,
    matching the CLI's own lifecycle.
    """
    progress = on_progress or (lambda _msg: None)
    if not chunks:
        return {
            "extracted": 0,
            "failed": 0,
            "failed_chunk_ids": [],
            "failed_details": [],
            "communities_summarized": 0,
            "counts": {},
        }

    llm = build_llm(settings)
    owns_store = store is None
    store = store or GraphStore.from_settings(settings)
    try:
        store.ensure_schema()
        if refresh:
            store.wipe()
            progress("Graph wiped.")

        extractor = GraphExtractor(llm, cache_dir=settings.graph_cache_dir)
        progress(f"Extracting {len(chunks)} chunks (cache: {settings.graph_cache_dir}) ...")
        extractions = extractor.extract_all(chunks, on_progress=progress)
        failed = [extraction for extraction in extractions if extraction.error is not None]
        for extraction in extractions:
            if extraction.error is None:
                store.upsert_extraction(extraction)

        summarized = 0
        if not skip_communities:
            progress("Detecting communities ...")
            summarized = build_communities(store, CommunitySummarizer(llm), on_progress=progress)

        return {
            "extracted": len(extractions) - len(failed),
            "failed": len(failed),
            "failed_chunk_ids": [extraction.chunk_id for extraction in failed],
            "failed_details": [(extraction.chunk_id, extraction.error) for extraction in failed],
            "communities_summarized": summarized,
            "counts": store.counts(),
        }
    finally:
        if owns_store:
            store.close()

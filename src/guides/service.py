"""Wire a :class:`GuideWriter` to the app's retrieval stack.

Shared by the CLI and the server so both build the writer the same way:
the hybrid provider for scoping and ``retrieve_chunks``, the chunk store for
whole-video export, and the Chroma metadata readers for verification.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

from src.config import Settings
from src.guides.catalog import GuidePaths
from src.guides.writer import GuideWriter, OnEvent, WriterConfig


def sdk_problem() -> str | None:
    """Why an agent run cannot start here, or ``None`` when it can."""
    try:
        import claude_agent_sdk  # noqa: F401
    except ImportError:
        return "claude-agent-sdk is not installed: run `uv sync --group guides` first."
    if not os.environ.get("CLAUDE_CODE_OAUTH_TOKEN") and not os.environ.get("ANTHROPIC_API_KEY"):
        return "No CLAUDE_CODE_OAUTH_TOKEN (or ANTHROPIC_API_KEY) in the environment or ~/.env."
    return None


def corpus_readers(
    settings: Settings,
) -> tuple[set[str], Callable[[str], list[dict[str, Any]]], list[dict[str, Any]]]:
    """``(known video ids, chunk_lookup, video records)`` from Chroma metadata alone."""
    from src.api.corpus import list_corpus, load_chunk_corpus

    corpus = list_corpus(
        settings.chroma_path, settings.raw_transcript_collection, settings.chunk_collection
    )
    videos = list(corpus.get("videos", []))
    known = {str(video["video_id"]) for video in videos}

    def chunk_lookup(video_id: str) -> list[dict[str, Any]]:
        return load_chunk_corpus(settings.chroma_path, settings.chunk_collection, video_id)

    return known, chunk_lookup, videos


def retrieval_fns(
    provider: Any,
) -> tuple[Callable[[str, int], list[Any]], Callable[[str, list[str] | None, int], list[Any]]]:
    """``(retrieve_whole, retrieve)`` over a ``MultiTranscriptRagContextProvider``."""

    def retrieve_whole(question: str, top_k: int) -> list[Any]:
        return list(provider.get_context(question, top_k=top_k).retrieved_chunks)

    def retrieve(question: str, video_ids: list[str] | None, top_k: int) -> list[Any]:
        if video_ids:
            return list(provider.chunk_store.query_by_video_ids(video_ids, question, top_k))
        return retrieve_whole(question, top_k)

    return retrieve_whole, retrieve


def build_writer(
    settings: Settings,
    provider: Any,
    paths: GuidePaths,
    *,
    config: WriterConfig,
    on_event: OnEvent | None = None,
) -> GuideWriter:
    known, chunk_lookup, _videos = corpus_readers(settings)
    _whole, retrieve = retrieval_fns(provider)
    if config.example_path is None:
        example = paths.root / "ship-like-a-studio" / "guide.html"
        config.example_path = example if example.is_file() else None
    return GuideWriter(
        paths,
        chunks_for=provider.chunk_store.chunks_for_videos,
        retrieve=retrieve,
        known_videos=known,
        chunk_lookup=chunk_lookup,
        config=config,
        on_event=on_event,
    )


def guides_root(settings: Settings, guides_dir: Path | None) -> Path:
    from src.guides.catalog import DEFAULT_GUIDES_DIR

    return guides_dir or DEFAULT_GUIDES_DIR

"""Contextual Retrieval: an LLM-written situating sentence per chunk.

Anthropic's Contextual Retrieval observes that a chunk embedded alone loses
whatever its document made obvious. That is acute for transcripts, which are
mid-conversation by construction: "...had to. So I'm just going to copy that
across" embeds against almost nothing. Chunking already prepends a deterministic
header (``[channel — title @ 12:03-13:40]``, see
:func:`src.rag.chunking.build_context_header`), which fixes *which video* a
chunk is from but not *what it is about*. This module adds the second half: one
LLM call per chunk that writes the sentence naming the topic, tool or step the
passage belongs to, using the surrounding transcript as evidence.

Three deliberate boundaries, so the result is a measurement and not a rewrite:

* **A parallel collection, never the live one.** Situated chunks are written to
  ``transcript_chunks_contextual``. The baseline index is untouched, so the two
  can answer the same golden questions and be compared — which is impossible
  once one has overwritten the other.
* **Only the embedding changes.** The situating sentence joins the context
  header, which :attr:`~src.rag.models.TranscriptChunk.embedding_text` embeds
  and the stored document excludes. Retrieval sees it; the answering LLM still
  receives the spoken text alone, and citations still point at what was said.
  So a win here is a *retrieval* win, with the generation side held fixed.
* **Cache by chunk hash.** Same rule as
  :mod:`src.rag.graph_extract`: keyed on the chunk id plus its text, so
  re-running only pays for chunks whose content changed, and a failed chunk is
  retried next run rather than pinned forever.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

from src.agents.prompts import CHUNK_CONTEXT_SYSTEM_PROMPT, build_chunk_context_prompt
from src.rag.models import TranscriptChunk
from src.rag.storage import TranscriptChunkStore

logger = logging.getLogger(__name__)

#: Characters of surrounding transcript shown to the model per chunk. Enough
#: for the passage to be placed within its section; far short of a whole
#: transcript, which would multiply the bill by an order of magnitude for
#: context the model does not need to write one sentence.
DEFAULT_WINDOW_CHARS = 4000


class ChatModel(Protocol):
    def invoke(self, messages: list) -> object: ...


@dataclass(frozen=True)
class ChunkContext:
    """The situating sentence for one chunk, or the failure to produce it."""

    chunk_id: str
    context: str = ""
    error: str | None = None
    #: Read back from the disk cache, so no LLM call was made for this chunk.
    cached: bool = False


def context_cache_key(chunk: TranscriptChunk) -> str:
    """Cache identity: the chunk id plus a hash of its text."""
    digest = hashlib.sha256(f"{chunk.chunk_id}\n{chunk.text}".encode("utf-8")).hexdigest()
    return digest[:32]


def document_window(
    chunks: list[TranscriptChunk], index: int, window_chars: int = DEFAULT_WINDOW_CHARS
) -> str:
    """The transcript around ``chunks[index]``, centred on it.

    Neighbours are added outward in turn and only when they fit whole, so the
    excerpt stays centred even at the start or end of a video, where one side
    runs out and the other absorbs the remaining budget. Nothing is cut
    mid-neighbour: trimming the joined text instead would drop from the end,
    which for a chunk late in the excerpt means trimming away the very passage
    the model is being asked to situate.
    """
    if not chunks:
        return ""
    parts: list[str] = [chunks[index].text[:window_chars]]
    used = len(parts[0])
    before, after = index - 1, index + 1
    while before >= 0 or after < len(chunks):
        added = False
        if before >= 0:
            text = chunks[before].text
            if used + len(text) + 1 <= window_chars:
                parts.insert(0, text)
                used += len(text) + 1
                before -= 1
                added = True
            else:
                before = -1
        if after < len(chunks):
            text = chunks[after].text
            if used + len(text) + 1 <= window_chars:
                parts.append(text)
                used += len(text) + 1
                after += 1
                added = True
            else:
                after = len(chunks)
        if not added:
            break
    return "\n".join(parts)


class ChunkContextualizer:
    """Write one situating sentence per chunk, with a disk cache."""

    def __init__(
        self,
        llm: ChatModel,
        cache_dir: Path | None = None,
        window_chars: int = DEFAULT_WINDOW_CHARS,
    ) -> None:
        self.llm = llm
        self.cache_dir = cache_dir
        self.window_chars = window_chars
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)

    def contextualize(self, chunk: TranscriptChunk, excerpt: str) -> ChunkContext:
        result, _cached = self._contextualize_cached(chunk, excerpt)
        return result

    def contextualize_video(
        self,
        chunks: list[TranscriptChunk],
        on_progress: Callable[[str], None] | None = None,
        max_workers: int = 8,
    ) -> list[ChunkContext]:
        """Situate every chunk of one video, concurrently, in input order.

        Grouped per video because the excerpt each call needs is built from the
        chunk's own neighbours — the calls themselves are independent, so they
        parallelize the same way extraction does.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from threading import Lock

        ordered = sorted(chunks, key=lambda chunk: chunk.chunk_index)
        results: list[ChunkContext | None] = [None] * len(ordered)
        progress_lock = Lock()
        completed = 0

        def work(index: int) -> tuple[int, ChunkContext, bool]:
            excerpt = document_window(ordered, index, self.window_chars)
            result, cached = self._contextualize_cached(ordered[index], excerpt)
            return index, result, cached

        if not ordered:
            return []
        workers = max(1, min(max_workers, len(ordered)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(work, index) for index in range(len(ordered))]
            for future in as_completed(futures):
                index, result, cached = future.result()
                results[index] = result
                if on_progress is not None:
                    detail = result.error or _clip(result.context)
                    with progress_lock:
                        completed += 1
                        on_progress(
                            f"[{completed}/{len(ordered)}] {ordered[index].chunk_id} "
                            f"({'cache' if cached else 'llm'}) {detail}"
                        )
        return [result for result in results if result is not None]

    # ── internals ────────────────────────────────────────────────────────────

    def _contextualize_cached(
        self, chunk: TranscriptChunk, excerpt: str
    ) -> tuple[ChunkContext, bool]:
        cached = self._read_cache(chunk)
        if cached is not None:
            return ChunkContext(chunk_id=chunk.chunk_id, context=cached, cached=True), True
        result = self._contextualize_uncached(chunk, excerpt)
        self._write_cache(chunk, result)
        return result, False

    def _contextualize_uncached(self, chunk: TranscriptChunk, excerpt: str) -> ChunkContext:
        """One situating call, retried once — same rule as graph extraction.

        A retry is worth it here because both failure modes seen in practice are
        transient: a dropped connection, and an occasional empty completion for a
        chunk the model situates fine on the next attempt. Over a corpus-wide
        pass, no retry means a handful of chunks land in the contextual index
        with no situating sentence for no better reason than luck.
        """
        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [
            SystemMessage(content=CHUNK_CONTEXT_SYSTEM_PROMPT),
            HumanMessage(
                content=build_chunk_context_prompt(
                    chunk_text=chunk.text,
                    document_excerpt=excerpt,
                    video_title=chunk.title,
                    channel_name=chunk.channel_name,
                    upload_date=chunk.upload_date,
                )
            ),
        ]
        last_error = "empty context"
        for attempt in range(2):
            try:
                response = self.llm.invoke(messages)
            except Exception as exc:  # noqa: BLE001 - one bad chunk must not abort the pass
                last_error = str(exc)
                logger.warning(
                    "contextualization call failed for %s (attempt %s): %s",
                    chunk.chunk_id,
                    attempt + 1,
                    exc,
                )
                continue
            context = " ".join(str(getattr(response, "content", response) or "").split())
            if context:
                return ChunkContext(chunk_id=chunk.chunk_id, context=context)
            logger.warning(
                "contextualization returned nothing for %s (attempt %s)",
                chunk.chunk_id,
                attempt + 1,
            )
        return ChunkContext(chunk_id=chunk.chunk_id, error=last_error)

    def _cache_path(self, chunk: TranscriptChunk) -> Path | None:
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"{context_cache_key(chunk)}.json"

    def _read_cache(self, chunk: TranscriptChunk) -> str | None:
        path = self._cache_path(chunk)
        if path is None or not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        context = data.get("context") if isinstance(data, dict) else None
        return str(context) if context else None

    def _write_cache(self, chunk: TranscriptChunk, result: ChunkContext) -> None:
        path = self._cache_path(chunk)
        # A failure is never served back, so writing one would only leave a file
        # that can never hit — the chunk is retried on the next run instead.
        if path is None or result.error is not None or not result.context:
            return
        path.write_text(
            json.dumps({"chunk_id": chunk.chunk_id, "context": result.context}, indent=2) + "\n",
            encoding="utf-8",
        )


def situate(chunk: TranscriptChunk, context: str) -> TranscriptChunk:
    """A copy of ``chunk`` whose context header carries the situating sentence.

    The deterministic header is kept and the sentence appended to it: the two
    say different things (*which video* versus *what about it*), and the
    contextual index is meant to add the second, not to trade away the first.
    """
    if not context:
        return chunk
    header = f"{chunk.context_header}\n{context}" if chunk.context_header else context
    return chunk.model_copy(update={"context_header": header})


@dataclass
class ContextualIndexResult:
    """What one ``index-contextual`` pass did."""

    videos: int = 0
    chunks: int = 0
    contextualized: int = 0
    cached: int = 0
    failed: int = 0
    #: Contextual chunks dropped because the baseline no longer has them — the
    #: mirror catching up with a video that was re-chunked smaller.
    removed: int = 0

    def summary(self) -> str:
        removed = f", {self.removed} stale removed" if self.removed else ""
        return (
            f"{self.contextualized}/{self.chunks} chunks situated across "
            f"{self.videos} videos ({self.cached} cached, {self.failed} failed{removed})"
        )


def build_contextual_index(
    chunk_store: TranscriptChunkStore,
    contextual_store: TranscriptChunkStore,
    contextualizer: ChunkContextualizer,
    *,
    on_progress: Callable[[str], None] | None = None,
    max_chunks: int | None = None,
    max_workers: int = 8,
) -> ContextualIndexResult:
    """Situate every indexed chunk and upsert it into the contextual store.

    Reads the baseline collection and writes the parallel one, video by video —
    ordering that is not cosmetic: the excerpt each call sees is built from the
    chunk's neighbours, so chunks must be grouped by video to be situated at
    all. Upserts happen per video too, so an interrupted pass leaves a
    consistent partial index that the next run completes from the cache.

    A chunk whose situating call failed is still indexed, with the
    deterministic header it already had. That keeps the contextual collection a
    complete copy of the corpus: a missing chunk would silently depress recall
    and read as a retrieval result rather than an indexing gap.

    This collection is a *mirror* of the baseline, so a full pass replaces each
    video's chunks rather than upserting over them — otherwise a video the
    baseline re-chunked smaller keeps its old tail here, and the contextual arm
    of an ablation retrieves chunks the baseline arm cannot. A ``max_chunks``
    pass upserts instead: it deliberately reads a prefix of the corpus, so the
    last video it reaches is partial and replacing from it would delete real
    chunks on the strength of a truncation.
    """
    result = ContextualIndexResult()
    chunks = chunk_store.all_chunks()
    if max_chunks is not None:
        chunks = chunks[:max_chunks]
    by_video: dict[str, list[TranscriptChunk]] = {}
    for chunk in chunks:
        by_video.setdefault(chunk.video_id, []).append(chunk)

    result.chunks = len(chunks)
    result.videos = len(by_video)
    for video_id, video_chunks in by_video.items():
        if on_progress is not None:
            on_progress(f"── {video_id} ({len(video_chunks)} chunks) ──")
        contexts = contextualizer.contextualize_video(
            video_chunks, on_progress=on_progress, max_workers=max_workers
        )
        by_id = {context.chunk_id: context for context in contexts}
        situated: list[TranscriptChunk] = []
        for chunk in sorted(video_chunks, key=lambda chunk: chunk.chunk_index):
            context = by_id.get(chunk.chunk_id)
            if context is None or context.error is not None or not context.context:
                result.failed += 1
                situated.append(chunk)
                continue
            result.contextualized += 1
            result.cached += 1 if context.cached else 0
            situated.append(situate(chunk, context.context))
        if max_chunks is None:
            result.removed += len(contextual_store.replace_chunks(video_id, situated))
        else:
            contextual_store.upsert_chunks(situated)
    return result


def _clip(text: str, limit: int = 60) -> str:
    collapsed = " ".join(text.split())
    return collapsed if len(collapsed) <= limit else f"{collapsed[: limit - 1]}…"

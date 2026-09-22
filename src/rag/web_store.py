"""Where web sources and their chunks live.

Two new Chroma collections, ``web_sources`` and ``web_chunks``, beside the four
that already exist. Every existing collection is left byte-identical, and that
is the point rather than a nicety: ``transcript_chunks`` is the collection the
committed eval snapshots in ``evals/runs/`` were measured against, so writing
articles into it would move what retrieval returns without moving any number the
CI gate checks — the worst shape a change can have.

Both stores go through this module rather than being used directly, so the
Postgres cutover designed in ``s54`` is a backend swap behind the same method
surface instead of a redesign.

**The re-embedding rule, which is most of why this file exists.** A re-ingest
matches chunks by *content hash*, not by position. A typo fixed in the first
paragraph of a ninety-chunk article therefore re-embeds one chunk, and an
inserted section shifts every following chunk's index while still re-embedding
only the section that was added. Matching by index would re-embed the whole
tail for nothing.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Sequence

import chromadb

from src.rag.embeddings import EmbeddingModel
from src.rag.storage import collection_write_lock
from src.rag.web_models import RetrievedWebChunk, WebChunk, WebSource, source_key

logger = logging.getLogger(__name__)

WEB_SOURCE_COLLECTION = "web_sources"
WEB_CHUNK_COLLECTION = "web_chunks"


def _scalar(value: Any) -> str | int | float | bool:
    """Chroma metadata accepts only scalars, so lists are JSON in a string."""
    if isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value)


def _source_metadata(source: WebSource) -> dict[str, str | int | float | bool]:
    data = source.model_dump()
    data["url_history"] = json.dumps(source.url_history)
    return {key: _scalar(value) for key, value in data.items() if value is not None}


def _source_from_metadata(metadata: dict[str, Any]) -> WebSource | None:
    payload = dict(metadata)
    history = payload.get("url_history")
    if isinstance(history, str):
        try:
            payload["url_history"] = json.loads(history)
        except ValueError:
            payload["url_history"] = []
    try:
        return WebSource(
            **{key: value for key, value in payload.items() if key in WebSource.model_fields}
        )
    except ValueError:
        # A half-written source must not take the whole listing down; it reads
        # as "not stored", which is what an unparseable record effectively is.
        logger.warning("discarding unreadable web source metadata")
        return None


class WebSourceStore:
    """One row per watched document, keyed on the channel's own id for it."""

    collection_name = WEB_SOURCE_COLLECTION

    def __init__(
        self,
        path: Path | str,
        embedding_model: EmbeddingModel,
        collection_name: str | None = None,
    ) -> None:
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.embedding_model = embedding_model
        self.collection_name = collection_name or self.collection_name
        self.client = chromadb.PersistentClient(path=str(self.path))
        self.collection = self.client.get_or_create_collection(self.collection_name)
        self.write_lock = collection_write_lock(self.path, self.collection_name)

    def upsert(self, source: WebSource) -> None:
        # The document text is the title plus URL rather than the article: the
        # article is in the chunks, and duplicating it here would double the
        # store for a row nothing retrieves against.
        body = " ".join(part for part in (source.title, source.live_url) if part)
        with self.write_lock:
            self.collection.upsert(
                ids=[source.key],
                documents=[body or source.external_id],
                embeddings=[[0.0]],
                metadatas=[_source_metadata(source)],
            )

    def get(self, external_id: str) -> WebSource | None:
        result = self.collection.get(ids=[source_key(external_id)], include=["metadatas"])
        metadatas = result.get("metadatas") or []
        if not metadatas:
            return None
        return _source_from_metadata(dict(metadatas[0] or {}))

    def all(self) -> list[WebSource]:
        result = self.collection.get(include=["metadatas"])
        sources = [
            source
            for metadata in (result.get("metadatas") or [])
            if (source := _source_from_metadata(dict(metadata or {}))) is not None
        ]
        sources.sort(key=lambda item: item.last_fetched_at, reverse=True)
        return sources

    def for_channel(self, channel_id: str) -> list[WebSource]:
        return [source for source in self.all() if source.channel_id == channel_id]

    def delete(self, external_id: str) -> bool:
        key = source_key(external_id)
        if not (self.collection.get(ids=[key]).get("ids") or []):
            return False
        with self.write_lock:
            self.collection.delete(ids=[key])
        return True

    def count(self) -> int:
        return int(self.collection.count())


class WebChunkStore:
    """Section-anchored article chunks, in their own collection."""

    collection_name = WEB_CHUNK_COLLECTION

    def __init__(
        self,
        path: Path | str,
        embedding_model: EmbeddingModel,
        collection_name: str | None = None,
        exclude_external_ids: Sequence[str] | None = None,
    ) -> None:
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.embedding_model = embedding_model
        self.collection_name = collection_name or self.collection_name
        self.client = chromadb.PersistentClient(path=str(self.path))
        self.collection = self.client.get_or_create_collection(self.collection_name)
        self.write_lock = collection_write_lock(self.path, self.collection_name)
        #: Sources this store must behave as if it had never indexed, for the
        #: same held-out-evaluation reason ``TranscriptChunkStore`` has one.
        #: On the store rather than per call because exclusion has to hold on
        #: every read path, and a per-call argument is what one path forgets.
        self.exclude_external_ids: list[str] = sorted(dict.fromkeys(exclude_external_ids or []))

    def scoped_where(self, where: dict[str, Any] | None = None) -> dict[str, Any] | None:
        if not self.exclude_external_ids:
            return where
        excluded: dict[str, Any] = {"external_id": {"$nin": list(self.exclude_external_ids)}}
        if where is None:
            return excluded
        return {"$and": [where, excluded]}

    def _existing_vectors(self, key: str) -> dict[str, list[float]]:
        """Content hash to embedding, for everything already stored for one source.

        Keyed by hash rather than by chunk id so a paragraph that merely moved
        keeps its vector. See the module docstring.
        """
        result = self.collection.get(where={"source_key": key}, include=["metadatas", "embeddings"])
        vectors: dict[str, list[float]] = {}
        embeddings = result.get("embeddings")
        embeddings = [] if embeddings is None else list(embeddings)
        for metadata, embedding in zip(result.get("metadatas") or [], embeddings):
            digest = (metadata or {}).get("content_hash")
            if digest and embedding is not None:
                vectors[str(digest)] = list(embedding)
        return vectors

    def chunk_ids_for(self, key: str) -> list[str]:
        result = self.collection.get(where={"source_key": key}, include=[])
        return sorted(result.get("ids") or [])

    def replace(self, key: str, chunks: list[WebChunk]) -> tuple[int, list[str]]:
        """Make ``chunks`` the whole truth for one source.

        Returns how many chunks had to be embedded and which stored ids were
        removed — the shrinking tail a re-chunk leaves behind. Replace rather
        than upsert for the same reason the transcript indexer replaces: the
        rebuilt set *is* the document, so anything left at a higher index has
        to go with it.
        """
        reusable = self._existing_vectors(key)
        to_embed = [chunk for chunk in chunks if chunk.content_hash not in reusable]
        if to_embed:
            fresh = self.embedding_model.embed_documents(
                [chunk.embedding_text for chunk in to_embed]
            )
            for chunk, embedding in zip(to_embed, fresh):
                reusable[chunk.content_hash] = list(embedding)

        stale = set(self.chunk_ids_for(key)) - {chunk.chunk_id for chunk in chunks}
        with self.write_lock:
            if chunks:
                self.collection.upsert(
                    ids=[chunk.chunk_id for chunk in chunks],
                    documents=[chunk.text for chunk in chunks],
                    embeddings=[reusable[chunk.content_hash] for chunk in chunks],
                    metadatas=[self._metadata(chunk) for chunk in chunks],
                )
            if stale:
                self.collection.delete(ids=sorted(stale))
        return len(to_embed), sorted(stale)

    @staticmethod
    def _metadata(chunk: WebChunk) -> dict[str, str | int | float | bool]:
        data = chunk.model_dump(exclude={"text"})
        return {key: _scalar(value) for key, value in data.items() if value is not None}

    def delete_source(self, key: str) -> list[str]:
        ids = self.chunk_ids_for(key)
        if ids:
            with self.write_lock:
                self.collection.delete(ids=ids)
        return ids

    def count(self) -> int:
        return int(self.collection.count())

    def has_any(self) -> bool:
        return self.count() > 0

    def query(
        self,
        query: str,
        top_k: int,
        where: dict[str, Any] | None = None,
    ) -> list[RetrievedWebChunk]:
        if not self.count():
            return []
        embedding = self.embedding_model.embed_query(query)
        result = self.collection.query(
            query_embeddings=[embedding],
            n_results=max(1, top_k),
            where=self.scoped_where(where),
            include=["documents", "metadatas", "distances"],
        )
        chunks: list[RetrievedWebChunk] = []
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        for text, metadata, distance in zip(documents, metadatas, distances):
            payload = dict(metadata or {})
            fields = {
                key: value
                for key, value in payload.items()
                if key in RetrievedWebChunk.model_fields
            }
            try:
                chunk = RetrievedWebChunk(**{**fields, "text": text or ""})
            except ValueError:
                continue
            chunk.score = None if distance is None else 1.0 - float(distance)
            chunks.append(chunk)
        return chunks

    def all_chunks(self) -> list[WebChunk]:
        result = self.collection.get(where=self.scoped_where(), include=["documents", "metadatas"])
        chunks: list[WebChunk] = []
        for text, metadata in zip(result.get("documents") or [], result.get("metadatas") or []):
            payload = dict(metadata or {})
            fields = {key: value for key, value in payload.items() if key in WebChunk.model_fields}
            try:
                chunks.append(WebChunk(**{**fields, "text": text or ""}))
            except ValueError:
                continue
        chunks.sort(key=lambda item: (item.source_key, item.chunk_index))
        return chunks

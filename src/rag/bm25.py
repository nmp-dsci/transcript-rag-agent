"""Okapi BM25 keyword ranking over stored transcript chunks.

The RAG pipeline retrieves semantically: a question is embedded and compared to
chunk embeddings by cosine distance. That finds paraphrases but can miss exact
terms, so the Retrieval Lab ranks the same chunk corpus lexically as well and
shows where the two disagree.

Scoring is delegated to ``rank_bm25`` (a small pure-Python package). The index
is built in memory from chunk texts and cached per corpus fingerprint, which is
appropriate at this project's scale — hundreds to a few thousand chunks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Sequence

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens — transcripts have no markup to strip."""
    return _TOKEN.findall(str(text).lower())


@dataclass
class Bm25Index:
    """A built BM25 index over a fixed list of chunk records."""

    records: list[dict[str, Any]]
    _bm25: Any
    _doc_tokens: list[set[str]]

    @classmethod
    def build(cls, records: Sequence[dict[str, Any]]) -> "Bm25Index":
        from rank_bm25 import BM25Okapi

        kept = [record for record in records if str(record.get("text", "")).strip()]
        corpus = [tokenize(record["text"]) for record in kept]
        # BM25Okapi divides by the corpus average length and rejects an empty
        # corpus, so an empty index carries no scorer and returns no results.
        return cls(
            records=list(kept),
            _bm25=BM25Okapi(corpus) if corpus else None,
            _doc_tokens=[set(tokens) for tokens in corpus],
        )

    def search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        """Top-k records containing at least one query term, best score first.

        Membership decides what counts as a hit, not the score. BM25's IDF term
        floors to zero for a term that appears in roughly half the corpus, so a
        zero score means "unremarkable term", not "no match" — filtering on
        score would silently drop real hits from a small corpus.
        """
        if self._bm25 is None or top_k <= 0:
            return []
        tokens = tokenize(query)
        if not tokens:
            return []
        wanted = set(tokens)
        scores = self._bm25.get_scores(tokens)
        hits = [i for i, doc in enumerate(self._doc_tokens) if wanted & doc]
        hits.sort(key=lambda i: float(scores[i]), reverse=True)
        return [
            {
                **self.records[index],
                "rank": rank,
                "score": round(float(scores[index]), 4),
            }
            for rank, index in enumerate(hits[:top_k], start=1)
        ]


_CACHE: dict[tuple[str | None, int], Bm25Index] = {}


def search(
    records: Sequence[dict[str, Any]],
    query: str,
    top_k: int,
    *,
    cache_key: str | None = None,
) -> list[dict[str, Any]]:
    """Rank ``records`` against ``query``, reusing a cached index when possible.

    The cache is keyed by ``(cache_key, len(records))`` so indexing new content
    invalidates it automatically — chunk counts only change when the corpus does.
    """
    key = (cache_key, len(records))
    index = _CACHE.get(key)
    if index is None:
        index = Bm25Index.build(records)
        _CACHE[key] = index
    return index.search(query, top_k)


def clear_cache() -> None:
    _CACHE.clear()

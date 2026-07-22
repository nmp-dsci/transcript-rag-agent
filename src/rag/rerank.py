"""Cross-encoder reranking of retrieved chunks.

Retrieval is a recall stage: bi-encoder embeddings are compared without ever
seeing the query and the chunk together, which is what makes them fast enough
to search a whole corpus. A cross-encoder scores the (query, chunk) *pair*
jointly, so it judges relevance far more accurately — but only at a cost that
is affordable over a handful of candidates. The standard arrangement, and the
one here, is retrieve wide then rerank narrow.

The model is loaded lazily on the first ``rerank`` call, never at import or
construction, matching ``RagasJudge.from_settings`` and the API's lazy holders:
importing this module must stay free for the CLI, the tests, and any code path
that never reranks.

``sentence-transformers`` is already a direct project dependency, so
``CrossEncoder`` adds nothing new to install.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, Sequence, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from src.rag.models import RetrievedChunk

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# model name -> a loaded model exposing .predict(pairs)
Loader = Callable[[str], Any]


class Reranker(Protocol):
    """The interface callers switch on, so config never becomes a branch."""

    def rerank(
        self, query: str, chunks: Sequence[Any], top_k: int
    ) -> list["RetrievedChunk"]: ...

    def rerank_with_scores(
        self, query: str, chunks: Sequence[Any], top_k: int
    ) -> list[tuple["RetrievedChunk", float]]: ...


def _load_cross_encoder(name: str) -> Any:
    # Imported inside the function: sentence_transformers pulls in torch, which
    # costs seconds of import time that non-reranking code paths must not pay.
    from sentence_transformers import CrossEncoder

    return CrossEncoder(name)


@dataclass
class CrossEncoderReranker:
    """Reorders chunks by a cross-encoder's (query, chunk) relevance score.

    Returned chunks keep their original retrieval ``.score`` untouched. The
    rerank score is *not* attached to the chunk: ``RetrievedChunk`` is a
    pydantic model with no such field, and ``model_copy(update=...)`` would set
    an attribute that silently disappears from ``model_dump()`` — the score
    would be lost the moment the API serialized the chunk. Callers that need
    the score use :meth:`rerank_with_scores`, which returns it alongside.

    ``loader`` exists so tests can inject a fake model; leaving it ``None``
    loads the real ``sentence_transformers.CrossEncoder``.
    """

    model_name: str = DEFAULT_MODEL
    loader: Loader | None = None
    _model: Any = field(default=None, init=False, repr=False)
    _lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )

    @classmethod
    def from_model_name(cls, name: str = DEFAULT_MODEL) -> "CrossEncoderReranker":
        """Build a reranker for ``name``. Loads nothing until first use."""
        return cls(model_name=name)

    @property
    def model_loaded(self) -> bool:
        """Whether the model has been loaded yet (laziness is testable)."""
        return self._model is not None

    def _get_model(self) -> Any:
        # Double-checked under a lock: concurrent requests must not each pay
        # for their own copy of the model.
        if self._model is None:
            with self._lock:
                if self._model is None:
                    load = self.loader or _load_cross_encoder
                    self._model = load(self.model_name)
        return self._model

    def rerank(
        self, query: str, chunks: Sequence[Any], top_k: int
    ) -> list["RetrievedChunk"]:
        """The ``top_k`` most relevant chunks to ``query``, best first."""
        return [chunk for chunk, _score in self.rerank_with_scores(query, chunks, top_k)]

    def rerank_with_scores(
        self, query: str, chunks: Sequence[Any], top_k: int
    ) -> list[tuple["RetrievedChunk", float]]:
        """:meth:`rerank`, paired with each chunk's cross-encoder score.

        Scores are raw model logits, not probabilities, and are comparable only
        within one call. Ties keep the input order.
        """
        if top_k <= 0 or not chunks:
            return []

        pairs = [(query, getattr(chunk, "text", "") or "") for chunk in chunks]
        raw = self._get_model().predict(pairs)
        scores = [float(value) for value in raw]
        if len(scores) != len(chunks):
            raise ValueError(
                f"reranker returned {len(scores)} scores for {len(chunks)} chunks"
            )

        ordered = sorted(
            enumerate(chunks), key=lambda item: (-scores[item[0]], item[0])
        )
        return [(chunk, scores[index]) for index, chunk in ordered[:top_k]]


@dataclass
class NullReranker:
    """Passthrough reranker: same interface, no model, no reordering.

    Lets a caller hold a reranker unconditionally and switch on config at
    construction instead of branching at every retrieval site.
    """

    def rerank(
        self, query: str, chunks: Sequence[Any], top_k: int
    ) -> list["RetrievedChunk"]:
        if top_k <= 0:
            return []
        return list(chunks[:top_k])

    def rerank_with_scores(
        self, query: str, chunks: Sequence[Any], top_k: int
    ) -> list[tuple["RetrievedChunk", float]]:
        """Scores mirror the incoming order (descending) so callers that sort
        or display them see the original ranking preserved rather than a flat
        tie that some other sort could silently reshuffle."""
        selected = self.rerank(query, chunks, top_k)
        return [(chunk, float(len(selected) - index)) for index, chunk in enumerate(selected)]

from __future__ import annotations

import threading
from typing import Any, Protocol


class EmbeddingModel(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class HuggingFaceEmbeddingModel:
    """Sentence-transformers embeddings, loaded on first embed call.

    The weights are hundreds of megabytes and take seconds to load, while
    several callers only need an ``EmbeddingModel`` to satisfy a store
    constructor whose read-only methods never embed anything (the ingestion
    graph hook's ``chunks_for_videos``, for one). Deferring the load to the
    first ``embed_*`` call keeps those paths free of it, and costs the paths
    that do embed nothing but the same load a moment later.
    """

    def __init__(self, model_name: str, device: str | None = None) -> None:
        self.model_name = model_name
        self.device = device
        self._lock = threading.Lock()
        self._embeddings: Any | None = None

    def _resolve_device(self) -> str:
        """Pin the torch device rather than letting sentence-transformers pick.

        Its auto-selection prefers MPS on Apple Silicon, where the first embed
        call never returns — it wedges inside the Metal driver, and under
        uvicorn that hangs the whole server with the port still bound. The
        default comes from settings (CPU unless overridden), so a machine with
        a working GPU can opt back in without touching call sites.
        """
        if self.device is not None:
            return self.device
        from src.config import load_settings

        return load_settings(require_keys=False).embedding_device

    def _model(self) -> Any:
        with self._lock:
            if self._embeddings is None:
                from langchain_huggingface import HuggingFaceEmbeddings

                self._embeddings = HuggingFaceEmbeddings(
                    model_name=self.model_name,
                    model_kwargs={"device": self._resolve_device()},
                )
            return self._embeddings

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        result: list[list[float]] = self._model().embed_documents(texts)
        return result

    def embed_query(self, text: str) -> list[float]:
        result: list[float] = self._model().embed_query(text)
        return result


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = sum(a * a for a in left) ** 0.5
    right_norm = sum(b * b for b in right) ** 0.5
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)

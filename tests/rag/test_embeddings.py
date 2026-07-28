"""The embedding wrapper loads its weights lazily.

Several call sites only need an ``EmbeddingModel`` to satisfy a store
constructor and never embed anything (the API's ingestion graph hook reads
chunks back by metadata), so paying for the sentence-transformers load at
construction time is pure waste on those paths.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.rag.embeddings import HuggingFaceEmbeddingModel

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class _FakeEmbeddings:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.0, 1.0]


@pytest.fixture
def loads(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    import langchain_huggingface

    names: list[str] = []

    def fake_embeddings(**kwargs: Any) -> _FakeEmbeddings:
        names.append(kwargs["model_name"])
        return _FakeEmbeddings()

    monkeypatch.setattr(langchain_huggingface, "HuggingFaceEmbeddings", fake_embeddings)
    return names


def test_construction_does_not_load_the_weights(loads: list[str]) -> None:
    HuggingFaceEmbeddingModel(MODEL_NAME)
    assert loads == []


def test_the_weights_load_once_on_first_use(loads: list[str]) -> None:
    model = HuggingFaceEmbeddingModel(MODEL_NAME)

    assert model.embed_query("what did they say") == [0.0, 1.0]
    assert loads == [MODEL_NAME]

    assert model.embed_documents(["a", "b"]) == [[1.0, 0.0], [1.0, 0.0]]
    assert loads == [MODEL_NAME]

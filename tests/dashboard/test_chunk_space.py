from __future__ import annotations

import numpy as np

from src.dashboard.chunk_space import (
    fit_chunk_projection,
    nearest_chunks_for_question,
    transform_question,
)


def test_fit_chunk_projection_and_transform_question_share_axes() -> None:
    embeddings = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    projection = fit_chunk_projection(embeddings, ["a", "b", "c"], "fake")
    question = transform_question(projection, np.asarray([1.0, 0.0, 0.0]))

    assert projection.method == "pca"
    assert projection.n_chunks == 3
    assert len(projection.chunk_coords) == 3
    assert len(question) == 2


def test_nearest_chunks_scores_original_embedding_space() -> None:
    embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.8, 0.2]])

    nearest = nearest_chunks_for_question(
        np.asarray([1.0, 0.0]),
        embeddings,
        ["exact", "orthogonal", "near"],
        top_k=2,
    )

    assert [item.chunk_id for item in nearest] == ["exact", "near"]
    assert nearest[0].score == 1.0

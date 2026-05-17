from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.decomposition import IncrementalPCA, PCA


@dataclass(frozen=True)
class ChunkProjection:
    method: str
    fitted_at: str
    embedding_model: str
    n_chunks: int
    explained_variance: tuple[float, float]
    components: list[list[float]]
    mean: list[float]
    chunk_coords: list[tuple[str, float, float]]


@dataclass(frozen=True)
class NearestChunk:
    chunk_id: str
    score: float


def fit_chunk_projection(
    chunk_embeddings: np.ndarray,
    chunk_ids: list[str],
    embedding_model: str,
) -> ChunkProjection:
    if len(chunk_ids) != len(chunk_embeddings):
        raise ValueError("chunk_ids and chunk_embeddings must have the same length")
    if len(chunk_ids) < 2:
        raise ValueError("At least two chunk embeddings are required for PCA")
    reducer = (
        IncrementalPCA(n_components=2)
        if len(chunk_ids) > 50_000
        else PCA(n_components=2)
    )
    coords = reducer.fit_transform(chunk_embeddings)
    return ChunkProjection(
        method="pca",
        fitted_at=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        embedding_model=embedding_model,
        n_chunks=len(chunk_ids),
        explained_variance=tuple(float(value) for value in reducer.explained_variance_ratio_[:2]),
        components=np.asarray(reducer.components_).astype(float).tolist(),
        mean=np.asarray(reducer.mean_).astype(float).tolist(),
        chunk_coords=[
            (chunk_id, float(coords[index][0]), float(coords[index][1]))
            for index, chunk_id in enumerate(chunk_ids)
        ],
    )


def transform_question(
    projection: ChunkProjection,
    question_embedding: np.ndarray,
) -> tuple[float, float]:
    centered = question_embedding.astype(float) - np.asarray(projection.mean, dtype=float)
    components = np.asarray(projection.components, dtype=float)
    coords = centered @ components.T
    return float(coords[0]), float(coords[1])


def nearest_chunks_for_question(
    question_embedding: np.ndarray,
    chunk_embeddings: np.ndarray,
    chunk_ids: list[str],
    top_k: int,
) -> list[NearestChunk]:
    if len(chunk_ids) != len(chunk_embeddings):
        raise ValueError("chunk_ids and chunk_embeddings must have the same length")
    query = question_embedding.astype(float)
    matrix = chunk_embeddings.astype(float)
    query_norm = np.linalg.norm(query)
    matrix_norms = np.linalg.norm(matrix, axis=1)
    denominators = matrix_norms * query_norm
    scores = np.divide(
        matrix @ query,
        denominators,
        out=np.zeros(len(matrix), dtype=float),
        where=denominators != 0,
    )
    order = np.argsort(scores)[::-1][:top_k]
    return [
        NearestChunk(chunk_id=chunk_ids[index], score=float(scores[index]))
        for index in order
    ]


def projection_to_json(projection: ChunkProjection) -> dict:
    return asdict(projection)


def projection_from_json(data: dict) -> ChunkProjection:
    return ChunkProjection(
        method=str(data["method"]),
        fitted_at=str(data["fitted_at"]),
        embedding_model=str(data["embedding_model"]),
        n_chunks=int(data["n_chunks"]),
        explained_variance=tuple(float(value) for value in data["explained_variance"][:2]),
        components=[[float(value) for value in row] for row in data["components"]],
        mean=[float(value) for value in data["mean"]],
        chunk_coords=[
            (str(item[0]), float(item[1]), float(item[2]))
            for item in data.get("chunk_coords", [])
        ],
    )


def write_projection_artifact(path: Path, projection: ChunkProjection) -> bool:
    payload = json.dumps(projection_to_json(projection), indent=2, sort_keys=True)
    if len(payload.encode("utf-8")) > 5_000_000:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
    return True

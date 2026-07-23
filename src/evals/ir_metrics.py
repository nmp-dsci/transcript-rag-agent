"""Information-retrieval metrics over the golden set's chunk-level labels.

:mod:`src.evals.golden` already computes ``context_recall`` — the fraction of a
question's expected chunks that retrieval surfaced *anywhere* in its output. That
answers "did we find the evidence?" but not "did we rank it well?". A retriever
that buries the one relevant chunk at position 30 scores the same ``context_recall``
as one that puts it first, yet the answer model only ever sees the top few.

This module adds the rank-aware half of the standard IR toolkit, all derived from
the same two inputs the golden set already provides — the *ordered* list of
retrieved chunk ids and the *set* of relevant (expected) chunk ids — under binary
relevance (a chunk is relevant iff it is in the expected set):

* :func:`recall_at_k` — recall restricted to the top ``k`` results. Swept across
  ``k`` by :func:`recall_curve`, this is the recall@k curve.
* :func:`reciprocal_rank` — ``1 / rank`` of the first relevant result. Averaged
  over the question set (see :func:`mean_metrics`) this is the familiar MRR.
* :func:`ndcg_at_k` — normalized discounted cumulative gain, which rewards putting
  relevant chunks *early*, not merely somewhere in the top ``k``.

Everything here is pure arithmetic over ids: deterministic, free, and reproducible,
exactly like :func:`src.evals.golden.context_recall`. No LLM, no embeddings.
"""

from __future__ import annotations

from math import log2

#: The k cut-offs reported for every golden run. recall@10 collapses to recall@5
#: when only 5 chunks were retrieved, which is intended — a run's top_k is part of
#: the configuration being compared, and the curve stays meaningful across runs.
DEFAULT_KS: tuple[int, ...] = (1, 3, 5, 10)

#: Per-entry metric keys :func:`entry_ir_metrics` emits, in report order. Averaged
#: over the entries (:func:`mean_metrics`), ``mrr`` becomes the set's true Mean
#: Reciprocal Rank; the rest keep their names.
IR_METRIC_NAMES: list[str] = [f"recall@{k}" for k in DEFAULT_KS] + ["mrr", "ndcg@10"]


def _dedupe(ids: list[str]) -> list[str]:
    """``ids`` with later duplicates dropped, first-seen order preserved.

    Rank metrics must not give a chunk two chances at being "the first relevant
    hit", and a chunk retrieved twice is still one retrieval. Order matters here
    (unlike :func:`src.evals.golden.context_recall`, which can use a set), so this
    keeps the earliest occurrence rather than collapsing to an unordered set.
    """
    seen: set[str] = set()
    unique: list[str] = []
    for item in ids:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


def recall_at_k(retrieved_chunk_ids: list[str], expected_chunk_ids: list[str], k: int) -> float:
    """Fraction of the expected chunks found within the top ``k`` retrieved.

    Convention: recall of nothing is ``1.0`` — with no expected ids there is
    nothing to miss, matching :func:`src.evals.golden.context_recall`. ``k <= 0``
    retrieves nothing, so it scores ``0.0`` unless there was nothing to find.
    """
    expected = set(expected_chunk_ids)
    if not expected:
        return 1.0
    if k <= 0:
        return 0.0
    top_k = set(_dedupe(retrieved_chunk_ids)[:k])
    return len(expected & top_k) / len(expected)


def recall_curve(
    retrieved_chunk_ids: list[str],
    expected_chunk_ids: list[str],
    ks: tuple[int, ...] = DEFAULT_KS,
) -> dict[int, float]:
    """:func:`recall_at_k` at each cut-off in ``ks`` — the recall@k curve."""
    return {k: recall_at_k(retrieved_chunk_ids, expected_chunk_ids, k) for k in ks}


def reciprocal_rank(retrieved_chunk_ids: list[str], expected_chunk_ids: list[str]) -> float:
    """``1 / rank`` of the first relevant chunk, or ``0.0`` if none was retrieved.

    Ranks are 1-based, so a relevant chunk in first position scores ``1.0`` and one
    in third scores ``1/3``. Averaged over the question set this is MRR; on its own
    it is that question's reciprocal rank.
    """
    if not expected_chunk_ids:
        return 1.0
    expected = set(expected_chunk_ids)
    for rank, chunk_id in enumerate(_dedupe(retrieved_chunk_ids), start=1):
        if chunk_id in expected:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_chunk_ids: list[str], expected_chunk_ids: list[str], k: int) -> float:
    """Normalized discounted cumulative gain over the top ``k``, binary relevance.

    ``DCG@k = Σ rel_i / log2(i + 1)`` rewards ranking relevant chunks early; the
    ideal ordering (every relevant chunk first) gives ``IDCG@k``, and the ratio
    normalizes to ``[0, 1]`` so runs with different numbers of relevant chunks are
    comparable. With nothing to find the result is ``1.0`` by the same convention
    as :func:`recall_at_k`.
    """
    expected = set(expected_chunk_ids)
    if not expected:
        return 1.0
    if k <= 0:
        return 0.0
    ranked = _dedupe(retrieved_chunk_ids)[:k]
    dcg = sum(1.0 / log2(rank + 1) for rank, cid in enumerate(ranked, start=1) if cid in expected)
    ideal_hits = min(len(expected), k)
    idcg = sum(1.0 / log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def entry_ir_metrics(
    retrieved_chunk_ids: list[str],
    expected_chunk_ids: list[str],
    ks: tuple[int, ...] = DEFAULT_KS,
    ndcg_k: int = 10,
) -> dict[str, float]:
    """All per-entry IR metrics for one answered golden question, rounded.

    Keys are :data:`IR_METRIC_NAMES`. ``mrr`` here is the entry's reciprocal rank;
    it earns its name once averaged across the set by :func:`mean_metrics`.
    """
    scores: dict[str, float] = {
        f"recall@{k}": round(recall_at_k(retrieved_chunk_ids, expected_chunk_ids, k), 4)
        for k in ks
    }
    scores["mrr"] = round(reciprocal_rank(retrieved_chunk_ids, expected_chunk_ids), 4)
    scores[f"ndcg@{ndcg_k}"] = round(ndcg_at_k(retrieved_chunk_ids, expected_chunk_ids, ndcg_k), 4)
    return scores


def mean_metrics(per_entry: list[dict[str, float | None]], names: list[str]) -> dict[str, float]:
    """Mean of each named metric over the entries that reported a number.

    An entry missing a metric (``None``, or absent) is skipped for that metric
    rather than counted as zero, so a crashed question never silently drags an
    average down. Averaging the per-entry ``mrr`` column yields the set's MRR.
    """
    averages: dict[str, float] = {}
    for name in names:
        values = [
            value
            for entry in per_entry
            if isinstance((value := entry.get(name)), (int, float))
        ]
        if values:
            averages[name] = round(sum(values) / len(values), 4)
    return averages

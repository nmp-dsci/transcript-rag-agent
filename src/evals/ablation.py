"""Retrieval ablations over the golden set — the "which retrieval wins" harness.

:mod:`src.evals.regression` snapshots one configuration end-to-end (retrieve →
answer → judge) so a change can be shown not to regress. This module answers the
adjacent, retrieval-science question the roadmap's P1 is about: *holding the
questions fixed, which retrieval configuration ranks the evidence best?* — e.g.
does hybrid+rerank beat plain semantic on recall@10 and NDCG, and by how much.

It measures **retrieval only**. No answer is generated and no judge runs, so every
metric here is the deterministic, id-based arithmetic of :mod:`src.evals.ir_metrics`
and :mod:`src.evals.golden`: free, fast, and reproducible without an API key. That
is what makes sweeping several configurations over the whole golden set cheap.

The heavy wiring (embeddings, Chroma, the cross-encoder) lives behind a ``retrieve``
callable so the aggregation logic can be unit-tested with a fake retriever; the CLI
supplies the real one via :func:`build_retrieve`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from src.evals.golden import GoldenEntry, evaluate_entry, load_golden
from src.evals.ir_metrics import IR_METRIC_NAMES, mean_metrics

#: Deterministic metrics reported per configuration, in table order. Coverage
#: (the recalls) then ranking quality (recall@k, MRR, NDCG). All id-based.
ABLATION_METRICS: list[str] = ["context_recall", "video_recall", *IR_METRIC_NAMES]

# (question, config) -> the ordered chunk ids that configuration retrieved.
RetrieveFn = Callable[[str, "AblationConfig"], list[str]]


@dataclass(frozen=True)
class AblationConfig:
    """One retrieval configuration to measure, identified by ``label``."""

    label: str
    retrieval_mode: str = "semantic"
    rerank: bool = False
    neighbor_span: int = 0
    top_k: int = 10

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "retrieval_mode": self.retrieval_mode,
            "rerank": self.rerank,
            "neighbor_span": self.neighbor_span,
            "top_k": self.top_k,
        }


#: The default sweep: isolate the two axes P1 cares about — lexical fusion and
#: cross-encoder reranking — against the plain-semantic baseline. ``semantic`` is
#: first, so it is the baseline every delta is measured from.
def default_configs(top_k: int = 10) -> list[AblationConfig]:
    return [
        AblationConfig(label="semantic", retrieval_mode="semantic", rerank=False, top_k=top_k),
        AblationConfig(label="hybrid", retrieval_mode="hybrid", rerank=False, top_k=top_k),
        AblationConfig(
            label="hybrid+rerank", retrieval_mode="hybrid", rerank=True, top_k=top_k
        ),
    ]


@dataclass
class _CellResult:
    config: AblationConfig
    entries: list[dict[str, Any]] = field(default_factory=list)


def _score_entry(entry: GoldenEntry, chunk_ids: list[str]) -> dict[str, float | None]:
    """The deterministic subset of :func:`evaluate_entry` for one retrieval."""
    scores = evaluate_entry(entry, "", chunk_ids)
    return {name: scores.get(name) for name in ABLATION_METRICS}


def run_ablation(
    entries: list[GoldenEntry],
    configs: list[AblationConfig],
    retrieve: RetrieveFn,
    *,
    now: datetime | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Measure every configuration over every entry and compare them.

    Each configuration retrieves for each golden question; the ordered chunk ids
    are scored with the deterministic id-based metrics and averaged, both overall
    and per domain (property vs ai-coding — segment-level reporting, not just a
    single number). The first configuration is the baseline every other config's
    ``deltas`` are measured against.
    """
    if not configs:
        raise ValueError("run_ablation needs at least one configuration")

    cells: list[_CellResult] = []
    for config in configs:
        cell = _CellResult(config=config)
        for index, entry in enumerate(entries, start=1):
            if on_progress is not None:
                on_progress(f"[{config.label}] [{index}/{len(entries)}] {entry.id}")
            chunk_ids = list(retrieve(entry.question, config))
            cell.entries.append(
                {
                    "id": entry.id,
                    "domain": entry.domain,
                    "retrieved_chunk_ids": chunk_ids,
                    "scores": _score_entry(entry, chunk_ids),
                }
            )
        cells.append(cell)

    summaries = [_summarize_cell(cell) for cell in cells]
    baseline = summaries[0]
    deltas = [_delta(baseline, summary) for summary in summaries[1:]]

    moment = now or datetime.now(timezone.utc)
    return {
        "run_id": f"ablation-{moment.strftime('%Y%m%d-%H%M%S')}",
        "created_at": moment.isoformat(),
        "kind": "retrieval-ablation",
        "entries": len(entries),
        "metrics": ABLATION_METRICS,
        "baseline": baseline["label"],
        "cells": summaries,
        "deltas": deltas,
    }


def _summarize_cell(cell: _CellResult) -> dict[str, Any]:
    per_entry = [entry["scores"] for entry in cell.entries]
    by_domain: dict[str, dict[str, float]] = {}
    domains = sorted({entry["domain"] for entry in cell.entries})
    for domain in domains:
        scoped = [e["scores"] for e in cell.entries if e["domain"] == domain]
        by_domain[domain] = mean_metrics(scoped, ABLATION_METRICS)
    return {
        "label": cell.config.label,
        "config": cell.config.to_dict(),
        "averages": mean_metrics(per_entry, ABLATION_METRICS),
        "by_domain": by_domain,
        "entries": cell.entries,
    }


def _delta(baseline: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    base_avg = baseline["averages"]
    avg = summary["averages"]
    return {
        "label": summary["label"],
        "vs_baseline": {
            metric: round(avg[metric] - base_avg[metric], 4)
            for metric in ABLATION_METRICS
            if metric in avg and metric in base_avg
        },
    }


def build_retrieve(settings: Any) -> RetrieveFn:
    """A ``retrieve`` over the real stack: one provider per configuration.

    Stores, the embedding model and the cross-encoder are built once and shared
    across configurations; only the lightweight provider wrapper varies per config,
    so a sweep loads each model at most once. Retrieval runs corpus-wide with no
    indexer, so the ablation measures the corpus exactly as it stands.
    """
    from src.rag.context import MultiTranscriptRagContextProvider
    from src.rag.embeddings import HuggingFaceEmbeddingModel
    from src.rag.storage import RawTranscriptStore, TranscriptChunkStore

    embedding_model = HuggingFaceEmbeddingModel(settings.embedding_model)
    raw_store = RawTranscriptStore(
        settings.chroma_path, collection_name=settings.raw_transcript_collection
    )
    chunk_store = TranscriptChunkStore(
        settings.chroma_path,
        embedding_model=embedding_model,
        collection_name=settings.chunk_collection,
    )

    reranker = None

    def _reranker() -> Any:
        nonlocal reranker
        if reranker is None:
            from src.rag.rerank import CrossEncoderReranker

            reranker = CrossEncoderReranker.from_model_name(settings.rerank_model)
        return reranker

    providers: dict[str, MultiTranscriptRagContextProvider] = {}

    def provider_for(config: AblationConfig) -> MultiTranscriptRagContextProvider:
        if config.label not in providers:
            providers[config.label] = MultiTranscriptRagContextProvider(
                raw_store=raw_store,
                chunk_store=chunk_store,
                retrieval_mode=config.retrieval_mode,
                retrieval_candidates=settings.retrieval_candidates,
                reranker=_reranker() if config.rerank else None,
                neighbor_span=config.neighbor_span,
            )
        return providers[config.label]

    def retrieve(question: str, config: AblationConfig) -> list[str]:
        context = provider_for(config).get_context(
            question, top_k=config.top_k, retrieval_mode=config.retrieval_mode
        )
        return [
            f"chunk:{chunk.video_id}:{chunk.chunk_index}"
            for chunk in (context.retrieved_chunks or [])
            if getattr(chunk, "video_id", None) is not None
        ]

    return retrieve


def run_default_ablation(
    settings: Any,
    *,
    top_k: int | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Load the golden set and sweep :func:`default_configs` over the real stack."""
    entries = load_golden()
    configs = default_configs(top_k=top_k or settings.rag_top_k)
    return run_ablation(entries, configs, build_retrieve(settings), on_progress=on_progress)


def format_table(result: dict[str, Any]) -> str:
    """A fixed-width comparison table for the terminal, baseline row first."""
    metrics = result["metrics"]
    header = "  ".join([f"{'config':<14}"] + [f"{m:>13}" for m in metrics])
    lines = [header, "-" * len(header)]
    for cell in result["cells"]:
        avg = cell["averages"]
        row = "  ".join(
            [f"{cell['label']:<14}"]
            + [f"{avg.get(m, float('nan')):>13.3f}" for m in metrics]
        )
        lines.append(row)
    if result["deltas"]:
        lines.append("")
        lines.append(f"deltas vs {result['baseline']}:")
        for delta in result["deltas"]:
            moves = "  ".join(
                f"{metric} {value:+.3f}" for metric, value in delta["vs_baseline"].items()
            )
            lines.append(f"  {delta['label']:<14} {moves}")
    return "\n".join(lines)

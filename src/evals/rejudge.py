"""Re-score a committed matrix run under a different rubric.

Judging is independent of generation, so changing the rubric does not need the
matrix re-run. This reads a committed ``matrix-*.json``, keeps every answer and
every stored grounding score exactly as committed, computes only the metrics
the new rubric adds, and writes a second committed run.

Reusing the stored ``faithfulness`` / ``answer_relevancy`` /
``context_precision`` is deliberate rather than a shortcut. The point of the
new run is a *ranking comparison*: depth-v2 against the ragas-v1 run it came
from. Re-judging grounding would move those three numbers by judge
nondeterminism alone, and any change in the ranking would then be
unattributable — the comparison has to differ in the rubric and nothing else.

Contexts are not stored on a committed cell (only ``retrieved_chunk_ids``), so
they are resolved back out of the chunk store by id. A chunk that no longer
exists is dropped rather than faked; the cell records how many were resolved,
because a depth judgement made against half the context is a weaker
measurement and should say so.

    uv run python -m src.cli rejudge --run matrix-20260729-061607 --rubric depth-v2
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from src.config import Settings
from src.evals.judge import DEPTH_V2, MetricBreakdown, Rubric
from src.evals.matrix import build_comparison
from src.evals.regression import DEFAULT_RUNS_DIR

#: ``(question, answer, contexts) -> {metric: MetricBreakdown}``.
DepthScoreFn = Callable[[str, str, list[str]], dict[str, MetricBreakdown]]
#: ``chunk_ids -> chunk texts``, in the order the ids were retrieved.
ContextsFn = Callable[[Sequence[str]], list[str]]


def find_run(run_id: str, runs_dir: Path = DEFAULT_RUNS_DIR) -> dict[str, Any]:
    """The committed run with this id (or this path). Raises if there is none."""
    candidate = Path(run_id)
    path = candidate if candidate.suffix == ".json" else runs_dir / f"{run_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"no committed run at {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("kind") != "matrix":
        raise ValueError(f"{path.name} is not a matrix run")
    return dict(data)


def chunk_context_lookup(settings: Settings) -> ContextsFn:
    """Resolve retrieved chunk ids back to their text from the chunk store.

    One store, one bulk read per call scoped to the videos the ids name — the
    ids of one cell touch a handful of videos, so this never walks the corpus.
    """
    from src.rag.embeddings import HuggingFaceEmbeddingModel
    from src.rag.storage import TranscriptChunkStore

    store = TranscriptChunkStore(
        settings.chroma_path,
        embedding_model=HuggingFaceEmbeddingModel(settings.embedding_model),
        collection_name=settings.chunk_collection,
    )

    def lookup(chunk_ids: Sequence[str]) -> list[str]:
        video_ids = sorted({parts[1] for cid in chunk_ids if len(parts := cid.split(":")) == 3})
        if not video_ids:
            return []
        by_id = {chunk.chunk_id: chunk.text for chunk in store.chunks_for_videos(video_ids)}
        return [by_id[cid] for cid in chunk_ids if cid in by_id]

    return lookup


def rejudge_cell(
    cell: dict[str, Any],
    *,
    depth_fn: DepthScoreFn,
    contexts_fn: ContextsFn,
    rubric: Rubric = DEPTH_V2,
) -> dict[str, Any]:
    """One cell re-scored: stored grounding kept, depth added, composite redone.

    A cell that errored, produced no answer, or was never judged cannot be
    graded for depth — there is nothing to read. Such a cell is **not** carried
    through with the composite it already had: that number was computed under
    the *source* run's rubric, and averaging a ragas-v1 composite into a
    depth-v2 leaderboard silently mixes two scales in one mean. It is marked
    ``rejudged: false``, its composite is cleared so every consumer treats it
    as unjudged under this rubric, and the number it used to have is preserved
    under ``source_composite`` so nothing is lost.
    """
    scores = dict(cell.get("scores") or {})
    answer = str(cell.get("answer") or "")
    if cell.get("error") or not answer or scores.get("composite") is None:
        skipped = dict(cell)
        skipped["rejudged"] = False
        skipped["rubric_version"] = rubric.version
        skipped["rejudge_skipped_reason"] = (
            "cell errored" if cell.get("error") else "no answer to grade for depth"
        )
        if scores.get("composite") is not None:
            skipped["source_composite"] = scores["composite"]
            scores = {**scores, "composite": None}
            skipped["scores"] = scores
        return skipped

    chunk_ids = list(cell.get("retrieved_chunk_ids") or [])
    contexts = contexts_fn(chunk_ids)
    details: dict[str, Any] = {}
    depth_error: str | None = None
    try:
        for metric, breakdown in depth_fn(
            str(cell.get("question") or ""), answer, contexts
        ).items():
            scores[metric] = breakdown.score
            details[metric] = breakdown.details
    except Exception as exc:
        # The grounding half still composites; the cell says what is missing
        # rather than dropping out of the run.
        depth_error = str(exc)

    result = rubric.composite(scores)
    scores["composite"] = result.composite
    rejudged = dict(cell)
    rejudged["scores"] = scores
    rejudged["rubric_version"] = rubric.version
    rejudged["rejudged"] = True
    rejudged["composite_uncapped"] = result.uncapped
    rejudged["composite_weight"] = result.weight_used
    rejudged["cap_applied"] = result.cap_applied
    rejudged["cap_reason"] = result.cap_reason
    rejudged["grounding_floor_breached"] = result.grounding_floor_breached
    rejudged["grounding_reason"] = result.grounding_reason
    rejudged["depth_details"] = details or None
    rejudged["depth_error"] = depth_error
    rejudged["contexts_resolved"] = len(contexts)
    rejudged["contexts_expected"] = len(chunk_ids)
    return rejudged


def rejudge_run(
    run: dict[str, Any],
    *,
    depth_fn: DepthScoreFn,
    contexts_fn: ContextsFn,
    rubric: Rubric = DEPTH_V2,
    judge_model: str | None = None,
    on_progress: Callable[[str], None] | None = None,
    now: datetime | None = None,
    max_workers: int = 1,
) -> dict[str, Any]:
    """A new committed-run document: the same answers under ``rubric``.

    ``max_workers`` > 1 runs the per-cell depth calls concurrently. They are
    independent (one answer, one prompt, one result), so the wall clock of a
    rejudge is dominated by how many can be in flight at once.
    """
    from src.evals.matrix import summarize_cells

    moment = now or datetime.now(timezone.utc)
    stamp = moment.strftime("%Y%m%d-%H%M%S")
    # The id carries the stamp and the rubric and deliberately not the depth
    # judge, which has been asked for more than once — a rejudge and the run it
    # rejudged can differ in exactly the one thing the id does not name. The
    # rubric is in there because ``describe_matrix_run`` needs the picker to
    # distinguish two rankings produced under different scoring; the judge is
    # not, because it is already on the record twice, in ``depth_judge_model``
    # at run and setup level below, which is what ``depth_judge_models`` on the
    # Scoreboard and ``_self_graded`` in ``src/api/matrix_runs.py`` read to
    # decide whether a ranking graded itself.
    #
    # Naming it here would cost more than it says. ``save_run`` uses the id as
    # the filename, and the judge actually used for the independent run was
    # ``gpt-5.5``: ``demo/validate/status.py`` scrapes run ids with a character
    # class that stops at the dot, so such a run would be read as
    # ``...-judge-gpt-5`` and reported as missing from disk forever after. Ids
    # are also pinned by evaluator scripts and chained through
    # ``rejudged_from`` — they are join keys, and a join key that encodes
    # configuration rots the moment the configuration does. A run that needs
    # its judge legible in a *filename* can be copied out under one, which is
    # what ``demo/validate/artifacts/v0_independent_judge/`` did.
    run_id = f"matrix-{stamp}-{rubric.version}"

    cells: list[tuple[str, int, dict[str, Any]]] = [
        (setup, index, cell)
        for setup, setup_run in (run.get("runs") or {}).items()
        for index, cell in enumerate(setup_run.get("entries") or [])
    ]
    total = len(cells)
    done = 0

    def score_one(item: tuple[str, int, dict[str, Any]]) -> tuple[str, int, dict[str, Any]]:
        setup, index, cell = item
        return (
            setup,
            index,
            rejudge_cell(cell, depth_fn=depth_fn, contexts_fn=contexts_fn, rubric=rubric),
        )

    results: dict[tuple[str, int], dict[str, Any]] = {}
    for setup, index, rejudged in _map(score_one, cells, max_workers):
        results[(setup, index)] = rejudged
        done += 1
        if on_progress is not None:
            if not rejudged.get("rejudged"):
                note = f"skipped — {rejudged.get('rejudge_skipped_reason')}"
            else:
                composite = (rejudged.get("scores") or {}).get("composite")
                flag = (
                    " (capped)"
                    if rejudged.get("cap_applied")
                    else " (ungrounded)"
                    if rejudged.get("grounding_floor_breached")
                    else ""
                )
                note = f"composite {composite}{flag}"
            on_progress(f"[{done}/{total}] {setup} {rejudged.get('id')}: {note}")

    # ``depth_judge_model`` goes on the per-setup config as well as the run's,
    # because the Scoreboard reads a cell's provenance from the setup config it
    # belongs to — put it only at run level and the tab names the grounding
    # judge while silently omitting the one that produced 60% of the composite.
    setup_config_extra = {
        "rubric_version": rubric.version,
        "depth_judge_model": judge_model,
    }

    setup_runs: dict[str, dict[str, Any]] = {}
    for setup, setup_run in (run.get("runs") or {}).items():
        entries = [
            results[(setup, index)] for index, _ in enumerate(setup_run.get("entries") or [])
        ]
        setup_runs[setup] = {
            **setup_run,
            "run_id": f"matrix-{stamp}-{rubric.version}-{setup}",
            "created_at": moment.isoformat(),
            "config": {**(setup_run.get("config") or {}), **setup_config_extra},
            "entries": entries,
            "summary": summarize_cells(entries),
        }

    rejudged_cells = sum(1 for cell in results.values() if cell.get("rejudged"))
    config = {**(run.get("config") or {}), **setup_config_extra}
    return {
        **run,
        "run_id": run_id,
        "created_at": moment.isoformat(),
        "kind": "matrix",
        "rubric_version": rubric.version,
        # Which run's answers and grounding scores these are. Without it the two
        # runs look like independent measurements instead of one comparison.
        "rejudged_from": run.get("run_id"),
        # How much of the source run this rubric actually covers. A cell that
        # could not be rejudged is excluded from every average rather than
        # carried over at its old rubric's score, so the counts have to be
        # stated or the leaderboard's n is unexplained.
        "rejudged_cells": rejudged_cells,
        "skipped_cells": total - rejudged_cells,
        "config": config,
        "runs": setup_runs,
        "comparison": build_comparison(setup_runs),
    }


def _map(fn: Callable[[Any], Any], items: list[Any], max_workers: int) -> Iterable[Any]:
    """``map`` sequentially, or over a thread pool when asked for concurrency."""
    if max_workers <= 1:
        for item in items:
            yield fn(item)
        return
    from concurrent.futures import ThreadPoolExecutor, as_completed

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(fn, item) for item in items]
        for future in as_completed(futures):
            yield future.result()

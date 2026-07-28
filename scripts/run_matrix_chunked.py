"""Resumable, parallel driver for the judged head-to-head matrix (s05 §06).

``eval-matrix`` already caches every ``(engine, question)`` cell by default
(see :mod:`src.evals.matrix_cache`) — adding one new engine variant and
re-running it only scores that engine's cells, reusing everything else. This
driver reads and writes the exact same cache, so a run started here can be
finished by plain ``eval-matrix`` and vice versa. What it adds on top:

* **Parallel across engines.** One worker thread per engine (each engine's
  own cells stay sequential — a single agent instance is not thread-safe),
  so 4 engines' worth of judging overlaps instead of running back to back.
* **A time budget.** A fully judged run over every engine can take well over
  an hour, and background processes have been killed mid-run in this project
  before. ``--max-seconds`` stops cleanly and exits 3 once the budget is
  spent, with everything scored so far already in the cache — re-running the
  same command resumes exactly where it left off.

    uv run python scripts/run_matrix_chunked.py
    uv run python scripts/run_matrix_chunked.py --setups rag_llm,graph_rag
    uv run python scripts/run_matrix_chunked.py --max-seconds 300
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_settings  # noqa: E402
from src.evals.golden import GoldenEntry, load_golden  # noqa: E402
from src.evals.judge import ragas_version  # noqa: E402
from src.evals.matrix import (  # noqa: E402
    DEFAULT_MATRIX_SETUPS,
    _summarize_dicts,
    build_comparison,
    config_snapshot,
    format_matrix_table,
)
from src.evals.matrix_cache import (  # noqa: E402
    DEFAULT_CACHE_DIR,
    cell_fingerprint,
    load_cell,
    save_cell,
)
from src.evals.regression import run_golden_eval, save_run  # noqa: E402

logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--setups", default=",".join(DEFAULT_MATRIX_SETUPS))
    parser.add_argument(
        "--refresh", action="store_true", help="Bypass the cache and rescore every cell"
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help=(
            "Stop scoring new cells after this many seconds and exit 3; the "
            "cache resumes on the next run. Lets the driver run inside a "
            "bounded shell without losing work."
        ),
    )
    args = parser.parse_args()
    setups = [token.strip() for token in args.setups.split(",") if token.strip()]

    settings = load_settings()
    entries = load_golden()
    judge_model_name = settings.judge_model or settings.deepseek_model
    ragas_version_str = ragas_version()

    def fingerprint(setup: str, entry: GoldenEntry) -> str:
        return cell_fingerprint(
            setup,
            entry,
            settings,
            judge_model=judge_model_name,
            judge_samples=settings.judge_samples,
            ragas_version=ragas_version_str,
            reference_scored=True,
        )

    cache: dict[tuple[str, str], dict] = {}
    todo: list[tuple[str, GoldenEntry]] = []
    for setup in setups:
        for entry in entries:
            cached = (
                None if args.refresh else load_cell(fingerprint(setup, entry), DEFAULT_CACHE_DIR)
            )
            if cached is not None:
                cache[(setup, entry.id)] = cached
            else:
                todo.append((setup, entry))
    cache_hits = len(cache)

    print(
        f"matrix: {len(setups)} setups × {len(entries)} entries — "
        f"{len(cache)} cells cached, {len(todo)} to score",
        flush=True,
    )

    scored_fresh = 0
    timed_out = False
    if todo:
        from src.chat.setups import RagSetupRunner
        from src.evals.golden import answer_correctness_fns
        from src.evals.judge import RagasJudge

        started = time.monotonic()
        write_lock = Lock()

        def score_setup(setup: str, queue: list[GoldenEntry], stack: tuple) -> None:
            """One worker per setup, owning its own runner/judge/metric stack.

            Nothing is shared between workers (agents keep per-answer state
            like ``last_context``), so setups parallelize safely even though a
            single setup's cells must stay sequential. The stacks are built
            sequentially in the main thread first — concurrent Chroma client
            construction against one path races in the rust bindings.
            """
            nonlocal timed_out, scored_fresh
            runner, judge, reference_fns = stack
            for entry in queue:
                if args.max_seconds is not None and time.monotonic() - started > args.max_seconds:
                    timed_out = True
                    return
                try:
                    run = run_golden_eval(
                        runner,
                        settings,
                        setup=setup,
                        judge=judge,
                        reference_fns=reference_fns,
                        entries=[entry],
                    )
                    result = run["entries"][0]
                except Exception as exc:  # noqa: BLE001 - one bad cell must not drop the queue
                    logger.warning("cell %s x %s failed: %s", setup, entry.id, exc)
                    result = {
                        "id": entry.id,
                        "question": entry.question,
                        "domain": entry.domain,
                        "question_type": getattr(entry, "question_type", "local"),
                        "answer": "",
                        "error": str(exc),
                        "scores": {},
                        "retrieved_chunk_ids": [],
                        "elapsed_seconds": 0.0,
                        "token_estimate": 0,
                    }
                # save_cell refuses to write a failed cell, so a transient error
                # is retried next run rather than pinned as a result.
                save_cell(fingerprint(setup, entry), result, DEFAULT_CACHE_DIR)
                scores = result["scores"]
                headline = {
                    key: scores.get(key)
                    for key in ("answer_correctness", "faithfulness", "context_recall")
                    if scores.get(key) is not None
                }
                error = result.get("error")
                with write_lock:
                    cache[(setup, entry.id)] = result
                    if not error:
                        scored_fresh += 1
                    print(
                        f"cell {setup} × {entry.id} ({entry.question_type}): "
                        + (f"ERROR {error}" if error else f"{headline}"),
                        flush=True,
                    )

        queues: dict[str, list[GoldenEntry]] = {}
        for setup, entry in todo:
            queues.setdefault(setup, []).append(entry)
        stacks = {
            setup: (
                RagSetupRunner.from_settings(settings),
                RagasJudge.from_settings(settings),
                answer_correctness_fns(settings),
            )
            for setup in queues
        }
        with ThreadPoolExecutor(max_workers=len(queues)) as pool:
            for future in [
                pool.submit(score_setup, setup, queue, stacks[setup])
                for setup, queue in queues.items()
            ]:
                future.result()
        if timed_out:
            done = sum(1 for setup in setups for entry in entries if (setup, entry.id) in cache)
            print(
                f"time budget reached — {done}/{len(setups) * len(entries)} "
                "cells cached; re-run to resume",
                flush=True,
            )
            return 3

    moment = datetime.now(timezone.utc)
    config = config_snapshot(settings, ragas_version=ragas_version_str)
    runs: dict = {}
    for setup in setups:
        setup_entries = [
            cache[(setup, entry.id)] for entry in entries if (setup, entry.id) in cache
        ]
        runs[setup] = {
            "run_id": f"matrix-{moment.strftime('%Y%m%d-%H%M%S')}-{setup}",
            "created_at": moment.isoformat(),
            "setup": setup,
            "config": config,
            "entries": setup_entries,
            "summary": _summarize_dicts(setup_entries),
        }

    result = {
        "run_id": f"matrix-{moment.strftime('%Y%m%d-%H%M%S')}",
        "created_at": moment.isoformat(),
        "kind": "matrix",
        "setups": setups,
        "config": config,
        "judged": True,
        "reference_scored": True,
        "entry_count": len(entries),
        "question_types": {
            question_type: sum(1 for entry in entries if entry.question_type == question_type)
            for question_type in dict.fromkeys(e.question_type for e in entries)
        },
        "cache_hits": cache_hits,
        "cache_misses": scored_fresh,
        "runs": runs,
        "comparison": build_comparison(runs),
    }
    path = save_run(result)
    print(f"\n{result['run_id']} — {len(entries)} questions × {len(setups)} setups\n")
    print(format_matrix_table(result))
    print(f"\nsaved {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

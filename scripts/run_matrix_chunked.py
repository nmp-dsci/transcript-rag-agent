"""Resumable driver for the judged head-to-head matrix (s05 §06).

``eval-matrix`` scores everything in one long-lived process; a judged run over
4 setups × 14 questions takes a couple of hours, and a killed process loses
the lot. This driver scores one (setup, entry) cell at a time through the
exact same pipeline (``run_golden_eval`` with judge + reference metrics),
appending each cell to a JSONL checkpoint the moment it is scored. Re-running
skips finished cells, so the run survives any interruption.

When every cell is present it assembles the same ``matrix-<ts>.json`` shape
``run_matrix`` produces and saves it under ``evals/runs/``.

    uv run python scripts/run_matrix_chunked.py
    uv run python scripts/run_matrix_chunked.py --setups rag_llm,graph_rag
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_settings  # noqa: E402
from src.evals.golden import load_golden  # noqa: E402
from src.evals.matrix import DEFAULT_MATRIX_SETUPS, build_comparison, format_matrix_table  # noqa: E402
from src.evals.regression import EntryResult, run_golden_eval, save_run, summarize  # noqa: E402

CHECKPOINT = Path(".yt-agent/matrix_checkpoint.jsonl")


def load_checkpoint() -> dict[tuple[str, str], dict]:
    cells: dict[tuple[str, str], dict] = {}
    if CHECKPOINT.exists():
        for line in CHECKPOINT.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                # A kill mid-append can truncate the trailing line; that cell
                # simply re-scores on this run rather than aborting resume.
                print(f"matrix: dropping truncated checkpoint line: {line!r}", file=sys.stderr)
                continue
            cells[(record["setup"], record["entry"]["id"])] = record
    return cells


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--setups", default=",".join(DEFAULT_MATRIX_SETUPS))
    parser.add_argument("--fresh", action="store_true", help="Discard the checkpoint")
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help=(
            "Stop scoring new cells after this many seconds and exit 3; the "
            "checkpoint resumes on the next run. Lets the driver run inside a "
            "bounded shell without losing work."
        ),
    )
    args = parser.parse_args()
    setups = [token.strip() for token in args.setups.split(",") if token.strip()]

    if args.fresh and CHECKPOINT.exists():
        CHECKPOINT.unlink()
    cells = load_checkpoint()
    entries = load_golden()
    todo = [
        (setup, entry) for setup in setups for entry in entries if (setup, entry.id) not in cells
    ]
    print(
        f"matrix: {len(setups)} setups × {len(entries)} entries — "
        f"{len(cells)} cells checkpointed, {len(todo)} to score",
        flush=True,
    )

    settings = load_settings()
    config: dict = {}
    if todo:
        import time
        from concurrent.futures import ThreadPoolExecutor
        from threading import Lock

        from src.chat.setups import RagSetupRunner
        from src.evals.golden import answer_correctness_fns
        from src.evals.judge import RagasJudge

        started = time.monotonic()
        write_lock = Lock()
        timed_out = False

        def score_setup(setup: str, queue: list, stack: tuple) -> None:
            """One worker per setup, owning its own runner/judge/metric stack.

            Nothing is shared between workers (agents keep per-answer state
            like ``last_context``), so setups parallelize safely even though a
            single setup's cells must stay sequential. The stacks are built
            sequentially in the main thread first — concurrent Chroma client
            construction against one path races in the rust bindings.
            """
            nonlocal config, timed_out
            runner, judge, reference_fns = stack
            for entry in queue:
                if args.max_seconds is not None and time.monotonic() - started > args.max_seconds:
                    timed_out = True
                    return
                run = run_golden_eval(
                    runner,
                    settings,
                    setup=setup,
                    judge=judge,
                    reference_fns=reference_fns,
                    entries=[entry],
                )
                record = {"setup": setup, "config": run["config"], "entry": run["entries"][0]}
                scores = record["entry"]["scores"]
                headline = {
                    key: scores.get(key)
                    for key in ("answer_correctness", "faithfulness", "context_recall")
                    if scores.get(key) is not None
                }
                error = record["entry"].get("error")
                with write_lock:
                    config = run["config"]
                    with CHECKPOINT.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(record) + "\n")
                    print(
                        f"cell {setup} × {entry.id} ({entry.question_type}): "
                        + (f"ERROR {error}" if error else f"{headline}"),
                        flush=True,
                    )

        CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
        queues: dict[str, list] = {}
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
            print(
                f"time budget reached — {len(load_checkpoint())}/{len(setups) * len(entries)} "
                "cells checkpointed; re-run to resume",
                flush=True,
            )
            return 3
        cells = load_checkpoint()

    # Assemble the run_matrix result shape from the checkpoint.
    moment = datetime.now(timezone.utc)
    runs: dict[str, dict] = {}
    for setup in setups:
        setup_entries = [
            cells[(setup, entry.id)]["entry"] for entry in entries if (setup, entry.id) in cells
        ]
        config = config or next((cells[key]["config"] for key in cells if key[0] == setup), {})
        results = [
            EntryResult(
                id=e["id"],
                question=e["question"],
                domain=e["domain"],
                question_type=e.get("question_type", "local"),
                answer=e.get("answer", ""),
                error=e.get("error"),
                scores=e.get("scores", {}),
                retrieved_chunk_ids=e.get("retrieved_chunk_ids", []),
                elapsed_seconds=e.get("elapsed_seconds", 0.0),
                token_estimate=e.get("token_estimate", 0),
            )
            for e in setup_entries
        ]
        runs[setup] = {
            "run_id": f"matrix-{moment.strftime('%Y%m%d-%H%M%S')}-{setup}",
            "created_at": moment.isoformat(),
            "setup": setup,
            "config": config,
            "entries": setup_entries,
            "summary": summarize(results),
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

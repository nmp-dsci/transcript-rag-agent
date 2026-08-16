"""Re-run the depth-v2 rejudge over the 20-question matrix run, off to one side.

    PYTHONPATH=. uv run python -m demo.validate.v0_rejudge_n20

The committed depth-v2 run covers five questions across six setups. Five is
exactly the app's own ``LOW_N`` threshold, so a ranking flip there is on the
boundary of what the UI itself is willing to call a ranking. This reproduces the
same rejudge over ``matrix-20260729-025133`` — twenty questions, four setups —
to see whether the flip survives four times the sample.

It calls ``src.evals.rejudge.rejudge_run`` exactly as the CLI does, but writes
the result under ``artifacts/`` instead of ``evals/runs/``: this is evidence for
a review, not a run the project is committing to.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from src.config import load_settings
from src.evals.judge import DEPTH_V2, DepthJudge
from src.evals.rejudge import chunk_context_lookup, find_run, rejudge_run

SOURCE_RUN = "matrix-20260729-025133"
OUT_DIR = Path(__file__).parent / "artifacts" / "v0_rejudge_n20"
MAX_WORKERS = 4


def ranking(run: dict[str, Any]) -> list[tuple[str, float]]:
    """Mean composite per setup, best first — the leaderboard's own ordering."""
    rows: list[tuple[str, float]] = []
    for setup, setup_run in (run.get("runs") or {}).items():
        values = [
            (cell.get("scores") or {}).get("composite") for cell in setup_run.get("entries") or []
        ]
        scored = [v for v in values if isinstance(v, (int, float))]
        if scored:
            rows.append((setup, round(sum(scored) / len(scored), 4)))
    return sorted(rows, key=lambda row: -row[1])


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    target = OUT_DIR / f"{SOURCE_RUN}-depth-v2.json"

    source = find_run(SOURCE_RUN)
    if target.exists():
        result = json.loads(target.read_text(encoding="utf-8"))
        print(f"reusing {target}")
    else:
        settings = load_settings()
        judge = DepthJudge.from_settings(settings)
        result = rejudge_run(
            source,
            depth_fn=judge.score,
            contexts_fn=chunk_context_lookup(settings),
            rubric=DEPTH_V2,
            judge_model=judge.judge_model,
            on_progress=print,
            max_workers=MAX_WORKERS,
        )
        target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    before = ranking(source)
    after = ranking(result)
    capped = sum(
        1
        for setup_run in result["runs"].values()
        for cell in setup_run["entries"]
        if cell.get("cap_applied")
    )
    depth_errors = [
        (setup, cell.get("id"), cell.get("depth_error"))
        for setup, setup_run in result["runs"].items()
        for cell in setup_run["entries"]
        if cell.get("depth_error")
    ]

    summary = {
        "source_run": SOURCE_RUN,
        "questions": source.get("entry_count"),
        "setups": source.get("setups"),
        "ragas_v1_ranking": before,
        "depth_v2_ranking": after,
        "order_changed": [s for s, _ in before] != [s for s, _ in after],
        "capped_cells": capped,
        "depth_errors": depth_errors,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"\nragas-v1 (n={source.get('entry_count')}):")
    for index, (setup, score) in enumerate(before, 1):
        print(f"  {index}. {setup:22s} {score}")
    print("depth-v2:")
    for index, (setup, score) in enumerate(after, 1):
        print(f"  {index}. {setup:22s} {score}")
    print(f"\norder changed: {summary['order_changed']}; capped cells: {capped}")
    if depth_errors:
        print(f"depth errors: {depth_errors}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

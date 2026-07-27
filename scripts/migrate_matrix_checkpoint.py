"""Back-fill the eval cell cache from the legacy matrix checkpoint.

Before :mod:`src.evals.matrix_cache` existed, the chunked matrix driver
resumed from a bespoke ``.yt-agent/matrix_checkpoint.jsonl`` append log. The
cache that replaced it is keyed by a fingerprint of the question plus the exact
answering/judging configuration, which is strictly better — but it starts
empty, so the first run after the switch would re-score every cell the
checkpoint had already paid for.

This migration reads those checkpoint rows and writes them into the cache under
the fingerprint they would have been stored with. It refuses to migrate a row
whose recorded ``config`` differs from the current settings: a cached cell
asserts "this question, under this configuration, scored this", and writing one
under a configuration that did not produce it would silently corrupt every
comparison drawn from it afterwards.

One-time operation. Safe to re-run — writing the same fingerprint twice is a
no-op — and safe to delete once the checkpoint file is gone.

    uv run python scripts/migrate_matrix_checkpoint.py
    uv run python scripts/migrate_matrix_checkpoint.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Settings, load_settings  # noqa: E402
from src.evals.golden import GoldenEntry, load_golden  # noqa: E402
from src.evals.judge import ragas_version  # noqa: E402
from src.evals.matrix import config_snapshot  # noqa: E402
from src.evals.matrix_cache import (  # noqa: E402
    DEFAULT_CACHE_DIR,
    cell_fingerprint,
    load_cell,
    save_cell,
)

DEFAULT_CHECKPOINT = Path(".yt-agent/matrix_checkpoint.jsonl")


def read_checkpoint(path: Path) -> list[dict[str, Any]]:
    """Every well-formed row; malformed lines are skipped, not fatal."""
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and "setup" in row and "entry" in row:
            rows.append(row)
    return rows


def migrate(
    rows: list[dict[str, Any]],
    entries: list[GoldenEntry],
    settings: Settings,
    *,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    dry_run: bool = False,
) -> dict[str, int]:
    """Write each checkpoint row into the cache, skipping ones we cannot vouch for.

    Returns counts by outcome: ``migrated``, ``already_cached``,
    ``config_mismatch``, ``unknown_entry`` and ``errored`` (a cell whose stored
    result is a failure — never cached, so it is retried on the next run).
    """
    by_id = {entry.id: entry for entry in entries}
    expected_config = config_snapshot(settings)
    judge_model = settings.judge_model or settings.deepseek_model
    version = ragas_version()
    counts = dict.fromkeys(
        ("migrated", "already_cached", "config_mismatch", "unknown_entry", "errored"), 0
    )

    for row in rows:
        result = row["entry"]
        entry = by_id.get(str(result.get("id")))
        if entry is None:
            counts["unknown_entry"] += 1
            continue
        # Compare only the fields the checkpoint actually recorded. Config has
        # gained keys since (``ragas_version``), and a key the old format never
        # wrote is not evidence of a different configuration — a key it wrote
        # with a different value is. But a row that recorded no config at all
        # gives us nothing to vouch for, so treat that as unverifiable rather
        # than an automatic match.
        recorded = row.get("config")
        if not recorded:
            counts["config_mismatch"] += 1
            continue
        if any(expected_config.get(key) != value for key, value in recorded.items()):
            counts["config_mismatch"] += 1
            continue
        if result.get("error"):
            counts["errored"] += 1
            continue
        fingerprint = cell_fingerprint(
            row["setup"],
            entry,
            settings,
            judge_model=judge_model,
            judge_samples=settings.judge_samples,
            ragas_version=version,
            reference_scored=True,
        )
        if load_cell(fingerprint, cache_dir) is not None:
            counts["already_cached"] += 1
            continue
        if not dry_run:
            save_cell(fingerprint, result, cache_dir)
        counts["migrated"] += 1
    return counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--dry-run", action="store_true", help="Report what would be migrated, write nothing"
    )
    args = parser.parse_args()

    rows = read_checkpoint(args.checkpoint)
    if not rows:
        print(f"no checkpoint rows at {args.checkpoint} — nothing to migrate")
        return 0

    counts = migrate(
        rows,
        load_golden(),
        load_settings(),
        cache_dir=args.cache_dir,
        dry_run=args.dry_run,
    )
    verb = "would migrate" if args.dry_run else "migrated"
    print(f"{len(rows)} checkpoint rows read from {args.checkpoint}")
    print(f"  {verb:<16} {counts['migrated']}")
    for key in ("already_cached", "config_mismatch", "unknown_entry", "errored"):
        if counts[key]:
            print(f"  {key:<16} {counts[key]}")
    if counts["config_mismatch"]:
        print(
            "\nRows recorded under a different configuration were skipped: a cached "
            "cell must describe the config that actually produced it."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

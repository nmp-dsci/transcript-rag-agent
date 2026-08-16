"""Repair the ``chunk_index`` on citations stored before references were reconciled.

Every other field of a stored citation was read off a real chunk. ``chunk_index``
was not: for a stretch of this project's history the answering LLM supplied it,
and the model has never been shown a chunk index — the context header gives it
``video=<id>``, a ``mm:ss`` window and a timestamp URL, and nothing else. So the
model wrote down the only number it *could* see, the citation label, and the
field ended up meaning "the third chunk of this answer" while claiming to mean
"chunk 3 of this video". Both live paths reconcile references against real
chunks now (:func:`src.agents.rag_transcript_agent.reconcile_references` and
:func:`src.agents.rag_agent.reconcile_agent_references`); this is the history
they were fixed too late for.

**How a reference is re-resolved.** By ``(video_id, start_seconds)`` against the
live chunk store, taking the nearest chunk and requiring it within
``MAX_DRIFT_SECONDS``. That is the same rule the agentic path uses at answer
time, and it works because ``start_seconds`` — unlike ``chunk_index`` — was
either copied from the chunk header or derived from the ``mm:ss`` the model was
shown, so it is at worst a fraction of a second low. A reference that does not
resolve within the tolerance is left exactly as it is and reported: this repairs
citations, it does not invent them.

**What can be checked afterwards, and how.**

* Re-run with ``--check``. It reports how many references disagree with the
  store; after a successful ``--apply`` that number is zero, and it stays zero
  on every later run because the operation is idempotent.
* The writer refuses to save if any value other than a ``chunk_index`` inside a
  ``references`` list differs from what it loaded. Nothing else in the file can
  change, whatever the resolution logic decides.
* ``--check`` also runs an independent second route on every reference whose
  label is positional: label ``[n]`` must name ``retrieved_chunk_ids[n-1]``, the
  order retrieval was stored in, which shares no input with the timestamp route.
  Disagreements are printed. ``graph_rag`` answers are exempt — their stored
  chunk ids are graph evidence and were never in citation order.
* Every rewrite is printed as ``entry / setup / label: old -> new``, so the
  change set is enumerable, and ``git diff`` on the history file should contain
  nothing but ``chunk_index`` lines.

    PYTHONPATH=. uv run python scripts/backfill_reference_chunk_index.py --check
    PYTHONPATH=. uv run python scripts/backfill_reference_chunk_index.py --apply
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.chat.history import DEFAULT_HISTORY_PATH  # noqa: E402
from src.config import load_settings  # noqa: E402

#: How far a stored ``start_seconds`` may sit from a chunk's real one and still
#: be that chunk. Matches ``src.agents.rag_agent.MAX_START_DRIFT_SECONDS``: the
#: model was shown whole seconds, so a faithful value is at most one second low,
#: and two is that with margin — still far inside the ~70s a chunk spans, so the
#: tolerance can never reach a neighbouring chunk.
MAX_DRIFT_SECONDS = 2.0

#: Setups whose ``retrieved_chunk_ids`` are in citation order, so the positional
#: cross-check applies. ``graph_rag`` records the graph evidence it traversed,
#: which is a different list in a different order.
POSITIONAL_SETUPS = frozenset(
    {
        "rag_llm",
        "rag_llm_recursive",
        "rag_llm_filtered",
        "rag_llm_hyde",
        "rag_llm_contextual",
        "rag_agent",
    }
)


def chunk_starts(settings) -> dict[str, dict[int, float]]:
    """``video_id -> {chunk_index: start_seconds}`` for the whole chunk store."""
    import chromadb

    client = chromadb.PersistentClient(path=str(settings.chroma_path))
    collection = client.get_or_create_collection(settings.chunk_collection)
    starts: dict[str, dict[int, float]] = {}
    for meta in collection.get(include=["metadatas"]).get("metadatas") or []:
        meta = meta or {}
        video_id = str(meta.get("video_id", ""))
        if not video_id:
            continue
        start = meta.get("start_seconds")
        starts.setdefault(video_id, {})[int(meta.get("chunk_index", 0) or 0)] = (
            0.0 if start is None else float(start)
        )
    return starts


def resolve(reference: dict[str, Any], starts: dict[str, dict[int, float]]) -> int | None:
    """The chunk index this citation's timestamp points at, or ``None``."""
    video_id = str(reference.get("video_id") or "")
    claimed = reference.get("start_seconds")
    by_index = starts.get(video_id)
    if not by_index or claimed is None:
        return None
    claimed = float(claimed)
    best = min(by_index, key=lambda index: abs(by_index[index] - claimed))
    if abs(by_index[best] - claimed) > MAX_DRIFT_SECONDS:
        return None
    return best


def label_number(label: object) -> int | None:
    text = str(label or "").strip()
    if text.startswith("[") and text.endswith("]") and text[1:-1].isdigit():
        return int(text[1:-1])
    return None


def audit(conversations: list[dict], starts: dict[str, dict[int, float]]) -> dict[str, Any]:
    """Every reference classified, without changing anything."""
    rewrites: list[tuple[str, str, str, int | None, int]] = []
    already_right = 0
    unresolvable: list[tuple[str, str, str]] = []
    no_index = 0
    cross_checked = 0
    cross_check_failures: list[tuple[str, str, str, str, str]] = []

    for entry in conversations:
        for answer in entry.get("answers", []):
            retrieved = answer.get("retrieved_chunk_ids") or []
            for reference in answer.get("references", []):
                stored = reference.get("chunk_index")
                if stored is None:
                    no_index += 1
                    continue
                resolved = resolve(reference, starts)
                if resolved is None:
                    unresolvable.append((entry["id"], answer["key"], str(reference.get("label"))))
                    continue
                if int(stored) == resolved:
                    already_right += 1
                else:
                    rewrites.append(
                        (entry["id"], answer["key"], str(reference.get("label")), stored, resolved)
                    )
                number = label_number(reference.get("label"))
                if (
                    answer["key"] in POSITIONAL_SETUPS
                    and number is not None
                    and 1 <= number <= len(retrieved)
                ):
                    cross_checked += 1
                    by_position = retrieved[number - 1]
                    by_timestamp = f"chunk:{reference.get('video_id')}:{resolved}"
                    if by_position != by_timestamp:
                        cross_check_failures.append(
                            (
                                entry["id"],
                                answer["key"],
                                str(reference.get("label")),
                                by_position,
                                by_timestamp,
                            )
                        )

    return {
        "rewrites": rewrites,
        "already_right": already_right,
        "unresolvable": unresolvable,
        "no_index": no_index,
        "cross_checked": cross_checked,
        "cross_check_failures": cross_check_failures,
    }


def apply_rewrites(conversations: list[dict], starts: dict[str, dict[int, float]]) -> int:
    """Rewrite resolvable ``chunk_index`` values in place; return how many changed."""
    changed = 0
    for entry in conversations:
        for answer in entry.get("answers", []):
            for reference in answer.get("references", []):
                if reference.get("chunk_index") is None:
                    continue
                resolved = resolve(reference, starts)
                if resolved is None or int(reference["chunk_index"]) == resolved:
                    continue
                reference["chunk_index"] = resolved
                changed += 1
    return changed


def assert_only_chunk_index_changed(before: Any, after: Any, path: str = "$") -> None:
    """Raise unless every difference is a ``chunk_index`` inside a ``references`` list.

    The guard is on the *written payload*, not on the logic that produced it, so
    a future edit to ``resolve`` that started touching timestamps or dropping
    references would fail here rather than in review.
    """
    if isinstance(before, dict) and isinstance(after, dict):
        if before.keys() != after.keys():
            raise ValueError(f"keys changed at {path}")
        for key in before:
            child = f"{path}.{key}"
            if key == "chunk_index" and ".references[" in path:
                continue
            assert_only_chunk_index_changed(before[key], after[key], child)
        return
    if isinstance(before, list) and isinstance(after, list):
        if len(before) != len(after):
            raise ValueError(f"list length changed at {path}")
        for index, (old, new) in enumerate(zip(before, after)):
            assert_only_chunk_index_changed(old, new, f"{path}[{index}]")
        return
    if before != after:
        raise ValueError(f"unexpected change at {path}: {before!r} -> {after!r}")


def report(result: dict[str, Any], total: int) -> None:
    rewrites = result["rewrites"]
    print(f"references                    : {total}")
    print(f"  already correct             : {result['already_right']}")
    print(f"  wrong chunk_index           : {len(rewrites)}")
    print(f"  no chunk_index (left alone) : {result['no_index']}")
    print(f"  unresolvable (left alone)   : {len(result['unresolvable'])}")
    print(
        f"cross-checked against retrieved_chunk_ids : {result['cross_checked']}"
        f" — {len(result['cross_check_failures'])} disagreements"
    )
    for row in result["cross_check_failures"]:
        print(f"    DISAGREE {row[0]} {row[1]} {row[2]}: position={row[3]} timestamp={row[4]}")
    for row in result["unresolvable"]:
        print(f"    unresolvable {row[0]} {row[1]} {row[2]}")
    for entry_id, key, label, old, new in rewrites:
        print(f"    {entry_id} / {key} / {label}: {old} -> {new}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY_PATH)
    parser.add_argument("--check", action="store_true", help="report only (the default)")
    parser.add_argument("--apply", action="store_true", help="write the repaired history")
    args = parser.parse_args(argv)

    settings = load_settings(require_keys=False)
    starts = chunk_starts(settings)
    if not starts:
        print("no chunks in the store — nothing to resolve against", file=sys.stderr)
        return 1

    raw = json.loads(args.history.read_text(encoding="utf-8"))
    conversations = raw["conversations"] if isinstance(raw, dict) else raw
    total = sum(
        len(answer.get("references", []))
        for entry in conversations
        for answer in entry.get("answers", [])
    )

    result = audit(conversations, starts)
    report(result, total)

    if result["cross_check_failures"]:
        print(
            "\nrefusing to write: the timestamp route and the retrieval-order route "
            "disagree, so one of them is wrong and a rewrite would be a guess",
            file=sys.stderr,
        )
        return 1
    if not args.apply:
        print("\ncheck only — nothing written (pass --apply to write)")
        return 0
    if not result["rewrites"]:
        print("\nnothing to do")
        return 0

    before = json.loads(json.dumps(raw))
    changed = apply_rewrites(conversations, starts)
    assert_only_chunk_index_changed(before, raw)
    args.history.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
    print(f"\nrewrote {changed} chunk_index values in {args.history}")
    print("re-run with --check: it must now report 0 wrong")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

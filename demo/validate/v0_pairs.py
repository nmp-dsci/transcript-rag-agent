"""Judge hand-written shallow/deep answer pairs under both rubrics.

    PYTHONPATH=. uv run python -m demo.validate.v0_pairs --shard 0 --shards 6  # x6
    PYTHONPATH=. uv run python -m demo.validate.v0_pairs --report

Gate 1 of the V0 review asks whether ``depth-v2`` measures something
``ragas-v1`` does not. A ranking flip on the committed matrix run is one piece
of evidence, but it is only five questions and those answers were not written to
isolate depth from every other difference. So this script judges pairs written
*for* the question: same question, same contexts, one answer a faithful
restatement of a single chunk and the other a genuine synthesis across the same
context set. Anything that separates them is depth, because nothing else varies.

The scoring path is the app's own. Grounding comes from :class:`RagasJudge`
exactly as a matrix run produces it; depth comes from :class:`DepthJudge`; and
the two composites are the two rubrics applied to that one set of scores — which
is what ``src.evals.rejudge`` does to a committed run.

Gate 2 reuses the same machinery on deliberately ungrounded answers: rich,
specific, multi-source-sounding, and unsupported by the contexts they were given.
Their composites must land at or below the rubric's cap.

**Sharded across processes, not threads.** ``RagasJudge``'s metric functions
drive ragas prompts through ``ragas.async_utils.run``, which opens an event loop
per call over a shared LLM wrapper; running several of those in one process's
thread pool wedged at 0% CPU rather than going faster. Each shard is therefore
its own process scoring its own slice, and results are written one file per
answer so the shards never contend for a cache file.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from src.config import load_settings
from src.evals.judge import DEPTH_V2, RAGAS_V1, DepthJudge, RagasJudge
from src.evals.rejudge import chunk_context_lookup

HERE = Path(__file__).parent
PAIRS_FILE = HERE / "pairs" / "v0_pairs.json"
OUT_DIR = HERE / "artifacts" / "v0_pairs"
CACHE_DIR = OUT_DIR / "cache"
REPORT_FILE = OUT_DIR / "report.json"


def cache_path(key: str) -> Path:
    return CACHE_DIR / f"{key.replace(':', '__')}.json"


def load_result(key: str) -> dict[str, Any]:
    path = cache_path(key)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def score_answer(
    key: str,
    question: str,
    answer: str,
    contexts: list[str],
    *,
    grounding: RagasJudge,
    depth: DepthJudge,
) -> dict[str, Any]:
    """Both rubrics' composites for one answer, from one set of metric scores."""
    record = grounding.score(question, answer, contexts)
    scores = dict(record["scores"])
    depth_error = None
    try:
        for metric, breakdown in depth.score(question, answer, contexts).items():
            scores[metric] = breakdown.score
    except Exception as exc:  # noqa: BLE001 - reported, not raised
        depth_error = str(exc)

    v1 = RAGAS_V1.composite(scores)
    v2 = DEPTH_V2.composite(scores)
    return {
        "key": key,
        "scores": scores,
        "grounding_error": record.get("error"),
        "depth_error": depth_error,
        "ragas_v1": {"composite": v1.composite},
        "depth_v2": {
            "composite": v2.composite,
            "uncapped": v2.uncapped,
            "cap_applied": v2.cap_applied,
            "cap_reason": v2.cap_reason,
        },
        "context_count": len(contexts),
    }


def build_jobs(
    data: dict[str, Any], lookup: Any
) -> tuple[list[tuple[str, str, str, list[str]]], list[str]]:
    jobs: list[tuple[str, str, str, list[str]]] = []
    gaps: list[str] = []

    def add(item_id: str, variant: str, question: str, answer: str, ids: list[str]) -> None:
        contexts = lookup(ids)
        if len(contexts) != len(ids):
            gaps.append(f"{item_id}: resolved {len(contexts)} of {len(ids)} chunk ids")
        jobs.append((f"{item_id}:{variant}", question, answer, contexts))

    for pair in data["pairs"]:
        for variant in ("shallow", "deep"):
            add(pair["id"], variant, pair["question"], pair[variant], pair["context_chunk_ids"])
    for item in data["ungrounded"]:
        add(item["id"], "ungrounded", item["question"], item["answer"], item["context_chunk_ids"])
    return jobs, gaps


def run_shard(shard: int, shards: int) -> int:
    data = json.loads(PAIRS_FILE.read_text(encoding="utf-8"))
    settings = load_settings()
    lookup = chunk_context_lookup(settings)
    jobs, gaps = build_jobs(data, lookup)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if gaps:
        print(f"[{shard}] context resolution gaps: {gaps}")

    mine = [job for index, job in enumerate(jobs) if index % shards == shard]
    todo = [job for job in mine if not cache_path(job[0]).exists()]
    print(f"[{shard}] {len(mine)} answers in shard, {len(todo)} to judge")
    if not todo:
        return 0

    grounding = RagasJudge.from_settings(settings)
    depth = DepthJudge.from_settings(settings)
    for key, question, answer, contexts in todo:
        try:
            result = score_answer(key, question, answer, contexts, grounding=grounding, depth=depth)
        except Exception as exc:  # noqa: BLE001 - one answer failing is reportable
            result = {"key": key, "error": str(exc)}
        cache_path(key).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        v1 = (result.get("ragas_v1") or {}).get("composite")
        v2 = (result.get("depth_v2") or {}).get("composite")
        print(f"[{shard}] {key}: ragas-v1 {v1} depth-v2 {v2}")
    return 0


def _winner(shallow: float | None, deep: float | None) -> str:
    if shallow is None or deep is None:
        return "unscored"
    if abs(deep - shallow) < 1e-9:
        return "tie"
    return "deep" if deep > shallow else "shallow"


def composites(record: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Both rubrics applied to a cached answer's metric scores, *now*.

    The cache stores what the judges said — the expensive, non-deterministic
    part — and the composites that the rubric code produced at the time. Only
    the first of those is evidence. Re-deriving the composites from the stored
    scores on every report means a change to ``Rubric.composite`` shows up in
    the gate rather than being masked by a number frozen into the cache, which
    is exactly what a re-verification after a fix round has to detect.
    """
    scores = record.get("scores") or {}
    if not scores:
        return {}, {}
    v1 = RAGAS_V1.composite(scores)
    v2 = DEPTH_V2.composite(scores)
    return (
        {"composite": v1.composite},
        {
            "composite": v2.composite,
            "uncapped": v2.uncapped,
            "cap_applied": v2.cap_applied,
            "cap_reason": v2.cap_reason,
            "grounding_floor_breached": v2.grounding_floor_breached,
            "grounding_reason": v2.grounding_reason,
        },
    )


def report() -> int:
    data = json.loads(PAIRS_FILE.read_text(encoding="utf-8"))

    rows: list[dict[str, Any]] = []
    for pair in data["pairs"]:
        shallow = load_result(f"{pair['id']}:shallow")
        deep = load_result(f"{pair['id']}:deep")
        s_v1, s_v2 = composites(shallow)
        d_v1, d_v2 = composites(deep)
        v1s = s_v1.get("composite")
        v1d = d_v1.get("composite")
        v2s = s_v2.get("composite")
        v2d = d_v2.get("composite")
        # The cap can decide a pair on its own, which is a different claim from
        # "the depth dimensions separated these two answers". Both are reported
        # so a gate result is attributable to one or the other.
        u2s = s_v2.get("uncapped")
        u2d = d_v2.get("uncapped")
        rows.append(
            {
                "pair": pair["id"],
                "question": pair["question"],
                "ragas_v1_shallow": v1s,
                "ragas_v1_deep": v1d,
                "ragas_v1_winner": _winner(v1s, v1d),
                "depth_v2_shallow": v2s,
                "depth_v2_deep": v2d,
                "depth_v2_winner": _winner(v2s, v2d),
                "depth_v2_uncapped_shallow": u2s,
                "depth_v2_uncapped_deep": u2d,
                "depth_v2_uncapped_winner": _winner(u2s, u2d),
                "shallow_capped": s_v2.get("cap_applied"),
                "deep_capped": d_v2.get("cap_applied"),
                "shallow_floor_breached": s_v2.get("grounding_floor_breached"),
                "deep_floor_breached": d_v2.get("grounding_floor_breached"),
                "faithfulness_shallow": (shallow.get("scores") or {}).get("faithfulness"),
                "faithfulness_deep": (deep.get("scores") or {}).get("faithfulness"),
                "insight_shallow": (shallow.get("scores") or {}).get("insight_depth"),
                "insight_deep": (deep.get("scores") or {}).get("insight_depth"),
            }
        )

    unscored = [
        r["pair"] for r in rows if "unscored" in (r["ragas_v1_winner"], r["depth_v2_winner"])
    ]
    missed = [r for r in rows if r["ragas_v1_winner"] != "deep"]
    fixed = [r for r in missed if r["depth_v2_winner"] == "deep"]
    rate = len(fixed) / len(missed) if missed else None
    gate1 = not unscored and rate is not None and rate >= 0.8
    # The same arithmetic before the cap, so a failure can be attributed either
    # to the depth dimensions not separating the pair or to the cap overriding
    # them.
    fixed_uncapped = [r for r in missed if r["depth_v2_uncapped_winner"] == "deep"]
    rate_uncapped = len(fixed_uncapped) / len(missed) if missed else None

    capped: list[dict[str, Any]] = []
    for item in data["ungrounded"]:
        rec = load_result(f"{item['id']}:ungrounded")
        _, v2 = composites(rec)
        capped.append(
            {
                "id": item["id"],
                "faithfulness": (rec.get("scores") or {}).get("faithfulness"),
                "depth_v2_composite": v2.get("composite"),
                "depth_v2_uncapped": v2.get("uncapped"),
                "cap_applied": v2.get("cap_applied"),
                "cap_reason": v2.get("cap_reason"),
                # The cap only fires when it lowers the number, so an answer
                # this ungrounded can pass the <= 0.5 bar without the cap ever
                # applying. The floor flag is what marks it either way.
                "grounding_floor_breached": v2.get("grounding_floor_breached"),
                "grounding_reason": v2.get("grounding_reason"),
                "ragas_v1_composite": composites(rec)[0].get("composite"),
            }
        )
    # The gate is the number; the marker is what makes the number legible to a
    # reviewer, so both are required rather than the composite alone.
    gate2 = bool(capped) and all(
        c["depth_v2_composite"] is not None
        and c["depth_v2_composite"] <= 0.5
        and c["grounding_floor_breached"] is True
        for c in capped
    )

    result = {
        "pairs": rows,
        "gate1": {
            "total_pairs": len(rows),
            "unscored_pairs": unscored,
            "ragas_v1_missed": len(missed),
            "depth_v2_fixed": len(fixed),
            "rate": rate,
            "threshold": 0.8,
            "passed": gate1,
            "missed_pairs": [r["pair"] for r in missed],
            "fixed_pairs": [r["pair"] for r in fixed],
            "fixed_pairs_before_cap": [r["pair"] for r in fixed_uncapped],
            "rate_before_cap": rate_uncapped,
        },
        "gate2": {"answers": capped, "passed": gate2},
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    header = f"{'pair':<14}{'v1 shallow':>11}{'v1 deep':>9}{'v1 win':>9}{'v2 shallow':>12}{'v2 deep':>9}{'v2 win':>9}"
    print(header)
    for r in rows:
        print(
            f"{r['pair']:<14}{r['ragas_v1_shallow']!s:>11}{r['ragas_v1_deep']!s:>9}"
            f"{r['ragas_v1_winner']:>9}{r['depth_v2_shallow']!s:>12}{r['depth_v2_deep']!s:>9}"
            f"{r['depth_v2_winner']:>9}"
        )
    print(
        f"\ngate 1: ragas-v1 failed to rank deep first on {len(missed)} of {len(rows)} pairs; "
        f"depth-v2 fixed {len(fixed)} of those = "
        f"{'n/a' if rate is None else round(rate, 3)} (need >= 0.80) -> "
        f"{'PASS' if gate1 else 'FAIL'}"
    )
    print(
        "        before the faithfulness cap, depth-v2 would have fixed "
        f"{len(fixed_uncapped)} of {len(missed)} = "
        f"{'n/a' if rate_uncapped is None else round(rate_uncapped, 3)}"
    )
    if unscored:
        print(f"        unscored pairs (gate not evaluable): {unscored}")
    for c in capped:
        print(
            f"gate 2: {c['id']} faithfulness={c['faithfulness']} "
            f"uncapped={c['depth_v2_uncapped']} composite={c['depth_v2_composite']} "
            f"cap_applied={c['cap_applied']} (ragas-v1 would have said "
            f"{c['ragas_v1_composite']})"
        )
    print(f"gate 2: {'PASS' if gate2 else 'FAIL'}")
    print(f"report -> {REPORT_FILE}")
    return 0 if (gate1 and gate2) else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--report", action="store_true", help="Only compute the gates")
    args = parser.parse_args(argv)
    if args.report:
        return report()
    run_shard(args.shard, args.shards)
    return report() if args.shards == 1 else 0


if __name__ == "__main__":
    sys.exit(main())

"""Reconcile the s11 slice plan against the verdicts actually on disk.

``demo/validate/README.md`` says the record of "this slice passed" is a committed
``artifacts/<slice>/verdict.json``, written by an independent evaluator. That is
the right place for the *evidence*, but it answers one slice at a time — and with
ten slices in flight across several agents, nobody could answer "where are we"
without opening ten directories and remembering which of them were even supposed
to exist yet.

So this reads them all and writes ``STATUS.md``. It is generated, never hand-
edited, for the same reason the verdicts are committed: a status page somebody
maintains by hand is a status page that is wrong by the second week.

    PYTHONPATH=. uv run python -m demo.validate.status

Two things it deliberately does **not** do. It does not write or modify a
verdict — those belong to the evaluator agents, and a builder marking its own
slice passed is exactly the conflict the README's separate-agents rule exists to
prevent. And it does not infer success from code existing: a slice with a module
and no verdict is reported as *built, unvalidated*, because under this plan's
standing rule a hypothesis that cannot be seen in the frontend has not shipped
however much code is behind it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

#: The plan itself lives in a Lavish artifact, not in markdown — §9 "Build order
#: — vertical slices, each with a loop gate". This is a *label* table so the
#: status page can name a slice that has produced nothing yet; the artifact stays
#: the source of truth for hypotheses and gate wording.
PLAN_ARTIFACT = ".lavish/s11_distilled-reviewer-deep-research.html"

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = Path(__file__).resolve().parent / "artifacts"


@dataclass(frozen=True)
class Slice:
    """One planned slice, and where its evidence would be if it had any."""

    id: str
    title: str
    #: Verdict directories that count as evidence for this slice. More than one
    #: where the evaluator split the gate across scripts (V0's judge, its
    #: negative controls, its pair study).
    artifact_dirs: tuple[str, ...] = ()
    #: Code that exists whether or not the slice was ever reviewed. Used only to
    #: tell "not started" from "built but unvalidated" — never as evidence of
    #: passing.
    build_paths: tuple[str, ...] = ()
    #: Anything known to be open against a slice that otherwise looks green.
    caveat: str | None = None


SLICES: tuple[Slice, ...] = (
    Slice(
        id="V0",
        title="Judge v2 on the Scoreboard",
        artifact_dirs=(
            "v0_judge",
            "v0_judge_negatives",
            "v0_pairs",
            "v0_rejudge_n20",
            "v0_independent_judge",
        ),
        build_paths=("src/evals/rejudge.py", "frontend/src/scoreboard/RubricPanel.tsx"),
    ),
    Slice(
        id="V1",
        title="First real document review in Chat",
        artifact_dirs=("v1_document",),
        build_paths=("src/documents/review.py",),
    ),
    Slice(
        id="V2",
        title="Summary pre-filter on the Scoreboard",
        artifact_dirs=("v2_filter",),
        build_paths=("src/rag/summaries.py",),
    ),
    Slice(
        id="V3",
        title="Critique eval harness in Experiments",
        artifact_dirs=("v3_critique",),
        build_paths=(
            "src/evals/critique.py",
            "src/evals/critique_run.py",
            "frontend/src/experiments/CritiquePanel.tsx",
        ),
        caveat=(
            "src/evals/KNOWN_GAP_attack2.md is open: the grounding gate checks that a "
            "quote resolves, never that the cited chunk supports the finding it is "
            "attached to. Reproduced 2026-08-10 — reciting the criteria with one "
            "distinct real chunk each scores evidence_precision 1.000 and "
            "criteria_recall 0.526 against the honest baseline's 0.556 / 0.158. "
            "V5, V6 and V8 are all scored on this metric."
        ),
    ),
    Slice(
        id="V4",
        title="Theme layer in RAG Pipeline",
        artifact_dirs=("v4_themes",),
        build_paths=("src/rag/themes.py", "frontend/src/pipeline/ThemesView.tsx"),
    ),
    Slice(
        id="V4b",
        title="Ingest the two empty topics (system design, app architecture)",
        artifact_dirs=("v4b_ingest",),
    ),
    Slice(
        id="V5",
        title="Rubric packs in the app",
        artifact_dirs=("v5_packs",),
        build_paths=(
            "src/rag/packs.py",
            "src/rag/pack_build.py",
            "src/evals/pack_ablation.py",
            "experts/resume-design/pack.json",
        ),
    ),
    Slice(
        id="V6",
        title="Rubric-driven reviewer in Chat",
        artifact_dirs=("v6_reviewer",),
    ),
    Slice(
        id="V7",
        title="Disagreements as a first-class view",
        artifact_dirs=("v7_conflicts",),
        build_paths=("src/rag/conflicts.py",),
    ),
    Slice(
        id="V8",
        title="Deep-research build loop",
        artifact_dirs=("v8_loop",),
    ),
)


@dataclass
class Evidence:
    """What one artifact directory actually contains."""

    name: str
    passed: bool | None = None
    checks_passed: int = 0
    checks_total: int = 0
    started_at: str = ""
    #: A directory holding a report/summary but no verdict — real work, but not a
    #: pass/fail claim, so it never counts towards a slice being validated.
    supporting_only: bool = False
    failed_checks: tuple[str, ...] = ()


def read_evidence(name: str) -> Evidence | None:
    directory = ARTIFACTS / name
    if not directory.exists():
        return None
    verdict = directory / "verdict.json"
    if not verdict.exists():
        others = [p.name for p in directory.iterdir() if p.suffix == ".json"]
        # An empty directory means a script ran and could not record — worth
        # distinguishing from a directory that was never created at all.
        return Evidence(name=name, supporting_only=bool(others))
    try:
        data = json.loads(verdict.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return Evidence(name=name)
    checks = data.get("checks") or []
    return Evidence(
        name=name,
        passed=bool(data.get("passed")),
        checks_passed=sum(1 for c in checks if c.get("passed")),
        checks_total=len(checks),
        started_at=str(data.get("started_at") or "")[:19],
        failed_checks=tuple(str(c.get("name") or "?") for c in checks if not c.get("passed")),
    )


def state_of(item: Slice, evidence: list[Evidence]) -> tuple[str, str]:
    """The slice's state and the one line that justifies it.

    Ordered most-serious-first: a failing verdict outranks a passing one, because
    a slice with one of each is not half-done, it is failing.
    """
    verdicts = [e for e in evidence if e.passed is not None]
    if any(e.passed is False for e in verdicts):
        failed = next(e for e in verdicts if e.passed is False)
        detail = failed.failed_checks[0] if failed.failed_checks else "see verdict"
        return "FAILING", f"{failed.name}: {failed.checks_passed}/{failed.checks_total} — {detail}"
    if verdicts:
        total = sum(e.checks_total for e in verdicts)
        passed = sum(e.checks_passed for e in verdicts)
        label = "PASSED (caveat)" if item.caveat else "PASSED"
        return label, f"{passed}/{total} checks across {len(verdicts)} verdict(s)"
    built = [p for p in item.build_paths if (ROOT / p).exists()]
    if any(e.supporting_only for e in evidence):
        return "UNVALIDATED", "artifacts present but no verdict.json recorded"
    if built:
        return "UNVALIDATED", f"code on disk ({len(built)} path(s)), no verdict recorded"
    return "NOT STARTED", "no code, no artifacts"


ORDER = {"FAILING": 0, "UNVALIDATED": 1, "NOT STARTED": 2, "PASSED (caveat)": 3, "PASSED": 4}


def build_status() -> str:
    rows: list[str] = []
    caveats: list[str] = []
    counts: dict[str, int] = {}
    for item in SLICES:
        evidence = [e for e in (read_evidence(n) for n in item.artifact_dirs) if e]
        state, why = state_of(item, evidence)
        counts[state] = counts.get(state, 0) + 1
        rows.append(f"| {item.id} | {item.title} | **{state}** | {why} |")
        if item.caveat and state.startswith("PASSED"):
            caveats.append(f"- **{item.id}** — {item.caveat}")

    headline = " · ".join(
        f"{n} {state.lower()}" for state, n in sorted(counts.items(), key=lambda kv: ORDER[kv[0]])
    )
    generated = datetime.now(timezone.utc).isoformat(timespec="seconds")
    lines = [
        "# s11 slice status",
        "",
        "**Generated — do not hand-edit.** Regenerate with:",
        "",
        "```bash",
        "PYTHONPATH=. uv run python -m demo.validate.status",
        "```",
        "",
        f"Generated {generated}. Plan: `{PLAN_ARTIFACT}` §9. Evidence: "
        "`demo/validate/artifacts/<slice>/verdict.json`, written by the independent "
        "evaluator for that slice — never by its builder.",
        "",
        f"**{len(SLICES)} slices — {headline}.**",
        "",
        "| Slice | Title | State | Evidence |",
        "| --- | --- | --- | --- |",
        *rows,
        "",
        "## What the states mean",
        "",
        "- **PASSED** — an evaluator recorded `verdict.json` with `passed: true`.",
        "- **PASSED (caveat)** — the frontend gate passed, but something known and "
        "open is recorded against it. Read the caveat before building on it.",
        "- **FAILING** — a verdict exists and records `passed: false`.",
        "- **UNVALIDATED** — code or artifacts exist, but no verdict was recorded. "
        "Under the plan's standing rule (a hypothesis that cannot be seen in the "
        "frontend fails review even if every automated gate passed) this is **not** "
        "done, however much code is behind it.",
        "- **NOT STARTED** — no code, no artifacts.",
        "",
    ]
    if caveats:
        lines += ["## Open caveats against slices that otherwise read green", "", *caveats, ""]
    return "\n".join(lines)


def main() -> int:
    path = Path(__file__).resolve().parent / "STATUS.md"
    path.write_text(build_status() + "\n", encoding="utf-8")
    print(build_status())
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())

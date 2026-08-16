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

Two qualifications on a green row are *computed* rather than written down, for
the same reason this page is generated at all — a hand-maintained note about a
verdict is a note that stops being true without anyone noticing.

* **Reduced independence.** A verdict whose notes carry the README's
  ``REDUCED INDEPENDENCE`` disclosure is marked as such on its row. The spend
  limit forced the orchestrating agent to re-check its own fix on three slices;
  a summary that folds those checks into a plain tally overstates them.
* **Evidence that has moved.** Run files get superseded and deleted. A verdict
  that named one is still a true record of what its evaluator saw, but it is no
  longer reproducible, so the row says which runs no longer resolve on disk.
"""

from __future__ import annotations

import json
import re
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

#: The disclosure the README asks for when an evaluation could not be independent.
INDEPENDENCE_FLAG = "REDUCED INDEPENDENCE"

#: Run ids as they are written in prose and in check details:
#: ``critique-<video>-<YYYYMMDD>-<HHMMSS>``, ``matrix-<YYYYMMDD>-<HHMMSS>-<label>``.
#: Matched loosely, then kept only if it carries a timestamp — that filter is what
#: stops ordinary words beginning "eval-" from being treated as run ids.
RUN_ID = re.compile(r"\b(?:critique|matrix|ablation|eval)-[A-Za-z0-9][A-Za-z0-9_-]*")
RUN_STAMP = re.compile(r"\d{8}-\d{6}")
#: Where a named run could legitimately live: the committed runs, the fixture runs
#: a validator serves from a second instance, and the artifact directory itself.
RUN_DIRS = ("evals/runs", "demo/validate/fixtures")


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
            "src/evals/KNOWN_GAP_attack2.md is now *partly* closed, and what it costs "
            "matters as much as what it fixed. The distinct-chunk attack — recite the "
            "criteria, give each recited finding its own distinct real chunk, score "
            "evidence_precision 1.000 / criteria_recall 0.526 against the honest "
            "baseline's 0.364 / 0.158 — was gated on 2026-08-11 by GATE_PROVENANCE: a "
            "citation grounds a finding only when the chunk it resolves to is one that "
            "finding's own retrieval returned. Every committed run was rescored in "
            "place with no model calls; no score moved and no within-run ranking "
            "flipped. The cost is that the retrieval arms (rag_llm_filtered, "
            "rag_conflict_aware) have no per-finding provenance and are now ungraded "
            "rather than scored, which is why V6's rubric_packs 0.000 has no certified "
            "baseline to be subtracted from. Still open is the relevance question "
            "itself: a chunk the finding really did retrieve, cited for a rule it does "
            "not support, still passes, and only an entailment check reaches it. "
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
        # V4b produces no module of its own — it is an ingestion slice, and the
        # only thing it leaves behind is corpus. A built pack for each of the two
        # previously-empty topics is the closest thing to "code on disk" it has.
        build_paths=(
            "experts/system-design/pack.json",
            "experts/app-architecture/pack.json",
        ),
    ),
    Slice(
        id="V5",
        title="Rubric packs in the app",
        artifact_dirs=("v5_packs",),
        build_paths=(
            "src/rag/packs.py",
            "src/rag/pack_build.py",
            "src/evals/pack_ablation.py",
            "src/api/packs.py",
            "frontend/src/experiments/PackPanel.tsx",
            "experts/resume-design/pack.json",
        ),
    ),
    Slice(
        id="V6",
        title="Rubric-driven reviewer in Chat",
        artifact_dirs=("v6_reviewer",),
        build_paths=(
            "src/documents/rubric_review.py",
            "src/agents/rubric_review_agent.py",
            "frontend/src/chat/RubricVerdicts.tsx",
        ),
    ),
    Slice(
        id="V7",
        title="Disagreements as a first-class view",
        # The plan named this ``v7_conflicts``; the slice shipped as
        # "Disagreements" in the UI and its validator followed the label a reader
        # sees. Both are listed so the reconciliation does not depend on which
        # name an evaluator picked — a status page that reports validated work as
        # unvalidated because of a directory name is worse than useless.
        artifact_dirs=("v7_conflicts", "v7_disagreements"),
        build_paths=(
            "src/rag/conflicts.py",
            "frontend/src/pipeline/DisagreementsView.tsx",
        ),
        # Carried while the slice reads FAILING so it is already true if the count
        # ever clears the gate: the row below states the failure, the caveat states
        # what would still be open once it is fixed.
        caveat=(
            "The count is short of the gate and that is the one failing check in a "
            "64/65 verdict — 2 disagreement cards render where the gate asks for >=3. "
            "What would still be open if the count rose is its reliability. It is "
            "measured at 3 looks per pair with no recorded spread, and two builds over "
            "a provably identical candidate pool returned 4 cards and then 2, with two "
            "cards moving 3/3 to 0/3. The layer now says so in its own voice rather "
            "than reading as confidence: a warn chip beside the count ('spread not "
            "recorded — nearer 2 ± 1 than exactly 2'), card chips reading 'agreed 3/3 "
            "looks — the most 3 can show' in grey rather than green, and a "
            "self-agreement chip that no longer calls itself an error bar. That "
            "fallback retires automatically on a build that records its own spread, "
            "verified against a 9-look artifact served from a second instance. "
            "Unaddressed: corpus_fingerprint is computed onto ConflictIndex.corpus and "
            "never forwarded by list_conflicts, so a reader cannot see which corpus "
            "the count was measured over."
        ),
    ),
    Slice(
        id="V8",
        title="Deep-research build loop",
        artifact_dirs=("v8_loop",),
        build_paths=(
            "src/rag/deep_research.py",
            "frontend/src/experiments/ResearchPanel.tsx",
            "experts/resume-design/research.json",
        ),
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
    #: The verdict discloses that some part of it was not written independently.
    reduced_independence: bool = False
    #: Run ids the verdict names that no longer resolve to a file on disk.
    absent_runs: tuple[str, ...] = ()


def absent_runs(text: str, directory: Path) -> tuple[str, ...]:
    """Run ids named in a verdict that nothing on disk answers to any more.

    Not a failure of the verdict — it recorded what its evaluator saw. But a
    reader who wants to re-derive a number from the run behind it cannot, and a
    row that reads PASSED without saying so is quietly overstating the evidence.
    """
    missing: list[str] = []
    for run_id in dict.fromkeys(RUN_ID.findall(text)):
        if not RUN_STAMP.search(run_id):
            continue
        if (directory / f"{run_id}.json").exists():
            continue
        if any((ROOT / d / f"{run_id}.json").exists() for d in RUN_DIRS):
            continue
        missing.append(run_id)
    return tuple(missing)


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
        raw = verdict.read_text(encoding="utf-8")
        data = json.loads(raw)
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
        reduced_independence=INDEPENDENCE_FLAG in raw,
        absent_runs=absent_runs(raw, directory),
    )


def qualifiers(verdicts: list[Evidence]) -> str:
    """The computed riders on a tally: who wrote it, and whether it can be redone."""
    parts: list[str] = []
    if any(e.reduced_independence for e in verdicts):
        parts.append("reduced independence disclosed")
    gone = sorted({run for e in verdicts for run in e.absent_runs})
    if gone:
        parts.append(f"names {len(gone)} run(s) no longer on disk: {', '.join(gone)}")
    return "".join(f" · {part}" for part in parts)


def state_of(item: Slice, evidence: list[Evidence]) -> tuple[str, str]:
    """The slice's state and the one line that justifies it.

    Ordered most-serious-first: a failing verdict outranks a passing one, because
    a slice with one of each is not half-done, it is failing.
    """
    verdicts = [e for e in evidence if e.passed is not None]
    if any(e.passed is False for e in verdicts):
        failed = next(e for e in verdicts if e.passed is False)
        detail = failed.failed_checks[0] if failed.failed_checks else "see verdict"
        why = f"{failed.name}: {failed.checks_passed}/{failed.checks_total} — {detail}"
        return "FAILING", why + qualifiers(verdicts)
    if verdicts:
        total = sum(e.checks_total for e in verdicts)
        passed = sum(e.checks_passed for e in verdicts)
        label = "PASSED (caveat)" if item.caveat else "PASSED"
        why = f"{passed}/{total} checks across {len(verdicts)} verdict(s)"
        return label, why + qualifiers(verdicts)
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
        "Two riders in the Evidence column are read out of the verdicts rather than "
        "written here by hand. *Reduced independence disclosed* means the verdict "
        "itself records that some part of it was not written by an independent "
        "evaluator — read the scope it gives before treating the whole tally as "
        "arm's-length. *Names N run(s) no longer on disk* means the verdict was "
        "decided against run files that have since been superseded or deleted: still "
        "a true record of what was seen, but no longer reproducible from the repo.",
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

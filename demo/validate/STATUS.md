# s11 slice status

**Generated — do not hand-edit.** Regenerate with:

```bash
PYTHONPATH=. uv run python -m demo.validate.status
```

Generated 2026-08-10T03:52:37+00:00. Plan: `.lavish/s11_distilled-reviewer-deep-research.html` §9. Evidence: `demo/validate/artifacts/<slice>/verdict.json`, written by the independent evaluator for that slice — never by its builder.

**10 slices — 1 failing · 3 unvalidated · 3 not started · 1 passed (caveat) · 2 passed.**

| Slice | Title | State | Evidence |
| --- | --- | --- | --- |
| V0 | Judge v2 on the Scoreboard | **PASSED** | 36/36 checks across 2 verdict(s) |
| V1 | First real document review in Chat | **PASSED** | 39/39 checks across 1 verdict(s) |
| V2 | Summary pre-filter on the Scoreboard | **UNVALIDATED** | code on disk (1 path(s)), no verdict recorded |
| V3 | Critique eval harness in Experiments | **PASSED (caveat)** | 30/30 checks across 1 verdict(s) |
| V4 | Theme layer in RAG Pipeline | **FAILING** | v4_themes: 23/24 — the link's offset matches the clock the row displays |
| V4b | Ingest the two empty topics (system design, app architecture) | **NOT STARTED** | no code, no artifacts |
| V5 | Rubric packs in the app | **UNVALIDATED** | code on disk (4 path(s)), no verdict recorded |
| V6 | Rubric-driven reviewer in Chat | **NOT STARTED** | no code, no artifacts |
| V7 | Disagreements as a first-class view | **UNVALIDATED** | code on disk (1 path(s)), no verdict recorded |
| V8 | Deep-research build loop | **NOT STARTED** | no code, no artifacts |

## What the states mean

- **PASSED** — an evaluator recorded `verdict.json` with `passed: true`.
- **PASSED (caveat)** — the frontend gate passed, but something known and open is recorded against it. Read the caveat before building on it.
- **FAILING** — a verdict exists and records `passed: false`.
- **UNVALIDATED** — code or artifacts exist, but no verdict was recorded. Under the plan's standing rule (a hypothesis that cannot be seen in the frontend fails review even if every automated gate passed) this is **not** done, however much code is behind it.
- **NOT STARTED** — no code, no artifacts.

## Open caveats against slices that otherwise read green

- **V3** — src/evals/KNOWN_GAP_attack2.md is open: the grounding gate checks that a quote resolves, never that the cited chunk supports the finding it is attached to. Reproduced 2026-08-10 — reciting the criteria with one distinct real chunk each scores evidence_precision 1.000 and criteria_recall 0.526 against the honest baseline's 0.556 / 0.158. V5, V6 and V8 are all scored on this metric.


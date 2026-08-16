# s11 slice status

**Generated — do not hand-edit.** Regenerate with:

```bash
PYTHONPATH=. uv run python -m demo.validate.status
```

Generated 2026-08-11T04:50:50+00:00. Plan: `.lavish/s11_distilled-reviewer-deep-research.html` §9. Evidence: `demo/validate/artifacts/<slice>/verdict.json`, written by the independent evaluator for that slice — never by its builder.

**10 slices — 2 passed (caveat) · 8 passed.**

| Slice | Title | State | Evidence |
| --- | --- | --- | --- |
| V0 | Judge v2 on the Scoreboard | **PASSED** | 36/36 checks across 2 verdict(s) |
| V1 | First real document review in Chat | **PASSED** | 39/39 checks across 1 verdict(s) |
| V2 | Summary pre-filter on the Scoreboard | **PASSED** | 37/37 checks across 1 verdict(s) |
| V3 | Critique eval harness in Experiments | **PASSED (caveat)** | 33/33 checks across 1 verdict(s) · names 3 run(s) no longer on disk: critique-15rTnqKBlO8-20260809-232656, critique-15rTnqKBlO8-20260810-003940, critique-15rTnqKBlO8-20260810-051413 |
| V4 | Theme layer in RAG Pipeline | **PASSED** | 24/24 checks across 1 verdict(s) |
| V4b | Ingest the two empty topics (system design, app architecture) | **PASSED** | 42/42 checks across 1 verdict(s) |
| V5 | Rubric packs in the app | **PASSED** | 37/37 checks across 1 verdict(s) · reduced independence disclosed |
| V6 | Rubric-driven reviewer in Chat | **PASSED** | 40/40 checks across 1 verdict(s) |
| V7 | Disagreements as a first-class view | **PASSED (caveat)** | 65/65 checks across 1 verdict(s) |
| V8 | Deep-research build loop | **PASSED** | 20/22 checks across 1 verdict(s) |

## What the states mean

- **PASSED** — an evaluator recorded `verdict.json` with `passed: true`.
- **PASSED (caveat)** — the frontend gate passed, but something known and open is recorded against it. Read the caveat before building on it.
- **FAILING** — a verdict exists and records `passed: false`.
- **UNVALIDATED** — code or artifacts exist, but no verdict was recorded. Under the plan's standing rule (a hypothesis that cannot be seen in the frontend fails review even if every automated gate passed) this is **not** done, however much code is behind it.
- **NOT STARTED** — no code, no artifacts.

Two riders in the Evidence column are read out of the verdicts rather than written here by hand. *Reduced independence disclosed* means the verdict itself records that some part of it was not written by an independent evaluator — read the scope it gives before treating the whole tally as arm's-length. *Names N run(s) no longer on disk* means the verdict was decided against run files that have since been superseded or deleted: still a true record of what was seen, but no longer reproducible from the repo.

## Open caveats against slices that otherwise read green

- **V3** — src/evals/KNOWN_GAP_attack2.md is now *partly* closed, and what it costs matters as much as what it fixed. The distinct-chunk attack — recite the criteria, give each recited finding its own distinct real chunk, score evidence_precision 1.000 / criteria_recall 0.526 against the honest baseline's 0.364 / 0.158 — was gated on 2026-08-11 by GATE_PROVENANCE: a citation grounds a finding only when the chunk it resolves to is one that finding's own retrieval returned. Every committed run was rescored in place with no model calls; no score moved and no within-run ranking flipped. The cost is that the retrieval arms (rag_llm_filtered, rag_conflict_aware) have no per-finding provenance and are now ungraded rather than scored, which is why V6's rubric_packs 0.000 has no certified baseline to be subtracted from. Still open is the relevance question itself: a chunk the finding really did retrieve, cited for a rule it does not support, still passes, and only an entailment check reaches it. V5, V6 and V8 are all scored on this metric.
- **V7** — The count is short of the gate and that is the one failing check in a 64/65 verdict — 2 disagreement cards render where the gate asks for >=3. What would still be open if the count rose is its reliability. It is measured at 3 looks per pair with no recorded spread, and two builds over a provably identical candidate pool returned 4 cards and then 2, with two cards moving 3/3 to 0/3. The layer now says so in its own voice rather than reading as confidence: a warn chip beside the count ('spread not recorded — nearer 2 ± 1 than exactly 2'), card chips reading 'agreed 3/3 looks — the most 3 can show' in grey rather than green, and a self-agreement chip that no longer calls itself an error bar. That fallback retires automatically on a build that records its own spread, verified against a 9-look artifact served from a second instance. Unaddressed: corpus_fingerprint is computed onto ConflictIndex.corpus and never forwarded by list_conflicts, so a reader cannot see which corpus the count was measured over.


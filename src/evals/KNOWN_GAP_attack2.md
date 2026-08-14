# Known gap: the grounding gate checks bookkeeping, not relevance

**Status: partly closed.** Found by the V3 evaluation agent, confirmed by hand on
2026-08-10 against `evals/runs/critique-15rTnqKBlO8-20260810-003940.json`. The
retrieval-provenance gate landed 2026-08-11 (`GATE_PROVENANCE`, the shipped
default). Recorded here rather than in a task tracker because anyone reading
`criteria_recall` needs to know what it does and does not certify.

## What is fixed

### The shared-quote attack (closed 2026-08-10)

N findings all citing **one** quote is dead. `ground_findings` gives a finding
only the chunks no other finding cites, so an output that recites every
criterion against a single shared quote grounds nothing and scores
`criteria_recall 0.000 · evidence_precision 0.000`. Regression test:
`test_reciting_every_criterion_on_one_shared_quote_scores_nothing`.

### The distinct-chunk attack (gated 2026-08-11)

Exclusivity closed the *sharing* loophole, not the *relevance* one, and the
attack survived one edit: give each recited finding its **own distinct** real
chunk. Reproduced against the committed baseline's own ten retrieved chunks, on
the real corpus, under the old gate:

| | baseline (`094922/rag_llm_filtered`) | 10 recited findings, one distinct chunk each |
|---|---|---|
| `evidence_precision` | 0.364 (4/11) | **1.000 (10/10)** |
| `criteria_recall` ceiling | 0.158 (3/19) | **0.526 (10/19)** |

`check_citation` asks whether the quoted words appear in the chunk at that
timestamp. Nothing asked whether that chunk **supports the finding it is
attached to**.

`GATE_PROVENANCE` is the deterministic answer, and it is option 2 of the three
listed below. A citation grounds a finding only when the chunk it resolves to is
one **that finding's own reasoning retrieved** — `Finding.retrieved_chunk_ids`,
written by the engine, never by a model (`parse_findings` ignores the field
unless `trust_provenance` is set, which only re-scoring a stored run does).
Grabbing an arbitrary chunk to decorate a recited rule no longer grounds
anything.

Regression tests, in `tests/evals/test_critique.py`:

* `test_the_distinct_chunk_attack_sweeps_the_old_gate` — reproduces the defect
  first, so the fix proves something.
* `test_the_distinct_chunk_attack_earns_nothing_under_the_provenance_gate`
* `test_a_finding_that_cites_a_chunk_its_own_retrieval_never_returned_grounds_nothing`
* `test_a_chunk_grabbed_off_retrieval_does_not_spoil_the_finding_that_did_retrieve_it`

## What it costs: the retrieval arms are now ungraded, not scored

The arms differ in what they can honestly record, and the gate refuses to paper
over that.

* **Pack arms** (`rubric_packs`, and the D2/deep-research arms `raptor`,
  `communities`, `merged`, `deep-r1`, `deep-r2`, `deep-oneshot`, and the later
  `deep-r2-admit` and `deep-frontier`) have real
  per-finding provenance: a rubric is distilled from exactly one pack unit, and
  that unit's `chunk_ids` are the entire corpus the distilling model saw for that
  rubric. All 273 citations across every committed pack arm land inside their own
  unit, so these arms pass with **their published numbers unchanged**.
* **Retrieval arms** (`rag_llm_filtered`, `rag_conflict_aware`) have none. One
  call sees one pool of ten chunks and emits every finding from it, so "what this
  finding retrieved" is the whole pool for every finding — and a gate against the
  pool passes the attack the pool was used to mount. Those cells score
  **`None` — ungraded** on `criteria_recall` and `evidence_precision`.

  This is a fact on disk, not a projection. The committed runs were rescored in
  place on 2026-08-11 and carry `grounding_gate: retrieval_provenance` with
  `ungraded_cells: ["rag_conflict_aware", "rag_llm_filtered"]`; the Critique eval
  panel renders those rows as **not measured** over their ungated figure, and
  drops the delta against them rather than subtracting from a number the scorer
  does not certify. The before/after table below is the record of that change,
  not a forecast of one.

`None`, not zero, deliberately. Zero asserts the findings have no corpus behind
them and nothing has shown that; `None` is already this harness's word for "this
cell was not measured" (`_ratio`, `contested_coverage`). And not exempt either:
an arm that cannot substantiate the claim the metric makes does not get to keep
the number. Every cell keeps `criteria_recall_ungated` and
`evidence_precision_ungated`, so nothing is lost and the published series stays
readable.

## Before / after across every committed run

Recomputed with no model calls: each cell's stored `match_runs` are replayed
(`replay_matcher`), and pack provenance is read back off the committed packs
(`provenance_for_run`). The replay reproduces every published number exactly
before the gate is changed, which is what makes the "after" column attributable
to the gate and nothing else.

| run | arm | recall published | recall gated | precision published | precision gated |
|---|---|---|---|---|---|
| `003940` (superseded) | `rag_llm_filtered` | 0.1579 | **—** | 0.5556 | **—** |
| `070727` | `rag_llm_filtered` | 0.1053 | **—** | 0.5714 | **—** |
| `070727` | `rag_conflict_aware` | 0.1053 | **—** | 0.5455 | **—** |
| `093357` | `merged` | 0.2632 | 0.2632 | 0.8235 | 0.8235 |
| `093357` | `deep-r1` | 0.3684 | 0.3684 | 0.8125 | 0.8125 |
| `093357` | `deep-oneshot` | 0.2632 | 0.2632 | 0.6154 | 0.6154 |
| `093357` | `deep-r2` | 0.3684 | 0.3684 | 0.7692 | 0.7692 |
| `094922` | `rag_llm_filtered` | 0.1579 | **—** | 0.3636 | **—** |
| `094922` | `rag_conflict_aware` | 0.1579 | **—** | 0.7500 | **—** |
| `094922` | `rubric_packs` | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| `experts/ablation.json` | `raptor` | 0.2632 | 0.2632 | 0.6667 | 0.6667 |
| `experts/ablation.json` | `communities` | 0.1053 | 0.1053 | 0.7778 | 0.7778 |
| `experts/ablation.json` | `merged` | 0.2632 | 0.2632 | 0.8235 | 0.8235 |

Zero citations landed off-retrieval on any gradable arm. **No score changed and
no ranking within a run flipped.** What changed is which comparisons exist:

* The **D2 ablation ranking is intact** — `merged` (0.824) still leads
  `communities` (0.778) leads `raptor` (0.667) on precision, `raptor` and
  `merged` still tie on recall, and `deep-r1`/`deep-r2` still lead the
  deep-research run on recall.
* The **V6 comparison no longer exists.** `rubric_packs 0.000` was published
  against `rag_llm_filtered 0.158`; the baseline is now ungraded, so there is no
  scored number to compare it against. The honest reading of that pair today is
  "the rubric arm reached none of the applicable held-out criteria, and the
  chunk-dump arm's 0.158 was never certified". Do not restate the 0.158 as a
  measured baseline.

## What is still open

Option 2 is the cheap half of the fix and it does not do option 1's job.

1. **A cited chunk the finding's own retrieval really did return, cited for a
   rule it does not support, still passes.** This is the *whole* remaining
   relevance question, and it is the common case for the pack arms: a rubric's
   unit holds five to twenty chunks, and the gate only asks that the quote came
   from one of them. Only an entailment check reaches it — one judged call per
   citation — and it is the check the metric always implied.
2. **Provenance narrower than the pool is not required.** An engine that
   declared the entire retrieved pool as every finding's own retrieval would
   satisfy the gate and it would be vacuous again. Nothing stops that
   definitionally, because every rule that would catch it also fails a
   legitimate arm whose findings genuinely came from one unit. It is *reported*
   instead: `provenance_distinct_sets` on every cell says how many distinct
   declarations there were, so a cell with one set across twenty findings is
   visible. The reason the retrieval arms are ungraded rather than
   pool-graded is precisely that this move is available and would mean nothing.
3. **A pack arm's provenance is build-declared.** It is the unit the build says
   the rubric came from. That is a genuine record and not a model's claim, but a
   broken or dishonest build could widen a unit and widen the gate with it. The
   `unit_id`-to-unit lookup fails closed (a rubric whose unit is missing is left
   ungradable, taking the whole arm to ungraded), which is the only part of this
   the scorer can enforce on its own.
4. **`contested_coverage` is deliberately outside the gate.** Its denominator is
   fixed by retrieval before the answering call, so verbosity cannot move it, and
   the distinct-chunk attack does not touch it.
5. **Novelty** — score a finding only when it says something the criterion text
   alone does not, so recitation earns nothing — remains unimplemented and
   remains the hardest of the three to specify.

## What `criteria_recall` certifies today

For an arm with per-finding provenance: *"reached the held-out expert's
conclusion in a finding whose own retrieval returned a chunk that resolves, and
which no other finding also rests on."* Not *"the corpus produced this
insight"* — the citation is still not known to **support** the finding, only to
have been in front of the reasoning that produced it.

For an arm without it: nothing. The cell reads `None`, and
`criteria_recall_ungated` is a lower-bound figure under the old rule, which the
distinct-chunk attack beats 3.3x. Do not quote it as a baseline.

# Known gap: the grounding gate checks bookkeeping, not relevance

**Status: open.** Found by the V3 evaluation agent, confirmed by hand on
2026-08-10 against `evals/runs/critique-15rTnqKBlO8-20260810-003940.json`.
Recorded here rather than in a task tracker because anyone reading
`criteria_recall` needs to know what it does and does not certify.

## What is already fixed

The first padding attack — N findings all citing **one** quote — is dead.
`ground_findings` gives a finding only the chunks no other finding cites, so an
output that recites every criterion against a single shared quote grounds
nothing and scores `criteria_recall 0.000 · evidence_precision 0.000`. There is
a regression test for it.

## What is not

Exclusivity closed the *sharing* loophole. It did not close the *relevance*
one. `check_citation` asks whether the quoted words appear in the chunk at that
timestamp. Nothing anywhere asks whether that chunk **supports the finding it
is attached to**.

So the attack survives one edit: give each recited finding its **own distinct**
real chunk.

Measured against the committed baseline's own 10 retrieved chunks:

| | baseline | 10 recited findings, one distinct chunk each |
|---|---|---|
| `evidence_precision` | 0.556 (5/9) | **1.000 (10/10)** |
| `criteria_recall` ceiling | 0.158 (3/19) | **0.526 (10/19)** |

Every recited finding keeps its evidence. A system that emits generic advice it
already knew, and staples one distinct resolving citation onto each line, beats
the honest baseline on both metrics — more than three times the recall.

The ceiling is set by `top_k`: with 10 retrieved chunks the attack tops out at
10/19. Raising `top_k` raises the ceiling.

## Why this matters more than a normal metric bug

`criteria_recall` carries the project's central claim — that the corpus reaches
an expert's conclusions **without having seen that expert**. An ungrounded
recitation of prior knowledge is the exact opposite of that claim, and this
metric currently rewards it. V5, V6 and V8 are all scored here.

## What would close it

The citation must be tied to the finding, not merely to the corpus. Options,
cheapest first:

1. **Entailment check** — does the cited chunk support this finding's claim?
   One judged call per citation, and it is the check the metric always implied.
2. **Retrieval provenance** — require the cited chunk to be one the finding's
   own reasoning retrieved, not any chunk in the pool. Deterministic, free, but
   only works for engines that record per-finding retrieval.
3. **Novelty gate** — score a finding only when it says something the criterion
   text alone does not, so recitation earns nothing. Hardest to specify.

Until one lands, read `criteria_recall` as *"reached the conclusion **and** cited
something that resolves"*, not as *"the corpus produced this insight"*.

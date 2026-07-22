# Growing the golden set

The golden set (`src/evals/golden_dataset.json`) is what makes retrieval
*measurable*: each entry pairs a real question with a grounded reference answer and
the `expected_chunk_ids` a good retriever must surface. Those labels are what the
reference-free RAGAS metrics cannot give you — recall (what retrieval *missed*) and
correctness (whether the answer is actually *right*).

It currently holds 9 curated entries. The target is **40+**, reported per domain
(property vs ai-coding), so a headline number is never an average hiding a segment
that fails. This guide is the process for getting there without diluting quality.

## Why it stays hand-curated

`expected_chunk_ids` name the specific chunks that answer a question. Only a human
who has read the transcript can say which chunks those are, and a wrong label
silently corrupts every recall number computed against it. So generation is
assisted, but curation is manual — a small, correct golden set beats a large, noisy
one.

## The schema

Each entry is a `GoldenEntry` (`src/evals/golden.py`), validated on load:

```json
{
  "id": "g010",
  "question": "A question a real user would ask of this corpus",
  "reference_answer": "A grounded answer written from the transcripts, not memory.",
  "expected_video_ids": ["<video_id>"],
  "expected_chunk_ids": ["chunk:<video_id>:<index>"],
  "domain": "property",
  "notes": "optional — why these chunks, any caveats"
}
```

Validation enforces the invariants that catch hand-edited drift: chunk ids must be
`chunk:<video_id>:<index>`, every `expected_chunk_id`'s video must appear in
`expected_video_ids` and vice versa, no duplicates, and `domain` must be one of the
known domains (`property`, `ai-coding`).

## Adding an entry by hand

1. **Pick a question** a real user would ask, that the corpus can actually answer.
   Aim for a spread — factual lookups, comparisons, multi-video questions.
2. **Find the supporting chunks.** Open the workbench **RAG Pipeline → Retrieval
   Lab**, search your question, and read the top chunks; or list a video's chunks
   with `GET /api/corpus/{video_id}/chunks`. Note the `chunk:<video_id>:<index>` id
   of each chunk that genuinely supports the answer.
3. **Write the reference answer** from those chunks — grounded, not from memory.
4. **Assign the domain** and add the entry to `entries` in
   `src/evals/golden_dataset.json`.
5. **Validate** it loads cleanly:
   ```bash
   uv run python -c "from src.evals.golden import load_golden; print(len(load_golden()), 'entries OK')"
   ```
6. **Re-baseline** the runs so the new question is reflected:
   ```bash
   uv run python -m src.cli eval-ablation
   ```

## Drafting candidates with RAGAS (the scaffold)

To start from a list instead of a blank page, draft candidate questions from the
corpus with RAGAS' `TestsetGenerator`:

```bash
uv run python scripts/generate_golden_candidates.py --size 20 --out golden_candidates.json
```

This calls an LLM (so it needs an API key and is never run in CI) and writes
**unverified candidates** — questions and reference answers only. Each still needs a
human to do step 2 above (find and add `expected_chunk_ids`), assign a domain, and
fact-check the reference before it is promoted into `golden_dataset.json`. Treat the
output as a worklist, not a dataset.

## Keeping segments balanced

Report and grow the set per domain. The ablation harness already breaks every
metric down by domain (`by_domain` in `ablation-*.json`, shown in the workbench
**Experiments** tab), so aim for enough questions in each domain that a per-segment
number is meaningful — not one domain carrying the set.

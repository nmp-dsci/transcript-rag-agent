# Committed evaluation runs

These JSON snapshots are the **committed evidence** behind the project's retrieval
and answer-quality claims. Unlike the working data under the gitignored
`.yt-agent/`, they live in source control so a reviewer can open the exact numbers
a configuration produced, and so CI can gate on them without a live corpus.

Two kinds of run land here:

| Prefix | Produced by | What it measures |
|--------|-------------|------------------|
| `ablation-*.json` | `uv run python -m src.cli eval-ablation` | Retrieval only — semantic vs hybrid vs hybrid+rerank across `recall@k`, `MRR`, `NDCG` over the golden set. Deterministic, no LLM, no API key. |
| `eval-*.json` | `uv run python -m src.cli eval-golden …` | End-to-end — answers generated and (optionally) RAGAS-judged, alongside the deterministic recall/IR metrics. |

## Provenance

Runs are stamped with the configuration that produced them (`config` block:
answer model, embedding model, retrieval mode, rerank, `top_k`, judge model,
`judge_samples`) and, for ablations, the sweep of configurations compared. They
were generated against the committed corpus recorded in
`src/evals/golden_dataset.json` (its `corpus` block names the videos and chunk
counts and the date they were verified).

Because `expected_chunk_ids` are chunking-dependent, re-indexing the corpus with a
different chunk size renumbers chunks and invalidates the golden labels — so the
committed runs are only comparable against an index built with the same chunking.

## Regenerating

```bash
# Retrieval ablation (free, deterministic — no API key needed)
uv run python -m src.cli eval-ablation

# End-to-end golden run under the current config (generates answers; judges unless --no-judge)
uv run python -m src.cli eval-golden --setup rag_llm --retrieval hybrid

# Grade with an independent (non-DeepSeek) judge — any OpenAI-compatible API
YT_AGENT_JUDGE_MODEL=gpt-4o-mini \
YT_AGENT_JUDGE_API_KEY=$OPENAI_API_KEY \
YT_AGENT_JUDGE_BASE_URL=https://api.openai.com/v1 \
  uv run python -m src.cli eval-golden --setup rag_llm --retrieval hybrid

# Compare the two most recent golden runs (same-config regression check)
uv run python -m src.cli eval-golden --diff
```

## The CI eval gate

`tests/evals/test_committed_runs.py` re-scores these snapshots on every CI run: it
checks the schema and provenance are complete and enforces floors on the headline
retrieval claims (for example, that hybrid fusion still improves early-rank recall
over plain semantic). The gate is deterministic and needs no corpus or API key —
only the committed JSON here. Regenerating the snapshots needs the local corpus;
validating them does not.

# Committed evaluation runs

These JSON snapshots are the **committed evidence** behind the project's retrieval
and answer-quality claims. Unlike the working data under the gitignored
`.yt-agent/`, they live in source control so a reviewer can open the exact numbers
a configuration produced, and so CI can gate on them without a live corpus.

Three kinds of run land here:

| Prefix | Produced by | What it measures |
|--------|-------------|------------------|
| `ablation-*.json` | `uv run python -m src.cli eval-ablation` | Retrieval only — semantic vs hybrid vs hybrid+rerank across `recall@k`, `MRR`, `NDCG` over the golden set. Deterministic, no LLM, no API key. |
| `eval-*.json` | `uv run python -m src.cli eval-golden …` | End-to-end for **one** setup — answers generated and (optionally) RAGAS-judged, alongside the deterministic recall/IR metrics. |
| `matrix-*.json` | `uv run python -m src.cli eval-matrix`, `scripts/run_matrix_chunked.py`, or the Experiments tab's **Run eval matrix** button | Head-to-head — every answer engine (`rag_llm`, `rag_llm_recursive`, `rag_agent`, `graph_rag`) answers the same golden questions under one config and one judge; a comparison pivot by metric × setup, overall and per `question_type`. |

### Why `matrix-*.json` is the important one

It is the only run shape where the setups are directly comparable: same questions,
same retrieval config, same judge, in one file. That is why it powers **both**
comparison tabs — Experiments renders its `comparison` pivot (metric × setup, overall
and by question type), and the **Scoreboard** aggregates its per-entry `scores` into
the leaderboard and win-rate table, selectable per run from a dropdown. The
Scoreboard also pivots the same `entries` the other way in its **Questions** panel —
one row per golden question, each setup's composite on it — so an aggregate row can
be read back down to the questions it averages.

The Scoreboard deliberately does **not** read `dashboard/chat_history.json` any more.
Chat history is the *live* set — whatever a human happened to ask and judge — so a
newly added engine stayed invisible there until someone manually re-asked every
question through it. Ranking on a matrix run instead means an engine appears as soon
as its cells exist.

Structure: `runs` is keyed by setup, each holding that setup's `entries` (one per
golden question, with `scores`, `answer`, `retrieved_chunk_ids`, `elapsed_seconds`,
`token_estimate`) and a `summary`; `comparison` holds the pivot; `cache_hits` /
`cache_misses` record how much of the run was reused rather than re-scored.

### Cell caching

Every `(setup, question)` cell is cached by a fingerprint of the question plus the
exact answering and judging configuration (`src/evals/matrix_cache.py`, under the
gitignored `.yt-agent/eval_cache/`). Adding one question or one engine re-scores only
the cells that do not already exist — changing the answer model, embedding model,
retrieval mode, `top_k`, rerank settings, judge, or a setting the answering engine
reads (its recursion budget, its iteration cap, retrieval breadth) changes the
fingerprint, so stale numbers are never silently reused. Engine-specific settings
are scoped to the engines that read them, so tuning one engine keeps the others'
cells valid. Pass `--refresh` to bypass the cache.


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

# Head-to-head across every setup (what the Scoreboard ranks). Cached by cell,
# so re-running after adding a question only pays for the new cells.
uv run python -m src.cli eval-matrix

# The same matrix, parallel across engines and resumable: every cell is cached
# as it completes, so a killed or time-budgeted process loses nothing
uv run python scripts/run_matrix_chunked.py --max-seconds 600
```

## The CI eval gate

`tests/evals/test_committed_runs.py` re-scores the `ablation-*.json` and
`eval-*.json` snapshots on every CI run: it checks the schema and provenance are
complete and enforces floors on the headline retrieval claims (for example, that
hybrid fusion still improves early-rank recall over plain semantic). The gate is
deterministic and needs no corpus or API key — only the committed JSON here.
Regenerating the snapshots needs the local corpus; validating them does not.
`matrix-*.json` runs are not part of this gate — they are comparison evidence for
the Experiments tab, not a regression floor.

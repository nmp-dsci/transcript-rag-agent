## YouTube Transcript RAG Demo

CLI prototype that demonstrates the value of RAG over full-transcript prompting for YouTube transcript Q&A.

The main demo compares one question across three transcript input types:

- `raw_single`: full raw transcript for one video.
- `rag_single`: top 10 retrieved chunks for that same video.
- `rag_all`: top 10 retrieved chunks across all indexed videos.

The demo writes `dashboard/evaluation.html`: one question answered three ways — `rag_llm` single-hop, `rag_llm` recursive, and the agentic `rag_agent` — laid out as three side-by-side columns in dark mode, each titled by its command with the full command in an expandable block.

### Setup

This project uses `uv`.

```bash
uv sync
```

The dashboard Chunk Space tab uses `scikit-learn` for deterministic PCA projection of stored chunk embeddings; it is installed by `uv sync`.

Create `~/.env`:

```text
SUPADATA_API_KEY=<Supadata API key>
# SUPERDATA_API_KEY is also supported for compatibility with earlier project wording.
DEEPSEEK_API_KEY=<DeepSeek API key>
DEEPSEEK_MODEL=deepseek-v4-flash
DEEPSEEK_BASE_URL=https://api.deepseek.com
YT_AGENT_CHROMA_PATH=.yt-agent/chroma
YT_AGENT_RAW_TRANSCRIPT_COLLECTION=raw_transcripts
YT_AGENT_CHUNK_COLLECTION=transcript_chunks
YT_AGENT_TRANSCRIPT_SUMMARY_COLLECTION=transcript_summaries
YT_AGENT_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
YT_AGENT_RAG_TOP_K=10
YT_AGENT_TRANSCRIPT_FILTER_TOP_K=5
YT_AGENT_TRANSCRIPT_FILTER_MIN_SCORE=0.25
YT_AGENT_RAG_RECURSIVE_DEFAULT=false
YT_AGENT_RAG_MAX_DEPTH=1
YT_AGENT_RAG_MAX_FOLLOWUPS=3
YT_AGENT_RAG_FOLLOWUP_TOP_K=
YT_AGENT_RAG_NOVELTY_MIN_CHUNKS=2
YT_AGENT_RAG_MAX_TOTAL_FOLLOWUPS=
YT_AGENT_RAG_AGENT_MAX_ITERATIONS=10
YT_AGENT_CHUNK_TARGET_CHARS=1200
YT_AGENT_CHUNK_OVERLAP_CHARS=150
YT_AGENT_RETRIEVAL_MODE=semantic
YT_AGENT_RETRIEVAL_CANDIDATES=30
YT_AGENT_RERANK_ENABLED=false
YT_AGENT_RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
YT_AGENT_NEIGHBOR_SPAN=0
YT_AGENT_JUDGE_SAMPLES=1
YT_AGENT_DISCOVERY_CACHE_TTL_HOURS=24
SUPADATA_TIMEOUT_SECONDS=120
SUPADATA_POLL_INTERVAL_SECONDS=2
SUPADATA_MAX_POLL_SECONDS=600
MLFLOW_TRACKING_URI=file:.yt-agent/mlruns
MLFLOW_EXPERIMENT_NAME=yt-agent-v1
YT_AGENT_LOG_TRANSCRIPT_ARTIFACTS=false
```

`SUPADATA_API_KEY` is used with the Supadata transcript API. DeepSeek is called through the OpenAI-compatible LangChain client.

### Retrieval strategy

`YT_AGENT_RETRIEVAL_MODE` selects how chunks are found:

- `semantic` (default) — embed the question, cosine-rank chunk embeddings.
- `hybrid` — rank semantically *and* with BM25, then fuse the two rankings with
  Reciprocal Rank Fusion. Keyword and embedding retrieval disagree most on exact
  terms (figures, names, dates), which is what fusion recovers.

Both modes pull `YT_AGENT_RETRIEVAL_CANDIDATES` chunks before narrowing to
`top_k`, because reranking can only reorder what it was given. Set
`YT_AGENT_RERANK_ENABLED=true` to rerank those candidates with a local
cross-encoder (`YT_AGENT_RERANK_MODEL`); it loads lazily on first use and adds
no API calls. `YT_AGENT_NEIGHBOR_SPAN=1` pastes the chunks either side of each
hit into the context, which stops answers being cut off mid-sentence at a chunk
boundary. Per-request overrides come from the workbench's ⚙ advanced panel, so a
setup can be compared under both modes with the same judge.

Retrieval can be scoped to a **channel** or a **single video**. Channel
filtering is a native metadata filter, which is why chunks carry
`channel_id`/`channel_name`. Chunks indexed before that existed need a one-off
backfill:

```bash
uv run python scripts/backfill_chunk_metadata.py --dry-run   # report only
uv run python scripts/backfill_chunk_metadata.py             # stamp metadata
uv run python scripts/backfill_chunk_metadata.py --re-embed  # + contextual headers
```

The plain form only rewrites metadata and never re-embeds. `--re-embed` also
rebuilds every chunk vector with a contextual header (`[channel — title @
mm:ss-mm:ss]`) prepended before embedding. Transcript chunks are conversational
fragments that frequently lose their subject ("had. So, I'm going to just
copy…"), and the header restores the context the speaker left implicit; the
header is embedded but is not part of the text shown to the answering LLM.

Supadata can return async jobs for longer videos. `SUPADATA_MAX_POLL_SECONDS=600` lets indexing wait up to 10 minutes for those jobs before timing out.

Recursion env vars are used only when recursive mode is effectively on via `--recursive` or `YT_AGENT_RAG_RECURSIVE_DEFAULT=true`. Empty `YT_AGENT_RAG_FOLLOWUP_TOP_K` defaults follow-up retrieval to `YT_AGENT_RAG_TOP_K`; empty `YT_AGENT_RAG_MAX_TOTAL_FOLLOWUPS` defaults to `max_depth * max_followups`.

`YT_AGENT_RAG_AGENT_MAX_ITERATIONS` (default `10`) is read only when the agentic RAG agent is used via `rag-ask --rag_agent`. It is the hard cap on the LangGraph ReAct loop and can be overridden per-run with `--max-iterations`. It has no effect on any other path.

### Interactive Chat

The recommended entry point is the menu-driven chat. It wraps the same agents
the individual commands use, captures every question and answer, and renders a
WhatsApp-style transcript you can browse in the dashboard.

```bash
uv run python -m src.cli chat
```

#### Main menu

On launch you get a top-level menu and pick one action by typing its key:

```text
Main menu:
  [1] Ask a question
  [2] Fetch / index a new URL
  [q] Quit
Choose:
```

`q` (or `quit`/`exit`, or Ctrl-D) leaves the session. After each action the
menu reappears so you can keep going.

#### [1] Ask a question

The ask flow has three prompts:

1. **Question** — the question to ask the indexed corpus.
2. **Restrict to a single video URL** — optional. Leave blank to search every
   indexed transcript, or paste one video URL to confine retrieval to it.
3. **RAG setup(s)** — pick one setup, several (e.g. `1,3`), or `a` for all to
   answer the same question every way and compare:

   ```text
   RAG setups:
     [1] rag_llm (single-hop) — One retrieval across all indexed transcripts, then a single LLM answer.
     [2] rag_llm (recursive)  — Multi-hop retrieval: follow-up queries fan out, then a final synthesis call.
     [3] rag_agent (agentic)  — LangGraph ReAct loop that retrieves across sub-topics until it has enough evidence.
     [a] all (compare every setup)
   Choose setup(s) (e.g. 1,3 or a; blank to cancel):
   ```

The selected setups run in order (the retrieval stack loads once, on the first
question of the session). Each answer is appended to the chat history and the
`chat.html` view is regenerated. Example session:

```text
Choose: 1
Question: Is the Gold Coast property market at risk of collapse, and why?
Restrict to a single video URL (optional, blank for all):
Choose setup(s) (e.g. 1,3 or a; blank to cancel): a
  Running rag_llm (single-hop) ...
  Running rag_llm (recursive) ...
  Running rag_agent (agentic) ...

Captured 3 answer(s) for: q-20260616-005733-5393
  - rag_llm (single-hop): 776 chars (18.44s)
  - rag_llm (recursive): 2533 chars (17.25s)
  - rag_agent (agentic): 11533 chars (34.59s)
Updated dashboard/chat.html — open it to read the conversation.
```

#### [2] Fetch / index a new URL

The fetch flow first asks whether to index a single video or a whole channel:

```text
Fetch a new URL:
  [1] Single video URL
  [2] Bulk (whole channel)
Choose:
```

- **[1] Single video URL** — prompts for one `Video URL:` and runs the same
  pipeline as `index-rag <url>`.
- **[2] Bulk (whole channel)** — prompts for a `Channel (URL or @handle):` and
  `How many latest videos? [5]:`, then runs `bulk-index channel --channel <c>
  --latest <n>`.

Both paths reuse the documented indexing commands below, so newly indexed
transcripts are immediately available to the ask flow.

#### Browsing the conversation

Each answered question is appended to `dashboard/chat_history.json` and the
`dashboard/chat.html` view is regenerated. Open it to read conversations:

```text
dashboard/chat.html
```

The left sidebar lists every question with its time and id (newest first);
clicking one loads that conversation in the main panel — your question as an
outgoing bubble, then one incoming bubble per RAG setup, each headed by the
setup name with the answer, retrieval metadata, references, and the equivalent
`rag-ask` command below it. Because the history and view regenerate after every
question, you can keep `chat.html` open and just refresh.

### Evaluation Workbench (browser)

The web app is a chat-first evaluation workbench: ask a question, read the
answer as a conversation, and have **RAGAS score every answer under the same
eval process** — faithfulness, answer relevancy, and context precision, plus a
composite — so retrieval methods are compared with numbers, not vibes.

The UI is a React 19 + TypeScript app under `frontend/`, built with Vite and
served by the same FastAPI process. It follows the OS light/dark preference by
default; the ☀/☾ toggle in the header overrides that and persists the choice
per browser.

```bash
cd frontend && npm install && npm run build && cd ..   # once, and after UI changes
uv run python -m src.cli serve                         # http://127.0.0.1:8000
uv run python -m src.cli serve --host 0.0.0.0 --port 9000
```

`frontend/dist/` is gitignored, so a fresh clone must run `npm run build`
before `serve` shows the React UI. Without a build, `/` falls back to the
legacy single-file page and `GET /api/health` reports `"ui": "legacy"` — the
API is unaffected either way.

Three views (the tab formerly called **Library** is now **RAG Pipeline**; old
`#library` links still resolve):

- **Chat** — the landing tab. Type a question and it is answered in a
  conversation thread with citations back to source timestamps. The default
  agent is `rag_agent` (agentic), whose retrieval loop streams into the bubble
  live — one line per iteration showing the query it chose and how many chunks
  came back — so a ~30s research run reads as progress rather than a stall.
  Composer chips switch the scope (whole corpus or one video) and the
  answering agent; **⚙ advanced** exposes `top_k`, the auto-judge toggle, and
  additional setups to run alongside the default. When several setups answer
  the same question they share **one bubble with tabs**, each carrying its own
  answer, citations, and RAGAS score, with the best composite badged TOP and a
  compare grid underneath. "Compare N more setups" runs the remaining ones
  into the *same* history entry so the scoreboard sees them as competing
  answers. Esc cancels a running ask.
- **Library** — an interactive corpus tree (all videos → channel → video →
  chunks) with a sort control for "top" ordering by views, recency, chunk
  count, or title. Expanding a video lazily loads its chunks; selecting one
  shows its full text, timestamp range, segment span, and a deep link into
  the video at that moment. The **Retrieval Lab** at the top ranks the corpus
  for any query with **BM25, semantic, or both side by side** — aligned rows
  show each chunk's rank in the other mode (`↑2`, `↓1`, `only here`) plus an
  overlap count, which is the fastest way to see where keyword and embedding
  retrieval disagree. Indexing (single video or latest-N channel) lives in a
  panel here.
- **Scoreboard** — per-setup aggregates across everything judged, groupable by
  **setup × answering model** so scores from different model versions are never
  silently averaged. Each row shows average score per RAGAS metric, composite,
  win rate, latency, and token estimate. A judge filter keeps self-graded and
  independently-graded runs apart, and a provenance bar states the judge model,
  ragas version, embedding model, metric definitions, and last-judged time.
  Answers captured before model identity was recorded appear as
  `— pre-provenance` and are excluded from cross-model comparison.

#### Frontend development

```bash
cd frontend
npm install
npm run dev        # Vite on :5173, proxies /api to uvicorn on :8000
npm test           # Vitest
npm run typecheck  # tsc --noEmit (strict)
npm run build      # emits frontend/dist for `serve`
```

Run `uv run python -m src.cli serve` in a second terminal while using
`npm run dev`; the dev server proxies `/api` to it and passes SSE straight
through. Restart `serve` after the first `npm run build` so it picks up the
newly created `frontend/dist`.

Retrieved chunk texts are persisted with each answer (`contexts` in the
history JSON) so judging can run at any time, including re-judging with
`force`. Questions asked in the browser are appended to the same
`dashboard/chat_history.json` and regenerate `dashboard/chat.html`, so the CLI
chat, the workbench, and the static viewer share one history. Entries recorded
before context persistence report "no stored retrieval contexts" instead of
scores.

Each answer also records the stack that produced it — `model`,
`embedding_model`, and the effective `top_k` — and each evaluation records
`ragas_version` and the judge's `embedding_model`. All of these default to
`null`, so histories written before they existed keep loading unchanged; the
scoreboard reports those rows as `— pre-provenance` rather than attributing
them to a model.

Endpoints (JSON unless noted):

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | The workbench UI (React bundle, else the legacy page) |
| `/api/health` | GET | Liveness, lazy-stack state, judge/answer/embedding models, `ui` mode |
| `/api/setups` | GET | The three RAG setup descriptors |
| `/api/history` | GET | All captured conversations (with evaluations) |
| `/api/corpus` | GET | Indexed videos with metadata and chunk counts |
| `/api/corpus/{video_id}/chunks` | GET | Stored chunks for one video, ordered by index |
| `/api/scoreboard` | GET | RAGAS aggregates; `group_by=setup\|setup_model`, `judge_model` filter |
| `/api/ask` | POST | Answer a question (streams SSE; `entry_id` appends to an existing entry) |
| `/api/rank` | POST | Rank the corpus for a query by `semantic` and/or `bm25` |
| `/api/judge` | POST | RAGAS-score an entry's answers (streams SSE; `force` re-judges) |
| `/api/index` | POST | Index a video (`mode=video`) or channel (`mode=channel`) |
| `/api/index/stream` | POST | Index with per-stage SSE progress and a summary of what changed |
| `/api/chunk-graph` | POST | kNN similarity graph over chunk embeddings; `query` highlights its retrieval neighbourhood |

`/api/ask` emits these SSE events: `progress` (per setup), `agent_step` (one
per `rag_agent` retrieval iteration, carrying its query and chunk count),
`answer` (a finished setup), `done` (the saved entry), and `error`.

Keyword ranking uses `rank-bm25`, a small pure-Python Okapi BM25 implementation
installed by `uv sync`. The index is built in memory from stored chunk texts
and cached per chunk count, which is appropriate at this project's scale;
`src/rag/bm25.py` treats a chunk as a hit when it contains a query term rather
than when it scores above zero, because BM25's IDF term floors to zero for a
term appearing in roughly half a small corpus.

The judge LLM defaults to the configured DeepSeek model (self-grading); set
`YT_AGENT_JUDGE_MODEL`, `YT_AGENT_JUDGE_API_KEY`, and `YT_AGENT_JUDGE_BASE_URL`
to grade with an independent provider instead — any OpenAI-compatible API
works. Answer-relevancy embeddings use the same local sentence-transformers
model as retrieval. Each evaluation records `self_graded`, so a score the model
gave its own answer is never quietly compared against an independently graded
one.

#### How a score is derived

Every evaluation persists the judge's workings under `evaluation.details`, and
the workbench renders them when a metric bar is clicked:

- **faithfulness** — the claims extracted from the answer, each with a 0/1
  verdict and the judge's reason. Score is `supported / total`.
- **answer relevancy** — the question the judge generated from the answer, its
  cosine similarity to the real question, and the noncommittal flag. Score is
  the mean cosine, zeroed if the answer was evasive.
- **context precision** — a usefulness verdict per retrieved chunk in rank
  order. Score is average precision, so a useful chunk ranked low costs more
  than one ranked high.

Scores are computed *from* these intermediates rather than captured alongside
them, so a breakdown can never disagree with the number above it. Evaluations
judged before this existed have `details: null` and fall back to a static
explainer of each metric.

`YT_AGENT_JUDGE_SAMPLES` (default `1`) runs each metric several times and
records the mean plus the spread. DeepSeek rejects `n>1`, so samples are
independent calls — raising it multiplies judge time and cost. A single sample
is noisy enough that the UI shows the spread rather than implying more precision
than one pass supports.

#### Golden set and regression runs

`src/evals/golden_dataset.json` holds curated questions with reference answers
and the chunk ids a good retriever must surface. It unlocks the two things the
reference-free RAGAS metrics cannot measure: what retrieval **missed**
(`context_recall`, `video_recall`) and whether an answer is actually **right**
(`answer_correctness`, `answer_similarity`).

```bash
uv run python -m src.cli eval-golden --setup rag_llm
uv run python -m src.cli eval-golden --setup rag_llm --retrieval hybrid
uv run python -m src.cli eval-golden --no-judge          # recall only, fast
uv run python -m src.cli eval-golden --reference-metrics # + LLM reference metrics
uv run python -m src.cli eval-golden --diff              # compare the last two runs
```

Each run snapshots to `.yt-agent/eval_runs/` together with the configuration
that produced it (models, retrieval mode, rerank, top_k). `--diff` reports
per-metric and per-question movement and exits non-zero when a metric regresses,
so a config change can be shown to have helped rather than assumed to have.
Movements under 0.02 are reported as unchanged: one judged sample does not
support reading meaning into the third decimal. A question that errors is
recorded with its error and excluded from the averages rather than scored zero.

Dependencies: `ragas` (the eval metrics) and `uvicorn` (the server), both
installed by `uv sync`. `src/evals/_ragas_compat.py` shims two legacy Vertex AI
imports that ragas 0.4 expects from older `langchain-community` releases; the
retrieval and judge stacks load lazily on first use, never at startup.

### Command Sequence

The chat menu above is the recommended entry point. The individual commands
below are the underlying building blocks it calls; run them from the project
root after `uv sync` and env setup.

Set a reusable URL and question:

```bash
url="https://www.youtube.com/watch?v=3hk7nO_q0a8"
question="what does this video say for capital gains tax, is it being grandfathered or every now under new rules, does that mean if I sell before 30 June 2027 I can still access 50% discount"
```

#### 1. Optional transcript fetch

Fetch and cache a transcript without building the RAG index:

```bash
uv run python -m src.cli fetch "$url"
```

Fetch raw timestamped transcript segments:

```bash
uv run python -m src.cli fetch-raw "$url"
```

Use `--no-refresh` with either command to read from cache only when available:

```bash
uv run python -m src.cli fetch "$url" --no-refresh
uv run python -m src.cli fetch-raw "$url" --no-refresh
```

#### 2. Index transcripts

Index one YouTube transcript for RAG:

```bash
uv run python -m src.cli index-rag "$url"
```

`index-rag` stores raw transcript segments, chunk embeddings, an LLM-generated transcript summary, and a transcript-level summary embedding used for optional summary-first filtering. Regenerate the summary and summary embedding with:

```bash
uv run python -m src.cli index-rag "$url" --refresh-summary
```

Force a full transcript refresh and rebuild chunks:

```bash
uv run python -m src.cli index-rag "$url" --refresh
```

Bulk-index the most recent videos from a YouTube channel via Supadata discovery:

```bash
uv run python -m src.cli bulk-index channel \
  --channel "https://www.youtube.com/@aiDotEngineer" \
  --latest 5 \
  --label "ai-engineer-latest-5"
```

Preview a channel discovery run without indexing:

```bash
uv run python -m src.cli bulk-index channel \
  --channel "https://www.youtube.com/@aiDotEngineer" \
  --latest 5 \
  --dry-run
```

Bulk-index every video a channel published in a date window:

```bash
uv run python -m src.cli bulk-index channel \
  --channel "@somechannel" \
  --since 2026-01-01 \
  --until 2026-05-17 \
  --max-results 50 \
  --label "somechannel-q1-q2"
```

Bulk-index the top N YouTube search results for a query:

```bash
uv run python -m src.cli bulk-index search \
  --query "australian capital gains tax reform" \
  --top-n 10 \
  --label "cgt-top10"
  --dry-run
```

Common `bulk-index` flags:

- `--dry-run` — run discovery only, do not index.
- `--skip-existing` / `--no-skip-existing` — default skips videos already fully indexed in both `raw_transcripts` and `transcript_chunks`.
- `--refresh-summary` — regenerate transcript summaries even when raw transcripts and chunks are reused.
- `--concurrency 1` — only sequential ingestion is currently supported.
- `--no-discovery-cache` — bypass the 24h discovery cache for this run.

Each `bulk-index` run writes one JSON record under `.yt-agent/ingestion_runs/` capturing per-candidate outcomes. The Ingestion Runs tab in `rag_pipeline.html` reads these records when any exist.

#### 3. Refresh the RAG dashboard

Render the local RAG pipeline review dashboard:

```bash
uv run python -m src.dashboard.rag_pipeline --output dashboard/rag_pipeline.html
```

Force-refit the chunk-space PCA projection:

```bash
uv run python -m src.dashboard.rag_pipeline \
  --output dashboard/rag_pipeline.html \
  --refresh-projection
```

Override the canonical question used in the Chunk Space tab:

```bash
uv run python -m src.dashboard.rag_pipeline \
  --output dashboard/rag_pipeline.html \
  --question "$question"
```

Open:

```text
dashboard/rag_pipeline.html
```

#### 4. Ask questions

Full transcript (raw): sends the whole single-video transcript to the LLM.

```bash
uv run python -m src.cli ask "$url" "$question" --context raw
```

Single-transcript RAG: retrieves chunks from one video before calling the LLM.

```bash
uv run python -m src.cli ask "$url" "$question" --context rag --top-k 10
```

Multi-transcript RAG (single-hop): retrieves chunks across every indexed video, or restricts the same agent to one URL with `--url`.

```bash
question="how do ai engineers leveage claude to fully develop features and only set & review"

uv run python -m src.cli rag-ask "$question" --top-k 20
uv run python -m src.cli rag-ask "$question" --url "$url" --top-k 10
```

Multi-transcript RAG (single-hop, summary-filtered): first selects relevant transcript summaries, then retrieves chunks only from those videos.

```bash
uv run python -m src.cli rag-ask "$question" --filter-transcripts --top-k 20
uv run python -m src.cli rag-ask "$question" --filter-transcripts \
  --transcript-filter-top-k 8 --transcript-filter-min-score 0.3 --top-k 20
```

Multi-transcript RAG (single-hop, show follow-ups): still performs one retrieval and one LLM call, but prints the model's proposed follow-up retrieval queries.

```bash
uv run python -m src.cli rag-ask "$question" --show-followups
uv run python -m src.cli rag-ask "$question" --url "$url" --show-followups
uv run python -m src.cli rag-ask "$question" --filter-transcripts --show-followups
```

Multi-transcript RAG (recursive): acts on follow-up queries with bounded fan-out retrieval, then runs a final synthesis call.

Recursive RAG is a `rag_llm` feature only. It has no effect with `--rag_agent` (the agentic agent runs its own research loop). Since `rag_llm` is the default, the examples below omit the agent flag.

```bash
uv run python -m src.cli rag-ask "$question" --recursive
uv run python -m src.cli rag-ask "$question" --recursive --url "$url"
uv run python -m src.cli rag-ask "$question" --recursive --filter-transcripts
uv run python -m src.cli rag-ask "$question" --recursive \
  --max-depth 1 --max-followups 4 --top-k 15 --followup-top-k 10
uv run python -m src.cli rag-ask "$question" --recursive \
  --max-total-followups 6 --novelty-min-chunks 3
uv run python -m src.cli rag-ask "$question" --recursive --print-trace
uv run python -m src.cli rag-ask "$question" --recursive --filter-transcripts \
  --url "$url" --max-followups 3 --print-trace
```

With `YT_AGENT_RAG_RECURSIVE_DEFAULT=true`, `rag-ask "$question"` runs recursively by default. Use `--no-recursive` to force the single-hop path.

Agentic RAG (`--rag_agent`): routes `rag-ask` to the agentic LangGraph RAG agent (`rag_agent`) instead of the default pipeline agent (`rag_llm`). The agent drives its own ReAct research loop: it retrieves on the original question, identifies sub-topics, and calls retrieval again per sub-topic until it judges it has enough evidence, then writes a single comprehensive answer.

```bash
uv run python -m src.cli rag-ask "$question" --rag_agent
uv run python -m src.cli rag-ask "$question" --rag_agent --url "$url" --top-k 10
uv run python -m src.cli rag-ask "$question" --rag_agent --filter-transcripts
uv run python -m src.cli rag-ask "$question" --rag_agent --max-iterations 8
```

The agent inherits `--url`, `--filter-transcripts`, and `--top-k` for every retrieval call; only the query string changes per iteration.

Agentic RAG flags (`--rag_llm` and `--rag_agent` are mutually exclusive):

- `--rag_agent` — use the agentic LangGraph RAG agent (`rag_agent`) instead of the pipeline agent (`rag_llm`).
- `--rag_llm` — use the pipeline RAG agent (`rag_llm`) explicitly. This is also the default when neither flag is passed.
- `--max-iterations N` — hard cap on ReAct loop iterations; only used with `--rag_agent`. Defaults to `YT_AGENT_RAG_AGENT_MAX_ITERATIONS` (or `10`). Ignored without `--rag_agent`.

With `--rag_agent`, output streams live to the terminal: a `Researching...` header, then one `[N] Retrieving: "<query>"  →  K chunks` line per retrieval iteration (color-cycled on a TTY, plain text when piped), followed by the standard `Answer` / `References` blocks and an `Agent: N iterations (rag_agent)` footer. The `Answer` body uses a `## Key Findings` summary followed by one `## Finding N: <title>` section per insight, each with inline citations.

Without `--rag_agent` (no flag, or `--rag_llm`), `rag-ask` behaves exactly as before; `rag_llm` is used and no footer or streaming output is printed.

Recursive RAG flags:

- `--recursive` — enable recursive multi-hop RAG; default is off unless `YT_AGENT_RAG_RECURSIVE_DEFAULT=true`.
- `--no-recursive` — disable recursive RAG even when the env default is on.
- `--max-depth N` — default `1`; S6 implements `0` and `1`, where `0` collapses to single-hop.
- `--max-followups N` — default `3`; maximum follow-up queries selected from the first pass.
- `--followup-top-k N` — default is `--top-k`; chunks retrieved for each follow-up query.
- `--novelty-min-chunks N` — default `2`; minimum new chunks required to include a follow-up in synthesis.
- `--max-total-followups N` — default `max_depth * max_followups`; hard cap on fan-out retrievals.
- `--show-followups` — print proposed follow-up queries in single-hop mode.
- `--print-trace` — print per-follow-up chunk previews in recursive mode.

Summarize one transcript:

```bash
uv run python -m src.cli summarize "$url"
```

#### 5. Compare and evaluate

Compare full-transcript prompting against single-transcript RAG in the terminal:

```bash
uv run python -m src.cli compare-context "$url" "$question" --top-k 10
```

Generate the HTML evaluation report:

```bash
uv run python -m src.evals.evaluation \
  --question "$question" \
  --output dashboard/evaluation.html \
  --json-output dashboard/evaluation.json
```

Open:

```text
dashboard/evaluation.html
```

The report runs one question across three agent setups and lays them out side by side, one column per setup. Each column is titled by the flags from the command that produced it, with the full command shown in an expandable `Command` section:

| Column | Command | Description |
|---|---|---|
| `--rag_llm --top-k 30` | `rag-ask "$question" --rag_llm --top-k 30` | Baseline `rag_llm` single-hop, wide retrieval. |
| `--rag_llm --recursive --top-k 10` | `rag-ask "$question" --rag_llm --recursive --top-k 10` | `rag_llm` with recursive multi-hop retrieval. |
| `--rag_agent --top-k 10` | `rag-ask "$question" --rag_agent --top-k 10` | Agentic `rag_agent` ReAct research loop. |

The report shows:

- The question at the top, then three answer columns underneath.
- The full bash command for each setup in an expandable `Command` block.
- Per-setup metadata: prompt token estimate, retrieved chunk count, answer length, LLM calls (single-hop/recursive), iteration count (rag_agent), and terminated reason.
- Expandable `References` with traceable timestamp links back to the source video.
- Dark theme, matching the other dashboards.

### Architecture

```text
src/
  transcripts/   # YouTube URL parsing, Supadata fetching, transcript models/storage
  rag/           # Raw segment storage, chunking, embeddings, retrieval, references, BM25,
                 #   RRF fusion, cross-encoder reranking, chunk similarity graph
  agents/        # Full-transcript agent and RAG agent with optional recursive multi-hop retrieval
  api/           # FastAPI workbench: ask/judge/index SSE, corpus, chunks, ranking,
                 #   scoreboard, chunk graph
  chat/          # Setup registry + runner, shared chat history, static chat.html viewer
  evals/         # Demo/evaluation scripts, RAGAS judge, golden set, regression runs
  dashboard/     # Local HTML dashboards for reviewing indexed RAG state
scripts/         # One-off maintenance, incl. chunk-metadata backfill
frontend/        # React 19 + TypeScript UI (Vite); dist/ is gitignored
  src/api/       # Typed endpoint client and SSE reader
  src/answers/   # Answer/citation renderer (TS port of the shared renderer)
  src/chat/      # Chat thread, grouped multi-agent bubbles, composer, score breakdowns
  src/pipeline/  # Corpus tree, chunk detail, Retrieval Lab, indexing panel, chunk graph
  src/scoreboard/# Grouped aggregates, provenance bar, efficiency panel
tests/
```

The answer renderer exists twice on purpose: `src/chat/frontend.py` holds the
JS used by the standalone `dashboard/chat.html` viewer, and
`frontend/src/answers/render.ts` is its TypeScript port used by the React app.
They must stay behaviourally identical — `frontend/src/answers/render.test.ts`
pins the parsing, citation-linking, and section rules that both implement.

Canonical storage:

- `raw_transcripts`: timestamped Supadata segment stream.
- `transcript_chunks`: embedded timestamped transcript chunks.
- `transcript_summaries`: embedded LLM transcript summaries for optional transcript-level filtering.

The legacy `transcripts` collection may exist from earlier prototype work, but current raw and RAG paths use `raw_transcripts` and `transcript_chunks`.

### Agent Architecture

There are two agent paths:

- `TranscriptAgent`: supports full raw transcript prompting and single-video RAG comparison.
- `RagTranscriptAgent`: RAG agent that can search all indexed transcript chunks, filter to one URL, and optionally run recursive multi-hop retrieval.

`RagTranscriptAgent` uses a unified first-pass LLM contract in both modes: the prompt always asks for an answer with references plus proposed subtopics and follow-up retrieval queries. Single-hop mode returns those follow-ups only when requested by `--show-followups`; recursive mode acts on them with extra retrieval and a final synthesis call.

A third path, the agentic RAG agent (`RagAgent`), is available via `rag-ask --rag_agent`.

#### rag_llm vs rag_agent

Two labels are used in the CLI, specs, and eval reports to distinguish the two `rag-ask` agent paths:

| Label | Class | File | Selected by | Behavior |
|---|---|---|---|---|
| `rag_llm` | `RagTranscriptAgent` | `src/agents/rag_transcript_agent.py` | `--rag_llm`, or no flag (default) | Single-shot pipeline: one retrieval (or bounded recursive fan-out), then an LLM answer. |
| `rag_agent` | `RagAgent` | `src/agents/rag_agent.py` | `rag-ask --rag_agent` | Agentic LangGraph ReAct loop: the LLM iteratively retrieves across sub-topics, accumulating evidence, until it decides it has enough, then writes a cited answer. |

`rag_llm` is a documentation and CLI label only — no class, file, or import path was renamed. It refers to the existing `RagTranscriptAgent` exactly as it is. `--rag_agent` selects `rag_agent` (`RagAgent`); `--rag_llm` (or no flag) keeps `rag-ask` on `rag_llm`. The two flags are mutually exclusive. Both agents accept the same question and return the same `RagTranscriptAnswer` shape, so the two approaches can be compared side-by-side.

Indexing flow:

```text
YouTube URL
  -> extract video_id
  -> Supadata transcript fetch with text=false
  -> timestamped segments
  -> raw_transcripts collection
  -> segment-aware chunking
  -> local embedding model
  -> transcript_chunks collection
```

Raw single-transcript Q&A flow:

```text
User question + URL
  -> src.cli ask --context raw
  -> TranscriptAgent
  -> RawTranscriptContextProvider
  -> raw_transcripts lookup by video_id
  -> join every segment into full transcript context
  -> DeepSeek LLM
  -> answer
```

Raw mode sends the whole transcript to the LLM. It is the quality baseline, but it uses the most prompt tokens.

Single-transcript RAG Q&A flow:

```text
User question + URL
  -> src.cli ask --context rag --top-k 10
  -> TranscriptAgent
  -> RagTranscriptContextProvider
  -> embed user question
  -> transcript_chunks vector search where video_id == URL video_id
  -> format top 10 chunks with timestamps
  -> DeepSeek LLM
  -> answer
```

Single-transcript RAG only sends the retrieved chunks to the LLM. This is the direct token-reduction comparison against `raw_single`.

All-transcript RAG Q&A flow:

```text
User question
  -> src.cli rag-ask --top-k 10
  -> RagTranscriptAgent
  -> MultiTranscriptRagContextProvider
  -> embed user question
  -> transcript_chunks vector search across all indexed videos
  -> format top 10 chunks with video URLs and timestamp links
  -> DeepSeek LLM
  -> answer with source references + proposed follow-ups
```

All-transcript RAG is the demo path for asking across the indexed corpus. It can be filtered back to one transcript with:

```bash
uv run python -m src.cli rag-ask "$question" --url "$url" --top-k 10
```

Optional transcript-summary filtered RAG flow:

```text
User question
  -> src.cli rag-ask --filter-transcripts
  -> embed user question
  -> vector search transcript_summaries
  -> keep selected transcript video IDs
  -> vector search transcript_chunks restricted to those video IDs
  -> DeepSeek LLM
  -> answer with source references
```

Recursive RAG flow:

```text
User question
  -> src.cli rag-ask --recursive
  -> retrieve initial chunks with MultiTranscriptRagContextProvider
  -> first-pass DeepSeek call: answer + references + follow-up subtopics
  -> for each selected follow-up query, retrieve more chunks through the same provider
  -> drop duplicate or low-novelty follow-up evidence
  -> final DeepSeek synthesis call
  -> layered answer + combined references + recursion trace
```

Recursive mode inherits `--url` and `--filter-transcripts` because every hop reuses the same context provider and request filters.

Evaluation flow:

```text
src.evals.evaluation
  -> run raw_single
  -> run rag_single
  -> run rag_all
  -> estimate prompt tokens from context length
  -> embed the three answers
  -> compute pairwise cosine similarity
  -> write dashboard/evaluation.html
```

The evaluation proves the demo claim when RAG answers remain similar to raw answers while using substantially fewer prompt tokens.

### Dashboard Outputs

Generated review artifacts live under:

```text
dashboard/
  evaluation.html
  evaluation.json
  chat.html             # WhatsApp-style view of interactive chat Q&A
  chat_history.json     # captured interactive chat questions and per-setup answers
  rag_pipeline.html
  chunk_space/
    projection.json     # PCA projection (chunk coords, components, mean) — committed
    question.json       # canonical question + nearest chunks — committed
```

`chat.html` and `chat_history.json` are produced by `src.cli chat` and capture
interactive questions and their per-setup answers.

`evaluation.html` compares answers for a question. `rag_pipeline.html` is a tabbed dashboard that reviews indexed transcripts, summaries, summary encodings, chunk inventory, ingestion history when run records exist, and the chunk-embedding scatter plot. The `chunk_space/` artifacts are committed so a fresh clone renders the Chunk Space tab without re-running ingestion.

### Observability

MLflow local tracking is written to:

```text
.yt-agent/mlruns
```

Each CLI command creates a run with command metadata, cache status, transcript metadata, and answer artifacts. Full transcript artifacts are disabled by default unless `YT_AGENT_LOG_TRANSCRIPT_ARTIFACTS=true`.

### Tests

```bash
uv run pytest                        # Python: pipeline, API, evals
cd frontend && npm test              # TypeScript: renderer, SSE, tree, chat UI, theme
```

External Supadata, DeepSeek/LangChain, and embedding calls are mocked in automated tests where appropriate. Frontend tests run in jsdom with no network access.

### Agent Work

Implementation specs and handoff notes live in `agent-work/`.

Generated dashboard outputs should live in `dashboard/`, not `agent-work/`.

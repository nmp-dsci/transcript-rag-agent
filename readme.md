## YouTube Transcript RAG Demo

CLI prototype that demonstrates the value of RAG over full-transcript prompting for YouTube transcript Q&A.

The main demo compares one question across three transcript input types:

- `raw_single`: full raw transcript for one video.
- `rag_single`: top 10 retrieved chunks for that same video.
- `rag_all`: top 10 retrieved chunks across all indexed videos.

The demo writes `evaluation/evaluation.html` with answers, token estimates, pairwise answer similarity, and retrieved chunks. The target outcome is similar answer quality with roughly 80%+ fewer prompt tokens for RAG.

### Setup

This project uses `uv`.

```bash
uv sync
```

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
YT_AGENT_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
YT_AGENT_RAG_TOP_K=10
YT_AGENT_CHUNK_TARGET_CHARS=1200
YT_AGENT_CHUNK_OVERLAP_CHARS=150
MLFLOW_TRACKING_URI=file:.yt-agent/mlruns
MLFLOW_EXPERIMENT_NAME=yt-agent-v1
YT_AGENT_LOG_TRANSCRIPT_ARTIFACTS=false
```

`SUPADATA_API_KEY` is used with the Supadata transcript API. DeepSeek is called through the OpenAI-compatible LangChain client.

### End-To-End Demo

Run from the project root after `uv sync` and env setup:

```bash
url="https://www.youtube.com/watch?v=3hk7nO_q0a8"
other_url="https://www.youtube.com/watch?v=Uc1yniFxg0o"
question="what does this video say  for capital gains tax, is it being grandfathered or every now under new rules, does that mean if I sell before 30 June 2027 I can still access 50% discount "

uv run python -m src.cli index-rag "$url"
uv run python -m src.cli index-rag "$other_url"

uv run python -m src.evals.evaluation \
  --url "$url" \
  --question "$question" \
  --output evaluation/evaluation.html \
  --json-output evaluation/evaluation.json
```

Open:

```text
evaluation/evaluation.html
```

The report shows:

- Raw answer from the full transcript.
- RAG answer from top 10 chunks for the selected video.
- RAG answer from top 10 chunks across all indexed videos.
- Prompt token estimates for each mode.
- Pairwise embedding similarity between answers.
- Expandable retrieved chunks with source URL and timestamp links.

Current local demo output has shown:

```text
raw_single tokens: 18295
rag_single tokens: 2997
rag_all tokens: 3188

raw_single__rag_single similarity: 0.8704
raw_single__rag_all similarity: 0.9033
rag_single__rag_all similarity: 0.9462
```

### Interactive Commands

Index any YouTube transcript for RAG:

```bash
url="https://www.youtube.com/watch?v=VIDEO_ID"
uv run python -m src.cli index-rag "$url"
```

Ask against a full raw transcript:

```bash
uv run python -m src.cli ask "$url" "$question" --context raw
```

Ask with RAG restricted to one transcript:

```bash
uv run python -m src.cli ask "$url" "$question" --context rag --top-k 10
```

Ask with the RAG-only multi-transcript agent across all indexed transcripts:

```bash
uv run python -m src.cli rag-ask "$question" --top-k 10
```

Ask with the RAG-only agent restricted to one transcript:

```bash
uv run python -m src.cli rag-ask "$question" --url "$url" --top-k 10
```

### Architecture

```text
src/
  transcripts/   # YouTube URL parsing, Supadata fetching, transcript models/storage
  rag/           # Raw segment storage, chunking, embeddings, retrieval, references
  agents/        # Full-transcript agent and RAG-only transcript agent
  evals/         # Demo/evaluation scripts and HTML report generation
tests/
```

Canonical storage:

- `raw_transcripts`: timestamped Supadata segment stream.
- `transcript_chunks`: embedded timestamped transcript chunks.

The legacy `transcripts` collection may exist from earlier prototype work, but current raw and RAG paths use `raw_transcripts` and `transcript_chunks`.

### Evaluation Outputs

Generated demo reports live under:

```text
evaluation/
  evaluation.html
  evaluation.json
```

`evaluation.html` is the primary artifact for demo review.

### Observability

MLflow local tracking is written to:

```text
.yt-agent/mlruns
```

Each CLI command creates a run with command metadata, cache status, transcript metadata, and answer artifacts. Full transcript artifacts are disabled by default unless `YT_AGENT_LOG_TRANSCRIPT_ARTIFACTS=true`.

### Tests

```bash
uv run pytest
```

External Supadata, DeepSeek/LangChain, and embedding calls are mocked in automated tests where appropriate.

### Agent Work

Implementation specs and handoff notes live in `agent-work/`.

Generated evaluation outputs should live in `evaluation/`, not `agent-work/`.

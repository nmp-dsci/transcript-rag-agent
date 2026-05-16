## YouTube Transcript RAG Demo

CLI prototype that demonstrates the value of RAG over full-transcript prompting for YouTube transcript Q&A.

The main demo compares one question across three transcript input types:

- `raw_single`: full raw transcript for one video.
- `rag_single`: top 10 retrieved chunks for that same video.
- `rag_all`: top 10 retrieved chunks across all indexed videos.

The demo writes `dashboard/evaluation.html` with answers, token estimates, pairwise answer similarity, and retrieved chunks. The target outcome is similar answer quality with roughly 80%+ fewer prompt tokens for RAG.

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
YT_AGENT_TRANSCRIPT_SUMMARY_COLLECTION=transcript_summaries
YT_AGENT_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
YT_AGENT_RAG_TOP_K=10
YT_AGENT_TRANSCRIPT_FILTER_TOP_K=5
YT_AGENT_TRANSCRIPT_FILTER_MIN_SCORE=0.25
YT_AGENT_CHUNK_TARGET_CHARS=1200
YT_AGENT_CHUNK_OVERLAP_CHARS=150
SUPADATA_TIMEOUT_SECONDS=120
SUPADATA_POLL_INTERVAL_SECONDS=2
SUPADATA_MAX_POLL_SECONDS=600
MLFLOW_TRACKING_URI=file:.yt-agent/mlruns
MLFLOW_EXPERIMENT_NAME=yt-agent-v1
YT_AGENT_LOG_TRANSCRIPT_ARTIFACTS=false
```

`SUPADATA_API_KEY` is used with the Supadata transcript API. DeepSeek is called through the OpenAI-compatible LangChain client.

Supadata can return async jobs for longer videos. `SUPADATA_MAX_POLL_SECONDS=600` lets indexing wait up to 10 minutes for those jobs before timing out.

### End-To-End Demo

Run from the project root after `uv sync` and env setup:

```bash
url="https://www.youtube.com/watch?v=3hk7nO_q0a8"
other_url="https://www.youtube.com/watch?v=Uc1yniFxg0o"
url3="https://www.youtube.com/watch?v=Q4gnTeHd1OM&t=39s"
question="what does this video say  for capital gains tax, is it being grandfathered or every now under new rules, does that mean if I sell before 30 June 2027 I can still access 50% discount "

uv run python -m src.cli index-rag "$url3"
uv run python -m src.cli index-rag "$other_url"

uv run python -m src.evals.evaluation \
  --url "$url" \
  --question "$question" \
  --output dashboard/evaluation.html \
  --json-output dashboard/evaluation.json
```

Open:

```text
dashboard/evaluation.html
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

`index-rag` stores raw transcript segments, chunk embeddings, an LLM-generated transcript summary, and a transcript-level summary embedding used for optional summary-first filtering. Regenerate the summary and summary embedding with:

```bash
uv run python -m src.cli index-rag "$url" --refresh-summary
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

Ask across indexed transcripts with transcript-summary filtering before chunk retrieval:

```bash
uv run python -m src.cli rag-ask "$question" --filter-transcripts --top-k 10
```

Ask with the RAG-only agent restricted to one transcript:

```bash
uv run python -m src.cli rag-ask "$question" --url "$url" --top-k 10
```

Render the local RAG pipeline review dashboard:

```bash
uv run python -m src.dashboard.rag_pipeline --output dashboard/rag_pipeline.html
```

### Architecture

```text
src/
  transcripts/   # YouTube URL parsing, Supadata fetching, transcript models/storage
  rag/           # Raw segment storage, chunking, embeddings, retrieval, references
  agents/        # Full-transcript agent and RAG-only transcript agent
  evals/         # Demo/evaluation scripts and HTML report generation
  dashboard/     # Local HTML dashboards for reviewing indexed RAG state
tests/
```

Canonical storage:

- `raw_transcripts`: timestamped Supadata segment stream.
- `transcript_chunks`: embedded timestamped transcript chunks.
- `transcript_summaries`: embedded LLM transcript summaries for optional transcript-level filtering.

The legacy `transcripts` collection may exist from earlier prototype work, but current raw and RAG paths use `raw_transcripts` and `transcript_chunks`.

### Agent Architecture

There are two agent paths:

- `TranscriptAgent`: supports full raw transcript prompting and single-video RAG comparison.
- `RagTranscriptAgent`: RAG-only agent that can search all indexed transcript chunks or filter to one URL.

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
  -> answer with source references
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
  rag_pipeline.html
```

`evaluation.html` compares answers for a question. `rag_pipeline.html` reviews indexed transcripts, transcript summaries, summary encodings, and chunk inventory.

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

Generated dashboard outputs should live in `dashboard/`, not `agent-work/`.

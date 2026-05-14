# Spec: S3 Multi-Transcript RAG Agent

Status: ready
Date: 2026-05-14

## Summary

Evolve the transcript agent from S2 raw-vs-RAG comparison into a RAG-only transcript agent that can answer questions over all indexed YouTube transcript chunks by default.

The S3 agent should use only `transcript_chunks` retrieval for Q&A. It should retrieve top-k chunks across all indexed transcripts unless the user provides a URL filter. When a URL filter is provided, retrieval must be limited to chunks from that single transcript.

Answers must include evidence references that let the user inspect the source video:

- YouTube URL.
- Timestamp or timestamp range.
- Short reference label that maps back to a retrieved chunk.

The S3 evaluation should run the required CGT question in three contexts:

1. Single URL A only.
2. Single URL B only.
3. All indexed transcripts.

The evaluation output must be an HTML diagnostics file showing the question, answers, similarity/token metrics, and top 10 retrieved chunks for each run in expandable sections.

## Current Source Of Truth

Build on the current S2 implementation:

- `src/rag/storage.py`
  - `RawTranscriptStore`
  - `TranscriptChunkStore`
  - Chroma collections `raw_transcripts` and `transcript_chunks`
- `src/rag/context.py`
  - `RagTranscriptContextProvider`
  - Timestamped retrieved chunk formatting
- `src/rag/indexing.py`
  - `RagIndexer`
- `src/rag/eval.py`
  - `compare_answers`
  - `estimate_tokens`
- `src/agents/transcript_agent.py`
  - Existing S2 raw/RAG comparison agent using `TranscriptContext.context_text`
- `src/cli.py`
  - Existing `index-rag`, `ask --context rag`, and `compare-context`
- `src/evals/s2_context_eval.py`
  - Existing JSON evaluation pattern for one question

S2 retrieval is still scoped to one video because `TranscriptChunkStore.query(...)` filters by `video_id`. S3 changes this by adding all-transcript retrieval.

## Goals

- Add a RAG-only agent path for Q&A across indexed transcript chunks.
- Retrieve from all indexed transcript chunks by default.
- Allow an optional single-URL filter to restrict retrieval to one transcript.
- Preserve timestamped evidence and video URL references in the LLM context.
- Require answers to cite evidence references from retrieved chunks.
- Add an HTML evaluation report that makes answer quality diagnosable.
- Preserve S2 raw-vs-RAG commands and `src/agents/transcript_agent.py` behavior.

## Non-Goals

- Do not implement raw full-transcript prompting in the S3 agent.
- Do not overwrite, rename, or replace `src/agents/transcript_agent.py`.
- Do not route existing `ask`, `ask --context raw`, `ask --context rag`, or `compare-context` through the new S3 agent.
- Do not add a web app.
- Do not add a new database or external vector service.
- Do not implement multi-turn chat memory.
- Do not summarize multiple transcripts in S3.
- Do not require the user to pass all URLs at query time. S3 should use already indexed chunks by default.

## User Workflows

Index each video once:

```bash
url_a="https://www.youtube.com/watch?v=FIRST_VIDEO_ID"
url_b="https://www.youtube.com/watch?v=SECOND_VIDEO_ID"

uv run python -m src.cli index-rag "$url_a"
uv run python -m src.cli index-rag "$url_b"
```

Ask across all indexed transcripts:

```bash
question="what does this video say  for capital gains tax, is it being grandfathered or every now under new rules, does that mean if I sell before 30 June 2027 I can still access 50% discount "

uv run python -m src.cli rag-ask "$question" --top-k 10
```

Ask against one transcript only:

```bash
uv run python -m src.cli rag-ask "$question" --url "$url_a" --top-k 10
```

Run the S3 diagnostics evaluation:

```bash
uv run python -m src.evals.s3_rag_agent_eval \
  --url "$url_a" \
  --url "$url_b" \
  --output agent-work/s3_rag_agent_eval.html
```

## Proposed Project Structure

Add:

```text
src/
  agents/
    rag_transcript_agent.py      # New RAG-only multi-transcript agent for transcript Q&A
  evals/
    s3_rag_agent_eval.py         # HTML diagnostics evaluation
  rag/
    references.py                # Video URL + timestamp reference formatting helpers

tests/
  agents/
    test_rag_transcript_agent.py
  evals/
    test_s3_rag_agent_eval.py
  rag/
    test_references.py
```

Update:

```text
src/rag/storage.py               # Add all-transcript query and optional URL/video filter
src/rag/context.py               # Add multi-transcript RAG context provider or extend existing provider
src/agents/prompts.py            # Add RAG-only citation instructions
src/cli.py                       # Add rag-ask command
readme.md                        # Add S3 multi-transcript RAG commands and eval command
```

Do not update:

```text
src/agents/transcript_agent.py   # Preserve existing S2 agent and raw-vs-RAG comparison behavior
```

## Data And Retrieval Behavior

S3 uses the existing `transcript_chunks` collection.

Each chunk already includes scalar metadata:

- `transcript_id`
- `video_id`
- `source_url`
- `source_collection`
- `chunk_index`
- `start_seconds`
- `end_seconds`
- `start_segment_index`
- `end_segment_index`
- `segment_count`

Add retrieval methods:

```python
class TranscriptChunkStore:
    def query_all(
        self,
        query: str,
        top_k: int,
    ) -> list[RetrievedChunk]:
        ...

    def query_by_url(
        self,
        source_url: str,
        query: str,
        top_k: int,
    ) -> list[RetrievedChunk]:
        ...

    def query_by_video_id(
        self,
        video_id: str,
        query: str,
        top_k: int,
    ) -> list[RetrievedChunk]:
        ...
```

Rules:

- `query_all` must not apply a `video_id` filter.
- `query_by_url` should derive `video_id` using `extract_video_id(source_url)` and filter by that `video_id`.
- Existing S2 `query(video_id, query, top_k)` can remain as an alias for `query_by_video_id`.
- Retrieval must return chunk metadata needed for citations.
- Retrieval must be deterministic enough for tests by using fake embeddings.

## Reference Formatting

Add `src/rag/references.py`.

Required helpers:

```python
def youtube_timestamp_url(source_url: str, seconds: float | None) -> str:
    ...

def format_chunk_reference(index: int, chunk: RetrievedChunk) -> str:
    ...
```

Reference format in context:

```text
[1] video=3hk7nO_q0a8 time=09:53-11:05 url=https://www.youtube.com/watch?v=3hk7nO_q0a8&t=593s
chunk text...
```

Rules:

- Use `start_seconds` for timestamp URL `t=` parameter.
- If timestamp is missing, use the base `source_url`.
- For seconds under one hour, display `MM:SS`.
- For seconds at least one hour, display `HH:MM:SS`.
- The answer should cite references like `[1]`, `[2]`.
- The answer should include enough URL/time detail for a user to inspect the video.

## Agent Design

Add `src/agents/rag_transcript_agent.py`. This is a new agent, not a replacement for `src/agents/transcript_agent.py`.

Naming rule:

- Keep the existing `TranscriptAgent` class and file name unchanged.
- Name the new class `RagTranscriptAgent`.
- Use the new agent only from the new `rag-ask` CLI command and S3 evaluation.
- Existing S2 commands continue to use `TranscriptAgent`.

Recommended shape:

```python
class RagTranscriptAgent:
    def __init__(
        self,
        llm: ChatModel,
        context_provider: MultiTranscriptRagContextProvider,
        max_context_chars: int = 40_000,
    ) -> None:
        ...

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        context_provider: MultiTranscriptRagContextProvider | None = None,
    ) -> "RagTranscriptAgent":
        ...

    def answer(self, request: RagQuestionRequest) -> RagTranscriptAnswer:
        ...
```

Add models either in `src/agents/models.py` or a new module:

```python
class RagQuestionRequest(BaseModel):
    question: str
    source_url: HttpUrl | None = None
    top_k: int = 10


class RagAnswerReference(BaseModel):
    label: str
    source_url: HttpUrl
    timestamp_url: HttpUrl
    start_seconds: float | None = None
    end_seconds: float | None = None
    chunk_index: int
    video_id: str


class RagTranscriptAnswer(BaseModel):
    question: str
    answer: str
    references: list[RagAnswerReference] = Field(default_factory=list)
```

Provider behavior:

```python
class MultiTranscriptRagContextProvider:
    def get_context(
        self,
        question: str,
        source_url: str | None = None,
        top_k: int = 10,
    ) -> TranscriptContext:
        ...
```

Rules:

- If `source_url` is `None`, retrieve across all chunks.
- If `source_url` is provided, retrieve only chunks for that video.
- If no chunks are found for a provided URL, auto-index that URL before retrying retrieval.
- If no chunks are found for all-transcript mode, return a clear error telling the user to run `index-rag` for at least one URL.
- `TranscriptContext.context_text` must contain formatted chunk references, URL, timestamp URL, and chunk text.
- `TranscriptContext.retrieved_chunks` must contain the exact chunks sent to the model.

Prompt requirements:

- The system prompt must say the agent uses only retrieved transcript chunks.
- The answer must cite relevant chunk labels like `[1]`.
- The answer must include a source reference section with URL/time links when possible.
- If retrieved context is insufficient, the answer must say so.
- Do not cite chunks that do not support the answer.

The agent should parse a JSON response from the LLM, consistent with the existing `TranscriptAgent` pattern.

Expected LLM response shape:

```json
{
  "question": "user question",
  "answer": "answer with inline citations like [1]",
  "references": [
    {
      "label": "[1]",
      "source_url": "https://www.youtube.com/watch?v=...",
      "timestamp_url": "https://www.youtube.com/watch?v=...&t=593s",
      "start_seconds": 593.36,
      "end_seconds": 665.44,
      "chunk_index": 10,
      "video_id": "..."
    }
  ]
}
```

Implementation note:

- The model may omit or mis-shape references. The agent should be defensive:
  - Validate JSON.
  - If references are missing, populate references from retrieved chunks that are cited inline.
  - If inline citations are missing but the answer uses retrieved context, include retrieved chunk references in a separate `references` list.

## CLI Interface

Add:

```bash
uv run python -m src.cli rag-ask "question" --top-k 10
uv run python -m src.cli rag-ask "question" --url "https://www.youtube.com/watch?v=..." --top-k 10
```

Behavior:

- No `--url`: search all indexed transcript chunks.
- With `--url`: search only that transcript's chunks.
- Print answer first.
- Then print references:

```text
References
[1] https://www.youtube.com/watch?v=3hk7nO_q0a8&t=593s 09:53-11:05
[2] https://www.youtube.com/watch?v=...&t=...
```

Error behavior:

- If all-transcript mode has zero indexed chunks, fail with:

```text
No indexed transcript chunks found. Run index-rag for one or more YouTube URLs first.
```

- If URL-filter mode has zero chunks and auto-index fails, include the failed URL and stage.

## S3 Evaluation

Add `src/evals/s3_rag_agent_eval.py`.

Required default question:

```text
what does this video say  for capital gains tax, is it being grandfathered or every now under new rules, does that mean if I sell before 30 June 2027 I can still access 50% discount 
```

Required command:

```bash
uv run python -m src.evals.s3_rag_agent_eval \
  --url "$url_a" \
  --url "$url_b" \
  --output agent-work/s3_rag_agent_eval.html
```

Evaluation runs:

1. `url_a_only`
   - `source_url=url_a`
   - `top_k=10`
2. `url_b_only`
   - `source_url=url_b`
   - `top_k=10`
3. `all_indexed`
   - `source_url=None`
   - `top_k=10`

For each run collect:

- Question.
- Answer.
- Prompt context token estimate.
- Retrieved chunk count.
- Top 10 retrieved chunks.
- For each chunk:
  - rank
  - score
  - video id
  - source URL
  - timestamp URL
  - start/end seconds
  - chunk index
  - chunk text

Comparison metrics:

- Embedding cosine similarity between `url_a_only` answer and `all_indexed` answer.
- Embedding cosine similarity between `url_b_only` answer and `all_indexed` answer.
- Pairwise similarity between `url_a_only` and `url_b_only`.
- Token estimate for each run.

HTML output requirements:

- Single self-contained HTML file.
- No external JS/CSS dependencies.
- Show the question at the top.
- Show a summary table:
  - run name
  - filter
  - answer length
  - token estimate
  - retrieved chunk count
  - similarity to all-indexed answer where applicable
- Show each run's answer in a clearly separated section.
- Show references below each answer.
- Show top 10 chunks in expandable `<details>` sections.
- Chunk details must include URL and timestamp link.
- Include a diagnostics section with pairwise similarity metrics.

HTML skeleton:

```html
<h1>S3 RAG Agent Evaluation</h1>
<section id="question">...</section>
<section id="summary-table">...</section>
<section id="answers">
  <article>
    <h2>url_a_only</h2>
    <pre>answer...</pre>
    <h3>References</h3>
    ...
    <h3>Retrieved chunks</h3>
    <details>
      <summary>[1] video_id 09:53-11:05 score=...</summary>
      <a href="...">Open video at timestamp</a>
      <pre>chunk text...</pre>
    </details>
  </article>
</section>
```

Also write an optional JSON sidecar when `--json-output` is provided:

```bash
uv run python -m src.evals.s3_rag_agent_eval \
  --url "$url_a" \
  --url "$url_b" \
  --output agent-work/s3_rag_agent_eval.html \
  --json-output agent-work/s3_rag_agent_eval.json
```

## Testing Requirements

Add focused tests:

- `TranscriptChunkStore.query_all` searches without `video_id` filter.
- `TranscriptChunkStore.query_by_url` filters to one video's chunks.
- URL timestamp helper appends `t=<seconds>s`.
- Missing timestamp helper returns base URL.
- Multi-transcript context provider retrieves across all chunks when no URL is supplied.
- Multi-transcript context provider retrieves only one video's chunks when URL is supplied.
- URL-filter mode auto-indexes when chunks are missing.
- All-transcript mode errors clearly when no chunks exist.
- RAG context includes chunk labels, source URLs, timestamp URLs, and chunk text.
- `RagTranscriptAgent.answer` returns answer and references.
- `rag-ask` CLI calls all-transcript retrieval when no `--url` is provided.
- `rag-ask --url` calls single-transcript retrieval.
- S3 eval creates an HTML file containing:
  - question
  - all three run labels
  - answers
  - `<details>` sections
  - retrieved chunk URLs
  - similarity metrics

External calls must be mocked in tests:

- Supadata transcript fetching.
- DeepSeek/LangChain calls.
- Embedding model calls.

Use deterministic fake embeddings for retrieval and similarity tests.

## Manual Verification

From project root:

```bash
uv sync

url_a="https://www.youtube.com/watch?v=FIRST_VIDEO_ID"
url_b="https://www.youtube.com/watch?v=SECOND_VIDEO_ID"
question="what does this video say  for capital gains tax, is it being grandfathered or every now under new rules, does that mean if I sell before 30 June 2027 I can still access 50% discount "

uv run python -m src.cli index-rag "$url_a"
uv run python -m src.cli index-rag "$url_b"

uv run python -m src.cli rag-ask "$question" --url "$url_a" --top-k 10
uv run python -m src.cli rag-ask "$question" --url "$url_b" --top-k 10
uv run python -m src.cli rag-ask "$question" --top-k 10

uv run python -m src.evals.s3_rag_agent_eval \
  --url "$url_a" \
  --url "$url_b" \
  --output agent-work/s3_rag_agent_eval.html

uv run pytest
```

Open:

```text
agent-work/s3_rag_agent_eval.html
```

Review:

- Does each single-URL answer cite only that URL?
- Does the all-indexed answer cite the most relevant chunks from either or both URLs?
- Are timestamp links usable?
- Are the top 10 chunks visible in expandable sections?
- Do the retrieved chunks explain the answer quality?

## Acceptance Criteria

- S3 has a RAG-only Q&A agent path.
- S3 adds `src/agents/rag_transcript_agent.py` and does not overwrite or rename `src/agents/transcript_agent.py`.
- Existing S2 `ask`, `ask --context raw`, `ask --context rag`, and `compare-context` commands continue to work through `TranscriptAgent`.
- Default S3 retrieval searches all indexed transcript chunks.
- Optional `--url` filter restricts retrieval to one transcript.
- Answers include references with URL and timestamp links.
- `rag-ask` works for all-indexed mode and single-URL mode.
- Existing `index-rag` remains the indexing path for each video.
- If a filtered URL has no chunks, the system auto-indexes that URL before answering.
- If all-indexed mode has no chunks, the system returns a clear indexing error.
- S3 eval runs the required CGT question for URL A, URL B, and all indexed transcripts.
- S3 eval writes a self-contained HTML diagnostics report.
- HTML report includes question, answers, token estimates, similarity metrics, references, and expandable top 10 chunks for each run.
- Tests pass with external calls mocked.

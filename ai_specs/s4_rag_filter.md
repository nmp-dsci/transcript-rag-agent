# Spec: S4 Transcript Summary Filter For RAG

Status: ready
Date: 2026-05-16

## Summary

Add an optional transcript-level filtering stage before chunk retrieval for the RAG-only agent.

Today, `rag-ask` without a URL searches all chunks in `transcript_chunks`. That behavior must remain the default for backward compatibility.

S4 adds a transcript summary layer:

1. Each unique transcript in `raw_transcripts` gets an LLM-generated summary focused on key topics.
2. That summary is embedded and the encoded array is stored on the transcript-level raw record.
3. The same summary embedding is also upserted into `transcript_summaries` for vector retrieval.
4. When the user opts into transcript filtering, the agent embeds the question, compares it to transcript summary embeddings, keeps only relevant transcripts, then runs chunk retrieval only over chunks from those transcripts.

The final test compares the same question with:

- no URL and no transcript filtering: current all-chunk retrieval behavior.
- no URL and transcript filtering enabled: transcript-summary filtered retrieval.

Compare answer text, answer similarity, token consumption, selected transcripts, retrieved chunks, and time taken.

## Current Source Of Truth

Build on the current implementation:

- `src/rag/models.py`
  - `RawTranscriptDocument`
  - `RawTranscriptSegment`
  - `TranscriptChunk`
  - `RetrievedChunk`
- `src/rag/storage.py`
  - `RawTranscriptStore`
  - `TranscriptChunkStore`
  - `raw_transcripts`
  - `transcript_chunks`
- `src/rag/context.py`
  - `MultiTranscriptRagContextProvider`
  - all-transcript and URL-filtered RAG context
- `src/agents/rag_transcript_agent.py`
  - RAG-only multi-transcript agent
- `src/cli.py`
  - `index-rag`
  - `rag-ask`
- `src/evals/evaluation.py`
  - canonical raw/RAG demo report

Important current behavior:

- `rag-ask "$question"` searches all indexed chunks.
- `rag-ask "$question" --url "$url"` searches only that URL's chunks.
- Existing behavior must remain available.

## Goals

- Add `summary` to the transcript-level raw data model.
- Add the encoded summary array to the transcript-level raw data model for audit/debug visibility.
- Add transcript-level summary embeddings for each unique URL/video.
- Use summary embeddings to optionally filter transcripts before chunk retrieval.
- Add CLI flags so users can opt into transcript filtering.
- Preserve current all-chunk RAG behavior when filtering is not selected.
- Add an evaluation that compares filtered vs unfiltered RAG retrieval for the same no-URL question.
- Report answer, similarity, token consumption, selected transcripts, retrieved chunks, and time taken.

## Non-Goals

- Do not remove or replace existing chunk retrieval.
- Do not make transcript filtering the default in S4.
- Do not overwrite `src/agents/transcript_agent.py`.
- Do not require a new database or external vector service.
- Do not implement a web UI.
- Do not require summary generation during every query. Summaries should be generated/indexed ahead of retrieval or lazily only when missing.
- Do not change raw transcript segment storage into relational tables. Continue using Chroma collections.

## Proposed Data Model

Update `RawTranscriptDocument` in `src/rag/models.py`:

```python
class RawTranscriptDocument(BaseModel):
    transcript_id: str
    video_id: str
    source_url: HttpUrl
    provider: str = "supadata"
    title: str | None = None
    language: str | None = None
    segments: list[RawTranscriptSegment] = Field(default_factory=list)
    fetched_at: str
    source_collection: str = "raw_transcripts"
    summary: str | None = None
    summary_model: str | None = None
    summary_generated_at: str | None = None
    summary_embedding: list[float] | None = None
    summary_embedding_model: str | None = None
    summary_embedded_at: str | None = None
```

Rules:

- `summary` is an LLM-generated description of the transcript's key topics.
- `summary` should be concise but specific enough for retrieval filtering.
- `summary_model` records the LLM model used.
- `summary_generated_at` records generation time.
- `summary_embedding` records the encoded array generated from `summary` for RAG filtering.
- `summary_embedding_model` records the embedding model used.
- `summary_embedded_at` records embedding generation time.
- The canonical transcript ID remains `raw_transcript:{video_id}`.
- Uniqueness remains by `video_id`, not full URL with timestamp parameters.
- `summary_embedding` is duplicated from the retrieval collection intentionally so the raw transcript record captures the exact encoded array used for filtering.

## Transcript Summary Embedding Storage

Add a new Chroma collection:

```text
transcript_summaries
```

Purpose:

- Store one embedded summary document per transcript.
- Enable fast transcript-level filtering before chunk retrieval.
- Mirror the `raw_transcripts.summary_embedding` value in Chroma's native vector index.

Store each transcript summary as:

- `id`: `summary:{video_id}`
- `document`: summary text
- `embedding`: embedding of summary text, using the same encoded array stored in `raw_transcripts.summary_embedding`
- metadata:
  - `transcript_id`
  - `video_id`
  - `source_url`
  - `provider`
  - `title`
  - `language`
  - `summary_model`
  - `summary_generated_at`
  - `summary_embedding_model`
  - `summary_embedded_at`
  - `segment_count`
  - `chunk_count` when available

Add config:

```text
YT_AGENT_TRANSCRIPT_SUMMARY_COLLECTION=transcript_summaries
YT_AGENT_TRANSCRIPT_FILTER_TOP_K=5
YT_AGENT_TRANSCRIPT_FILTER_MIN_SCORE=0.25
```

Use existing embedding model config:

```text
YT_AGENT_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

## Summary Generation

Add a transcript summary service, for example:

```text
src/rag/summaries.py
```

Recommended classes:

```python
class TranscriptSummaryGenerator:
    def __init__(self, llm: ChatModel, model_name: str) -> None:
        ...

    def summarize(self, raw_document: RawTranscriptDocument) -> TranscriptSummaryRecord:
        ...


class TranscriptSummaryStore:
    def upsert_summary(record: TranscriptSummaryRecord) -> None:
        ...

    def get_summary(video_id: str) -> TranscriptSummaryRecord | None:
        ...

    def ensure_summary(raw_document: RawTranscriptDocument, refresh: bool = False) -> TranscriptSummaryRecord:
        ...

    def query_relevant_transcripts(
        question: str,
        top_k: int,
        min_score: float,
    ) -> list[RetrievedTranscriptSummary]:
        ...
```

Add models:

```python
class TranscriptSummaryRecord(BaseModel):
    transcript_id: str
    video_id: str
    source_url: HttpUrl
    summary: str
    summary_model: str
    summary_generated_at: str
    summary_embedding: list[float]
    summary_embedding_model: str
    summary_embedded_at: str
    title: str | None = None
    language: str | None = None
    segment_count: int = 0
    chunk_count: int | None = None


class RetrievedTranscriptSummary(TranscriptSummaryRecord):
    score: float | None = None
```

Summary system prompt requirements:

- Use only the transcript text.
- Produce a concise summary of the transcript's key topics.
- Include domain-specific terms, entities, policy names, dates, and claims that would help route future questions.
- Avoid generic filler.
- Return JSON:

```json
{
  "summary": "key topics and claims..."
}
```

The implementation should pass these instructions as the LLM system prompt, with the raw transcript text supplied as user/input content. The summary will be embedded and used as the transcript-level document for RAG filtering, so it should optimize for routing questions to the right transcript rather than for final-answer prose.

Implementation notes:

- The summary input may use the full joined raw transcript.
- Respect existing transcript-size guardrails or add a summarization max char limit if needed.
- If the transcript is too long for a direct summary call, use a simple deterministic fallback:
  - summarize first N characters and last N characters in S4, or
  - produce a summary from transcript chunk texts.
- Keep this boring and demo-oriented.

## Indexing Behavior

Update `index-rag`:

```bash
uv run python -m src.cli index-rag "$url"
```

Behavior:

1. Ensure raw transcript exists in `raw_transcripts`.
2. Build/upsert chunks in `transcript_chunks`.
3. Ensure transcript summary exists in `raw_transcripts.summary`.
4. Embed the summary and persist the encoded array to `raw_transcripts.summary_embedding`.
5. Upsert the same summary text and encoded array to `transcript_summaries`.
6. Print chunk count and whether summary/summary embedding was created/refreshed.

Add optional refresh flag:

```bash
uv run python -m src.cli index-rag "$url" --refresh-summary
```

Rules:

- Do not regenerate summaries on every index unless missing or `--refresh-summary` is set.
- Do not regenerate summary embeddings on every index unless missing, the summary changed, the embedding model changed, or `--refresh-summary` is set.
- If summary generation fails, fail indexing clearly with URL and stage.
- Do not create summary embeddings from an empty summary.
- Do not let `raw_transcripts.summary_embedding` and `transcript_summaries.embedding` diverge for the same summary/model.

## Retrieval Behavior

Current all-chunk behavior must remain:

```bash
uv run python -m src.cli rag-ask "$question" --top-k 10
```

This continues to query all chunks in `transcript_chunks`.

New filtered behavior:

```bash
uv run python -m src.cli rag-ask "$question" --filter-transcripts --top-k 10
```

Flow:

```text
question
  -> embed question
  -> query transcript_summaries
  -> keep summaries with score >= min_score, up to transcript_filter_top_k
  -> get selected video_ids
  -> query transcript_chunks restricted to selected video_ids
  -> format retrieved chunks
  -> RagTranscriptAgent answer
```

Add flags:

```bash
--filter-transcripts
--transcript-filter-top-k 5
--transcript-filter-min-score 0.25
```

Rules:

- If `--url` is provided, URL filter takes precedence and transcript filtering is not used.
- If no `--url` and `--filter-transcripts` is not set, preserve current all-chunk behavior.
- If transcript filtering returns zero transcripts, return a clear error:

```text
No transcript summaries matched the question. Try lowering --transcript-filter-min-score or run without --filter-transcripts.
```

- If summaries are missing for indexed transcripts, the agent should either:
  - tell the user to run `index-rag` for those URLs, or
  - lazily create missing summaries only when raw transcripts are available.

For S4, prefer indexing-time summary creation to keep query behavior predictable.

## Storage API Changes

Update `TranscriptChunkStore` to support restricting retrieval to multiple video IDs:

```python
def query_by_video_ids(
    self,
    video_ids: list[str],
    query: str,
    top_k: int,
) -> list[RetrievedChunk]:
    ...
```

Chroma filter implementation:

- Use an `$in` filter if supported by the installed Chroma version.
- If `$in` is not supported or awkward, query each selected `video_id` separately with `top_k`, merge by score, and return global top-k.
- Tests should not depend on Chroma internals.

Add transcript summary store:

```python
class TranscriptSummaryStore:
    collection_name = "transcript_summaries"
    ...
```

## Agent Context Changes

Update `MultiTranscriptRagContextProvider.get_context`:

```python
def get_context(
    self,
    question: str,
    source_url: str | None = None,
    top_k: int = 10,
    filter_transcripts: bool = False,
    transcript_filter_top_k: int = 5,
    transcript_filter_min_score: float = 0.25,
) -> TranscriptContext:
    ...
```

Add context metadata if needed:

```python
TranscriptContext(
    ...
    selected_transcripts=[RetrievedTranscriptSummary(...)]
)
```

If changing `TranscriptContext` is too broad, define a sidecar attribute/model in the new RAG-only path. Do not break S2 tests.

Context text should include selected transcript summary diagnostics before chunks only if useful:

```text
Selected transcripts:
- video=... score=... url=... summary=...

Retrieved chunks:
[1] video=... time=... url=...
chunk text...
```

For the LLM context, retrieved chunks remain the primary evidence. Transcript summaries are for filtering and diagnostics, not final answer evidence, unless explicitly included and cited.

## CLI Interface

Update `rag-ask`:

```bash
uv run python -m src.cli rag-ask "$question" --top-k 10
uv run python -m src.cli rag-ask "$question" --filter-transcripts --top-k 10
uv run python -m src.cli rag-ask "$question" --filter-transcripts --transcript-filter-top-k 3 --transcript-filter-min-score 0.3 --top-k 10
uv run python -m src.cli rag-ask "$question" --url "$url" --top-k 10
```

Output should include selected transcript diagnostics when filtering is used:

```text
Selected transcripts
1. score=0.72 video=... url=...
2. score=0.51 video=... url=...

Answer
...

References
...
```

Update `index-rag` output:

```text
RAG index updated
Raw transcript collection: raw_transcripts
Chunk collection: transcript_chunks
Transcript summary collection: transcript_summaries
Chunks: 100
Summary: created
Chroma path: ...
```

## MLflow Logging

Extend the existing MLflow observability so S4 runs capture the transcript filtering decision and the final RAG evidence.

Current code already logs retrieved chunks through `log_context_details(..., retrieved_chunks=...)` as `rag_chunks.json`. S4 should keep that behavior and add summary-filter-specific logging.

Add an observability helper, for example:

```python
def log_transcript_filter_details(
    enabled: bool,
    selected_transcripts: list[RetrievedTranscriptSummary] | None = None,
    filter_top_k: int | None = None,
    min_score: float | None = None,
    retrieved_chunks: list[RetrievedChunk] | None = None,
) -> None:
    ...
```

When `rag-ask --filter-transcripts` is used, log:

- MLflow params:
  - `transcript_filter_enabled=true`
  - `transcript_filter_top_k`
  - `transcript_filter_min_score`
- MLflow metrics:
  - `selected_transcript_count`
  - `retrieved_chunk_count`
  - `selected_transcript_score_max`, when available
  - `selected_transcript_score_min`, when available
- MLflow tags:
  - `selected_video_ids`, comma-separated
  - `selected_transcript_ids`, comma-separated
  - `retrieved_chunk_ids`, comma-separated, preserving current chunk logging behavior
- MLflow artifacts:
  - `transcript_filter.json`, containing selected transcript summaries, scores, `video_id`, `source_url`, `summary_model`, `summary_embedding_model`, and filter config.
  - `rag_chunks.json`, containing the final chunks passed to the LLM after transcript filtering.

When transcript filtering is not enabled, log `transcript_filter_enabled=false` and continue logging retrieved chunks through the existing RAG context logging path.

For `src/evals/s4_rag_filter_eval.py`, wrap the evaluation in an MLflow run and log:

- params:
  - `evaluation=s4_rag_filter`
  - `question`
  - `top_k`
  - filter config for the filtered run
- metrics:
  - unfiltered and filtered prompt token estimates
  - unfiltered and filtered time seconds
  - selected transcript count
  - retrieved chunk counts
  - answer similarity
  - token delta and percent change
  - time delta and percent change
- artifacts:
  - generated HTML report as `s4_rag_filter.html`
  - generated JSON report as `s4_rag_filter.json`
  - selected transcript summaries as `s4_selected_transcripts.json`
  - retrieved chunks for each run as `s4_rag_all_unfiltered_chunks.json` and `s4_rag_all_filtered_chunks.json`

MLflow logging must not change RAG behavior. If MLflow artifact logging fails, fail the run clearly during eval, but CLI answer generation should not lose the answer after the LLM has completed.

## Evaluation

Add:

```text
src/evals/s4_rag_filter_eval.py
```

Required default question:

```text
what does this video say  for capital gains tax, is it being grandfathered or every now under new rules, does that mean if I sell before 30 June 2027 I can still access 50% discount 
```

Required command:

```bash
uv run python -m src.evals.s4_rag_filter_eval \
  --question "$question" \
  --output dashboard/s4_rag_filter.html \
  --json-output dashboard/s4_rag_filter.json
```

Evaluation runs:

1. `rag_all_unfiltered`
   - no URL
   - no transcript filtering
   - current all-chunk retrieval behavior
2. `rag_all_filtered`
   - no URL
   - `filter_transcripts=True`
   - transcript-summary filter before chunk retrieval

Capture for each run:

- Answer.
- Prompt token estimate.
- Retrieved chunks.
- Selected transcript summaries, if any.
- Time taken in seconds.

Comparison metrics:

- Embedding cosine similarity between answers.
- Token difference and percent change.
- Time difference and percent change.
- Overlap of retrieved chunk video IDs.
- Selected transcript IDs for filtered run.

HTML output requirements:

- Single self-contained HTML file.
- Show the question at the top.
- Summary table:
  - run name
  - transcript filter enabled
  - selected transcript count
  - retrieved chunk count
  - prompt token estimate
  - time seconds
- Comparison table:
  - answer similarity
  - token delta
  - time delta
- Answer sections for both runs.
- Expandable selected transcript summaries for filtered run.
- Expandable top retrieved chunks for both runs.
- Chunk details include video ID, source URL, timestamp URL, score, chunk index, and text.

## Testing Requirements

Add tests for:

- `RawTranscriptDocument` supports summary fields.
- `RawTranscriptDocument` supports summary embedding fields.
- `RawTranscriptStore` persists and loads summary and summary embedding fields.
- `TranscriptSummaryStore` upserts one summary per video.
- `TranscriptSummaryStore.query_relevant_transcripts` returns top-k with scores.
- `index-rag` creates summary and summary embedding when missing.
- `index-rag` stores the same encoded summary array in `raw_transcripts.summary_embedding` and `transcript_summaries.embedding`.
- `index-rag --refresh-summary` regenerates summary and summary embedding.
- `TranscriptChunkStore.query_by_video_ids` restricts chunks to selected videos.
- `MultiTranscriptRagContextProvider` preserves current all-chunk behavior when `filter_transcripts=False`.
- `MultiTranscriptRagContextProvider` filters transcripts before chunk retrieval when `filter_transcripts=True`.
- `rag-ask --filter-transcripts` passes filtering options to provider.
- `rag-ask --url` ignores transcript filtering and uses URL filtering.
- MLflow logging records transcript filter params, selected transcript metrics, selected transcript artifacts, and retrieved chunk artifacts.
- MLflow logging records `transcript_filter_enabled=false` for unfiltered RAG runs.
- `s4_rag_filter_eval` writes HTML with:
  - question
  - filtered and unfiltered answers
  - selected transcript summaries
  - retrieved chunks
  - token estimates
  - similarity score
  - time taken
- `s4_rag_filter_eval` logs the HTML report, JSON report, selected transcript summaries, retrieved chunks, token metrics, timing metrics, and answer similarity to MLflow.

Mock external calls:

- Supadata fetching.
- DeepSeek/LangChain summary generation and answer generation.
- Embeddings.

Use deterministic fake embeddings for filtering tests.

## Manual Verification

From project root:

```bash
uv sync

url="https://www.youtube.com/watch?v=3hk7nO_q0a8"
other_url="https://www.youtube.com/watch?v=Uc1yniFxg0o"
url3="https://www.youtube.com/watch?v=Q4gnTeHd1OM&t=39s"
question="what does this video say  for capital gains tax, is it being grandfathered or every now under new rules, does that mean if I sell before 30 June 2027 I can still access 50% discount "

uv run python -m src.cli index-rag "$url"
uv run python -m src.cli index-rag "$other_url"
uv run python -m src.cli index-rag "$url3"

uv run python -m src.cli rag-ask "$question" --top-k 10
uv run python -m src.cli rag-ask "$question" --filter-transcripts --top-k 10

uv run python -m src.evals.s4_rag_filter_eval \
  --question "$question" \
  --output dashboard/s4_rag_filter.html \
  --json-output dashboard/s4_rag_filter.json

uv run pytest
```

Open:

```text
dashboard/s4_rag_filter.html
```

Review:

- Which transcripts were selected by the summary filter?
- Did filtering remove irrelevant transcripts?
- Did answer similarity remain acceptable?
- Did prompt token consumption change?
- Did runtime improve or worsen?
- Did retrieved chunks still support the answer?

## Acceptance Criteria

- `raw_transcripts` transcript-level model includes summary fields.
- `raw_transcripts` transcript-level model records the encoded summary array used for transcript filtering.
- Transcript summaries are LLM-generated and persisted.
- Transcript summary generation uses a system prompt that tells the LLM to summarize the raw transcript for RAG filtering.
- Transcript summary embeddings are stored in both `raw_transcripts.summary_embedding` and `transcript_summaries`.
- `index-rag` ensures chunk embeddings, raw transcript summary embeddings, and transcript summary retrieval embeddings exist.
- `rag-ask` remains backward compatible by default.
- `rag-ask --filter-transcripts` filters transcript candidates before chunk retrieval.
- `rag-ask --url` continues to restrict retrieval to one transcript.
- Filtered retrieval surfaces selected transcript diagnostics.
- MLflow captures selected transcript summaries and retrieved RAG chunks for filtered and unfiltered S4 runs.
- S4 evaluation compares unfiltered all-chunk RAG against summary-filtered RAG.
- Evaluation reports answer similarity, token consumption, selected transcripts, retrieved chunks, and time taken.
- Tests pass with external calls mocked.

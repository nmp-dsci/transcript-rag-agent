/** Wire types for the FastAPI workbench API (see src/api/main.py). */

export interface SetupSpec {
  key: string;
  title: string;
  description: string;
  /** Query rewritten before embedding ("hyde"/"multi_query"), or null for none. */
  query_transform?: string | null;
  /** Answers over the Contextual Retrieval index rather than the baseline one. */
  contextual?: boolean;
}

export interface Reference {
  label?: string;
  video_id?: string;
  source_url?: string;
  timestamp_url?: string;
  start_seconds?: number | null;
}

/** One claim the judge extracted from an answer, and whether the chunks back it. */
export interface FaithfulnessClaim {
  claim: string;
  verdict: 0 | 1;
  reason: string;
}

/** One retrieved chunk's usefulness verdict, in retrieval rank order. */
export interface PrecisionVerdict {
  rank: number;
  verdict: 0 | 1;
  reason: string;
  chunk_preview: string;
}

/**
 * The judge's workings behind each score.
 *
 * Persisted by src/evals/judge.py, which computes each score FROM these
 * intermediates — so a breakdown always reconciles with the number above it.
 * Null per metric when capture failed, and null overall on evaluations written
 * before derivations were recorded.
 */
export interface EvaluationDetails {
  faithfulness: {
    claims: FaithfulnessClaim[];
    supported: number;
    total: number;
  } | null;
  answer_relevancy: {
    generated_questions: string[];
    noncommittal: boolean;
    similarities: number[];
  } | null;
  context_precision: {
    verdicts: PrecisionVerdict[];
    average_precision: number;
  } | null;
}

export interface Evaluation {
  judge: string;
  judge_model: string;
  rubric_version: string;
  ragas_version?: string | null;
  embedding_model?: string | null;
  scores: Record<string, number>;
  composite: number | null;
  elapsed_seconds: number;
  scored_at: string;
  error: string | null;
  /** Optional: absent on evaluations written before each field existed. */
  spread?: Record<string, number>;
  sample_scores?: Record<string, number[]>;
  /** Per-metric count of attempts that actually succeeded; absent on older records. */
  sample_counts?: Record<string, number>;
  judge_samples?: number;
  /** True when the judge model also wrote the answer; null when unknown. */
  self_graded?: boolean | null;
  details?: EvaluationDetails | null;
}

/** One persisted execution step of an answer path (serialized TraceStep). */
export interface TraceStep {
  phase: "route" | "filter" | "retrieve" | "rerank" | "merge" | "llm";
  label: string;
  detail: string;
  chunk_ids: string[];
  model: string | null;
  elapsed_ms: number | null;
  iteration: number | null;
}

export interface Answer {
  key: string;
  title: string;
  command: string;
  answer: string;
  references: Reference[];
  token_estimate: number;
  chunk_count: number;
  llm_calls: number | null;
  iterations: number | null;
  terminated_reason: string | null;
  elapsed_seconds: number;
  error: string | null;
  contexts: string[];
  evaluation: Evaluation | null;
  model: string | null;
  embedding_model: string | null;
  top_k: number | null;
  /** Retrieval scope and strategy; null on answers predating scoping. */
  channel_id?: string | null;
  retrieval_mode?: string | null;
  /** Follow-up questions the answering LLM proposed for this answer. */
  followups?: Followup[];
  /** Persisted execution steps; empty/absent on answers predating tracing. */
  trace?: TraceStep[];
}

export interface Followup {
  topic: string;
  rationale: string;
  followup_query: string;
  confidence: number;
}

export interface Entry {
  id: string;
  question: string;
  url: string | null;
  asked_at: string;
  answers: Answer[];
  /**
   * The document this question reviewed, by id. The text is not here on
   * purpose — this file is committed, so the document lives server-side and is
   * fetched with `api.document(id)`.
   */
  document_id?: string | null;
}

export interface DocumentSection {
  index: number;
  heading: string | null;
  text: string;
}

/** A page fetched from a URL in a chat message, extracted for review. */
export interface ReviewedDocument {
  id: string;
  url: string;
  requested_url: string;
  title: string | null;
  sections: DocumentSection[];
  /** The page exceeded the fetch cap and is missing its end. */
  truncated: boolean;
  fetched_at: string;
  /** Read from the store rather than fetched again — only on the SSE event. */
  reused?: boolean;
  /** Only some sections were sent to the model, chosen for this question. */
  narrowed?: boolean;
  sections_selected?: number[];
  detail?: string;
}

export interface Health {
  status: string;
  runner_loaded: boolean;
  judge_loaded: boolean;
  judge_model: string;
  answer_model: string;
  embedding_model: string;
  ui: string;
}

export interface Video {
  video_id: string;
  title: string | null;
  channel_name: string | null;
  /** Matches the channel_id stamped on chunks, so it is safe to filter with. */
  channel_id: string | null;
  thumbnail_url: string | null;
  source_url: string | null;
  duration_seconds: number | null;
  upload_date: string | null;
  view_count: number | null;
  summary: string | null;
  fetched_at: string | null;
  chunk_count: number;
}

export interface Channel {
  channel_id: string;
  channel_name: string;
  video_count: number;
  chunk_count: number;
  video_ids: string[];
}

/** An observation about corpus shape that affects retrieval quality. */
export interface CorpusInsight {
  kind: "channel_skew" | "missing_summaries" | "unindexed" | "size_spread";
  level: "info" | "warn" | "bad";
  message: string;
  channel_id?: string;
  video_ids?: string[];
}

export interface Corpus {
  videos: Video[];
  channels: Channel[];
  totals: { videos: number; chunks: number; channels: number };
  insights: CorpusInsight[];
}

export interface GraphNode {
  id: string;
  video_id: string;
  chunk_index: number;
  channel_id: string | null;
  channel_name: string | null;
  title: string | null;
  preview: string;
  start_seconds: number | null;
  end_seconds: number | null;
  source_url: string | null;
  degree: number;
  x: number;
  y: number;
}

export interface GraphEdge {
  source: string;
  target: string;
  similarity: number;
}

export interface ChunkGraph {
  nodes: GraphNode[];
  edges: GraphEdge[];
  stats: {
    nodes: number;
    edges: number;
    k: number;
    min_similarity: number;
    channels: number;
    mean_similarity: number;
    isolated_nodes: number;
  };
  /** Present only when a query was supplied: its retrieval neighbourhood. */
  query?: {
    text: string;
    nearest: { chunk_id: string; similarity: number }[];
  };
}

/** One entity node in the GraphRAG knowledge graph (GET /api/graph/knowledge). */
export interface EntityNode {
  id: string;
  name: string;
  type: string;
  mentions: number;
  community_id: number | null;
  x: number;
  y: number;
}

export interface EntityEdge {
  source: string;
  target: string;
  weight: number;
}

export interface GraphCommunity {
  id: number;
  summary: string | null;
  entity_count: number;
  claim_count: number;
}

export interface KnowledgeGraph {
  nodes: EntityNode[];
  edges: EntityEdge[];
  communities: GraphCommunity[];
}

/** One entity's metadata plus its dated claim timeline. */
export interface GraphClaim {
  id: string;
  text: string;
  entities: string[];
  chunk_id: string;
  video_id: string;
  source_url: string;
  video_title: string | null;
  upload_date: string | null;
  start_seconds: number | null;
  end_seconds: number | null;
  polarity: string;
}

export interface EntityDetail {
  entity: {
    id: string;
    name: string;
    type: string;
    aliases: string[];
    mentions: number;
    community_id: number | null;
  } | null;
  claims: GraphClaim[];
}

export interface Chunk {
  chunk_index: number;
  text: string;
  start_seconds: number | null;
  end_seconds: number | null;
  start_segment_index: number | null;
  end_segment_index: number | null;
  segment_count: number;
  source_url: string | null;
}

export interface ChunkList {
  video_id: string;
  chunks: Chunk[];
  total: number;
}

export type RankMode = "semantic" | "bm25" | "graph";

export interface RankRow {
  chunk_id: string;
  video_id: string | null;
  chunk_index: number | null;
  rank: number;
  score: number | null;
  preview: string;
  start_seconds: number | null;
  end_seconds: number | null;
  source_url: string | null;
  /** Best rank this chunk holds in any other selected mode; null when only
   * this mode found it. */
  other_rank: number | null;
  /** Graph mode only: which of the query's resolved entities this chunk's
   * claims cover. */
  matched_entities?: string[];
}

export interface Rankings {
  query: string;
  video_id: string | null;
  top_k: number;
  modes: Partial<Record<RankMode, RankRow[]>>;
  /** Why a selected mode produced no ranking — graph mode reports an
   * unreachable Neo4j here, so an unavailable column stays distinguishable
   * from one that legitimately matched nothing. A mode listed here is left
   * out of `overlap`. */
  errors?: Partial<Record<RankMode, string>>;
  overlap: { count: number; of: number; chunk_ids: string[] };
}

/** One chunk's GraphRAG enrichment — entities and claims the extraction pass
 * read into it (GET /api/graph/knowledge/videos/{video_id}/chunks). */
export interface ChunkEnrichment {
  entities: string[];
  claims: { id: string; text: string; polarity: string }[];
}

export interface VideoChunkEnrichment {
  chunks: Record<string, ChunkEnrichment>;
}

export interface ScoreboardRow {
  key: string;
  title: string;
  model: string | null;
  legacy: boolean;
  answers: number;
  judged: number;
  avg_scores: Record<string, number>;
  avg_composite: number | null;
  wins: number;
  contests: number;
  win_rate: number | null;
  avg_latency_seconds: number | null;
  avg_token_estimate: number | null;
}

export interface Provenance {
  judge_models: string[];
  ragas_versions: string[];
  embedding_models: string[];
  last_judged: string | null;
  metrics: string[];
  composite: string;
}

/** One committed matrix run, as the Scoreboard's run picker lists it. */
export interface MatrixRunOption {
  run_id: string;
  created_at: string | null;
  setups: string[];
  entry_count: number | null;
  judged: boolean;
}

/** One setup's answer to one golden question within the selected matrix run. */
export interface ScoreboardQuestionSetup {
  key: string;
  title: string;
  composite: number | null;
  judged: boolean;
  error: string | null;
  /** The answer this setup produced for this question — the text the judge graded. */
  answer: string;
  /** Answering model recorded for the setup's run, not the judge's. */
  model: string | null;
  elapsed_seconds: number | null;
  token_estimate: number | null;
  chunk_count: number;
}

/** One golden question judged in the selected matrix run, with every setup's score on it. */
export interface ScoreboardQuestion {
  id: string;
  question: string;
  domain: string | null;
  question_type: string | null;
  setups: ScoreboardQuestionSetup[];
}

export interface Scoreboard {
  setups: ScoreboardRow[];
  entries_total: number;
  entries_judged: number;
  group_by: string;
  judge_model: string;
  provenance: Provenance;
  /** The matrix run these rows were aggregated from; null when none exists. */
  run_id: string | null;
  /** Every committed run, newest first — the picker's options. */
  runs: MatrixRunOption[];
  /** Every golden question in the selected run, for the question-level breakdown. */
  questions: ScoreboardQuestion[];
}

/** Per-iteration research step emitted by the agentic setup while it runs. */
export interface AgentStep {
  key: string;
  iteration: number;
  event_type: "retrieval_start" | "retrieval_complete" | "answer_start";
  query: string | null;
  chunk_count: number | null;
}

export type RetrievalMode = "semantic" | "hybrid";

export interface AskRequest {
  question: string;
  setups: string[];
  url?: string | null;
  top_k?: number | null;
  entry_id?: string | null;
  /** A follow-up's source entry, to reuse its pinned document without
   * requiring the follow-up's (different) question to match that entry. */
  document_entry_id?: string | null;
  /** Ignored by the server when `url` pins a single video. */
  channel_id?: string | null;
  retrieval_mode?: RetrievalMode | null;
  filter_transcripts?: boolean;
  history?: string[];
}

/** One retrieval configuration measured by `eval-ablation` (src/evals/ablation.py). */
export interface AblationConfig {
  label: string;
  retrieval_mode: string;
  rerank: boolean;
  neighbor_span: number;
  top_k: number;
  /** Query rewritten before embedding ("hyde"/"multi_query"), or null for none. */
  query_transform?: string | null;
  /** Retrieved from the Contextual Retrieval index rather than the baseline one. */
  contextual?: boolean;
}

export interface AblationCell {
  label: string;
  config: AblationConfig;
  averages: Record<string, number>;
  by_domain: Record<string, Record<string, number>>;
}

export interface AblationDelta {
  label: string;
  vs_baseline: Record<string, number>;
}

export interface AblationRun {
  run_id: string;
  created_at: string;
  entries: number;
  metrics: string[];
  baseline: string;
  /** Which config set was swept. Absent on runs predating the extended sweep. */
  sweep?: string | null;
  cells: AblationCell[];
  deltas: AblationDelta[];
}

export interface GoldenRunSummary {
  run_id: string;
  created_at: string;
  setup: string;
  config: Record<string, unknown>;
  summary: {
    entries?: number;
    scored?: number;
    failed?: number;
    averages?: Record<string, number>;
    avg_elapsed_seconds?: number | null;
    avg_token_estimate?: number | null;
  };
}

/** Committed eval snapshots for the Experiments tab (GET /api/experiments). */
export interface Experiments {
  ablations: AblationRun[];
  golden_runs: GoldenRunSummary[];
  matrix_runs: MatrixRunSummary[];
}

/** A matrix run started from the Experiments tab (POST/GET /api/eval/matrix). */
export interface MatrixJob {
  id: string;
  setups: string[];
  status: "running" | "done" | "error";
  message: string | null;
  cells_done: number;
  cells_total: number;
  cache_hits: number;
  cache_misses: number;
  /** The committed run this produced, once it finished. */
  run_id: string | null;
  error: string | null;
}

/** One head-to-head matrix run (eval-matrix), summarized for the tab. */
export interface MatrixRunSummary {
  run_id: string;
  created_at: string;
  setups: string[];
  entry_count: number;
  judged: boolean;
  reference_scored: boolean;
  question_types: Record<string, number>;
  comparison: {
    overall: Record<string, Record<string, number>>;
    by_question_type: Record<string, Record<string, Record<string, number>>>;
    ops: Record<
      string,
      {
        avg_elapsed_seconds?: number | null;
        avg_token_estimate?: number | null;
        answered: number;
        failed: number;
      }
    >;
  };
}

/** One prompt from the live registry (GET /api/prompts). */
export interface PromptEntry {
  name: string;
  system: string;
  role: string;
  template_vars: string[];
  text: string;
  module: string;
}

export interface PromptSystem {
  key: string;
  title: string;
  description: string;
  count: number;
  prompts: PromptEntry[];
}

export interface Prompts {
  systems: PromptSystem[];
  total: number;
  notes: string[];
}

/** One step of an agent node's answer-path flow. */
export interface SystemDesignFlowStep {
  label: string;
  detail: string;
  branch: string | null;
}

/** One node of the System Design graph (GET /api/system-design). */
export interface SystemDesignNode {
  id: string;
  label: string;
  kind: "agent" | "stage" | "model" | "store";
  x: number;
  y: number;
  description: string;
  prompts: PromptEntry[];
  config: Record<string, unknown>;
  flow: SystemDesignFlowStep[];
}

export interface SystemDesignEdge {
  source: string;
  target: string;
}

export interface SystemDesign {
  nodes: SystemDesignNode[];
  edges: SystemDesignEdge[];
}

/** One stage of an indexing run, streamed by POST /api/index/stream. */
export interface IndexStage {
  stage: "discover" | "fetch" | "chunk" | "embed" | "summarize";
  message: string;
}

/** entities/claims extraction the ingestion queue ran automatically for
 * newly added videos, once the vector index step succeeded. Absent when no
 * videos were added; `ok: false` means extraction failed (vector RAG for the
 * new video is unaffected — only GraphRAG stays uncaught-up until the next
 * `index-graph` run). */
export interface GraphIngestResult {
  ok: boolean;
  extracted?: number;
  failed?: number;
  error?: string;
}

export interface IndexResult {
  ok: boolean;
  target: string;
  added_videos: Video[];
  added_video_count: number;
  added_chunk_count: number;
  totals: { videos: number; chunks: number; channels: number };
  insights: CorpusInsight[];
  channels: Channel[];
  graph?: GraphIngestResult;
}

/** One queued/running/finished ingestion job (GET/POST /api/index/queue). */
export interface IngestionJob {
  id: string;
  mode: "video" | "channel";
  target: string;
  latest: number | null;
  status: "queued" | "running" | "done" | "error";
  stage: string | null;
  message: string | null;
  result: IndexResult | null;
  error: string | null;
}

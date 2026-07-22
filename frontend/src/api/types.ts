/** Wire types for the FastAPI workbench API (see src/api/main.py). */

export interface SetupSpec {
  key: string;
  title: string;
  description: string;
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
  kind: 'channel_skew' | 'missing_summaries' | 'unindexed' | 'size_spread';
  level: 'info' | 'warn' | 'bad';
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

export type RankMode = 'semantic' | 'bm25';

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
  /** Rank of this chunk in the other mode; null when only this mode found it. */
  other_rank: number | null;
}

export interface Rankings {
  query: string;
  video_id: string | null;
  top_k: number;
  modes: Partial<Record<RankMode, RankRow[]>>;
  overlap: { count: number; of: number; chunk_ids: string[] };
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

export interface Scoreboard {
  setups: ScoreboardRow[];
  entries_total: number;
  entries_judged: number;
  group_by: string;
  judge_model: string;
  provenance: Provenance;
}

/** Per-iteration research step emitted by the agentic setup while it runs. */
export interface AgentStep {
  key: string;
  iteration: number;
  event_type: 'retrieval_start' | 'retrieval_complete' | 'answer_start';
  query: string | null;
  chunk_count: number | null;
}

export type RetrievalMode = 'semantic' | 'hybrid';

export interface AskRequest {
  question: string;
  setups: string[];
  url?: string | null;
  top_k?: number | null;
  entry_id?: string | null;
  /** Ignored by the server when `url` pins a single video. */
  channel_id?: string | null;
  retrieval_mode?: RetrievalMode | null;
  filter_transcripts?: boolean;
  history?: string[];
}

/** One stage of an indexing run, streamed by POST /api/index/stream. */
export interface IndexStage {
  stage: 'discover' | 'fetch' | 'chunk' | 'embed' | 'summarize';
  message: string;
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
}

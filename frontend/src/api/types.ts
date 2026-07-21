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
  source_url: string | null;
  duration_seconds: number | null;
  upload_date: string | null;
  view_count: number | null;
  summary: string | null;
  fetched_at: string | null;
  chunk_count: number;
}

export interface Corpus {
  videos: Video[];
  totals: { videos: number; chunks: number };
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

export interface AskRequest {
  question: string;
  setups: string[];
  url?: string | null;
  top_k?: number | null;
}

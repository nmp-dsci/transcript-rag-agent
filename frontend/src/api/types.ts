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
  /** True when the rubric's cap decided the composite; false/absent otherwise. */
  cap_applied?: boolean;
  cap_reason?: string | null;
  grounding_floor_breached?: boolean;
  grounding_reason?: string | null;
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
  /**
   * What was actually embedded/searched. Not always the user's question — a
   * follow-up is rewritten and a document review searches for the criteria the
   * document should be judged against — so it is rendered in full rather than
   * folded into the truncated detail line. Absent on non-search steps and on
   * traces written before this existed.
   */
  query?: string | null;
  /**
   * Long-form text this step has to show in full — a coverage caveat, the list
   * of videos a filter routed to. `detail` renders on the single-line row and
   * ellipsises at whatever width is left; anything a reader has to actually
   * read comes through here and gets its own wrapping line. Absent on steps
   * with nothing long to say, and on traces written before this existed.
   */
  note?: string | null;
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
  /**
   * How much of that document the answer read — "whole document — all 9
   * sections in context", or the narrowed equivalent. Recorded on the entry
   * rather than derived on reload, so a reopened conversation reports what
   * actually happened at the time it was answered.
   */
  document_detail?: string | null;
  /** Section indices the answer call actually received. */
  document_sections_selected?: number[] | null;
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

/* ── Themes: RAPTOR level 2, clusters that cross video boundaries ────────── */

/** One video contributing members to a theme (GET /api/themes). */
export interface ThemeVideo {
  video_id: string;
  title: string | null;
  channel_name: string | null;
  member_count: number;
  domain: string;
}

export interface Theme {
  theme_id: string;
  title: string;
  summary: string;
  member_count: number;
  video_count: number;
  channel_count: number;
  /** False when every member came from one video — no lift over level 1. */
  cross_video: boolean;
  domain: string;
  domain_mix: Record<string, number>;
  property_share: number;
  videos: ThemeVideo[];
}

export interface ThemeStats {
  themes?: number;
  cross_video_themes?: number;
  single_video_themes?: number;
  video_count_distribution?: Record<string, number>;
  max_videos_in_a_theme?: number;
  chunks_clustered?: number;
  videos_covered?: number;
  job_search_themes?: number;
  impure_job_search_themes?: string[];
  max_property_share_in_job_search_theme?: number;
}

export interface ThemeList {
  themes: Theme[];
  stats: ThemeStats;
  generated_at: string | null;
  summary_model: string | null;
  embedding_model: string | null;
  build_command: string;
}

/** A member chunk, hydrated from the chunk store (GET /api/themes/{id}). */
export interface ThemeChunk {
  chunk_id: string;
  chunk_index: number;
  probability: number;
  text: string;
  start_seconds: number | null;
  end_seconds: number | null;
  source_url: string | null;
}

export interface ThemeVideoGroup extends ThemeVideo {
  chunks: ThemeChunk[];
}

export interface ThemeDetail {
  theme: Theme;
  videos: ThemeVideoGroup[];
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
  /** Judged answers in this row that hit the rubric's cap; 0 without a cap. */
  capped?: number;
  /** Judged answers that breached the grounding floor — a superset of `capped`. */
  ungrounded?: number;
}

/** A named block of metrics and the share of the composite it owns. */
export interface MetricGroup {
  key: string;
  label: string;
  weight: number;
  metrics: string[];
}

export interface Provenance {
  judge_models: string[];
  /** The judge behind the depth metrics; empty under a rubric without them. */
  depth_judge_models?: string[];
  ragas_versions: string[];
  embedding_models: string[];
  last_judged: string | null;
  /** True when the answering model also graded these answers. */
  self_graded?: boolean;
  self_graded_answers?: number;
  /** Which rubric composited these rows; drives the metric columns below. */
  rubric_version?: string;
  /** Every rubric seen in the run — more than one means mixed records. */
  rubric_versions?: string[];
  metrics: string[];
  /** Each metric's share of the composite. Absent on older payloads. */
  metric_weights?: Record<string, number>;
  /** Grouping for the rubric panel; empty for an ungrouped rubric. */
  metric_groups?: MetricGroup[];
  composite: string;
}

/** One committed matrix run, as the Scoreboard's run picker lists it. */
export interface MatrixRunOption {
  run_id: string;
  created_at: string | null;
  setups: string[];
  entry_count: number | null;
  judged: boolean;
  /** Which rubric composited it — two runs can rank the same setups differently. */
  rubric_version?: string;
  /** The run whose answers and grounding scores this one re-scored. */
  rejudged_from?: string | null;
  rejudged_cells?: number | null;
  /** Cells this run's rubric could not score, excluded from every average. */
  skipped_cells?: number | null;
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
  /** Which rubric composited this cell. Absent on older payloads (= ragas-v1). */
  rubric_version?: string;
  /** True when the rubric's cap decided this composite. */
  cap_applied?: boolean;
  /** Why, in a sentence — rendered as page text, not only as a tooltip. */
  cap_reason?: string | null;
  /** True whenever the grounding floor was breached, capped or not. */
  grounding_floor_breached?: boolean;
  grounding_reason?: string | null;
  /** False on a cell this run's rubric could not score. */
  rejudged?: boolean | null;
  rejudge_skipped_reason?: string | null;
  /** How much of the retrieved context the judge actually saw. */
  contexts_resolved?: number | null;
  contexts_expected?: number | null;
  depth_error?: string | null;
  /** The weighted sum before the cap, so the cost of the cap is readable. */
  composite_uncapped?: number | null;
  /** Every metric the judge scored for this cell. */
  scores?: Record<string, number>;
  /** The judge's one-sentence reason per depth metric. */
  rationales?: Record<string, string>;
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
  /** That run's descriptor, for the coverage facts the averages depend on. */
  run?: MatrixRunOption | null;
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

/** One setup's row in a held-out critique run, without the per-finding detail. */
export interface CritiqueCell {
  setup: string;
  scores: Record<string, number | null>;
  /** Min/median/max of criteria_recall across the matcher's own repeats. */
  score_spread: {
    criteria_recall_min: number | null;
    criteria_recall_median: number | null;
    criteria_recall_max: number | null;
  };
  match_repeats: number | null;
  criteria_recall_all: number | null;
  criteria_recall_grouped: number | null;
  criteria_groups: number | null;
  criteria_applicable: number | null;
  criteria_matched: number | null;
  criteria_matched_ungrounded: number | null;
  findings_total: number | null;
  findings_grounded: number | null;
  findings_sharing_evidence: number | null;
  citations_total: number | null;
  citations_resolved: number | null;
  contested_findings: number | null;
  held_out_leaks: number | null;
  elapsed_seconds: number | null;
  token_estimate: number | null;
  error: string | null;
  retrieved_video_ids: string[];
}

/** Can the corpus reach an expert's criteria without having read that expert? */
export interface CritiqueRunSummary {
  run_id: string;
  created_at: string;
  held_out_video_id: string;
  held_out_title: string;
  artifact_url: string;
  artifact_kind: string;
  criteria_total: number;
  criteria_applicable: number;
  criteria_groups: number | null;
  match_repeats: number | null;
  metrics: string[];
  baseline: string;
  held_out_leaks: number;
  exclusion_version: string | null;
  config: Record<string, unknown>;
  cells: CritiqueCell[];
}

/** One held-out criterion and the finding (if any) that independently reached it. */
export interface CritiqueMatch {
  id: string;
  criterion: string;
  quote: string;
  video_id: string;
  start_seconds: number;
  applies_to: string[];
  note: string | null;
  group: string;
  matched: boolean;
  /** Paired with a finding whose evidence no other finding also claims. */
  counted: boolean;
  /** Paired, but that finding rests on no exclusive corpus evidence. */
  ungrounded: boolean;
  /** Share of matcher repeats that backed this verdict. */
  agreement: number;
  finding_id: string | null;
  finding_criterion: string | null;
  score: number;
  why: string | null;
}

export interface CritiqueCitationCheck {
  video_id: string;
  start_seconds: number;
  quote: string;
  resolved: boolean;
  reason: string;
  ratio: number;
  chunk_id: string | null;
  /** Another finding also rests on this chunk, so neither may claim it. */
  shared: boolean;
}

export interface CritiqueFinding {
  id: string;
  criterion: string;
  detail: string;
  contested: boolean;
  grounded: boolean;
  exclusive_chunk_ids: string[];
  citation_checks: CritiqueCitationCheck[];
}

/** The full run behind one expanded row (GET /api/experiments/critique/{id}). */
export interface CritiqueRunDetail extends CritiqueRunSummary {
  cells: (CritiqueCell & {
    matches: CritiqueMatch[];
    findings: CritiqueFinding[];
  })[];
}

/** Committed eval snapshots for the Experiments tab (GET /api/experiments). */
export interface Experiments {
  ablations: AblationRun[];
  golden_runs: GoldenRunSummary[];
  matrix_runs: MatrixRunSummary[];
  critique_runs: CritiqueRunSummary[];
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

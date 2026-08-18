/** Wire types for the FastAPI workbench API (see src/api/main.py). */

export interface SetupSpec {
  key: string;
  title: string;
  description: string;
  /** Query rewritten before embedding ("hyde"/"multi_query"), or null for none. */
  query_transform?: string | null;
  /** Answers over the Contextual Retrieval index rather than the baseline one. */
  contextual?: boolean;
  /** Reviews an attached document; has nothing to answer without one. */
  document_only?: boolean;
  /** Threads an attached document into its answer call rather than ignoring it. */
  document_capable?: boolean;
}

/** One creator quote behind a rubric, with the second it was said at. */
export interface RubricEvidence {
  video_id: string;
  chunk_id: string;
  quote: string;
  channel_name: string | null;
  title: string | null;
  start_seconds: number;
  /** Built server-side from `video_id`, never by appending to a stored url. */
  url: string;
}

/**
 * One rubric applied to the reviewed document, and what it decided.
 *
 * Everything above `verdict` is copied from the shipped pack; everything from
 * `verdict` down is what the reviewer said about this document. A row can be
 * read in either direction — finding back to the creator's sentence, or rubric
 * forward to the section it lands on.
 */
export interface RubricVerdict {
  rubric_id: string;
  topic: string;
  pack_name: string;
  criterion: string;
  check: string;
  why: string;
  unit_title: string;
  creators: string[];
  verdict: "pass" | "fail" | "n-a" | "unjudged";
  severity: "blocker" | "major" | "minor" | "none";
  finding: string;
  /** Zero-based section indices, rendered as §N+1 to match the answer. */
  sections: number[];
  evidence: RubricEvidence[];
  /** Why a row was not counted, when it was rejected rather than decided. */
  note: string;
}

export interface RubricPackOutcome {
  topic: string;
  name: string;
  artifact: string;
  rubrics: number;
  elapsed_seconds: number;
  error: string | null;
  unknown_rubric_ids: string[];
  duplicate_rubric_ids: string[];
  missing_rubric_ids: string[];
  unanchored_failures: string[];
}

export interface RubricReviewStats {
  rubrics_total: number;
  verdicts: Record<string, number>;
  severities: Record<string, number>;
  packs_used: number;
  packs_failed: number;
  with_rubric_id: number;
  with_timestamp: number;
  with_id_and_timestamp: number;
  id_and_timestamp_share: number;
  evidence_links: number;
  sections_cited: number[];
}

export interface RubricReview {
  document_id: string;
  document_url: string;
  document_kind: string;
  verdicts: RubricVerdict[];
  packs: RubricPackOutcome[];
  stats: RubricReviewStats;
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
  /**
   * Per-rubric verdicts, on answers from the rubric reviewer. Absent on every
   * other setup — the panel renders from this rather than re-parsing the prose,
   * so the rows and the summary above them cannot disagree.
   */
  rubric_review?: RubricReview | null;
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
  /** "demo" on the public read-only deployment, "full" everywhere else.
   * Optional so the UI degrades to full behaviour against an older server. */
  mode?: "demo" | "full";
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

/* ── Disagreements: where the corpus contradicts itself ──────────────────── */

/**
 * One creator's end of an axis (GET /api/conflicts).
 *
 * `quote` is cut out of the stored transcript server-side rather than typed by
 * the adjudicator, and `quote_ratio` says how much of what the model claimed
 * was quoting was actually found there — so the UI can show provenance as a
 * number instead of as a promise.
 */
export interface ConflictSide {
  video_id: string;
  chunk_id: string;
  channel_name: string;
  title: string;
  start_seconds: number;
  end_seconds: number;
  position: string;
  quote: string;
  quote_ratio: number;
  watch_url: string;
}

/**
 * One disagreement. There is deliberately no winner field: the payload cannot
 * express "the corpus says X", only "these two answer this question
 * differently".
 */
export interface Conflict {
  conflict_id: string;
  axis: string;
  why_incompatible: string;
  left: ConflictSide;
  right: ConflictSide;
  similarity: number;
  /**
   * Different uploading channels — which is not quite "different people". A
   * guest in a cold-open montage is attributed to the channel owner.
   */
  cross_channel: boolean;
  /**
   * `"axis"` when reasonable people could land on either side; `"factual"` when
   * the question has one true answer and one side is simply wrong. Rendered
   * differently because even-handed framing is honest for the first and
   * misleading for the second.
   */
  kind: string;
  /**
   * How many of `repeats` adjudications called this a conflict. On the card
   * because the adjudicator does not agree with itself: a 2/3 and a 3/3 are
   * different claims and a reader cannot tell them apart otherwise.
   */
  votes: number;
  repeats: number;
}

/** One calibration pair with a known answer, run through the live adjudicator. */
export interface ConflictProbe {
  probe_id: string;
  expect: string;
  why: string;
  verdicts: string[];
  passed: boolean;
  unanimous: boolean;
  axis: string;
  position_a: string;
  position_b: string;
}

export interface ConflictStats {
  conflicts?: number;
  cross_channel_conflicts?: number;
  within_channel_conflicts?: number;
  unanimous_conflicts?: number;
  split_conflicts?: number;
  factual_conflicts?: number;
  axis_conflicts?: number;
  candidates_adjudicated?: number;
  conflict_precision?: number | null;
  channels_involved?: number;
  videos_involved?: number;
  channels?: string[];
  rejected?: Record<string, number>;
  quotes_resolved?: number;
  min_quote_ratio?: number;
  claims?: number;
  probes_passed?: boolean;
  probe_repeats?: number;
  adjudicate_repeats?: number;
  adjudications?: number;
  /** Pairs where the adjudicator disagreed with itself across repeats. */
  pairs_with_split_verdicts?: number;
  verdict_agreement?: number | null;
  /** Conflicts carried by at least two thirds of looks, not just a majority. */
  firm_conflicts?: number;
  /** Pairs the judge is genuinely undecided about (a third to two thirds). */
  undecided_pairs?: number;
  /**
   * Modelled run-to-run spread on `conflicts`, in conflicts.
   *
   * Absent on any artifact built before `stability_statistics` existed, which
   * includes the one this app ships with. Absent is not zero and the view may
   * not render it as either zero or nothing — see `Spread` in
   * `DisagreementsView`. A recorded `0` is a real measurement and prints.
   */
  count_sd_estimate?: number;
  /** Conflict votes drawn, as `votes -> number of pairs`. */
  vote_histogram?: Record<string, number>;
  /**
   * What independent three-look builds of this same data would have reported,
   * measured by splitting this run's draws into disjoint groups of three.
   */
  subsample_counts_at_3?: number[];
}

/** The population a conflict count was taken over — see `corpus_fingerprint`. */
export interface ConflictCorpus {
  videos?: number;
  chunks?: number;
  digest?: string;
  chunks_with_claims?: number;
  videos_with_claims?: number;
}

export interface ConflictList {
  conflicts: Conflict[];
  stats: ConflictStats;
  probes: ConflictProbe[];
  generated_at: string | null;
  adjudicator_model: string | null;
  embedding_model: string | null;
  config: Record<string, unknown>;
  build_command: string;
  corpus?: ConflictCorpus;
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
  /**
   * The corpus this run was scored over. Absent on runs predating corpus
   * identity in the fingerprint — see `scoreboard/corpus.ts`, which treats
   * absence as *unknown* rather than as *the same corpus*.
   */
  corpus?: string | null;
  /** Size of that corpus. A run can record its digest without its counts. */
  corpus_videos?: number | null;
  corpus_chunks?: number | null;
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
  /**
   * What `chunk_count` counts, when it is not chunks. The rubric reviewer
   * reports verdicts per pack on this channel; rendering "17 chunks" for
   * seventeen verdicts would be a measurement of the wrong thing.
   */
  unit?: string | null;
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
/** One disagreement that was in a cell's context, and who named it. */
export interface CritiqueConflict {
  conflict_id: string;
  axis: string;
  video_ids: string[];
  chunk_ids: string[];
  /** Finding ids that cited both sides; empty means the cell averaged it away. */
  named_by: string[];
}

/** One citation the model invented, with its own words kept for reading. */
export interface FabricatedCitation {
  finding_id: string;
  video_id: string;
  start_seconds: number;
  claimed_quote: string;
  ratio: number | null;
  chunk_id: string | null;
  reason: string;
}

/** How many matcher repeats voted for one pairing (or for no pairing at all). */
export interface CritiqueBallotVote {
  /** null is a candidate like any other — "most repeats did not raise this". */
  finding_id: string | null;
  finding_criterion: string | null;
  count: number;
}

/**
 * One criterion's raw per-repeat ballots, beside what the vote made of them.
 *
 * Only criteria at least one repeat paired appear: a criterion every repeat
 * left alone has a ballot of blanks and the missed column already says so.
 * `consensus_finding_id` is read from the run's own `matches`, never recomputed
 * here — see src/evals/critique.py:consensus.
 */
export interface CritiqueBallot {
  criterion_id: string;
  criterion: string | null;
  applies_to: string[];
  /** One entry per matcher repeat: the finding it paired, or null for none. */
  draws: (string | null)[];
  votes: CritiqueBallotVote[];
  consensus_finding_id: string | null;
  consensus_finding_criterion: string | null;
  agreement: number | null;
}

export interface CritiqueCell {
  setup: string;
  scores: Record<string, number | null>;
  /**
   * Whether the grounding gate could grade this arm at all.
   *
   * `false` is not a bad score, it is the absence of one: `criteria_recall` and
   * `evidence_precision` come back null, and the panel must render that as "not
   * measured" rather than as a dash a reader fills in with the old number.
   * Only an engine that records what each *finding's own* reasoning retrieved
   * can be gated; an arm that emits every finding from one shared retrieval
   * pool cannot, and gating it against the pool would pass the padding attack
   * the gate exists for. See src/evals/critique.py:gate_verdict.
   */
  gradable?: boolean | null;
  /** Why the gate could not grade this arm, in the scorer's own words. */
  ungradable_reason?: string | null;
  /** The rule that produced `scores`, e.g. "retrieval_provenance". */
  grounding_gate?: string | null;
  /**
   * What the previous, weaker rule scored — the figures this run published.
   *
   * Kept on every cell whatever the gate decides, so nothing is lost when an
   * arm becomes ungraded. For a graded arm these are the same two numbers; for
   * an ungraded one they are the only figures there are, and they are a lower
   * bound the distinct-chunk attack beats 3.3x, so they are never a baseline.
   */
  criteria_recall_ungated?: number | null;
  evidence_precision_ungated?: number | null;
  /** Findings that record their own retrieval, out of `findings_total`. */
  findings_with_provenance?: number | null;
  /** Distinct retrieval declarations across those findings. One is a pool. */
  provenance_distinct_sets?: number | null;
  /** Resolved citations landing outside their own finding's retrieval. */
  citations_off_retrieval?: number | null;
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
  /**
   * Disagreements this cell's context held both sides of, and how many the
   * findings actually named.
   *
   * The **denominator** is fixed by retrieval before the answering call, so no
   * output can widen the field it is scored against. The numerator is not
   * immune to volume — more findings are more chances one cites both sides —
   * but it is capped by that fixed denominator and each conflict counts once,
   * so verbosity can only walk the score towards a ceiling retrieval already
   * set. See src/evals/critique.py.
   */
  conflicts_in_context: number | null;
  conflicts_named: number | null;
  conflicts?: CritiqueConflict[];
  /** The model's own `contested` boolean, recorded and deliberately not scored. */
  self_declared_contested: number | null;
  /**
   * Citations whose quoted words are not in the transcript they point at — a
   * model inventing evidence. Named at cell level rather than left inside a
   * finding's checks, because this is the failure the harness exists to catch.
   */
  fabricated_citations?: FabricatedCitation[];
  held_out_leaks: number | null;
  elapsed_seconds: number | null;
  token_estimate: number | null;
  error: string | null;
  retrieved_video_ids: string[];
  /**
   * The matcher's raw per-repeat ballots, for the criteria some repeat paired.
   *
   * On the list rather than behind the expand, because "0.000 because the vote
   * discarded two pairings" and "0.000 because nothing was found" are different
   * results and the score alone cannot tell them apart.
   */
  match_ballots?: CritiqueBallot[];
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
  /** Total invented citations across every cell. Zero is the only good value. */
  fabricated_citations?: number;
  /**
   * The grounding rule every score in this run was produced under.
   *
   * Two runs scored under different gates are not comparable, so this is on the
   * run and not only inside a cell. Null on a snapshot committed before the gate
   * existed.
   */
  grounding_gate?: string | null;
  /** Setups the gate refused to grade. Their scores are null, not zero. */
  ungraded_cells?: string[];
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

/* ── Expert rubric packs (GET /api/packs) ────────────────────────────────
   A pack is a JSON file under experts/ that `build-packs` wrote. The panel
   renders it directly, so these types mirror src/api/packs.py rather than the
   pydantic models — the server already resolved every deep link and recomputed
   the gate numbers from the rubrics on screen. */

/** The deterministic claims a pack's header states, recomputed from its rubrics. */
export interface PackChecks {
  rubrics: number;
  evidence_total: number;
  evidence_resolved: number;
  quote_resolution: number;
  multi_creator_rubrics: number;
  /** Distinct *creators*, not distinct video ids — a theme can span four
   *  videos and still be one podcast talking. */
  multi_creator_share: number;
  contested_rubrics: number;
  creators: number;
  excluded_video_citations: number;
}

/** One transcript quote behind a rubric, with the link that opens it. */
export interface PackEvidence {
  video_id: string;
  chunk_id: string;
  quote: string;
  model_quote: string;
  channel_name: string | null;
  title: string | null;
  start_seconds: number;
  quote_start_seconds: number;
  ratio: number;
  resolved: boolean;
  /** Built server-side from the video id, never by appending to source_url —
   *  eight corpus urls already carry a `t=` and YouTube honours the first. */
  url: string;
}

export interface PackRubric {
  rubric_id: string;
  criterion: string;
  check: string;
  why: string;
  contested: boolean;
  unit_id: string;
  unit_kind: string;
  unit_title: string;
  creators: string[];
  videos: string[];
  evidence: PackEvidence[];
}

/** One video the router put in (or a human pinned in) a pack. */
export interface PackMember {
  video_id: string;
  title: string | null;
  channel_name: string | null;
  score: number;
  routed: boolean;
  override: boolean | null;
  included?: boolean;
  chunk_count: number;
  /** Whether any of this video's chunks reached a source unit. False means it
   *  routed into the pack but had no say in its rules — in this corpus that
   *  happens when a video was ingested after the last `index-themes`. */
  in_units?: boolean;
  /** Whether any rubric quotes this video. */
  cited?: boolean;
}

/** A source unit that produced no rubric, and why — silence, logged. */
export interface PackGap {
  unit_id: string;
  unit_kind: string;
  unit_title: string;
  videos: number;
  chunks: number;
  creator_count?: number;
  top_creator?: string;
  top_creator_share?: number;
  reason: string;
}

export interface PackUnit {
  unit_id: string;
  kind: string;
  title: string;
  chunks: number;
  videos: number;
  creator_count: number;
  top_creator: string;
  top_creator_share: number;
}

export interface PackProvenance {
  corpus_digest: string;
  chunk_count: number;
  video_count: number;
  theme_count: number;
  graph_extractions: number;
  excluded_video_ids: string[];
  held_out_video_ids: string[];
  embedding_model: string;
  rubric_model: string;
  units_budget: number;
  routing_top_k: number;
  routing_min_score: number;
  community_min_entities: number;
}

/** One arm of the D2 three-way ablation, scored on the held-out harness. */
export interface PackAblationCell {
  arm: string;
  scores: Record<string, number>;
  score_spread: Record<string, number>;
  rubrics: number | null;
  criteria_recall_all: number | null;
  criteria_recall_grouped: number | null;
  findings_total: number | null;
  findings_grounded: number | null;
  citations_total: number | null;
  citations_resolved: number | null;
  held_out_leaks: number | null;
  units_by_kind: Record<string, number> | null;
}

/**
 * How far a pack's build corpus is behind the one being browsed.
 *
 * Absent when no comparison was available — which must not render the same as
 * "up to date". `behind_*` are signed, because a re-chunk can shrink the corpus.
 */
export interface PackStaleness {
  build_digest: string;
  build_videos: number;
  build_chunks: number;
  live_digest: string;
  live_videos: number;
  live_chunks: number;
  behind_videos: number;
  behind_chunks: number;
  current: boolean;
}

/**
 * Which arm led a metric, and on what evidence.
 *
 * `decisive` alone was ambiguous: only `criteria_recall` is scored repeatedly,
 * so every other metric is a single draw with no spread to clear. `basis` says
 * which situation produced the verdict, so the badge can stop describing a
 * comparison that was never performed.
 */
export type PackAblationBasis =
  | "cleared-spread"
  | "inside-spread"
  | "unrepeated"
  | "tied"
  | "single-arm"
  | "no-scored-arm";

export interface PackAblationVerdict {
  metric: string;
  leader: string | null;
  value?: number;
  tied?: string[];
  runner_up_max?: number | null;
  decisive: boolean;
  /** Absent on runs committed before the three-way verdict landed. */
  basis?: PackAblationBasis;
  reason: string;
}

export interface PackAblation {
  generated_at: string | null;
  metrics: string[];
  baseline: string | null;
  held_out_title: string | null;
  artifact_url: string | null;
  criteria_total: number | null;
  criteria_applicable: number | null;
  match_repeats: number | null;
  cells: PackAblationCell[];
  verdicts: Record<string, PackAblationVerdict>;
}

/** A row in the pack list: declared, and built or not. */
export interface PackSummary {
  topic: string;
  name: string;
  blurb: string;
  artifact: string;
  built: boolean;
  arm?: string;
  generated_at?: string;
  corpus_digest?: string;
  checks?: PackChecks;
  members?: number;
}

export interface PackList {
  packs: PackSummary[];
  excluded_video_ids?: string[];
  exclusion_reason?: string;
  held_out_video_ids?: string[];
  build_command: string;
}

export interface PackArmSummary {
  arm: string;
  checks: PackChecks;
  units_by_kind: Record<string, number>;
  unit_budget: number | null;
  gaps: number | null;
}

export interface PackDetail {
  topic: string;
  name: string;
  blurb: string;
  artifact: string;
  routing_text: string;
  built: boolean;
  overrides: Record<string, boolean>;
  arms: Record<string, PackArmSummary>;
  ablation: PackAblation | null;
  excluded_video_ids: string[];
  held_out_video_ids: string[];
  build_command: string;
  /** Absent when the live corpus could not be read — not the same as current. */
  staleness?: PackStaleness | null;
  arm?: string;
  version?: number;
  generated_at?: string;
  provenance?: PackProvenance;
  stats?: Record<string, unknown>;
  checks?: PackChecks;
  rubrics?: PackRubric[];
  members?: PackMember[];
  units?: PackUnit[];
  gaps?: PackGap[];
}

/* ── Deep-research build loop (experts/<topic>/research.json) ───────────── */

/** One sub-question the loop spent an executor call on.
 *
 *  `origin` is the causal link the whole slice turns on: `plan` means the
 *  planner wrote it up front, `gap:<id>` names the critic finding that caused
 *  it. Without it, "a second round happened" and "the critic caused a second
 *  round" look identical in the report. */
export interface ResearchProbe {
  probe_id: string;
  facet: string;
  question: string;
  why: string;
  origin: string;
  rank: number;
}

/** What one executor call produced, including when it produced nothing. */
export interface ResearchProbeOutcome {
  probe: ResearchProbe;
  unit_id: string;
  chunks: number;
  videos: number;
  creator_count: number;
  top_creator: string;
  top_creator_share: number;
  eligible: boolean;
  reject_reason: string;
  rubric_ids: string[];
  reason: string;
  spent_call: boolean;
}

/** One thing the gap critic says a reviewer still cannot check. */
export interface ResearchGap {
  gap_id: string;
  missing: string;
  why: string;
  probe: string;
}

export interface ResearchRound {
  round: number;
  arm: string;
  label: string;
  caused_by: string;
  probes: ResearchProbeOutcome[];
  executor_calls: number;
  planner_calls: number;
  critic_calls: number;
  rubrics: number;
  citations: number;
  multi_creator_share: number;
  deduped_against_previous: number;
  added_rubric_ids: string[];
}

export interface ResearchDiffRow {
  rubric_id: string;
  criterion: string;
  check: string;
  unit_id: string;
  creators: string[];
  citations: number;
  /**
   * Shipped-pack rules citing a chunk this added rule also cites. Empty means
   * genuinely new ground; absent means the comparison was not available. An
   * added rule is new against *round one*, which is all the gap critic sees —
   * this says whether it is new against the pack it is measured against.
   */
  already_in_shipped?: string[];
}

export interface ResearchDiff {
  before_arm: string;
  after_arm: string;
  kept: ResearchDiffRow[];
  added: ResearchDiffRow[];
  removed: ResearchDiffRow[];
}

/** Calls per arm, stated rather than implied — a loop that beat a one-shot on
 *  more calls has reported its budget, not its architecture. */
export interface ResearchBudgetRow {
  arm: string;
  label: string;
  planner_calls: number;
  critic_calls: number;
  executor_calls: number;
  probes_budgeted: number;
  total_llm_calls: number;
  rubrics: number;
  citations: number;
}

/** One arm on the held-out harness. The finding and citation counts sit beside
 *  the scores because `criteria_recall` rises with finding count, so the shape
 *  of an arm that gained has to be visible next to the number. */
export interface ResearchScoreRow {
  arm: string;
  scores: Record<string, number | null>;
  score_spread: Record<string, number | null>;
  findings_total: number | null;
  findings_grounded: number | null;
  citations_total: number | null;
  citations_resolved: number | null;
  criteria_matched: number | null;
  criteria_applicable: number | null;
  criteria_recall_grouped: number | null;
  executor_calls: number | null;
  unit_budget: number | null;
  held_out_leaks: number | null;
}

/** One arm against the run's **baseline**, not against the runner-up.
 *
 *  `winner()` answers "which arm leads and does the lead survive the noise",
 *  which is the wrong question for a gate phrased as "does the loop-built pack
 *  reach the hand-built one" — when two loop arms tie for the lead it reports
 *  `tied` and says nothing about the pack they were built to beat. Computed in
 *  `src/evals/pack_ablation.py` and committed into the run, so the comparison
 *  exists outside the browser. */
export interface ResearchBaselineRow {
  arm: string;
  baseline: string;
  metric: string;
  value: number | null;
  baseline_value: number | null;
  arm_min?: number | null;
  baseline_max?: number | null;
  delta: number | null;
  beats_baseline: boolean | null;
  /** `ranges-disjoint` · `ranges-overlap` · `deterministic` · `unrepeated` · `ungraded` */
  basis: string;
  reason: string;
}

export interface ResearchScores {
  generated_at: string | null;
  run_id: string | null;
  metrics: string[];
  baseline: string | null;
  held_out_title: string | null;
  held_out_video_id: string | null;
  criteria_total: number | null;
  criteria_applicable: number | null;
  match_repeats: number | null;
  /** The corpus the arms were built from, and the one their citations were
   *  resolved against. Two fields because ingestion does not stop for an
   *  experiment, and a score measured on a corpus that has moved has to say so. */
  build_corpus_digest: string | null;
  scoring_corpus_digest: string | null;
  scoring_chunk_count: number | null;
  scored_on_build_corpus: boolean | null;
  rows: ResearchScoreRow[];
  verdicts: Record<string, PackAblationVerdict>;
  /** Absent on reports written before the comparison existed; the server
   *  re-derives it from the rows, so in practice it is always present. */
  against_baseline?: Record<string, ResearchBaselineRow[]>;
}

/** Did round 2 get nearer to the gap than round 1 already was?
 *
 *  A gap is *asked about* by construction — its probe is the critic's own
 *  words. Whether it was *closed* is a different question, and the deliberately
 *  weak deterministic test is this: a rule admitted from the gap's own probe
 *  sitting closer to the critic's statement than every rule round 1 had. */
export interface ResearchGapClosure {
  gap_id: string;
  missing: string;
  probe: string;
  rules_from_this_probe: number;
  best_new_rubric_id: string | null;
  best_new_criterion: string | null;
  best_new_cosine: number | null;
  round_one_best_cosine: number;
  closed: boolean;
}

/** Each new rule's nearest surviving rule, and how near it sat.
 *
 *  The dedupe threshold is a cliff and a cliff has a shadow: two statements of
 *  the same rule landing just under it both survive. Reported, never filtered —
 *  a threshold tuned here would make the loop's numbers a function of a knob
 *  the loop owns. */
export interface ResearchRestatement {
  rubric_id: string;
  criterion: string;
  unit_id: string;
  nearest_prior_id: string | null;
  nearest_prior_criterion: string | null;
  nearest_prior_cosine: number | null;
  dedupe_threshold: number;
}

/** One member video, and how much of it a round's probes actually read.
 *
 *  The whole of what V8's coverage-aware critic is told beyond the criteria it
 *  already read: counts, and the video titles the planner was handed anyway. No
 *  transcript, no evidence, nothing from the held-out expert. */
export interface ResearchCoverageRow {
  video_id: string;
  title: string;
  channel_name: string;
  chunks: number;
  read: number;
}

/** A rule refused for resting on no transcript passage of its own.
 *
 *  The complement of the cosine dedupe, on evidence identity rather than
 *  wording — an identity check has no cliff and therefore no shadow under it. */
export interface ResearchRefusal {
  rubric_id: string;
  criterion: string;
  unit_id: string;
  chunk_ids: string[];
  already_rested_on_by: string[];
}

/** How many of an arm's additions cite a chunk the shipped pack already cites.
 *
 *  The number V8 turns on. `deep-r2` scored 4 of 10 and the one-shot control
 *  scored the same 4 of 10, which is what said the loop was searching more
 *  rather than better. `rediscovered: null` means no shipped pack was on disk
 *  to compare against — unmeasured, not zero. */
export interface ResearchRediscovery {
  shipped_arm?: string;
  added: number;
  rediscovered: number | null;
  rate: number | null;
  rows: Array<{
    rubric_id: string;
    criterion: string;
    already_in_shipped: string[];
  }>;
}

/** V8's frontier round: a coverage-aware critic, retrieval kept off round 1's
 *  ground, and rules admitted only when they rest on a passage of their own. */
export interface ResearchFrontier {
  generated_at: string;
  arm: string;
  admission_arm: string;
  model: string;
  corpus_digest: string;
  chunk_count: number;
  video_count: number;
  member_videos: number;
  member_chunks: number;
  round_one_read: number;
  settings: Record<string, number | string>;
  coverage: ResearchCoverageRow[];
  gaps: ResearchGap[];
  probes: ResearchProbeOutcome[];
  probe_overlap: Array<{
    unit_id: string;
    chunks: number;
    from_round_one_ground: number;
  }>;
  refused: ResearchRefusal[];
  /** This arm's row in the Rounds table, so its spend sits beside what it added
   *  like every other round's. Derived server-side for reports written before
   *  the field existed. */
  round?: ResearchRound;
  /** What the admission rule would refuse in the pack being beaten. The new
   *  arms get a rule the hand build was never put through, so the honest
   *  question is whether the hand build has anything to lose to it. */
  admission_on_shipped: {
    arm: string | null;
    rubrics: number;
    would_refuse: number | null;
    refused_ids: string[];
  };
  deduped_against_previous: string[];
  diff: ResearchDiff;
  rediscovery: Record<string, ResearchRediscovery>;
  gap_closure: ResearchGapClosure[];
  restatements: ResearchRestatement[];
  budget: ResearchBudgetRow[];
}

export interface ResearchReport {
  kind: string;
  topic: string;
  name: string;
  artifact: string;
  generated_at: string;
  corpus_digest: string;
  chunk_count: number;
  model: string;
  members: number;
  excluded_video_ids: string[];
  held_out_video_ids: string[];
  settings: Record<string, number>;
  plan: ResearchProbe[];
  gaps: ResearchGap[];
  gap_closure: ResearchGapClosure[];
  restatements: ResearchRestatement[];
  rounds: ResearchRound[];
  control: ResearchRound;
  diff: ResearchDiff;
  control_diff: ResearchDiff;
  budget: ResearchBudgetRow[];
  arms: Record<string, Record<string, number | null>>;
  scores: ResearchScores | null;
  /** Absent on every report written before V8 — the panel renders the original
   *  three-arm experiment unchanged when it is not there. */
  frontier?: ResearchFrontier;
}

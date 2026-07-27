/** Typed wrappers for every workbench endpoint. */

import { readEvents } from "./sse";
import type {
  AgentStep,
  Answer,
  AskRequest,
  ChunkGraph,
  ChunkList,
  Corpus,
  Entry,
  EntityDetail,
  Evaluation,
  Experiments,
  Health,
  IndexResult,
  IndexStage,
  IngestionJob,
  KnowledgeGraph,
  MatrixJob,
  Prompts,
  RankMode,
  Rankings,
  Scoreboard,
  SetupSpec,
  SystemDesign,
  VideoChunkEnrichment,
} from "./types";

async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`${path} → HTTP ${response.status}`);
  return (await response.json()) as T;
}

async function postJson<T>(path: string, body: unknown): Promise<T> {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(`${path} → HTTP ${response.status}: ${detail}`);
  }
  return (await response.json()) as T;
}

async function postStream(
  path: string,
  body: unknown,
  handlers: Record<string, (data: any) => void>,
  signal?: AbortSignal,
): Promise<void> {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!response.ok || !response.body) {
    throw new Error(
      `${path} → HTTP ${response.status}: ${await response.text()}`,
    );
  }
  await readEvents(response.body, handlers);
}

/** Same shape as postStream, for endpoints with no request body — the live
 * ingestion queue feed is a persistent GET stream, not a one-shot POST. */
async function getStream(
  path: string,
  handlers: Record<string, (data: any) => void>,
  signal?: AbortSignal,
): Promise<void> {
  const response = await fetch(path, { signal });
  if (!response.ok || !response.body) {
    throw new Error(
      `${path} → HTTP ${response.status}: ${await response.text()}`,
    );
  }
  await readEvents(response.body, handlers);
}

export const api = {
  health: () => getJson<Health>("/api/health"),
  setups: () =>
    getJson<{ setups: SetupSpec[] }>("/api/setups").then((r) => r.setups),
  history: () =>
    getJson<{ conversations: Entry[] }>("/api/history").then(
      (r) => r.conversations,
    ),
  corpus: () => getJson<Corpus>("/api/corpus"),
  chunks: (videoId: string) =>
    getJson<ChunkList>(`/api/corpus/${encodeURIComponent(videoId)}/chunks`),

  experiments: () => getJson<Experiments>("/api/experiments"),

  /** Start a judged eval matrix. Returns the run already in flight, if any. */
  startMatrixRun: (setups: string[] = []) =>
    postJson<MatrixJob>("/api/eval/matrix", { setups }),

  matrixRunSnapshot: () =>
    getJson<{ job: MatrixJob | null }>("/api/eval/matrix").then((r) => r.job),

  /** Live progress for the current matrix run. Never ends on its own — the
   * caller aborts `signal` to disconnect. */
  subscribeMatrixRun: (
    handlers: {
      snapshot?: (data: { job: MatrixJob | null }) => void;
      job?: (data: { job: MatrixJob }) => void;
    },
    signal?: AbortSignal,
  ) => getStream("/api/eval/matrix/stream", handlers, signal),
  prompts: () => getJson<Prompts>("/api/prompts"),
  systemDesign: () => getJson<SystemDesign>("/api/system-design"),

  scoreboard: (
    groupBy: string,
    judgeModel?: string | null,
    runId?: string | null,
  ) => {
    const params = new URLSearchParams({ group_by: groupBy });
    if (judgeModel) params.set("judge_model", judgeModel);
    if (runId) params.set("run_id", runId);
    return getJson<Scoreboard>(`/api/scoreboard?${params}`);
  },

  rank: (
    query: string,
    modes: RankMode[],
    topK: number,
    videoId?: string | null,
  ) =>
    postJson<Rankings>("/api/rank", {
      query,
      modes,
      top_k: topK,
      video_id: videoId ?? null,
    }),

  index: (payload: {
    mode: "video" | "channel";
    url?: string;
    channel?: string;
    latest?: number;
  }) =>
    postJson<{
      ok: boolean;
      exit_code: number;
      target: string;
      detail?: string;
    }>("/api/index", payload),

  /** Index with per-stage progress, ending in a summary of what changed. */
  indexStream: (
    payload: {
      mode: "video" | "channel";
      url?: string;
      channel?: string;
      latest?: number;
    },
    handlers: {
      stage?: (data: IndexStage) => void;
      done?: (data: IndexResult) => void;
      error?: (data: { message: string }) => void;
    },
    signal?: AbortSignal,
  ) => postStream("/api/index/stream", payload, handlers, signal),

  /** Add a job to the ingestion queue. Returns immediately — never blocks on
   * the job actually running, so the caller can enqueue another right away. */
  enqueueIndex: (payload: {
    mode: "video" | "channel";
    url?: string;
    channel?: string;
    latest?: number;
  }) => postJson<IngestionJob>("/api/index/queue", payload),

  indexQueueSnapshot: () =>
    getJson<{ jobs: IngestionJob[] }>("/api/index/queue").then((r) => r.jobs),

  /** Live queue progress for every job, from every browser tab. The stream
   * never ends on its own — the caller aborts `signal` to disconnect. */
  subscribeIndexQueue: (
    handlers: {
      snapshot?: (data: { jobs: IngestionJob[] }) => void;
      job?: (data: { job: IngestionJob }) => void;
    },
    signal?: AbortSignal,
  ) => getStream("/api/index/queue/stream", handlers, signal),

  chunkGraph: (
    opts: {
      k?: number;
      min_similarity?: number;
      query?: string | null;
      top_k?: number;
    } = {},
  ) => postJson<ChunkGraph>("/api/chunk-graph", opts),

  knowledgeGraph: () => getJson<KnowledgeGraph>("/api/graph/knowledge"),
  knowledgeGraphEntity: (entityId: string) =>
    getJson<EntityDetail>(
      `/api/graph/knowledge/entities/${encodeURIComponent(entityId)}`,
    ),
  chunkEnrichment: (videoId: string) =>
    getJson<VideoChunkEnrichment>(
      `/api/graph/knowledge/videos/${encodeURIComponent(videoId)}/chunks`,
    ),

  ask: (
    request: AskRequest,
    handlers: {
      progress?: (data: { key?: string; message: string }) => void;
      agent_step?: (data: AgentStep) => void;
      answer?: (data: Answer) => void;
      done?: (data: Entry) => void;
      error?: (data: { message: string }) => void;
    },
    signal?: AbortSignal,
  ) => postStream("/api/ask", request, handlers, signal),

  judge: (
    entryId: string,
    handlers: {
      progress?: (data: { key?: string; message: string }) => void;
      scored?: (data: { key: string; evaluation: Evaluation }) => void;
      done?: (data: Entry) => void;
      error?: (data: { message: string }) => void;
    },
    force = false,
  ) => postStream("/api/judge", { entry_id: entryId, force }, handlers),
};

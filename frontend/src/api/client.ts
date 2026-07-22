/** Typed wrappers for every workbench endpoint. */

import { readEvents } from './sse';
import type {
  AgentStep,
  Answer,
  AskRequest,
  ChunkGraph,
  ChunkList,
  Corpus,
  Entry,
  Evaluation,
  Experiments,
  Health,
  IndexResult,
  IndexStage,
  RankMode,
  Rankings,
  Scoreboard,
  SetupSpec,
} from './types';

async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`${path} → HTTP ${response.status}`);
  return (await response.json()) as T;
}

async function postJson<T>(path: string, body: unknown): Promise<T> {
  const response = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
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
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal,
  });
  if (!response.ok || !response.body) {
    throw new Error(`${path} → HTTP ${response.status}: ${await response.text()}`);
  }
  await readEvents(response.body, handlers);
}

export const api = {
  health: () => getJson<Health>('/api/health'),
  setups: () => getJson<{ setups: SetupSpec[] }>('/api/setups').then((r) => r.setups),
  history: () =>
    getJson<{ conversations: Entry[] }>('/api/history').then((r) => r.conversations),
  corpus: () => getJson<Corpus>('/api/corpus'),
  chunks: (videoId: string) =>
    getJson<ChunkList>(`/api/corpus/${encodeURIComponent(videoId)}/chunks`),

  experiments: () => getJson<Experiments>('/api/experiments'),

  scoreboard: (groupBy: string, judgeModel?: string | null) => {
    const params = new URLSearchParams({ group_by: groupBy });
    if (judgeModel) params.set('judge_model', judgeModel);
    return getJson<Scoreboard>(`/api/scoreboard?${params}`);
  },

  rank: (query: string, modes: RankMode[], topK: number, videoId?: string | null) =>
    postJson<Rankings>('/api/rank', {
      query,
      modes,
      top_k: topK,
      video_id: videoId ?? null,
    }),

  index: (payload: { mode: 'video' | 'channel'; url?: string; channel?: string; latest?: number }) =>
    postJson<{ ok: boolean; exit_code: number; target: string; detail?: string }>(
      '/api/index',
      payload,
    ),

  /** Index with per-stage progress, ending in a summary of what changed. */
  indexStream: (
    payload: { mode: 'video' | 'channel'; url?: string; channel?: string; latest?: number },
    handlers: {
      stage?: (data: IndexStage) => void;
      done?: (data: IndexResult) => void;
      error?: (data: { message: string }) => void;
    },
    signal?: AbortSignal,
  ) => postStream('/api/index/stream', payload, handlers, signal),

  chunkGraph: (opts: {
    k?: number;
    min_similarity?: number;
    query?: string | null;
    top_k?: number;
  } = {}) => postJson<ChunkGraph>('/api/chunk-graph', opts),

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
  ) => postStream('/api/ask', request, handlers, signal),

  judge: (
    entryId: string,
    handlers: {
      progress?: (data: { key?: string; message: string }) => void;
      scored?: (data: { key: string; evaluation: Evaluation }) => void;
      done?: (data: Entry) => void;
      error?: (data: { message: string }) => void;
    },
    force = false,
  ) => postStream('/api/judge', { entry_id: entryId, force }, handlers),
};

/**
 * Ingestion stage tracking.
 *
 * `POST /api/index/stream` emits one `stage` event per step, and a channel run
 * cycles through fetch→chunk→embed→summarize once per video. So progress is
 * derived from the stage currently reporting rather than from a counter: every
 * earlier stage is done, every later one is pending, and a stage arriving out
 * of order simply means the next video started.
 */

import type { IndexStage } from '../api/types';

export type StageName = IndexStage['stage'];
export type StageStatus = 'pending' | 'active' | 'done';

export const STAGES: { name: StageName; label: string; hint: string }[] = [
  { name: 'discover', label: 'Discover', hint: 'resolve videos' },
  { name: 'fetch', label: 'Fetch', hint: 'pull transcripts' },
  { name: 'chunk', label: 'Chunk', hint: 'split by timing' },
  { name: 'embed', label: 'Embed', hint: 'vectorise chunks' },
  { name: 'summarize', label: 'Summarize', hint: 'per-video digest' },
];

const ORDER: StageName[] = STAGES.map((stage) => stage.name);

/**
 * Status of every stage given the one now reporting.
 *
 * `active` is null before the first event (all pending) and after completion
 * the caller passes `finished` so the whole sequence reads as done.
 */
export function stageStatuses(
  active: StageName | null,
  finished = false,
): Record<StageName, StageStatus> {
  const statuses = {} as Record<StageName, StageStatus>;
  const activeIndex = active ? ORDER.indexOf(active) : -1;
  ORDER.forEach((name, index) => {
    if (finished) statuses[name] = 'done';
    else if (activeIndex < 0) statuses[name] = 'pending';
    else if (index < activeIndex) statuses[name] = 'done';
    else if (index === activeIndex) statuses[name] = 'active';
    else statuses[name] = 'pending';
  });
  return statuses;
}

/** Cap the streamed message log so a long channel run cannot grow unbounded. */
export const LOG_LIMIT = 60;

export function appendLog(log: string[], line: string): string[] {
  const next = [...log, line];
  return next.length > LOG_LIMIT ? next.slice(next.length - LOG_LIMIT) : next;
}

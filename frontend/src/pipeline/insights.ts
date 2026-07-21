/**
 * Corpus insights → badge styling and tree filters.
 *
 * An insight is only worth surfacing if the user can act on it, so each kind
 * that carries a target (a channel, or a list of videos) turns into a filter
 * the corpus tree can apply. Kinds without a target stay informational.
 */

import type { CorpusInsight, Video } from '../api/types';

/** Existing `.badge` variants, so insights inherit the workbench palette. */
export function insightBadgeClass(level: CorpusInsight['level']): string {
  if (level === 'bad') return 'bad';
  if (level === 'warn') return 'warn';
  return 'acc';
}

export interface TreeFilter {
  /** Which insight opened this filter, so the same chip can toggle it off. */
  key: string;
  label: string;
  channelId?: string;
  videoIds?: string[];
}

/** Stable identity for an insight — the list has no ids of its own. */
export function insightKey(insight: CorpusInsight, index: number): string {
  return `${insight.kind}:${insight.channel_id ?? index}`;
}

const FILTER_LABELS: Record<CorpusInsight['kind'], string> = {
  channel_skew: 'channel',
  missing_summaries: 'missing summaries',
  unindexed: 'unindexed',
  size_spread: 'size spread',
};

/**
 * The tree filter an insight opens, or null when it names no target.
 *
 * `channel_skew` carries a channel_id; `missing_summaries` and `unindexed`
 * carry video_ids. `size_spread` describes the whole corpus and so is inert.
 */
export function insightFilter(insight: CorpusInsight, index: number): TreeFilter | null {
  const key = insightKey(insight, index);
  const label = FILTER_LABELS[insight.kind];
  if (insight.channel_id) return { key, label, channelId: insight.channel_id };
  if (insight.video_ids && insight.video_ids.length > 0) {
    return { key, label: `${label} · ${insight.video_ids.length}`, videoIds: insight.video_ids };
  }
  return null;
}

/** Narrow the corpus to a filter's target. An absent filter shows everything. */
export function applyFilter(videos: Video[], filter: TreeFilter | null): Video[] {
  if (!filter) return videos;
  if (filter.channelId) return videos.filter((video) => video.channel_id === filter.channelId);
  if (filter.videoIds) {
    const wanted = new Set(filter.videoIds);
    return videos.filter((video) => wanted.has(video.video_id));
  }
  return videos;
}

/** How many videos carry a stored transcript summary. */
export function summarisedCount(videos: Video[]): number {
  return videos.filter((video) => Boolean(video.summary)).length;
}

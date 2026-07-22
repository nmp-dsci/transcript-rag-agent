/**
 * Test fixtures for the pipeline view.
 *
 * Only imported by *.test.* files. Kept out of the test files themselves so the
 * wire types have a single place to be satisfied when the API contract grows.
 */

import type { Corpus, CorpusInsight, GraphNode, IndexResult, Video } from '../api/types';

const EMPTY_VIDEO: Omit<Video, 'video_id'> = {
  title: null,
  channel_name: null,
  channel_id: null,
  thumbnail_url: null,
  source_url: null,
  duration_seconds: null,
  upload_date: null,
  view_count: null,
  summary: null,
  fetched_at: null,
  chunk_count: 0,
};

/**
 * Overrides are merged key by key rather than spread: `Partial<Video>` widens
 * every field with `undefined`, which the strict wire types reject.
 */
export function video(overrides: Partial<Video> & { video_id: string }): Video {
  const built: Video = { ...EMPTY_VIDEO, video_id: overrides.video_id };
  for (const [key, value] of Object.entries(overrides)) {
    if (value !== undefined) (built as unknown as Record<string, unknown>)[key] = value;
  }
  return built;
}

const EMPTY_NODE: Omit<GraphNode, 'id'> = {
  video_id: 'v',
  chunk_index: 0,
  channel_id: null,
  channel_name: null,
  title: null,
  preview: '',
  start_seconds: null,
  end_seconds: null,
  source_url: null,
  degree: 0,
  x: 0,
  y: 0,
};

export function graphNode(overrides: Partial<GraphNode> & { id: string }): GraphNode {
  const built: GraphNode = { ...EMPTY_NODE, id: overrides.id };
  for (const [key, value] of Object.entries(overrides)) {
    if (value !== undefined) (built as unknown as Record<string, unknown>)[key] = value;
  }
  return built;
}

export function corpus(videos: Video[], insights: CorpusInsight[] = []): Corpus {
  const channels = new Map<string, { name: string; ids: string[]; chunks: number }>();
  for (const item of videos) {
    const id = item.channel_id ?? '';
    const entry = channels.get(id) ?? { name: item.channel_name ?? 'Unknown', ids: [], chunks: 0 };
    entry.ids.push(item.video_id);
    entry.chunks += item.chunk_count;
    channels.set(id, entry);
  }
  return {
    videos,
    insights,
    channels: [...channels.entries()].map(([channel_id, entry]) => ({
      channel_id,
      channel_name: entry.name,
      video_count: entry.ids.length,
      chunk_count: entry.chunks,
      video_ids: entry.ids,
    })),
    totals: {
      videos: videos.length,
      chunks: videos.reduce((total, item) => total + item.chunk_count, 0),
      channels: channels.size,
    },
  };
}

export function indexResult(overrides: Partial<IndexResult> = {}): IndexResult {
  return {
    ok: true,
    target: 'https://youtu.be/new',
    added_videos: [],
    added_video_count: 0,
    added_chunk_count: 0,
    totals: { videos: 1, chunks: 10, channels: 1 },
    insights: [],
    channels: [],
    ...overrides,
  };
}

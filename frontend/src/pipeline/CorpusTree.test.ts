import { describe, expect, it } from 'vitest';

import type { Video } from '../api/types';
import { groupByChannel } from './CorpusTree';
import { video } from './fixtures';

const VIDEOS: Video[] = [
  video({ video_id: 'a', channel_name: 'Alpha', view_count: 100, chunk_count: 10, upload_date: '2026-01-01' }),
  video({ video_id: 'b', channel_name: 'Alpha', view_count: 900, chunk_count: 5, upload_date: '2026-05-01' }),
  video({ video_id: 'c', channel_name: 'Beta', view_count: 300, chunk_count: 40, upload_date: '2026-03-01' }),
];

describe('groupByChannel', () => {
  it('groups videos under their channel', () => {
    const channels = groupByChannel(VIDEOS, 'title');
    expect(channels.map((channel) => channel.name)).toEqual(['Alpha', 'Beta']);
    expect(channels[0]?.videos.map((v) => v.video_id)).toEqual(['a', 'b']);
  });

  it('totals chunks and views per channel', () => {
    const [alpha] = groupByChannel(VIDEOS, 'title');
    expect(alpha?.chunkTotal).toBe(15);
    expect(alpha?.viewTotal).toBe(1000);
  });

  it('orders channels and videos by view count for "top"', () => {
    const channels = groupByChannel(VIDEOS, 'views');
    expect(channels.map((channel) => channel.name)).toEqual(['Alpha', 'Beta']);
    expect(channels[0]?.videos.map((v) => v.video_id)).toEqual(['b', 'a']);
  });

  it('orders by chunk count', () => {
    const channels = groupByChannel(VIDEOS, 'chunks');
    expect(channels.map((channel) => channel.name)).toEqual(['Beta', 'Alpha']);
  });

  it('orders videos by upload date when sorting by recency', () => {
    const channels = groupByChannel(VIDEOS, 'recent');
    const alpha = channels.find((channel) => channel.name === 'Alpha');
    expect(alpha?.videos.map((v) => v.video_id)).toEqual(['b', 'a']);
  });

  it('buckets videos with no channel under a placeholder', () => {
    const channels = groupByChannel([video({ video_id: 'x' })], 'title');
    expect(channels[0]?.name).toBe('Unknown channel');
  });

  it('handles an empty corpus', () => {
    expect(groupByChannel([], 'views')).toEqual([]);
  });
});

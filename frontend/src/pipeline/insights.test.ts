import { describe, expect, it } from 'vitest';

import type { CorpusInsight } from '../api/types';
import { video } from './fixtures';
import {
  applyFilter,
  insightBadgeClass,
  insightFilter,
  insightKey,
  summarisedCount,
} from './insights';

const VIDEOS = [
  video({ video_id: 'a', channel_id: 'ch1', summary: 'done' }),
  video({ video_id: 'b', channel_id: 'ch1' }),
  video({ video_id: 'c', channel_id: 'ch2', summary: 'done' }),
];

describe('insightBadgeClass', () => {
  it('maps each level onto an existing badge variant', () => {
    expect(insightBadgeClass('info')).toBe('acc');
    expect(insightBadgeClass('warn')).toBe('warn');
    expect(insightBadgeClass('bad')).toBe('bad');
  });
});

describe('insightFilter', () => {
  it('turns a channel skew into a channel filter', () => {
    const insight: CorpusInsight = {
      kind: 'channel_skew',
      level: 'warn',
      message: 'ch1 is 80% of the corpus',
      channel_id: 'ch1',
    };
    expect(insightFilter(insight, 0)).toEqual({
      key: 'channel_skew:ch1',
      label: 'channel',
      channelId: 'ch1',
    });
  });

  it('turns missing summaries into a video filter', () => {
    const insight: CorpusInsight = {
      kind: 'missing_summaries',
      level: 'info',
      message: '1 video has no summary',
      video_ids: ['b'],
    };
    expect(insightFilter(insight, 2)).toEqual({
      key: 'missing_summaries:2',
      label: 'missing summaries · 1',
      videoIds: ['b'],
    });
  });

  it('leaves a corpus-wide insight inert', () => {
    const insight: CorpusInsight = {
      kind: 'size_spread',
      level: 'info',
      message: 'chunk counts vary widely',
    };
    expect(insightFilter(insight, 0)).toBeNull();
  });

  it('keeps keys stable across insights of the same kind', () => {
    const first: CorpusInsight = { kind: 'unindexed', level: 'bad', message: 'x', video_ids: ['a'] };
    const second: CorpusInsight = { kind: 'unindexed', level: 'bad', message: 'y', video_ids: ['b'] };
    expect(insightKey(first, 0)).not.toBe(insightKey(second, 1));
  });
});

describe('applyFilter', () => {
  it('shows everything when no filter is set', () => {
    expect(applyFilter(VIDEOS, null)).toHaveLength(3);
  });

  it('narrows to a channel', () => {
    const filtered = applyFilter(VIDEOS, { key: 'k', label: 'channel', channelId: 'ch1' });
    expect(filtered.map((item) => item.video_id)).toEqual(['a', 'b']);
  });

  it('narrows to named videos', () => {
    const filtered = applyFilter(VIDEOS, { key: 'k', label: 'unindexed', videoIds: ['c', 'zz'] });
    expect(filtered.map((item) => item.video_id)).toEqual(['c']);
  });

  it('returns nothing when the named videos have gone', () => {
    expect(applyFilter(VIDEOS, { key: 'k', label: 'x', videoIds: ['gone'] })).toEqual([]);
  });
});

describe('summarisedCount', () => {
  it('counts videos carrying a stored summary', () => {
    expect(summarisedCount(VIDEOS)).toBe(2);
    expect(summarisedCount([])).toBe(0);
  });
});

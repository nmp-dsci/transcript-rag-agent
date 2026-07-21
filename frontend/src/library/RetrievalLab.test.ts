import { describe, expect, it } from 'vitest';

import type { RankRow } from '../api/types';
import { movement } from './RetrievalLab';

function row(rank: number, otherRank: number | null): RankRow {
  return {
    chunk_id: 'v:1',
    video_id: 'v',
    chunk_index: 1,
    rank,
    score: 1,
    preview: 'text',
    start_seconds: null,
    end_seconds: null,
    source_url: null,
    other_rank: otherRank,
  };
}

describe('movement', () => {
  it('flags chunks only one mode found', () => {
    expect(movement(row(1, null))).toEqual({ className: 'only', label: 'only here' });
  });

  it('shows a rise when the other mode ranked it lower', () => {
    expect(movement(row(1, 3))).toEqual({ className: 'up', label: '↑2' });
  });

  it('shows a fall when the other mode ranked it higher', () => {
    expect(movement(row(4, 2))).toEqual({ className: 'dn', label: '↓2' });
  });

  it('marks agreement when both modes rank it the same', () => {
    expect(movement(row(2, 2))).toEqual({ className: 'same', label: '=' });
  });
});

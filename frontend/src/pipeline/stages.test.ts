import { describe, expect, it } from 'vitest';

import { LOG_LIMIT, STAGES, appendLog, stageStatuses } from './stages';

describe('stageStatuses', () => {
  it('leaves every stage pending before the first event', () => {
    const statuses = stageStatuses(null);
    expect(Object.values(statuses).every((status) => status === 'pending')).toBe(true);
  });

  it('marks earlier stages done and later ones pending', () => {
    expect(stageStatuses('chunk')).toEqual({
      discover: 'done',
      fetch: 'done',
      chunk: 'active',
      embed: 'pending',
      summarize: 'pending',
    });
  });

  it('rewinds when a channel run starts the next video', () => {
    expect(stageStatuses('fetch').embed).toBe('pending');
    expect(stageStatuses('fetch').fetch).toBe('active');
  });

  it('shows the whole sequence done once the run finishes', () => {
    const statuses = stageStatuses('embed', true);
    expect(Object.values(statuses).every((status) => status === 'done')).toBe(true);
  });

  it('covers every stage the API can emit', () => {
    expect(STAGES.map((stage) => stage.name)).toEqual([
      'discover',
      'fetch',
      'chunk',
      'embed',
      'summarize',
    ]);
  });
});

describe('appendLog', () => {
  it('appends in arrival order', () => {
    expect(appendLog(['a'], 'b')).toEqual(['a', 'b']);
  });

  it('drops the oldest lines past the cap', () => {
    const full = Array.from({ length: LOG_LIMIT }, (_, index) => `line ${index}`);
    const next = appendLog(full, 'newest');
    expect(next).toHaveLength(LOG_LIMIT);
    expect(next[0]).toBe('line 1');
    expect(next[next.length - 1]).toBe('newest');
  });
});

import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import type { GuideJob } from '../api/types';
import { ResearchMap, clusterMembership, deriveResearch, stageLabel } from './ResearchMap';

export function job(overrides: Partial<GuideJob> = {}): GuideJob {
  return {
    id: 'j1',
    kind: 'write',
    slug: 'llm-as-a-judge',
    topic: 'LLM-as-a-judge',
    title: 'LLM-as-a-Judge',
    video_ids: ['a', 'b', 'c'],
    allow_web: false,
    comment_ids: [],
    status: 'running',
    stage: 'extract',
    stage_order: ['scope', 'export', 'extract', 'compose', 'verify', 'publish'],
    stages: {
      scope: { status: 'done', message: '3 videos' },
      export: { status: 'done', message: '30 chunks' },
      extract: { status: 'progress', message: 'cluster-1: reading' },
    },
    message: 'cluster-1: reading',
    counters: { chunk_count: 30, clusters: 2, tool_calls: 3, files_read: 2, retrieval_queries: 1 },
    clusters: {
      videos: {
        a: { video_id: 'a', title: 'Judge 101', channel_name: 'A', chunk_count: 10 },
        b: { video_id: 'b', title: 'Pairwise', channel_name: 'A', chunk_count: 12 },
        c: { video_id: 'c', title: 'Calibration', channel_name: 'B', chunk_count: 8 },
      },
      'cluster-1': { status: 'done', claims: 21 },
      'cluster-2': { status: 'reading', claims: 0 },
    },
    activity: [
      { at: '2026-09-14T11:18:52+00:00', label: 'extract:cluster-1', name: 'Read', message: 'Read corpus/a.md' },
      { at: '2026-09-14T11:18:53+00:00', label: 'extract:cluster-1', name: 'Read', message: 'Read corpus/b.md' },
      { at: '2026-09-14T11:18:54+00:00', label: 'extract:cluster-2', name: 'mcp__corpus__retrieve_chunks', message: 'q', question: 'how to calibrate?', results: 4 },
    ],
    gaps: [],
    version: null,
    cite_valid: null,
    cite_total: null,
    error: null,
    started_at: '2026-09-14T11:18:43+00:00',
    finished_at: null,
    ...overrides,
  };
}

describe('deriveResearch', () => {
  it('reads clusters, membership and counters from the snapshot alone', () => {
    const j = job();
    const membership = clusterMembership(j);
    expect(membership).toEqual({ 'cluster-1': ['a', 'b'] });
    const view = deriveResearch(j, membership);
    expect(view.clusters.map((c) => c.name)).toEqual(['cluster-1', 'cluster-2']);
    expect([...(view.clusters[0]?.read ?? [])]).toEqual(['a', 'b']);
    expect(view.chunksRead).toBe(22);
    expect(view.chunksTotal).toBe(30);
    expect(view.claims).toBe(21);
    expect(view.retrievalQueries).toBe(1);
    expect(view.webFetches).toBe(0);
  });

  it('groups exported videos before any cluster reports', () => {
    const j = job({ clusters: { videos: job().clusters.videos ?? {} }, counters: { chunk_count: 30 }, activity: [] });
    const view = deriveResearch(j, {});
    expect(view.clusters).toHaveLength(1);
    expect(view.clusters[0]?.name).toBe('all');
    expect(view.clusters[0]?.videos).toHaveLength(3);
    expect(view.chunksRead).toBe(0);
  });

  it('labels stages by status', () => {
    expect(stageLabel('export', { status: 'done' })).toBe('✓ export');
    expect(stageLabel('extract', { status: 'progress' })).toBe('● extract');
    expect(stageLabel('verify', { status: 'error' })).toBe('✕ verify');
    expect(stageLabel('publish')).toBe('publish');
  });
});

describe('ResearchMap', () => {
  it('renders the stage rail, the map, the counters and the activity log', () => {
    render(<ResearchMap job={job()} />);
    expect(screen.getByRole('heading', { level: 2, name: 'LLM-as-a-Judge' })).toBeInTheDocument();
    expect(screen.getByText('✓ export')).toBeInTheDocument();
    expect(screen.getByText('● extract')).toBeInTheDocument();
    expect(screen.getByText('22 / 30')).toBeInTheDocument();
    expect(screen.getByText('chunks read in full')).toBeInTheDocument();
    expect(screen.getByText('21 claims')).toBeInTheDocument();
    expect(screen.getByText('reading')).toBeInTheDocument();
    expect(screen.getByText('retrieve_chunks "how to calibrate?"')).toBeInTheDocument();
    expect(screen.getByText('→ 4 chunks')).toBeInTheDocument();
    expect(screen.getByText(/corpus only/)).toBeInTheDocument();
  });

  it('says so when the run failed or published', () => {
    const { rerender } = render(<ResearchMap job={job({ status: 'error', error: 'extract:cluster-2: rate_limit' })} />);
    expect(screen.getByText(/failed: extract:cluster-2: rate_limit/)).toBeInTheDocument();
    rerender(<ResearchMap job={job({ status: 'done', version: 1, cite_valid: 40, cite_total: 41 })} />);
    expect(screen.getByText(/published v1/)).toBeInTheDocument();
    expect(screen.getByText('cites 40/41 resolve')).toBeInTheDocument();
  });
});

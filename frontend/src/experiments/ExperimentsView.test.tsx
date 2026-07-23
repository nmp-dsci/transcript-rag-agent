import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import type { AblationRun, Experiments, GoldenRunSummary } from '../api/types';
import { ExperimentsView } from './ExperimentsView';

const experiments = vi.fn();
vi.mock('../api/client', () => ({ api: { experiments: () => experiments() } }));

function ablation(overrides: Partial<AblationRun> = {}): AblationRun {
  return {
    run_id: 'ablation-20260722-093000',
    created_at: '2026-07-22T09:30:00+00:00',
    entries: 9,
    metrics: ['context_recall', 'recall@3', 'ndcg@10'],
    baseline: 'semantic',
    cells: [
      {
        label: 'semantic',
        config: { label: 'semantic', retrieval_mode: 'semantic', rerank: false, neighbor_span: 0, top_k: 10 },
        averages: { context_recall: 0.628, 'recall@3': 0.263, 'ndcg@10': 0.526 },
        by_domain: {
          property: { context_recall: 0.7, 'recall@3': 0.3, 'ndcg@10': 0.55 },
          'ai-coding': { context_recall: 0.5, 'recall@3': 0.2, 'ndcg@10': 0.48 },
        },
      },
      {
        label: 'hybrid',
        config: { label: 'hybrid', retrieval_mode: 'hybrid', rerank: false, neighbor_span: 0, top_k: 10 },
        averages: { context_recall: 0.612, 'recall@3': 0.39, 'ndcg@10': 0.554 },
        by_domain: {
          property: { context_recall: 0.65, 'recall@3': 0.45, 'ndcg@10': 0.58 },
          'ai-coding': { context_recall: 0.55, 'recall@3': 0.32, 'ndcg@10': 0.5 },
        },
      },
    ],
    deltas: [
      { label: 'hybrid', vs_baseline: { context_recall: -0.016, 'recall@3': 0.127, 'ndcg@10': 0.028 } },
    ],
    ...overrides,
  };
}

function goldenRun(overrides: Partial<GoldenRunSummary> = {}): GoldenRunSummary {
  return {
    run_id: 'eval-20260722-101500',
    created_at: '2026-07-22T10:15:00+00:00',
    setup: 'rag_llm',
    config: { retrieval_mode: 'hybrid', rerank_enabled: true, judge_model: 'deepseek-v4-flash' },
    summary: {
      entries: 9,
      scored: 9,
      failed: 0,
      averages: { context_recall: 0.61, faithfulness: 0.82, answer_relevancy: 0.9, composite: 0.75 },
    },
    ...overrides,
  };
}

function data(overrides: Partial<Experiments> = {}): Experiments {
  return { ablations: [ablation()], golden_runs: [goldenRun()], ...overrides };
}

describe('ExperimentsView', () => {
  beforeEach(() => experiments.mockReset());

  it('renders the ablation table with configs and metrics', async () => {
    experiments.mockResolvedValue(data());
    render(<ExperimentsView />);

    const table = await screen.findByRole('table');
    expect(within(table).getByText('semantic')).toBeInTheDocument();
    expect(within(table).getByText('hybrid')).toBeInTheDocument();
    expect(within(table).getByText('recall@3')).toBeInTheDocument();
    expect(within(table).getByText('0.390')).toBeInTheDocument();
  });

  it('highlights the best value in each metric column', async () => {
    experiments.mockResolvedValue(data());
    render(<ExperimentsView />);

    // hybrid wins recall@3 (0.390 > 0.263), so that cell is marked best.
    const best = await screen.findByText('0.390');
    expect(best).toHaveClass('best');
    // semantic wins context_recall (0.628 > 0.612).
    expect(screen.getByText('0.628')).toHaveClass('best');
    expect(screen.getByText('0.612')).not.toHaveClass('best');
  });

  it('shows deltas versus the baseline', async () => {
    experiments.mockResolvedValue(data());
    render(<ExperimentsView />);

    await screen.findByRole('table');
    expect(screen.getByText(/deltas vs semantic/i)).toBeInTheDocument();
    expect(screen.getByText('recall@3 +0.127')).toBeInTheDocument();
  });

  it('switches the table to a per-domain view', async () => {
    experiments.mockResolvedValue(data());
    render(<ExperimentsView />);
    await screen.findByRole('table');

    await userEvent.click(screen.getByRole('tab', { name: 'ai-coding' }));

    // ai-coding recall@3 for hybrid is 0.320, only visible in the domain view.
    await waitFor(() => expect(screen.getByText('0.320')).toBeInTheDocument());
  });

  it('lists end-to-end golden runs with their headline metrics', async () => {
    experiments.mockResolvedValue(data());
    render(<ExperimentsView />);

    expect(await screen.findByText('End-to-end golden runs')).toBeInTheDocument();
    expect(screen.getByText('+rerank')).toBeInTheDocument();
    expect(screen.getByText('judge deepseek-v4-flash')).toBeInTheDocument();
    expect(screen.getByText('0.820')).toBeInTheDocument(); // faithfulness
  });

  it('shows an empty state when nothing is committed', async () => {
    experiments.mockResolvedValue({ ablations: [], golden_runs: [] });
    render(<ExperimentsView />);

    expect(await screen.findByText(/No committed experiment runs yet/i)).toBeInTheDocument();
  });
});

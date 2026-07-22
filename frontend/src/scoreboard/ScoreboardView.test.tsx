import { render, screen, waitFor, within } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import type { Scoreboard, ScoreboardRow } from '../api/types';
import { ScoreboardView } from './ScoreboardView';

const scoreboard = vi.fn();
vi.mock('../api/client', () => ({ api: { scoreboard: (...args: unknown[]) => scoreboard(...args) } }));

function row(overrides: Partial<ScoreboardRow> = {}): ScoreboardRow {
  return {
    key: 'rag_llm',
    title: 'rag_llm (single-hop)',
    model: 'deepseek-v4',
    legacy: false,
    answers: 8,
    judged: 8,
    avg_scores: { faithfulness: 0.7, answer_relevancy: 0.6, context_precision: 0.55 },
    avg_composite: 0.62,
    wins: 5,
    contests: 8,
    win_rate: 0.625,
    avg_latency_seconds: 4,
    avg_token_estimate: 3000,
    ...overrides,
  };
}

function board(setups: ScoreboardRow[]): Scoreboard {
  return {
    setups,
    entries_total: 10,
    entries_judged: 9,
    group_by: 'setup_model',
    judge_model: 'deepseek-v4',
    provenance: {
      judge_models: ['deepseek-v4'],
      ragas_versions: ['0.4.3'],
      embedding_models: ['all-MiniLM-L6-v2'],
      last_judged: '2026-07-21T00:00:00+00:00',
      metrics: ['faithfulness', 'answer_relevancy', 'context_precision'],
      composite: 'mean of the three metrics',
    },
  };
}

describe('ScoreboardView', () => {
  beforeEach(() => {
    scoreboard.mockReset();
  });

  it('shows n for every aggregated row', async () => {
    scoreboard.mockResolvedValue(board([row()]));
    render(<ScoreboardView />);
    const table = await screen.findByRole('table');
    expect(within(table).getByText('n=8')).toBeInTheDocument();
    expect(within(table).getByText('of 8 answers')).toBeInTheDocument();
  });

  it('de-emphasises a row averaged over too few judged questions', async () => {
    scoreboard.mockResolvedValue(board([row({ judged: 3, answers: 3 })]));
    const { container } = render(<ScoreboardView />);
    const table = await screen.findByRole('table');
    expect(within(table).getByText('n=3')).toBeInTheDocument();
    expect(container.querySelectorAll('tr.lown')).toHaveLength(1);
    expect(within(table).getByText('thin')).toBeInTheDocument();
  });

  it('leaves a well-evidenced row at full strength', async () => {
    scoreboard.mockResolvedValue(board([row({ judged: 12, answers: 12 })]));
    const { container } = render(<ScoreboardView />);
    const table = await screen.findByRole('table');
    expect(within(table).getByText('n=12')).toBeInTheDocument();
    expect(container.querySelectorAll('tr.lown')).toHaveLength(0);
    expect(within(table).queryByText('thin')).not.toBeInTheDocument();
  });

  it('marks a win rate decided by only a handful of contests', async () => {
    scoreboard.mockResolvedValue(board([row({ judged: 9, contests: 2, wins: 2, win_rate: 1 })]));
    render(<ScoreboardView />);
    await screen.findByText('n=9');
    expect(screen.getByText('n=2')).toBeInTheDocument();
  });

  it('surfaces the token-efficiency comparison beneath the table', async () => {
    scoreboard.mockResolvedValue(
      board([
        row(),
        row({
          key: 'rag_agent',
          title: 'rag_agent (agentic)',
          avg_composite: 0.55,
          avg_token_estimate: 19000,
        }),
      ]),
    );
    render(<ScoreboardView />);
    expect(await screen.findByText('Efficiency — composite per 1k tokens')).toBeInTheDocument();
    expect(screen.getByText('0.207')).toBeInTheDocument();
    expect(screen.getByText('0.029')).toBeInTheDocument();
  });

  it('explains the metrics even with nothing to derive from', async () => {
    scoreboard.mockResolvedValue(board([row()]));
    render(<ScoreboardView />);
    expect(await screen.findByText('What the metrics mean')).toBeInTheDocument();
    expect(screen.getByText('supported claims ÷ total claims')).toBeInTheDocument();
    expect(
      screen.getByText('average precision — mean of precision@k over the ranks judged useful'),
    ).toBeInTheDocument();
  });

  it('keeps the provenance bar and the methodology note', async () => {
    scoreboard.mockResolvedValue(board([row()]));
    const { container } = render(<ScoreboardView />);
    await waitFor(() => expect(container.querySelector('.provbar')).not.toBeNull());
    expect(container.querySelector('.board-note')).not.toBeNull();
  });
});

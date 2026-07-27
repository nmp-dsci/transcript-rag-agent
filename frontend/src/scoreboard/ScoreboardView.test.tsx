import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
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

function board(setups: ScoreboardRow[], overrides: Partial<Scoreboard> = {}): Scoreboard {
  return {
    setups,
    entries_total: 10,
    entries_judged: 9,
    group_by: 'setup_model',
    judge_model: 'deepseek-v4',
    run_id: 'matrix-20260727-015519',
    runs: [
      {
        run_id: 'matrix-20260727-015519',
        created_at: '2026-07-27T01:55:19+00:00',
        setups: ['rag_llm', 'rag_agent'],
        entry_count: 10,
        judged: true,
      },
    ],
    provenance: {
      judge_models: ['deepseek-v4'],
      ragas_versions: ['0.4.3'],
      embedding_models: ['all-MiniLM-L6-v2'],
      last_judged: '2026-07-21T00:00:00+00:00',
      metrics: ['faithfulness', 'answer_relevancy', 'context_precision'],
      composite: 'mean of the three metrics',
    },
    ...overrides,
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

  it('lists every committed matrix run in the picker', async () => {
    scoreboard.mockResolvedValue(
      board([row()], {
        runs: [
          {
            run_id: 'matrix-newer',
            created_at: '2026-07-27T00:00:00+00:00',
            setups: ['rag_llm', 'rag_agent'],
            entry_count: 20,
            judged: true,
          },
          {
            run_id: 'matrix-older',
            created_at: '2026-07-01T00:00:00+00:00',
            setups: ['rag_llm'],
            entry_count: 14,
            judged: false,
          },
        ],
      }),
    );
    render(<ScoreboardView />);
    const picker = await screen.findByLabelText('Matrix run');
    expect(picker).toBeEnabled();
    expect(screen.getByRole('option', { name: 'newest run' })).toBeInTheDocument();
    expect(
      screen.getByRole('option', { name: 'matrix-newer · 20 questions · 2 setups' }),
    ).toBeInTheDocument();
    // An unjudged run is labelled as such, so it is not mistaken for a ranking.
    expect(
      screen.getByRole('option', { name: 'matrix-older · 14 questions · 1 setup · unjudged' }),
    ).toBeInTheDocument();
  });

  it('refetches the board for the run the user picks', async () => {
    scoreboard.mockResolvedValue(board([row()]));
    render(<ScoreboardView />);
    const picker = await screen.findByLabelText('Matrix run');
    await waitFor(() =>
      expect(scoreboard).toHaveBeenCalledWith('setup_model', null, null),
    );

    await userEvent.selectOptions(picker, 'matrix-20260727-015519');
    await waitFor(() =>
      expect(scoreboard).toHaveBeenCalledWith(
        'setup_model',
        null,
        'matrix-20260727-015519',
      ),
    );
  });

  it('points at the Experiments tab when no run has been committed', async () => {
    scoreboard.mockResolvedValue(
      board([], { runs: [], run_id: null, entries_judged: 0, entries_total: 0 }),
    );
    render(<ScoreboardView />);
    expect(await screen.findByText(/No committed matrix run yet/)).toBeInTheDocument();
    expect(screen.getByLabelText('Matrix run')).toBeDisabled();
  });
});

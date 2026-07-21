import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';

import type { CorpusInsight } from '../api/types';
import { CorpusSummary } from './CorpusSummary';
import { corpus, video } from './fixtures';

const INSIGHTS: CorpusInsight[] = [
  {
    kind: 'channel_skew',
    level: 'warn',
    message: 'Alpha holds 80% of chunks',
    channel_id: 'ch1',
  },
  {
    kind: 'missing_summaries',
    level: 'bad',
    message: '1 video has no summary',
    video_ids: ['b'],
  },
  { kind: 'size_spread', level: 'info', message: 'chunk counts vary widely' },
];

const CORPUS = corpus(
  [
    video({ video_id: 'a', channel_id: 'ch1', channel_name: 'Alpha', chunk_count: 8, summary: 's' }),
    video({ video_id: 'b', channel_id: 'ch1', channel_name: 'Alpha', chunk_count: 4 }),
    video({ video_id: 'c', channel_id: 'ch2', channel_name: 'Beta', chunk_count: 3, summary: 's' }),
  ],
  INSIGHTS,
);

function renderSummary(overrides: Partial<Parameters<typeof CorpusSummary>[0]> = {}) {
  const onFilterChange = vi.fn();
  render(
    <CorpusSummary
      corpus={CORPUS}
      embeddingModel="text-embedding-3-small"
      filter={null}
      onFilterChange={onFilterChange}
      {...overrides}
    />,
  );
  return { onFilterChange };
}

describe('CorpusSummary', () => {
  it('summarises the corpus, including summary coverage and the embedding model', () => {
    renderSummary();
    expect(screen.getByText('videos').previousSibling).toHaveTextContent('3');
    expect(screen.getByText('chunks').previousSibling).toHaveTextContent('15');
    expect(screen.getByText('channels').previousSibling).toHaveTextContent('2');
    expect(screen.getByText('with summaries').previousSibling).toHaveTextContent('2/3');
    expect(screen.getByText('text-embedding-3-small')).toBeInTheDocument();
  });

  it('styles each insight with the badge variant for its level', () => {
    renderSummary();
    expect(screen.getByText(/Alpha holds 80%/).closest('.badge')).toHaveClass('warn');
    expect(screen.getByText(/no summary/).closest('.badge')).toHaveClass('bad');
    expect(screen.getByText(/vary widely/).closest('.badge')).toHaveClass('acc');
  });

  it('opens a channel filter from a channel_skew chip', async () => {
    const { onFilterChange } = renderSummary();
    await userEvent.click(screen.getByRole('button', { name: /Alpha holds 80%/ }));
    expect(onFilterChange).toHaveBeenCalledWith(
      expect.objectContaining({ channelId: 'ch1', label: 'channel' }),
    );
  });

  it('opens a video filter from a missing_summaries chip', async () => {
    const { onFilterChange } = renderSummary();
    await userEvent.click(screen.getByRole('button', { name: /no summary/ }));
    expect(onFilterChange).toHaveBeenCalledWith(
      expect.objectContaining({ videoIds: ['b'] }),
    );
  });

  it('leaves a corpus-wide insight as plain text, not a button', () => {
    renderSummary();
    expect(screen.queryByRole('button', { name: /vary widely/ })).not.toBeInTheDocument();
  });

  it('clears the filter when the active chip is clicked again', async () => {
    const filter = { key: 'channel_skew:ch1', label: 'channel', channelId: 'ch1' };
    const { onFilterChange } = renderSummary({ filter });
    const chip = screen.getByRole('button', { name: /Alpha holds 80%/ });
    expect(chip).toHaveAttribute('aria-pressed', 'true');
    await userEvent.click(chip);
    expect(onFilterChange).toHaveBeenCalledWith(null);
  });

  it('offers a standalone clear control while a filter is active', async () => {
    const filter = { key: 'k', label: 'unindexed · 1', videoIds: ['b'] };
    const { onFilterChange } = renderSummary({ filter });
    await userEvent.click(screen.getByRole('button', { name: /filtered: unindexed/ }));
    expect(onFilterChange).toHaveBeenCalledWith(null);
  });

  it('renders zeroes and no insight row for an empty corpus', () => {
    renderSummary({ corpus: null, embeddingModel: null });
    expect(screen.getByText('videos').previousSibling).toHaveTextContent('0');
    expect(screen.queryByText('corpus health')).not.toBeInTheDocument();
  });
});

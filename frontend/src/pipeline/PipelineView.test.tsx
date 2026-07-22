import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import type { CorpusInsight } from '../api/types';
import { PipelineView } from './PipelineView';
import { corpus, video } from './fixtures';

const health = vi.fn();
const chunks = vi.fn();
const chunkGraph = vi.fn();
const rank = vi.fn();

vi.mock('../api/client', () => ({
  api: {
    health: (...args: never[]) => health(...args),
    chunks: (...args: never[]) => chunks(...args),
    chunkGraph: (...args: never[]) => chunkGraph(...args),
    rank: (...args: never[]) => rank(...args),
    indexStream: () => new Promise<void>(() => undefined),
  },
}));

const INSIGHTS: CorpusInsight[] = [
  { kind: 'channel_skew', level: 'warn', message: 'Alpha holds most chunks', channel_id: 'ch1' },
  { kind: 'unindexed', level: 'bad', message: '1 video is unindexed', video_ids: ['c'] },
];

const CORPUS = corpus(
  [
    video({ video_id: 'a', channel_id: 'ch1', channel_name: 'Alpha', title: 'Alpha one', chunk_count: 8 }),
    video({ video_id: 'b', channel_id: 'ch1', channel_name: 'Alpha', title: 'Alpha two', chunk_count: 4 }),
    video({ video_id: 'c', channel_id: 'ch2', channel_name: 'Beta', title: 'Beta one', chunk_count: 0 }),
  ],
  INSIGHTS,
);

function tree(): HTMLElement {
  return screen.getByRole('navigation', { name: 'Corpus' });
}

function renderView(overrides: Partial<Parameters<typeof PipelineView>[0]> = {}) {
  const onCorpusChange = vi.fn();
  const onAskAbout = vi.fn();
  render(
    <PipelineView
      corpus={CORPUS}
      onCorpusChange={onCorpusChange}
      onAskAbout={onAskAbout}
      embeddingModel="text-embedding-3-small"
      {...overrides}
    />,
  );
  return { onCorpusChange, onAskAbout };
}

describe('PipelineView', () => {
  beforeEach(() => {
    health.mockResolvedValue({ embedding_model: 'from-health' });
    chunks.mockResolvedValue({ video_id: 'a', chunks: [], total: 0 });
    chunkGraph.mockResolvedValue({
      nodes: [],
      edges: [],
      stats: {
        nodes: 0,
        edges: 0,
        k: 6,
        min_similarity: 0.55,
        channels: 0,
        mean_similarity: 0,
        isolated_nodes: 0,
      },
    });
  });

  it('opens on the corpus summary rather than a bare tree', () => {
    renderView();
    expect(screen.getByText('corpus health')).toBeInTheDocument();
    expect(screen.getByText('text-embedding-3-small')).toBeInTheDocument();
    expect(screen.getByRole('navigation', { name: 'Corpus' })).toBeInTheDocument();
  });

  it('fetches the embedding model itself when App does not supply one', async () => {
    render(
      <PipelineView corpus={CORPUS} onCorpusChange={vi.fn()} onAskAbout={vi.fn()} />,
    );
    expect(await screen.findByText('from-health')).toBeInTheDocument();
  });

  it('narrows the tree to the videos an insight names', async () => {
    renderView();
    expect(within(tree()).getByText('Beta one')).toBeInTheDocument();

    await userEvent.click(screen.getByRole('button', { name: /1 video is unindexed/ }));

    await waitFor(() => expect(within(tree()).queryByText('Alpha one')).not.toBeInTheDocument());
    expect(within(tree()).getByText('Beta one')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /filtered:/ })).toBeInTheDocument();
    // The first affected video opens in the detail pane too.
    expect(screen.getAllByText('Beta one').length).toBeGreaterThan(1);
  });

  it('narrows the tree to a channel and clears again', async () => {
    renderView();
    await userEvent.click(screen.getByRole('button', { name: /Alpha holds most chunks/ }));
    await waitFor(() => expect(within(tree()).queryByText('Beta one')).not.toBeInTheDocument());
    expect(within(tree()).getByText('Alpha one')).toBeInTheDocument();

    await userEvent.click(screen.getByRole('button', { name: /filtered: channel/ }));
    await waitFor(() => expect(within(tree()).getByText('Beta one')).toBeInTheDocument());
  });

  it('keeps the chunk graph behind a sub-tab', async () => {
    renderView();
    expect(chunkGraph).not.toHaveBeenCalled();

    await userEvent.click(screen.getByRole('button', { name: 'Chunk graph' }));
    await waitFor(() => expect(chunkGraph).toHaveBeenCalledTimes(1));

    // Returning to the corpus and back must not rebuild the projection.
    await userEvent.click(screen.getByRole('button', { name: 'Corpus & retrieval' }));
    await userEvent.click(screen.getByRole('button', { name: 'Chunk graph' }));
    expect(chunkGraph).toHaveBeenCalledTimes(1);
  });

  it('prompts for content when the corpus is empty', () => {
    renderView({ corpus: corpus([]) });
    expect(screen.getByText('The library is empty')).toBeInTheDocument();
    expect(screen.getByText('videos').previousSibling).toHaveTextContent('0');
  });

  it('handles a corpus that has not loaded yet', () => {
    renderView({ corpus: null });
    expect(screen.getByText('The library is empty')).toBeInTheDocument();
  });
});

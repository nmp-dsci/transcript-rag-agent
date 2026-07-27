import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { RetrievalLab } from './RetrievalLab';

const rank = vi.fn();

vi.mock('../api/client', () => ({
  api: {
    rank: (...args: never[]) => rank(...args),
  },
}));

describe('RetrievalLab graph mode', () => {
  beforeEach(() => {
    rank.mockReset();
  });

  it('ranks by graph and shows matched entities', async () => {
    rank.mockResolvedValue({
      query: 'negative gearing',
      video_id: null,
      top_k: 10,
      modes: {
        graph: [
          {
            chunk_id: 'v1:2',
            video_id: 'v1',
            chunk_index: 2,
            rank: 1,
            score: 1,
            preview: 'raw chunk text',
            start_seconds: 30,
            end_seconds: 40,
            source_url: null,
            other_rank: null,
            matched_entities: ['negative gearing', 'budget'],
          },
        ],
      },
      overlap: { count: 0, of: 0, chunk_ids: [] },
    });

    render(
      <RetrievalLab
        scopeVideoId={null}
        scopeLabel="Whole corpus"
        onSelectChunk={vi.fn()}
        selectedChunk={null}
      />,
    );

    await userEvent.click(screen.getByRole('button', { name: 'Graph' }));
    await userEvent.type(screen.getByLabelText('Retrieval query'), 'negative gearing');
    await userEvent.click(screen.getByRole('button', { name: 'Rank' }));

    await waitFor(() => expect(rank).toHaveBeenCalledWith('negative gearing', ['graph'], 10, null));
    expect(await screen.findByText('2 entities')).toBeInTheDocument();
  });

  it('labels an unavailable mode and keeps the surviving modes agreement', async () => {
    rank.mockResolvedValue({
      query: 'negative gearing',
      video_id: null,
      top_k: 10,
      modes: {
        semantic: [
          {
            chunk_id: 'v1:2',
            video_id: 'v1',
            chunk_index: 2,
            rank: 1,
            score: 0.9,
            preview: 'semantic hit',
            start_seconds: null,
            end_seconds: null,
            source_url: null,
            other_rank: 1,
          },
        ],
        bm25: [
          {
            chunk_id: 'v1:2',
            video_id: 'v1',
            chunk_index: 2,
            rank: 1,
            score: 3.1,
            preview: 'keyword hit',
            start_seconds: null,
            end_seconds: null,
            source_url: null,
            other_rank: 1,
          },
        ],
        graph: [],
      },
      errors: { graph: 'Neo4j query failed — is the container up?' },
      overlap: { count: 1, of: 1, chunk_ids: ['v1:2'] },
    });

    render(
      <RetrievalLab
        scopeVideoId={null}
        scopeLabel="Whole corpus"
        onSelectChunk={vi.fn()}
        selectedChunk={null}
      />,
    );

    await userEvent.click(screen.getByRole('button', { name: 'All 3' }));
    await userEvent.type(screen.getByLabelText('Retrieval query'), 'negative gearing');
    await userEvent.click(screen.getByRole('button', { name: 'Rank' }));

    // The graph column says why it is empty rather than reading as "no matches".
    expect(await screen.findByText('unavailable')).toBeInTheDocument();
    expect(screen.getByText(/Neo4j query failed/)).toBeInTheDocument();
    expect(screen.queryByText('No matches for this query.')).not.toBeInTheDocument();
    // Semantic vs BM25 agreement survives the outage.
    expect(screen.getByText('overlap 1/1')).toBeInTheDocument();
  });
});

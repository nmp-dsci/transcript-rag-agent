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
});

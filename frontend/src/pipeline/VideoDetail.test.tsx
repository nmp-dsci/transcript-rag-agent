import { render, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import type { Chunk } from '../api/types';
import { video } from './fixtures';
import { VideoDetail } from './VideoDetail';

const chunkEnrichment = vi.fn();

vi.mock('../api/client', () => ({
  api: {
    chunkEnrichment: (...args: never[]) => chunkEnrichment(...args),
  },
}));

function chunk(overrides: Partial<Chunk> & { chunk_index: number }): Chunk {
  return {
    text: 'chunk text',
    start_seconds: 0,
    end_seconds: 10,
    start_segment_index: null,
    end_segment_index: null,
    segment_count: 0,
    source_url: null,
    ...overrides,
  };
}

const VIDEO = video({ video_id: 'v1', title: 'A video' });

describe('VideoDetail graph enrichment', () => {
  beforeEach(() => {
    chunkEnrichment.mockReset();
  });

  it('shows entities and claims for a chunk the graph extracted', async () => {
    chunkEnrichment.mockResolvedValue({
      chunks: {
        '2': {
          entities: ['negative gearing', 'budget'],
          claims: [{ id: 'c1', text: 'Gearing is grandfathered.', polarity: 'asserts' }],
        },
      },
    });

    render(
      <VideoDetail
        video={VIDEO}
        chunks={[chunk({ chunk_index: 2 })]}
        selectedChunk={null}
        onAskAbout={vi.fn()}
      />,
    );

    await waitFor(() => expect(chunkEnrichment).toHaveBeenCalledWith('v1'));
    expect(await screen.findByText('negative gearing')).toBeInTheDocument();
    expect(screen.getByText('budget')).toBeInTheDocument();
    expect(screen.getByText('Gearing is grandfathered.')).toBeInTheDocument();
  });

  it('marks a chunk the graph never extracted anything from', async () => {
    chunkEnrichment.mockResolvedValue({ chunks: {} });

    render(
      <VideoDetail
        video={VIDEO}
        chunks={[chunk({ chunk_index: 0 })]}
        selectedChunk={null}
        onAskAbout={vi.fn()}
      />,
    );

    expect(await screen.findByText('not extracted')).toBeInTheDocument();
  });

  it('refetches enrichment when the selected video changes', async () => {
    chunkEnrichment.mockResolvedValue({ chunks: {} });
    const { rerender } = render(
      <VideoDetail video={VIDEO} chunks={[]} selectedChunk={null} onAskAbout={vi.fn()} />,
    );
    await waitFor(() => expect(chunkEnrichment).toHaveBeenCalledTimes(1));

    rerender(
      <VideoDetail
        video={video({ video_id: 'v2' })}
        chunks={[]}
        selectedChunk={null}
        onAskAbout={vi.fn()}
      />,
    );
    await waitFor(() => expect(chunkEnrichment).toHaveBeenCalledTimes(2));
    expect(chunkEnrichment).toHaveBeenLastCalledWith('v2');
  });
});

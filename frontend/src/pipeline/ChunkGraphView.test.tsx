import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import type { ChunkGraph } from '../api/types';
import { ChunkGraphView } from './ChunkGraphView';
import { graphNode } from './fixtures';

const chunkGraph = vi.fn();

vi.mock('../api/client', () => ({
  api: { chunkGraph: (...args: never[]) => chunkGraph(...args) },
}));

const NODES = [
  graphNode({
    id: 'v1:0',
    video_id: 'v1',
    chunk_index: 0,
    channel_id: 'ch1',
    channel_name: 'Alpha',
    title: 'Scaling laws',
    preview: 'the loss curve keeps bending',
    start_seconds: 91,
    source_url: 'https://www.youtube.com/watch?v=v1',
    degree: 4,
    x: -0.5,
    y: 0.5,
  }),
  graphNode({ id: 'v1:1', video_id: 'v1', chunk_index: 1, channel_id: 'ch1', channel_name: 'Alpha', degree: 2, x: 0.1, y: -0.2 }),
  graphNode({ id: 'v2:0', video_id: 'v2', chunk_index: 0, channel_id: 'ch2', channel_name: 'Beta', degree: 0, x: 0.9, y: 0.9 }),
];

function graph(overrides: Partial<ChunkGraph> = {}): ChunkGraph {
  return {
    nodes: NODES,
    edges: [
      { source: 'v1:0', target: 'v1:1', similarity: 0.9 },
      { source: 'v1:1', target: 'v2:0', similarity: 0.6 },
    ],
    stats: {
      nodes: 3,
      edges: 2,
      k: 6,
      min_similarity: 0.55,
      channels: 2,
      mean_similarity: 0.75,
      isolated_nodes: 1,
    },
    ...overrides,
  };
}

const EMPTY: ChunkGraph = {
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
};

function nodeGroups(container: HTMLElement): SVGGElement[] {
  return [...container.querySelectorAll<SVGGElement>('.graph-nodes > g')];
}

describe('ChunkGraphView', () => {
  beforeEach(() => {
    chunkGraph.mockReset();
    chunkGraph.mockResolvedValue(graph());
  });

  it('builds one node and one edge per record from the API', async () => {
    const { container } = render(<ChunkGraphView />);
    await waitFor(() => expect(nodeGroups(container)).toHaveLength(3));
    expect(container.querySelectorAll('.graph-edges line')).toHaveLength(2);
  });

  it('draws stronger edges more opaquely', async () => {
    const { container } = render(<ChunkGraphView />);
    await waitFor(() => expect(container.querySelectorAll('.graph-edges line')).toHaveLength(2));
    const [strong, weak] = [...container.querySelectorAll('.graph-edges line')];
    expect(Number(strong?.getAttribute('opacity'))).toBeGreaterThan(
      Number(weak?.getAttribute('opacity')),
    );
  });

  it('sizes nodes by degree and colours them by channel', async () => {
    const { container } = render(<ChunkGraphView />);
    await waitFor(() => expect(nodeGroups(container)).toHaveLength(3));
    const visible = [...container.querySelectorAll<SVGCircleElement>('.graph-nodes > g > circle:last-of-type')];
    expect(Number(visible[0]?.getAttribute('r'))).toBeGreaterThan(
      Number(visible[2]?.getAttribute('r')),
    );
    // Alpha and Beta must not share a colour.
    expect(visible[0]?.getAttribute('fill')).not.toBe(visible[2]?.getAttribute('fill'));
  });

  it('reports the graph statistics', async () => {
    render(<ChunkGraphView />);
    expect(await screen.findByText('3 nodes')).toBeInTheDocument();
    expect(screen.getByText('2 edges')).toBeInTheDocument();
    expect(screen.getByText('1 isolated')).toBeInTheDocument();
    expect(screen.getByText('mean sim 0.750')).toBeInTheDocument();
  });

  it('lists each channel in the legend', async () => {
    render(<ChunkGraphView />);
    expect(await screen.findByText('Alpha')).toBeInTheDocument();
    expect(screen.getByText('Beta')).toBeInTheDocument();
  });

  it('dims everything outside a query neighbourhood', async () => {
    const { container } = render(<ChunkGraphView />);
    await waitFor(() => expect(nodeGroups(container)).toHaveLength(3));

    chunkGraph.mockResolvedValue(
      graph({
        query: { text: 'scaling', nearest: [{ chunk_id: 'v1:0', similarity: 0.82 }] },
      }),
    );
    await userEvent.type(screen.getByLabelText('Graph query'), 'scaling');
    await userEvent.click(screen.getByRole('button', { name: 'Trace query' }));

    await waitFor(() => expect(screen.getByText(/nearest to/)).toBeInTheDocument());
    const groups = nodeGroups(container);
    expect(groups[0]).toHaveAttribute('opacity', '1');
    expect(Number(groups[1]?.getAttribute('opacity'))).toBeLessThan(0.2);
    expect(Number(groups[2]?.getAttribute('opacity'))).toBeLessThan(0.2);
  });

  it('ranks the query neighbourhood alongside the plot', async () => {
    chunkGraph.mockResolvedValue(
      graph({ query: { text: 'scaling', nearest: [{ chunk_id: 'v1:0', similarity: 0.82 }] } }),
    );
    render(<ChunkGraphView />);
    expect(await screen.findByText('0.82')).toBeInTheDocument();
    expect(screen.getByText('the loss curve keeps bending')).toBeInTheDocument();
  });

  it('sends the query and the current controls to the API', async () => {
    render(<ChunkGraphView />);
    await waitFor(() => expect(chunkGraph).toHaveBeenCalled());
    expect(chunkGraph.mock.calls[0]?.[0]).toMatchObject({ query: null, k: 6 });

    fireEvent.change(screen.getByLabelText('Neighbours per chunk'), { target: { value: '9' } });
    await userEvent.type(screen.getByLabelText('Graph query'), 'retrieval');
    await userEvent.click(screen.getByRole('button', { name: 'Trace query' }));

    await waitFor(() => expect(chunkGraph).toHaveBeenCalledTimes(2));
    expect(chunkGraph.mock.calls[1]?.[0]).toMatchObject({ query: 'retrieval', k: 9 });
  });

  it('shows a chunk preview with a deep link when a node is picked', async () => {
    const { container } = render(<ChunkGraphView />);
    await waitFor(() => expect(nodeGroups(container)).toHaveLength(3));

    fireEvent.click(nodeGroups(container)[0]!);

    expect(await screen.findByText('the loss curve keeps bending')).toBeInTheDocument();
    expect(screen.getByText('Scaling laws')).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /open at/ })).toHaveAttribute(
      'href',
      'https://www.youtube.com/watch?v=v1&t=91s',
    );
  });

  it('previews a chunk on hover without pinning it', async () => {
    const { container } = render(<ChunkGraphView />);
    await waitFor(() => expect(nodeGroups(container)).toHaveLength(3));

    fireEvent.mouseEnter(nodeGroups(container)[0]!);
    expect(await screen.findByText('the loss curve keeps bending')).toBeInTheDocument();

    fireEvent.mouseLeave(nodeGroups(container)[0]!);
    await waitFor(() =>
      expect(screen.queryByText('the loss curve keeps bending')).not.toBeInTheDocument(),
    );
  });

  it('invites indexing when the corpus has no chunks', async () => {
    chunkGraph.mockResolvedValue(EMPTY);
    const { container } = render(<ChunkGraphView />);
    expect(await screen.findByText('Nothing to plot yet')).toBeInTheDocument();
    expect(container.querySelector('svg')).toBeNull();
  });

  it('still plots the nodes when no edge clears the threshold', async () => {
    chunkGraph.mockResolvedValue(graph({ edges: [] }));
    const { container } = render(<ChunkGraphView />);
    await waitFor(() => expect(nodeGroups(container)).toHaveLength(3));
    expect(screen.getByText(/No edges clear a similarity/)).toBeInTheDocument();
  });

  it('surfaces a failed request', async () => {
    chunkGraph.mockRejectedValue(new Error('chunk graph unavailable'));
    render(<ChunkGraphView />);
    expect(await screen.findByText('chunk graph unavailable')).toBeInTheDocument();
  });
});

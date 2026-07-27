import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';

import type { EntityDetail, KnowledgeGraph } from '../api/types';
import { KnowledgeGraphView } from './KnowledgeGraphView';

const knowledgeGraph = vi.fn();
const knowledgeGraphEntity = vi.fn();
vi.mock('../api/client', () => ({
  api: {
    knowledgeGraph: () => knowledgeGraph(),
    knowledgeGraphEntity: (id: string) => knowledgeGraphEntity(id),
  },
}));

function data(): KnowledgeGraph {
  return {
    nodes: [
      { id: 'a', name: 'Negative gearing', type: 'policy', mentions: 34, community_id: 53, x: 0.1, y: 0.2 },
      { id: 'b', name: 'Budget 2026', type: 'event', mentions: 29, community_id: 53, x: -0.3, y: 0.1 },
      { id: 'c', name: 'Agentic coding', type: 'concept', mentions: 12, community_id: 7, x: 0.5, y: -0.4 },
    ],
    edges: [{ source: 'a', target: 'b', weight: 0.8 }],
    communities: [
      { id: 53, summary: 'Property tax reform.', entity_count: 2, claim_count: 30 },
      { id: 7, summary: null, entity_count: 1, claim_count: 5 },
    ],
  };
}

function detail(overrides: Partial<EntityDetail> = {}): EntityDetail {
  return {
    entity: { id: 'a', name: 'Negative gearing', type: 'policy', aliases: ['gearing'], mentions: 34, community_id: 53 },
    claims: [
      {
        id: 'c1',
        text: 'Gearing is grandfathered for pre-budget properties.',
        entities: ['a'],
        chunk_id: 'chunk:v1:3',
        video_id: 'v1',
        source_url: 'https://www.youtube.com/watch?v=v1',
        video_title: 'Budget special',
        upload_date: '2026-06-10',
        start_seconds: 120,
        end_seconds: 180,
        polarity: 'asserts',
      },
    ],
    ...overrides,
  };
}

describe('KnowledgeGraphView', () => {
  it('shows an empty state with the index-graph command when the graph is empty', async () => {
    knowledgeGraph.mockResolvedValue({ nodes: [], edges: [], communities: [] });
    render(<KnowledgeGraphView />);
    expect(await screen.findByText(/No knowledge graph built yet/)).toBeInTheDocument();
    expect(screen.getByText('uv run python -m src.cli index-graph')).toBeInTheDocument();
  });

  it('shows an error state when the request fails', async () => {
    knowledgeGraph.mockRejectedValue(new Error('Neo4j unreachable'));
    render(<KnowledgeGraphView />);
    expect(await screen.findByText('Neo4j unreachable')).toBeInTheDocument();
  });

  it('renders entity counts and a placeholder before any selection', async () => {
    knowledgeGraph.mockResolvedValue(data());
    render(<KnowledgeGraphView />);
    expect(await screen.findByText(/3 entities · 1 relations · 2 communities/)).toBeInTheDocument();
    expect(screen.getByText(/Click an entity to see its mentions/)).toBeInTheDocument();
  });

  it('clicking a node loads and shows its claims in the detail panel', async () => {
    knowledgeGraph.mockResolvedValue(data());
    knowledgeGraphEntity.mockResolvedValue(detail());
    const { container } = render(<KnowledgeGraphView />);
    await screen.findByText(/3 entities/);

    const nodeGroups = container.querySelectorAll('.kg-node');
    expect(nodeGroups.length).toBe(3);
    await userEvent.click(nodeGroups[0]!);

    expect(knowledgeGraphEntity).toHaveBeenCalledWith('a');
    expect(await screen.findByText('Negative gearing')).toBeInTheDocument();
    expect(await screen.findByText('Property tax reform.')).toBeInTheDocument();
    expect(await screen.findByText(/Gearing is grandfathered/)).toBeInTheDocument();
    expect(screen.getByText(/watch at 2:00/)).toBeInTheDocument();
  });

  it('says so when a selected entity has no claims', async () => {
    knowledgeGraph.mockResolvedValue(data());
    knowledgeGraphEntity.mockResolvedValue(detail({ claims: [] }));
    const { container } = render(<KnowledgeGraphView />);
    await screen.findByText(/3 entities/);

    await userEvent.click(container.querySelectorAll('.kg-node')[0]!);
    expect(await screen.findByText('No claims recorded for this entity.')).toBeInTheDocument();
  });

  it('the search box filters visually without removing nodes from the DOM', async () => {
    knowledgeGraph.mockResolvedValue(data());
    render(<KnowledgeGraphView />);
    await screen.findByText(/3 entities/);

    await userEvent.type(screen.getByLabelText('Search entities'), 'gearing');
    // Filtering dims non-matches via opacity rather than unmounting them, so
    // the graph never re-renders positions on every keystroke.
    expect(document.querySelectorAll('.kg-node').length).toBe(3);
  });
});

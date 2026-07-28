import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';

import type { SystemDesign } from '../api/types';
import { SystemDesignView } from './SystemDesignView';

const systemDesign = vi.fn();
vi.mock('../api/client', () => ({ api: { systemDesign: () => systemDesign() } }));

function data(): SystemDesign {
  return {
    nodes: [
      {
        id: 'vector_rag',
        label: 'Vector RAG',
        kind: 'agent',
        x: 340,
        y: 70,
        description: 'Single-hop RAG over the corpus.',
        prompts: [
          {
            name: 'RAG_SYSTEM_PROMPT',
            system: 'vector_rag',
            role: 'system',
            template_vars: [],
            text: 'You are a YouTube transcript RAG agent.',
            module: 'src/agents/prompts.py',
          },
        ],
        config: { model: 'deepseek-v4-flash', top_k: 10, rerank_enabled: true },
        flow: [
          { label: 'Retrieve', detail: 'Semantic search — top 10 candidates.', branch: null },
          { label: 'Answer', detail: 'One LLM call over the retrieved chunks.', branch: null },
        ],
      },
      {
        id: 'graph_rag',
        label: 'GraphRAG',
        kind: 'agent',
        x: 1000,
        y: 70,
        description: 'Routes local/global/temporal.',
        prompts: [
          {
            name: 'GRAPH_ROUTER_PROMPT',
            system: 'graph_rag',
            role: 'user_template',
            template_vars: ['question'],
            text: 'Classify: "{question}"',
            module: 'src/agents/prompts.py',
          },
        ],
        config: { neo4j_uri: 'bolt://localhost:7687' },
        flow: [
          { label: 'Route', detail: 'Classify the question as local, global, or temporal.', branch: null },
          { label: 'Retrieve', detail: 'Up to 30 graph claims plus a vector retrieval.', branch: 'local' },
          { label: 'Reduce', detail: 'Read pre-built community summaries.', branch: 'global' },
          { label: 'Answer', detail: 'One LLM call narrates over the evidence.', branch: null },
        ],
      },
      {
        id: 'neo4j',
        label: 'Neo4j · knowledge graph',
        kind: 'store',
        x: 1000,
        y: 520,
        description: 'Entities, relations and claims.',
        prompts: [],
        config: { uri: 'bolt://localhost:7687', user: 'neo4j' },
        flow: [],
      },
    ],
    edges: [
      { source: 'vector_rag', target: 'graph_rag' },
      { source: 'graph_rag', target: 'neo4j' },
    ],
  };
}

describe('SystemDesignView', () => {
  it('renders every node on the graph with no selection initially', async () => {
    systemDesign.mockResolvedValue(data());
    render(<SystemDesignView />);

    expect(await screen.findByRole('button', { name: /Vector RAG/ })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /GraphRAG/ })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Neo4j/ })).toBeInTheDocument();
    expect(
      screen.getByText(/Click a node on the left to see its system prompts/),
    ).toBeInTheDocument();
  });

  it('clicking a node shows its config and prompts in the detail panel', async () => {
    systemDesign.mockResolvedValue(data());
    render(<SystemDesignView />);

    await userEvent.click(await screen.findByRole('button', { name: /Vector RAG/ }));

    expect(screen.getByText('Single-hop RAG over the corpus.')).toBeInTheDocument();
    expect(screen.getByText('deepseek-v4-flash')).toBeInTheDocument();
    expect(screen.getByText('RAG_SYSTEM_PROMPT')).toBeInTheDocument();
    expect(screen.getByText('You are a YouTube transcript RAG agent.')).toBeInTheDocument();
  });

  it('store nodes show config with no prompts section', async () => {
    systemDesign.mockResolvedValue(data());
    render(<SystemDesignView />);

    await userEvent.click(await screen.findByRole('button', { name: /Neo4j/ }));
    expect(screen.getByText('bolt://localhost:7687')).toBeInTheDocument();
    expect(screen.queryByText('RAG_SYSTEM_PROMPT')).not.toBeInTheDocument();
  });

  it('switching selection updates the panel to the newly clicked node', async () => {
    systemDesign.mockResolvedValue(data());
    render(<SystemDesignView />);

    await userEvent.click(await screen.findByRole('button', { name: /Vector RAG/ }));
    expect(screen.getByText('RAG_SYSTEM_PROMPT')).toBeInTheDocument();

    await userEvent.click(screen.getByRole('button', { name: /GraphRAG/ }));
    expect(screen.queryByText('RAG_SYSTEM_PROMPT')).not.toBeInTheDocument();
    expect(screen.getByText('GRAPH_ROUTER_PROMPT')).toBeInTheDocument();
  });

  it('highlights the template variable inside a prompt template', async () => {
    systemDesign.mockResolvedValue(data());
    render(<SystemDesignView />);

    await userEvent.click(await screen.findByRole('button', { name: /GraphRAG/ }));
    expect(screen.getByText('{question}')).toBeInTheDocument();
  });

  it('shows the flow steps for an agent node, numbered in order', async () => {
    systemDesign.mockResolvedValue(data());
    render(<SystemDesignView />);

    await userEvent.click(await screen.findByRole('button', { name: /Vector RAG/ }));
    expect(screen.getByText('How a question flows through this path')).toBeInTheDocument();
    expect(screen.getByText('Semantic search — top 10 candidates.')).toBeInTheDocument();
    expect(screen.getByText('One LLM call over the retrieved chunks.')).toBeInTheDocument();
  });

  it('groups branching flow steps under a branch header', async () => {
    systemDesign.mockResolvedValue(data());
    render(<SystemDesignView />);

    await userEvent.click(await screen.findByRole('button', { name: /GraphRAG/ }));
    expect(screen.getByText('local route')).toBeInTheDocument();
    expect(screen.getByText('global route')).toBeInTheDocument();
    expect(screen.getByText('Up to 30 graph claims plus a vector retrieval.')).toBeInTheDocument();
  });

  it('marks the step where the branches converge so it reads as shared, not temporal-only', async () => {
    systemDesign.mockResolvedValue(data());
    render(<SystemDesignView />);

    await userEvent.click(await screen.findByRole('button', { name: /GraphRAG/ }));
    const converge = screen.getByText('all routes converge');
    expect(converge).toBeInTheDocument();
    // It must sit between the last branch's step and the shared answer step.
    const steps = Array.from(document.querySelectorAll('.ds-flow > div')).map(
      (node) => node.textContent ?? '',
    );
    const convergeIndex = steps.findIndex((text) => text.includes('all routes converge'));
    expect(steps[convergeIndex]).toContain('One LLM call narrates over the evidence.');
    expect(steps[convergeIndex - 1]).toContain('Read pre-built community summaries.');
    // A flow with no branches at all gets no marker.
    await userEvent.click(screen.getByRole('button', { name: /Vector RAG/ }));
    expect(screen.queryByText('all routes converge')).not.toBeInTheDocument();
  });

  it('store nodes with no flow show no flow section', async () => {
    systemDesign.mockResolvedValue(data());
    render(<SystemDesignView />);

    await userEvent.click(await screen.findByRole('button', { name: /Neo4j/ }));
    expect(screen.queryByText('How a question flows through this path')).not.toBeInTheDocument();
  });

  it('shows an error state when the request fails', async () => {
    systemDesign.mockRejectedValue(new Error('boom'));
    render(<SystemDesignView />);
    expect(
      await screen.findByText(/Could not load the system design graph/),
    ).toBeInTheDocument();
  });
});

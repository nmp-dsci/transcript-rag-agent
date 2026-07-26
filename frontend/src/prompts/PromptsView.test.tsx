import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';

import type { Prompts } from '../api/types';
import { PromptsView } from './PromptsView';

const prompts = vi.fn();
vi.mock('../api/client', () => ({ api: { prompts: () => prompts() } }));

function data(): Prompts {
  return {
    total: 2,
    notes: ['RAGAS judge prompts live inside the ragas library.'],
    systems: [
      {
        key: 'vector_rag',
        title: 'Vector RAG — single-hop',
        description: 'One retrieval, one answer call.',
        count: 1,
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
      },
      {
        key: 'graph_rag',
        title: 'GraphRAG — knowledge graph (P4)',
        description: 'Extraction, communities, router, answers.',
        count: 1,
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
      },
    ],
  };
}

describe('PromptsView', () => {
  it('renders one collapsed group per system with counts', async () => {
    prompts.mockResolvedValue(data());
    render(<PromptsView />);

    expect(await screen.findByText('Vector RAG — single-hop')).toBeInTheDocument();
    expect(screen.getByText('GraphRAG — knowledge graph (P4)')).toBeInTheDocument();
    // Collapsed by default: prompt bodies are not in the document yet.
    expect(screen.queryByText('RAG_SYSTEM_PROMPT')).not.toBeInTheDocument();
    expect(screen.getAllByText('1 prompt')).toHaveLength(2);
  });

  it('expands a group to show prompt text with highlighted template vars', async () => {
    prompts.mockResolvedValue(data());
    render(<PromptsView />);

    await userEvent.click(await screen.findByText('GraphRAG — knowledge graph (P4)'));
    expect(screen.getByText('GRAPH_ROUTER_PROMPT')).toBeInTheDocument();
    expect(screen.getByText('user template')).toBeInTheDocument();
    // The {question} placeholder renders twice: the vars row and inside the text.
    expect(screen.getAllByText('{question}').length).toBeGreaterThanOrEqual(2);
  });

  it('shows the library link-out note', async () => {
    prompts.mockResolvedValue(data());
    render(<PromptsView />);
    expect(
      await screen.findByText(/RAGAS judge prompts live inside the ragas library/),
    ).toBeInTheDocument();
  });
});

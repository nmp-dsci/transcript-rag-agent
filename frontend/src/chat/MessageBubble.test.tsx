import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';

import type { Answer, Evaluation } from '../api/types';
import { MessageBubble } from './MessageBubble';

function evaluation(composite: number): Evaluation {
  return {
    judge: 'ragas',
    judge_model: 'deepseek-v4',
    rubric_version: 'ragas-v1',
    ragas_version: '0.4.3',
    embedding_model: 'all-MiniLM-L6-v2',
    scores: { faithfulness: composite, answer_relevancy: composite, context_precision: composite },
    composite,
    elapsed_seconds: 1,
    scored_at: '2026-07-21T00:00:00+00:00',
    error: null,
  };
}

function answer(key: string, text: string, composite?: number): Answer {
  return {
    key,
    title: `${key} (setup)`,
    command: `rag-ask --${key}`,
    answer: text,
    references: [],
    token_estimate: 100,
    chunk_count: 5,
    llm_calls: 1,
    iterations: null,
    terminated_reason: null,
    elapsed_seconds: 2,
    error: null,
    contexts: [],
    evaluation: composite == null ? null : evaluation(composite),
    model: 'deepseek-v4',
    embedding_model: 'all-MiniLM-L6-v2',
    top_k: 10,
  };
}

describe('MessageBubble', () => {
  it('renders a single answer without tabs', () => {
    render(
      <MessageBubble answers={[answer('rag_llm', 'Only answer.')]} running={[]} judging={false} remainingSetups={0} />,
    );
    expect(screen.getByText(/Only answer/)).toBeInTheDocument();
    expect(screen.queryByRole('tab')).not.toBeInTheDocument();
  });

  it('groups several agents into one bubble with tabs', () => {
    render(
      <MessageBubble
        answers={[answer('rag_llm', 'Fast answer.', 0.7), answer('rag_agent', 'Deep answer.', 0.9)]}
        running={[]}
        judging={false}
        remainingSetups={0}
      />,
    );
    expect(screen.getAllByRole('tab')).toHaveLength(2);
  });

  it('opens on the highest-scoring answer and marks it TOP', () => {
    render(
      <MessageBubble
        answers={[answer('rag_llm', 'Fast answer.', 0.7), answer('rag_agent', 'Deep answer.', 0.9)]}
        running={[]}
        judging={false}
        remainingSetups={0}
      />,
    );
    expect(screen.getByText(/Deep answer/)).toBeInTheDocument();
    expect(screen.queryByText(/Fast answer/)).not.toBeInTheDocument();
    // Marked TOP on both its tab and its column in the compare grid.
    expect(screen.getAllByText(/TOP/).length).toBeGreaterThan(0);
  });

  it('switches the visible answer when another tab is clicked', async () => {
    render(
      <MessageBubble
        answers={[answer('rag_llm', 'Fast answer.', 0.7), answer('rag_agent', 'Deep answer.', 0.9)]}
        running={[]}
        judging={false}
        remainingSetups={0}
      />,
    );
    await userEvent.click(screen.getByRole('tab', { name: /rag_llm/ }));
    expect(screen.getByText(/Fast answer/)).toBeInTheDocument();
    expect(screen.queryByText(/Deep answer/)).not.toBeInTheDocument();
  });

  it('shows a research trace while the agentic setup runs', () => {
    render(
      <MessageBubble
        answers={[]}
        running={[
          {
            key: 'rag_agent',
            title: 'rag_agent (agentic)',
            startedAt: Date.now(),
            steps: [
              {
                key: 'rag_agent',
                iteration: 1,
                event_type: 'retrieval_start',
                query: 'capital gains',
                chunk_count: null,
              },
            ],
          },
        ]}
        judging={false}
        remainingSetups={0}
      />,
    );
    expect(screen.getByText('capital gains')).toBeInTheDocument();
    expect(screen.getByText('searching…')).toBeInTheDocument();
  });

  it('reports the chunk count once a retrieval completes', () => {
    render(
      <MessageBubble
        answers={[]}
        running={[
          {
            key: 'rag_agent',
            title: 'rag_agent (agentic)',
            startedAt: Date.now(),
            steps: [
              { key: 'rag_agent', iteration: 1, event_type: 'retrieval_start', query: 'q', chunk_count: null },
              { key: 'rag_agent', iteration: 1, event_type: 'retrieval_complete', query: 'q', chunk_count: 8 },
            ],
          },
        ]}
        judging={false}
        remainingSetups={0}
      />,
    );
    expect(screen.getByText('8 chunks')).toBeInTheDocument();
  });

  it('offers to run the setups that have not answered yet', async () => {
    const onCompare = vi.fn();
    render(
      <MessageBubble
        answers={[answer('rag_llm', 'One.')]}
        running={[]}
        judging={false}
        onCompare={onCompare}
        remainingSetups={2}
      />,
    );
    await userEvent.click(screen.getByText(/Compare 2 more setups/));
    expect(onCompare).toHaveBeenCalledOnce();
  });

  it('hides the compare action when every setup has answered', () => {
    render(
      <MessageBubble
        answers={[answer('rag_llm', 'One.')]}
        running={[]}
        judging={false}
        onCompare={vi.fn()}
        remainingSetups={0}
      />,
    );
    expect(screen.queryByText(/Compare/)).not.toBeInTheDocument();
  });

  it('keeps a finished answer’s research trace available, collapsed', () => {
    render(
      <MessageBubble
        answers={[answer('rag_agent', 'Done researching.')]}
        running={[]}
        judging={false}
        remainingSetups={0}
        traces={{
          rag_agent: [
            { key: 'rag_agent', iteration: 1, event_type: 'retrieval_start', query: 'first query', chunk_count: null },
            { key: 'rag_agent', iteration: 1, event_type: 'retrieval_complete', query: 'first query', chunk_count: 9 },
          ],
        }}
      />,
    );
    expect(screen.getByText(/research trace — 1 retrieval/)).toBeInTheDocument();
    expect(screen.getByText('first query')).toBeInTheDocument();
  });

  it('shows no trace for a setup that never reported steps', () => {
    render(
      <MessageBubble
        answers={[answer('rag_llm', 'Single hop.')]}
        running={[]}
        judging={false}
        remainingSetups={0}
        traces={{ rag_agent: [] }}
      />,
    );
    expect(screen.queryByText(/research trace/)).not.toBeInTheDocument();
  });

  it('renders an errored answer without crashing', () => {
    const broken = { ...answer('rag_llm', ''), error: 'stack exploded' };
    render(<MessageBubble answers={[broken]} running={[]} judging={false} remainingSetups={0} />);
    expect(screen.getByText(/stack exploded/)).toBeInTheDocument();
  });
});

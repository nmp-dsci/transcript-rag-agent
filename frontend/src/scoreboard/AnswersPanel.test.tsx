import { fireEvent, render, screen, within } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import type { ScoreboardQuestion, ScoreboardQuestionSetup } from '../api/types';
import { AnswersPanel } from './AnswersPanel';

function setup(overrides: Partial<ScoreboardQuestionSetup> = {}): ScoreboardQuestionSetup {
  return {
    key: 'rag_llm',
    title: 'rag_llm (single-hop)',
    composite: 0.71,
    judged: true,
    error: null,
    answer: 'Negative gearing changes start July 2028.',
    model: 'deepseek-v4-flash',
    elapsed_seconds: 12.5,
    token_estimate: 3211,
    chunk_count: 10,
    ...overrides,
  };
}

const TAX_QUESTION: ScoreboardQuestion = {
  id: 'q1',
  question: 'What are the key tax changes?',
  domain: 'tax',
  question_type: 'local',
  setups: [
    setup(),
    setup({
      key: 'graph_rag',
      title: 'graph_rag (knowledge graph)',
      composite: 0.64,
      answer: 'The graph links negative gearing to the 2028 budget.',
    }),
  ],
};

const QUESTIONS: ScoreboardQuestion[] = [
  TAX_QUESTION,
  {
    id: 'q2',
    question: 'How did sentiment shift over time?',
    domain: 'market',
    question_type: 'temporal',
    setups: [setup({ answer: 'Sentiment cooled through 2026.' })],
  },
];

function open() {
  render(<AnswersPanel questions={QUESTIONS} />);
  // <details> renders its children regardless of open state, so the rows are
  // queryable without simulating the disclosure toggle.
}

describe('AnswersPanel', () => {
  it('renders one row per question x setup cell', () => {
    open();
    expect(screen.getByText('Answers (3)')).toBeInTheDocument();
    expect(screen.getByText('Negative gearing changes start July 2028.')).toBeInTheDocument();
    expect(screen.getByText('Sentiment cooled through 2026.')).toBeInTheDocument();
  });

  it('filters by setup', () => {
    open();
    fireEvent.change(screen.getByLabelText('Filter by setup'), {
      target: { value: 'graph_rag' },
    });
    expect(screen.getByText('The graph links negative gearing to the 2028 budget.')).toBeInTheDocument();
    expect(screen.queryByText('Sentiment cooled through 2026.')).not.toBeInTheDocument();
  });

  it('filters by question', () => {
    open();
    fireEvent.change(screen.getByLabelText('Filter by question'), { target: { value: 'q2' } });
    const table = screen.getByRole('table');
    expect(within(table).getAllByRole('row')).toHaveLength(2); // header + one match
    expect(screen.getByText('Sentiment cooled through 2026.')).toBeInTheDocument();
  });

  it('searches answer text', () => {
    open();
    fireEvent.change(screen.getByLabelText('Search answer text'), {
      target: { value: 'sentiment' },
    });
    expect(screen.getByText('Sentiment cooled through 2026.')).toBeInTheDocument();
    expect(screen.queryByText('Negative gearing changes start July 2028.')).not.toBeInTheDocument();
  });

  it('hides unjudged cells when judged-only is checked', () => {
    render(
      <AnswersPanel
        questions={[
          {
            ...TAX_QUESTION,
            setups: [setup(), setup({ key: 'rag_agent', judged: false, composite: null })],
          },
        ]}
      />,
    );
    expect(screen.getByText('unjudged')).toBeInTheDocument();
    fireEvent.click(screen.getByLabelText(/judged only/i));
    expect(screen.queryByText('unjudged')).not.toBeInTheDocument();
  });

  it('reports when filters match nothing', () => {
    open();
    fireEvent.change(screen.getByLabelText('Search answer text'), {
      target: { value: 'no such phrase' },
    });
    expect(screen.getByText('No answers match these filters.')).toBeInTheDocument();
  });

  it('renders nothing when the run has no cells', () => {
    const { container } = render(<AnswersPanel questions={[]} />);
    expect(container).toBeEmptyDOMElement();
  });
});

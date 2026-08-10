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

describe('AnswersPanel under a capped rubric', () => {
  const CAPPED: ScoreboardQuestion[] = [
    {
      id: 'q1',
      question: 'What are the key tax changes?',
      domain: 'tax',
      question_type: 'local',
      setups: [
        setup({
          composite: 0.5,
          rubric_version: 'depth-v2',
          cap_applied: true,
          cap_reason:
            'faithfulness 0.38 below 0.6 — depth cannot rescue an ungrounded answer',
          composite_uncapped: 0.78,
          scores: { faithfulness: 0.38, insight_depth: 0.9 },
          rationales: { insight_depth: 'It connects three creators into one conclusion.' },
        }),
      ],
    },
  ];

  it('badges a capped cell in the composite column', () => {
    render(<AnswersPanel questions={CAPPED} />);
    expect(screen.getByText('0.50')).toBeInTheDocument();
    expect(screen.getAllByText('capped').length).toBeGreaterThan(0);
  });

  it('reads out the cap reason as page text once the cell is expanded', () => {
    render(<AnswersPanel questions={CAPPED} />);
    // Collapsed, the reason is not on the page — only the badge is.
    expect(
      screen.queryByText(/depth cannot rescue an ungrounded answer/),
    ).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: 'Expand' }));

    expect(
      screen.getByText(/faithfulness 0\.38 below 0\.6 — depth cannot rescue an ungrounded answer/),
    ).toBeInTheDocument();
    // And what the cap cost, so the number is not just low but explained.
    expect(screen.getByText(/Composite capped at 0\.50 from 0\.78/)).toBeInTheDocument();
  });

  it('shows the judge reason behind each depth metric on expand', () => {
    render(<AnswersPanel questions={CAPPED} />);
    fireEvent.click(screen.getByRole('button', { name: 'Expand' }));
    expect(screen.getByText('Insight')).toBeInTheDocument();
    expect(
      screen.getByText(/It connects three creators into one conclusion\./),
    ).toBeInTheDocument();
  });

  it('leaves an uncapped cell without a capped badge or a reason block', () => {
    render(<AnswersPanel questions={QUESTIONS} />);
    expect(screen.queryByText('capped')).not.toBeInTheDocument();
  });
});

describe('AnswersPanel grounding and coverage signals', () => {
  function one(overrides: Partial<ScoreboardQuestionSetup>): ScoreboardQuestion[] {
    return [
      {
        id: 'q1',
        question: 'What are the key tax changes?',
        domain: 'tax',
        question_type: 'local',
        setups: [setup({ rubric_version: 'depth-v2', ...overrides })],
      },
    ];
  }

  it('badges an ungrounded answer the cap never touched', () => {
    // faithfulness 0.00 with a low composite: the cap changes nothing, so
    // without this the worst answer in the run carries no mark at all.
    render(
      <AnswersPanel
        questions={one({
          composite: 0.14,
          cap_applied: false,
          grounding_floor_breached: true,
          grounding_reason: 'faithfulness 0.00 below 0.6 — depth cannot rescue an ungrounded answer',
        })}
      />,
    );
    expect(screen.getAllByText('ungrounded').length).toBeGreaterThan(0);
    expect(screen.queryByText('capped')).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: 'Expand' }));
    expect(screen.getByText(/faithfulness 0\.00 below 0\.6/)).toBeInTheDocument();
    expect(screen.getByText(/scored below it on its own/)).toBeInTheDocument();
  });

  it('prefers the capped badge when the cap actually moved the number', () => {
    render(
      <AnswersPanel
        questions={one({
          composite: 0.5,
          cap_applied: true,
          cap_reason: 'faithfulness 0.38 below 0.6 — depth cannot rescue an ungrounded answer',
          grounding_floor_breached: true,
        })}
      />,
    );
    expect(screen.getAllByText('capped').length).toBeGreaterThan(0);
    expect(screen.queryByText('ungrounded')).not.toBeInTheDocument();
  });

  it('says when a cell was judged on only part of its retrieved context', () => {
    render(<AnswersPanel questions={one({ contexts_resolved: 24, contexts_expected: 36 })} />);
    fireEvent.click(screen.getByRole('button', { name: 'Expand' }));
    expect(screen.getAllByText('partial context').length).toBeGreaterThan(0);
    expect(screen.getByText(/Judged on 24 of 36 retrieved chunks/)).toBeInTheDocument();
  });

  it('badges partial context on the collapsed row, not only once expanded', () => {
    // Otherwise the four affected cells of eighty can only be found by
    // expanding every one of them.
    render(<AnswersPanel questions={one({ contexts_resolved: 24, contexts_expected: 36 })} />);
    expect(screen.getByText('partial context')).toBeInTheDocument();
  });

  it('stays quiet when every retrieved chunk resolved', () => {
    render(<AnswersPanel questions={one({ contexts_resolved: 10, contexts_expected: 10 })} />);
    fireEvent.click(screen.getByRole('button', { name: 'Expand' }));
    expect(screen.queryByText('partial context')).not.toBeInTheDocument();
  });

  it('marks a cell the rubric could not rescore, rather than showing a stale score', () => {
    render(
      <AnswersPanel
        questions={one({
          composite: null,
          judged: false,
          rejudged: false,
          rejudge_skipped_reason: 'no answer to grade for depth',
        })}
      />,
    );
    expect(screen.getByText('not rescored')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Expand' }));
    expect(screen.getByText(/excluded from every average/)).toBeInTheDocument();
  });

  it('explains a not-rescored cell that has no answer to expand', () => {
    // The only skip reason that occurs is "no answer to grade" — and a cell
    // with no answer has no Expand button, so the explanation lived in a panel
    // it could never reach.
    render(
      <AnswersPanel
        questions={one({
          answer: '',
          composite: null,
          judged: false,
          rejudged: false,
          rejudge_skipped_reason: 'no answer to grade for depth',
        })}
      />,
    );
    expect(screen.queryByRole('button', { name: 'Expand' })).not.toBeInTheDocument();
    expect(screen.getByText(/excluded from every average/)).toBeInTheDocument();
    expect(screen.getByText(/no answer to grade for depth/)).toBeInTheDocument();
  });

  it('terminates the judge reason before the sentence that follows it', () => {
    render(
      <AnswersPanel
        questions={one({
          composite: 0.14,
          grounding_floor_breached: true,
          grounding_reason: 'faithfulness 0.00 below 0.6 — depth cannot rescue an ungrounded answer',
        })}
      />,
    );
    fireEvent.click(screen.getByRole('button', { name: 'Expand' }));
    expect(
      screen.getByText(/ungrounded answer\. The cap changed nothing here/),
    ).toBeInTheDocument();
  });

  it('reports a failed depth call instead of passing the composite off as whole', () => {
    render(<AnswersPanel questions={one({ depth_error: 'judge timeout' })} />);
    fireEvent.click(screen.getByRole('button', { name: 'Expand' }));
    expect(screen.getByText('depth judging failed')).toBeInTheDocument();
    expect(screen.getByText(/grounding half only, renormalised/)).toBeInTheDocument();
  });
});

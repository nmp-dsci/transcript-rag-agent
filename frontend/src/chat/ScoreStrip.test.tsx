import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it } from 'vitest';

import type { Evaluation, EvaluationDetails } from '../api/types';
import { ScoreStrip } from './ScoreStrip';

const DETAILS: EvaluationDetails = {
  faithfulness: {
    claims: [
      { claim: 'The bill passed in March.', verdict: 1, reason: 'Chunk 2 states the March date.' },
      { claim: 'It raised rates by 4%.', verdict: 1, reason: 'Chunk 1 gives the 4% figure.' },
      { claim: 'The vote was unanimous.', verdict: 0, reason: 'No chunk mentions the vote split.' },
      { claim: 'It takes effect in 2027.', verdict: 1, reason: 'Chunk 3 names 2027.' },
    ],
    supported: 3,
    total: 4,
  },
  answer_relevancy: {
    generated_questions: ['When did the bill pass?', 'What did the bill change?'],
    noncommittal: false,
    similarities: [0.91, 0.79],
  },
  context_precision: {
    verdicts: [
      { rank: 1, verdict: 1, reason: 'Directly answers it.', chunk_preview: 'The bill passed…' },
      { rank: 2, verdict: 0, reason: 'Unrelated tangent.', chunk_preview: 'Meanwhile in sport…' },
      { rank: 3, verdict: 1, reason: 'Gives the effective date.', chunk_preview: 'From 2027…' },
    ],
    average_precision: 0.8333,
  },
};

function evaluation(overrides: Partial<Evaluation> = {}): Evaluation {
  return {
    judge: 'ragas',
    judge_model: 'deepseek-v4',
    rubric_version: 'ragas-v1',
    ragas_version: '0.4.3',
    embedding_model: 'all-MiniLM-L6-v2',
    scores: { faithfulness: 0.75, answer_relevancy: 0.85, context_precision: 0.83 },
    composite: 0.81,
    elapsed_seconds: 4,
    scored_at: '2026-07-21T00:00:00+00:00',
    error: null,
    details: DETAILS,
    ...overrides,
  };
}

function dialog() {
  return screen.getByRole('dialog');
}

describe('ScoreStrip metric breakdown', () => {
  it('opens the faithfulness breakdown with every claim and its verdict', async () => {
    render(<ScoreStrip evaluation={evaluation()} />);
    await userEvent.click(screen.getByRole('button', { name: /^Faithful 0\.75/ }));

    const panel = dialog();
    expect(panel).toHaveAccessibleName('Faithful breakdown');
    expect(within(panel).getByText('The vote was unanimous.')).toBeInTheDocument();
    expect(within(panel).getByText('No chunk mentions the vote split.')).toBeInTheDocument();
    expect(within(panel).getAllByText('supported')).toHaveLength(3);
    expect(within(panel).getByText('not supported')).toBeInTheDocument();
  });

  it('shows faithfulness arithmetic that matches supported over total', async () => {
    render(<ScoreStrip evaluation={evaluation()} />);
    await userEvent.click(screen.getByRole('button', { name: /^Faithful/ }));
    expect(within(dialog()).getByText('3 ÷ 4 = 0.75')).toBeInTheDocument();
    expect(within(dialog()).getByText(/matches the 0.75 reported/)).toBeInTheDocument();
  });

  it('marks the unsupported claim so it stands out from the rest', async () => {
    const { container } = render(<ScoreStrip evaluation={evaluation()} />);
    await userEvent.click(screen.getByRole('button', { name: /^Faithful/ }));
    expect(container.ownerDocument.querySelectorAll('.bd-claim.no')).toHaveLength(1);
    expect(container.ownerDocument.querySelectorAll('.bd-claim.ok')).toHaveLength(3);
  });

  it('shows the relevancy formula, the reconstructed questions and the cosines', async () => {
    render(<ScoreStrip evaluation={evaluation()} question="When did the bill pass and what changed?" />);
    await userEvent.click(screen.getByRole('button', { name: /^Relevant/ }));

    const panel = dialog();
    expect(within(panel).getByText('When did the bill pass and what changed?')).toBeInTheDocument();
    expect(within(panel).getByText('When did the bill pass?')).toBeInTheDocument();
    expect(within(panel).getByText('0.910')).toBeInTheDocument();
    expect(within(panel).getByText(/mean cosine.*noncommittal/i)).toBeInTheDocument();
    expect(within(panel).getByText('mean(0.910, 0.790) = 0.850 × 1 = 0.85')).toBeInTheDocument();
  });

  it('zeroes relevancy and says so when the judge flagged a noncommittal answer', async () => {
    render(
      <ScoreStrip
        evaluation={evaluation({
          scores: { faithfulness: 0.75, answer_relevancy: 0, context_precision: 0.83 },
          details: {
            ...DETAILS,
            answer_relevancy: {
              generated_questions: ['What is being asked?'],
              noncommittal: true,
              similarities: [0.88],
            },
          },
        })}
      />,
    );
    await userEvent.click(screen.getByRole('button', { name: /^Relevant/ }));
    expect(within(dialog()).getByText('noncommittal → ×0')).toBeInTheDocument();
    expect(within(dialog()).getByText('mean(0.880) = 0.880 × 0 = 0.00')).toBeInTheDocument();
  });

  it('lists context precision verdicts in rank order with their precision@k', async () => {
    render(<ScoreStrip evaluation={evaluation()} />);
    await userEvent.click(screen.getByRole('button', { name: /^Precision/ }));

    const panel = dialog();
    expect(within(panel).getByText('Meanwhile in sport…')).toBeInTheDocument();
    expect(within(panel).getByText('1/1 = 1.00')).toBeInTheDocument();
    expect(within(panel).getByText('— adds nothing')).toBeInTheDocument();
    expect(within(panel).getByText('2/3 = 0.67')).toBeInTheDocument();
    expect(within(panel).getByText('(1/1 + 2/3) ÷ 2 = 1.667 ÷ 2 = 0.83')).toBeInTheDocument();
  });

  it('falls back to the metric explainer when nothing was persisted', async () => {
    render(<ScoreStrip evaluation={evaluation({ details: null })} />);
    await userEvent.click(screen.getByRole('button', { name: /^Faithful/ }));

    const panel = dialog();
    expect(within(panel).getByText(/stored no workings/)).toBeInTheDocument();
    expect(within(panel).getByText('supported claims ÷ total claims')).toBeInTheDocument();
    expect(within(panel).queryByText(/3 ÷ 4/)).not.toBeInTheDocument();
  });

  it('falls back per metric when only one metric failed to capture', async () => {
    render(
      <ScoreStrip evaluation={evaluation({ details: { ...DETAILS, context_precision: null } })} />,
    );
    await userEvent.click(screen.getByRole('button', { name: /^Precision/ }));
    expect(within(dialog()).getByText(/stored no workings/)).toBeInTheDocument();
  });

  it('renders an evaluation with no details field at all', () => {
    const legacy = evaluation();
    delete legacy.details;
    render(<ScoreStrip evaluation={legacy} />);
    expect(screen.getByRole('button', { name: /^Faithful 0\.75/ })).toBeInTheDocument();
  });

  it('closes on Escape and returns focus to the bar that opened it', async () => {
    render(<ScoreStrip evaluation={evaluation()} />);
    const bar = screen.getByRole('button', { name: /^Faithful/ });
    await userEvent.click(bar);
    expect(screen.getByRole('dialog')).toBeInTheDocument();

    await userEvent.keyboard('{Escape}');
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
    expect(bar).toHaveFocus();
  });

  it('opens from the keyboard', async () => {
    render(<ScoreStrip evaluation={evaluation()} />);
    screen.getByRole('button', { name: /^Faithful/ }).focus();
    await userEvent.keyboard('{Enter}');
    expect(screen.getByRole('dialog')).toHaveAccessibleName('Faithful breakdown');
  });

  it('closes from the Close button', async () => {
    render(<ScoreStrip evaluation={evaluation()} />);
    await userEvent.click(screen.getByRole('button', { name: /^Faithful/ }));
    await userEvent.click(screen.getByRole('button', { name: 'Close' }));
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
  });
});

describe('ScoreStrip uncertainty signals', () => {
  it('shows no spread for a single-sample judgement', () => {
    const { container } = render(
      <ScoreStrip evaluation={evaluation({ judge_samples: 1, spread: { faithfulness: 0 } })} />,
    );
    expect(screen.queryByText(/judge samples/)).not.toBeInTheDocument();
    expect(container.querySelector('.whisk')).toBeNull();
    expect(screen.getByRole('button', { name: /^Faithful 0\.75\./ })).toBeInTheDocument();
  });

  it('shows the spread and a whisker once the judge sampled more than once', () => {
    const { container } = render(
      <ScoreStrip
        evaluation={evaluation({
          judge_samples: 3,
          spread: { faithfulness: 0.06 },
          sample_scores: { faithfulness: [0.72, 0.78, 0.75] },
        })}
      />,
    );
    expect(screen.getByText('3 judge samples')).toBeInTheDocument();
    expect(screen.getByText(/±0\.06/)).toBeInTheDocument();
    expect(container.querySelectorAll('.whisk')).toHaveLength(1);
    expect(
      screen.getByRole('button', { name: /Faithful 0\.75, spread plus or minus 0\.06 across 3 judge samples/ }),
    ).toBeInTheDocument();
  });

  it('badges a self-graded evaluation and explains what it means', () => {
    render(<ScoreStrip evaluation={evaluation({ self_graded: true })} />);
    const badge = screen.getByText('self-graded');
    expect(badge).toBeInTheDocument();
    expect(badge).toHaveAttribute('title', expect.stringContaining('self-assessment'));
  });

  it('does not badge an independently graded evaluation', () => {
    render(<ScoreStrip evaluation={evaluation({ self_graded: false })} />);
    expect(screen.queryByText('self-graded')).not.toBeInTheDocument();
  });

  it('does not badge when self-grading is unknown', () => {
    render(<ScoreStrip evaluation={evaluation({ self_graded: null })} />);
    expect(screen.queryByText('self-graded')).not.toBeInTheDocument();
  });

  it('still renders the plain states', () => {
    const { rerender } = render(<ScoreStrip evaluation={null} judging />);
    expect(screen.getByText('judging…')).toBeInTheDocument();
    rerender(<ScoreStrip evaluation={null} />);
    expect(screen.getByText('not judged')).toBeInTheDocument();
  });
});

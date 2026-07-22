import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import type { ScoreboardRow } from '../api/types';
import { EfficiencyPanel, rankByEfficiency } from './EfficiencyPanel';

function row(overrides: Partial<ScoreboardRow> = {}): ScoreboardRow {
  return {
    key: 'rag_llm',
    title: 'rag_llm (single-hop)',
    model: 'deepseek-v4',
    legacy: false,
    answers: 8,
    judged: 8,
    avg_scores: { faithfulness: 0.7, answer_relevancy: 0.6, context_precision: 0.55 },
    avg_composite: 0.62,
    wins: 5,
    contests: 8,
    win_rate: 0.625,
    avg_latency_seconds: 4,
    avg_token_estimate: 3000,
    ...overrides,
  };
}

const SINGLE_HOP = row();
const AGENTIC = row({
  key: 'rag_agent',
  title: 'rag_agent (agentic)',
  avg_composite: 0.55,
  avg_token_estimate: 19000,
  avg_latency_seconds: 31,
});

describe('rankByEfficiency', () => {
  it('divides composite by tokens per thousand', () => {
    const [best, worst] = rankByEfficiency([AGENTIC, SINGLE_HOP]);
    expect(best?.row.key).toBe('rag_llm');
    expect(best?.value).toBeCloseTo(0.2067, 4);
    expect(worst?.row.key).toBe('rag_agent');
    expect(worst?.value).toBeCloseTo(0.0289, 4);
  });

  it('drops rows that cannot be placed on the axis', () => {
    expect(
      rankByEfficiency([row({ avg_composite: null }), row({ avg_token_estimate: null })]),
    ).toHaveLength(0);
  });
});

describe('EfficiencyPanel', () => {
  it('shows composite per 1k tokens for each setup', () => {
    render(<EfficiencyPanel rows={[AGENTIC, SINGLE_HOP]} />);
    expect(screen.getByText('0.207')).toBeInTheDocument();
    expect(screen.getByText('0.029')).toBeInTheDocument();
  });

  it('ranks the best value first regardless of input order', () => {
    render(<EfficiencyPanel rows={[AGENTIC, SINGLE_HOP]} />);
    const names = screen.getAllByRole('img').map((bar) => bar.getAttribute('aria-label') ?? '');
    expect(names[0]).toMatch(/rag_llm/);
    expect(names[1]).toMatch(/rag_agent/);
  });

  it('spells out the tradeoff when the expensive setup also scores lower', () => {
    render(<EfficiencyPanel rows={[AGENTIC, SINGLE_HOP]} />);
    expect(screen.getByText(/19000 tokens per answer/)).toBeInTheDocument();
    expect(screen.getByText(/7\.1× less quality per token/)).toBeInTheDocument();
  });

  it('stays quiet about a tradeoff when the expensive setup earns its tokens', () => {
    const worthIt = row({ key: 'rag_agent', title: 'rag_agent', avg_composite: 0.9, avg_token_estimate: 19000 });
    render(<EfficiencyPanel rows={[worthIt, SINGLE_HOP]} />);
    expect(screen.queryByText(/less quality per token/)).not.toBeInTheDocument();
  });

  it('flags rows whose average rests on too few questions', () => {
    render(<EfficiencyPanel rows={[row({ judged: 2 })]} />);
    expect(screen.getByText(/n=2 \(low n\)/)).toBeInTheDocument();
  });

  it('renders nothing when no row has both numbers', () => {
    const { container } = render(<EfficiencyPanel rows={[row({ avg_composite: null })]} />);
    expect(container).toBeEmptyDOMElement();
  });
});

import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import type { ScoreboardRow } from '../api/types';
import { VerdictStrip, computeVerdict } from './VerdictStrip';

function row(over: Partial<ScoreboardRow> = {}): ScoreboardRow {
  return {
    key: 'a',
    title: 'rag_llm (single-hop)',
    model: 'deepseek-v4-flash',
    legacy: false,
    answers: 20,
    judged: 20,
    avg_scores: {},
    avg_composite: 0.79,
    wins: 8,
    contests: 20,
    win_rate: 0.4,
    avg_latency_seconds: 41,
    avg_token_estimate: 3070,
    ...over,
  };
}

describe('computeVerdict', () => {
  it('needs at least two rankable rows to reach a verdict', () => {
    expect(computeVerdict([row()])).toBeNull();
    expect(computeVerdict([])).toBeNull();
  });

  it('excludes thin rows rather than ranking on them', () => {
    // A row judged on 5 questions or fewer swings several points on one bad
    // answer. It is dimmed in the table; it must not decide the verdict.
    const thin = row({ key: 'b', title: 'thin', judged: 3, avg_composite: 0.99 });
    expect(computeVerdict([row(), thin])).toBeNull();

    const third = row({ key: 'c', title: 'third', avg_composite: 0.6 });
    const verdict = computeVerdict([row(), thin, third]);
    expect(verdict?.best.key).toBe('a');
  });

  it('reports a tie as a tie when composites match at printed precision', () => {
    // 0.789 and 0.794 both print as 0.79, so the table shows one number twice
    // and the strip must not claim a lead the reader cannot see. Which row is
    // named first follows the raw value and is not part of the contract — the
    // claim is that the two are tied.
    const other = row({ key: 'b', title: 'rag_llm (summary-filtered)', avg_composite: 0.794 });
    const verdict = computeVerdict([row({ avg_composite: 0.789 }), other]);
    expect([verdict!.best.key, ...verdict!.tiedWith.map((r) => r.key)].sort()).toEqual(['a', 'b']);
    expect(verdict?.tiedWith).toHaveLength(1);
  });

  it('does not call a tie when the table would show two different numbers', () => {
    const other = row({ key: 'b', avg_composite: 0.72 });
    const verdict = computeVerdict([row(), other]);
    expect(verdict?.tiedWith).toEqual([]);
    expect(verdict?.best.key).toBe('a');
  });

  it('flags a split decision when the leader is not the best value', () => {
    // Same composite, far more tokens: the ranking and the bill disagree.
    const thrifty = row({ key: 'b', avg_composite: 0.78, avg_token_estimate: 900 });
    const verdict = computeVerdict([row(), thrifty]);
    expect(verdict?.best.key).toBe('a');
    expect(verdict?.cheapest?.key).toBe('b');
    expect(verdict?.splitDecision).toBe(true);
  });

  it('reports no split when the leader is also the best value', () => {
    const worse = row({ key: 'b', avg_composite: 0.5, avg_token_estimate: 9000 });
    const verdict = computeVerdict([row(), worse]);
    expect(verdict?.splitDecision).toBe(false);
  });
});

describe('VerdictStrip', () => {
  it('renders nothing when there is no verdict to state', () => {
    const { container } = render(<VerdictStrip rows={[row()]} />);
    expect(container).toBeEmptyDOMElement();
  });

  it('omits the win rate when contests are thin even though judged is not', () => {
    // judged=20 clears the ranking filter, but contests=3 is separately thin
    // — the composite is trustworthy, the head-to-head record is not, so the
    // strip must not state it as fact.
    const thinContests = row({
      key: 'b',
      title: 'rag_llm (summary-filtered)',
      avg_composite: 0.9,
      wins: 3,
      contests: 3,
      win_rate: 1,
    });
    render(<VerdictStrip rows={[row(), thinContests]} />);
    const line = screen.getByRole('status');
    expect(line).toHaveTextContent('rag_llm (summary-filtered)');
    expect(line).not.toHaveTextContent('winning');
    expect(line).not.toHaveTextContent('100%');
  });

  it('omits the tie-breaker win rate when a tied row has thin contests', () => {
    const thinContests = row({
      key: 'b',
      title: 'rag_llm (summary-filtered)',
      avg_composite: 0.791,
      wins: 3,
      contests: 3,
      win_rate: 1,
    });
    render(<VerdictStrip rows={[row({ avg_composite: 0.786 }), thinContests]} />);
    const line = screen.getByRole('status');
    expect(line).toHaveTextContent('tie on composite at 0.79');
    expect(line).not.toHaveTextContent('wins more');
    expect(line).not.toHaveTextContent('100%');
  });

  it('states the leader, its win rate, and the value disagreement', () => {
    const thrifty = row({
      key: 'b',
      title: 'rag_llm (summary-filtered)',
      avg_composite: 0.78,
      avg_token_estimate: 900,
    });
    render(<VerdictStrip rows={[row(), thrifty]} />);
    const line = screen.getByRole('status');
    expect(line).toHaveTextContent('rag_llm (single-hop) · deepseek-v4-flash');
    expect(line).toHaveTextContent('0.79');
    expect(line).toHaveTextContent('winning 40% of questions (8/20)');
    expect(line).toHaveTextContent('the ranking and the bill disagree');
  });

  it('says two setups tie rather than inventing a winner', () => {
    const other = row({ key: 'b', title: 'rag_llm (summary-filtered)', avg_composite: 0.791 });
    render(<VerdictStrip rows={[row({ avg_composite: 0.786 }), other]} />);
    expect(screen.getByRole('status')).toHaveTextContent('tie on composite at 0.79');
  });

  it('separates tied setups by win rate instead of quoting one of them', () => {
    // Quoting only the leader's win rate straight after "these two tie" reads
    // as if the figure described both. On a tie it is the thing that tells
    // them apart, so it is reported as a comparison.
    const other = row({
      key: 'b',
      title: 'rag_llm (summary-filtered)',
      avg_composite: 0.791,
      wins: 12,
      win_rate: 0.6,
    });
    render(<VerdictStrip rows={[row({ avg_composite: 0.786 }), other]} />);
    const line = screen.getByRole('status');
    expect(line).toHaveTextContent('rag_llm (summary-filtered) · deepseek-v4-flash wins more');
    expect(line).toHaveTextContent('60% (12/20) vs 40% (8/20)');
  });
});

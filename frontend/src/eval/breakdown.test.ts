import { describe, expect, it } from 'vitest';

import type { Evaluation } from '../api/types';
import {
  efficiency,
  faithfulnessArithmetic,
  precisionArithmetic,
  relevancyArithmetic,
  spreadRange,
} from './breakdown';

function evaluation(overrides: Partial<Evaluation> = {}): Evaluation {
  return {
    judge: 'ragas',
    judge_model: 'deepseek-v4',
    rubric_version: 'ragas-v1',
    ragas_version: '0.4.3',
    embedding_model: 'all-MiniLM-L6-v2',
    scores: { faithfulness: 0.74, answer_relevancy: 0.9, context_precision: 0.5 },
    composite: 0.71,
    elapsed_seconds: 3,
    scored_at: '2026-07-21T00:00:00+00:00',
    error: null,
    ...overrides,
  };
}

describe('faithfulnessArithmetic', () => {
  it('divides supported claims by total', () => {
    expect(faithfulnessArithmetic(3, 4)).toBe(0.75);
  });

  it('has no score when the judge extracted no claims', () => {
    expect(faithfulnessArithmetic(0, 0)).toBeNull();
  });
});

describe('relevancyArithmetic', () => {
  it('averages the cosines when the answer commits', () => {
    const result = relevancyArithmetic([0.9, 0.8], false);
    expect(result.mean).toBeCloseTo(0.85, 6);
    expect(result.multiplier).toBe(1);
    expect(result.score).toBeCloseTo(0.85, 6);
  });

  it('zeroes the score when the judge calls the answer noncommittal', () => {
    const result = relevancyArithmetic([0.9, 0.8], true);
    expect(result.mean).toBeCloseTo(0.85, 6);
    expect(result.multiplier).toBe(0);
    expect(result.score).toBe(0);
  });
});

describe('precisionArithmetic', () => {
  it('reproduces ragas average precision', () => {
    // useful at ranks 1 and 3: (1/1 + 2/3) / 2
    const result = precisionArithmetic([1, 0, 1]);
    expect(result.usefulCount).toBe(2);
    expect(result.score).toBeCloseTo((1 + 2 / 3) / 2, 6);
  });

  it('penalises a useful chunk that ranked low', () => {
    const top = precisionArithmetic([1, 0, 0, 0, 0]);
    const bottom = precisionArithmetic([0, 0, 0, 0, 1]);
    expect(top.score).toBe(1);
    expect(bottom.score).toBeCloseTo(0.2, 6);
  });

  it('scores zero when nothing retrieved was useful', () => {
    expect(precisionArithmetic([0, 0]).score).toBe(0);
  });

  it('reports precision@k only on the useful ranks', () => {
    const { steps } = precisionArithmetic([0, 1]);
    expect(steps[0]?.precisionAtK).toBeNull();
    expect(steps[1]?.precisionAtK).toBeCloseTo(0.5, 6);
  });
});

describe('spreadRange', () => {
  it('is null for a single-sample judgement', () => {
    expect(spreadRange(evaluation({ judge_samples: 1, spread: { faithfulness: 0 } }), 'faithfulness')).toBeNull();
  });

  it('is null when no sampling metadata was recorded', () => {
    expect(spreadRange(evaluation(), 'faithfulness')).toBeNull();
  });

  it('uses the real sample scores when they are stored', () => {
    const range = spreadRange(
      evaluation({ judge_samples: 3, sample_scores: { faithfulness: [0.7, 0.8, 0.72] } }),
      'faithfulness',
    );
    expect(range?.min).toBeCloseTo(0.7, 6);
    expect(range?.max).toBeCloseTo(0.8, 6);
    expect(range?.spread).toBeCloseTo(0.1, 6);
    expect(range?.samples).toBe(3);
  });

  it('centres the recorded spread on the mean when samples were not kept', () => {
    const range = spreadRange(
      evaluation({ judge_samples: 3, spread: { faithfulness: 0.06 } }),
      'faithfulness',
    );
    expect(range?.spread).toBeCloseTo(0.06, 6);
    expect(range?.min).toBeCloseTo(0.71, 6);
    expect(range?.max).toBeCloseTo(0.77, 6);
  });
});

describe('efficiency', () => {
  it('reports composite bought per 1k tokens', () => {
    expect(efficiency(0.62, 3000)).toBeCloseTo(0.2067, 4);
    expect(efficiency(0.55, 19000)).toBeCloseTo(0.0289, 4);
  });

  it('has no value without both a composite and a token count', () => {
    expect(efficiency(null, 3000)).toBeNull();
    expect(efficiency(0.6, null)).toBeNull();
    expect(efficiency(0.6, 0)).toBeNull();
  });
});

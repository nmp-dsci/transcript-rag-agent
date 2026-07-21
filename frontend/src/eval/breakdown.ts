/**
 * The arithmetic behind each RAGAS score, reproduced from the judge's workings.
 *
 * `src/evals/judge.py` computes every score *from* the intermediates it
 * persists, so recomputing here must land on the number the strip already
 * shows. These helpers exist so the UI can display the working out, not to
 * second-guess the backend — where a stored score is present it is still what
 * gets reported; the derived value is shown beside it as the explanation.
 */

import type { Evaluation, EvaluationDetails } from '../api/types';

/** Below this many judged questions an average is a hint, not a verdict. */
export const LOW_N = 5;

export interface PrecisionStep {
  rank: number;
  verdict: 0 | 1;
  /** Useful-so-far / rank. Null on ranks the judge rejected — they add nothing. */
  precisionAtK: number | null;
  usefulSoFar: number;
}

export interface PrecisionArithmetic {
  steps: PrecisionStep[];
  usefulCount: number;
  /** Sum of precision@k over the useful ranks — the average's numerator. */
  sum: number;
  score: number;
}

/**
 * Ragas' context-precision arithmetic: mean of precision@k over useful ranks.
 *
 * A useful chunk at rank 5 with nothing useful above it contributes 1/5, not 1
 * — which is exactly why burying a good chunk costs score.
 */
export function precisionArithmetic(verdicts: readonly number[]): PrecisionArithmetic {
  const steps: PrecisionStep[] = [];
  let usefulSoFar = 0;
  let sum = 0;
  verdicts.forEach((raw, index) => {
    const verdict: 0 | 1 = raw ? 1 : 0;
    if (verdict) usefulSoFar += 1;
    const precisionAtK = verdict ? usefulSoFar / (index + 1) : null;
    if (precisionAtK != null) sum += precisionAtK;
    steps.push({ rank: index + 1, verdict, precisionAtK, usefulSoFar });
  });
  const usefulCount = steps.filter((step) => step.verdict === 1).length;
  // Ragas guards an all-rejected list with a 1e-10 denominator, which lands on
  // 0; expressing that as a plain zero keeps the displayed sum honest.
  return { steps, usefulCount, sum, score: usefulCount === 0 ? 0 : sum / usefulCount };
}

export interface RelevancyArithmetic {
  mean: number;
  /** 0 when the judge called the answer noncommittal, else 1. */
  multiplier: 0 | 1;
  score: number;
}

/** mean(cosine similarity) × (0 if noncommittal else 1). */
export function relevancyArithmetic(
  similarities: readonly number[],
  noncommittal: boolean,
): RelevancyArithmetic {
  const mean = similarities.length
    ? similarities.reduce((total, value) => total + value, 0) / similarities.length
    : 0;
  const multiplier: 0 | 1 = noncommittal ? 0 : 1;
  return { mean, multiplier, score: mean * multiplier };
}

/** supported / total, or null when the judge extracted no claims to divide by. */
export function faithfulnessArithmetic(supported: number, total: number): number | null {
  return total === 0 ? null : supported / total;
}

/** The observed range across judge samples, for drawing a whisker on a bar. */
export interface SpreadRange {
  min: number;
  max: number;
  spread: number;
  samples: number;
}

/**
 * Where the judge's samples actually landed for one metric.
 *
 * Prefers the real per-sample scores; falls back to centring the recorded
 * spread on the mean when only the width was persisted. Null whenever a single
 * sample was taken — one run has no spread to show, and inventing a whisker
 * would make it look more measured than it is.
 */
export function spreadRange(evaluation: Evaluation, metric: string): SpreadRange | null {
  const samples = evaluation.judge_samples ?? 0;
  if (samples < 2) return null;
  const values = evaluation.sample_scores?.[metric];
  if (values && values.length > 1) {
    const min = Math.min(...values);
    const max = Math.max(...values);
    return { min, max, spread: max - min, samples: values.length };
  }
  const spread = evaluation.spread?.[metric];
  const value = evaluation.scores?.[metric];
  if (spread == null || value == null || spread <= 0) return null;
  return {
    min: Math.max(0, value - spread / 2),
    max: Math.min(1, value + spread / 2),
    spread,
    samples,
  };
}

/** True when this evaluation stored workings for the given metric. */
export function hasDetails(
  details: EvaluationDetails | null | undefined,
  metric: string,
): boolean {
  if (!details) return false;
  return Boolean(details[metric as keyof EvaluationDetails]);
}

/**
 * Composite score bought per 1k tokens spent.
 *
 * The comparison the scoreboard otherwise hides: a setup burning 19k tokens for
 * a lower composite than a 3k-token setup is losing twice, and only this ratio
 * says so.
 */
export function efficiency(
  composite: number | null | undefined,
  tokens: number | null | undefined,
): number | null {
  if (composite == null || tokens == null || tokens <= 0) return null;
  return (composite * 1000) / tokens;
}

/**
 * What each RAGAS metric measures, for evaluations with no stored workings.
 *
 * Deliberately short: this is the fallback a user reads when there is no
 * derivation to show, so it has to teach the metric in a glance rather than
 * explain the framework.
 */

export interface MetricExplainerCopy {
  /** Metric key as it appears in `Evaluation.scores`. */
  name: string;
  label: string;
  measures: string;
  formula: string;
  lowMeans: string;
}

export const METRIC_EXPLAINERS: MetricExplainerCopy[] = [
  {
    name: 'faithfulness',
    label: 'Faithfulness',
    measures:
      'Whether the answer stays inside the retrieved chunks. The judge splits the answer into standalone claims and checks each one against the chunks.',
    formula: 'supported claims ÷ total claims',
    lowMeans: 'Low means the model asserted things the transcripts never said — hallucination.',
  },
  {
    name: 'answer_relevancy',
    label: 'Answer relevancy',
    measures:
      'Whether the answer addresses the question asked. The judge reads only the answer, writes the question it thinks was asked, and compares that to the real one.',
    formula: 'mean cosine(original question, generated questions) × (0 if noncommittal else 1)',
    lowMeans:
      'Low means the answer drifted off-question, or hedged so hard the judge marked it noncommittal and zeroed it.',
  },
  {
    name: 'context_precision',
    label: 'Context precision',
    measures:
      'Whether retrieval put useful chunks at the top. Each retrieved chunk gets a useful / not-useful verdict in rank order.',
    formula: 'average precision — mean of precision@k over the ranks judged useful',
    lowMeans:
      'Low means the useful chunks were buried under noise; a good chunk at rank 5 counts far less than the same chunk at rank 1.',
  },
];

export function explainerFor(name: string): MetricExplainerCopy | null {
  return METRIC_EXPLAINERS.find((item) => item.name === name) ?? null;
}

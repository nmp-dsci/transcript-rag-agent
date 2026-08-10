/**
 * What each metric measures, for evaluations with no stored workings.
 *
 * Deliberately short: this is the fallback a user reads when there is no
 * derivation to show, so it has to teach the metric in a glance rather than
 * explain the framework.
 *
 * The first three are RAGAS' and derive their score from persisted
 * intermediates. The five depth metrics are judged straight to a number with a
 * one-sentence reason, so their "formula" line says what the judge was asked
 * rather than pretending to arithmetic that does not exist.
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
  {
    name: 'insight_depth',
    label: 'Insight depth',
    measures:
      'Whether the answer synthesises or restates. The judge asks whether it connects claims across sources, explains a mechanism, or resolves a tension the chunks only imply.',
    formula: 'judged 0–1 in one pass, with a one-sentence reason',
    lowMeans:
      'Low means a paraphrase of a single passage — accurate, and worth no more than the chunk it copied.',
  },
  {
    name: 'specificity',
    label: 'Specificity',
    measures:
      'Whether the answer is checkable. Named schemes, figures, thresholds, dates and who held which position all count; generic advice does not.',
    formula: 'judged 0–1 in one pass, with a one-sentence reason',
    lowMeans:
      'Low means the answer would read the same against any transcript on the topic, so it says nothing about this corpus.',
  },
  {
    name: 'coverage',
    label: 'Coverage',
    measures:
      'How much of what the retrieved chunks actually offer on the question the answer uses, rather than how much it says.',
    formula: 'judged 0–1 in one pass, with a one-sentence reason',
    lowMeans:
      'Low means retrieval found more than the answer used — one facet reported out of several available.',
  },
  {
    name: 'evidence_breadth',
    label: 'Evidence breadth',
    measures:
      'How many distinct sources or speakers the answer really draws on, relative to how many the context offered, and whether it makes clear who said what.',
    formula: 'judged 0–1 in one pass, with a one-sentence reason',
    lowMeans:
      'Low means one creator’s view presented as the answer, which is exactly what a multi-transcript corpus is supposed to avoid.',
  },
  {
    name: 'calibration',
    label: 'Calibration',
    measures:
      'Whether confidence matches evidence: asserting what the chunks support, flagging where they disagree or are thin, and marking predictions as predictions.',
    formula: 'judged 0–1 in one pass, with a one-sentence reason',
    lowMeans:
      'Low means flat assertions the context does not carry — or hedging so total the answer commits to nothing.',
  },
];

export function explainerFor(name: string): MetricExplainerCopy | null {
  return METRIC_EXPLAINERS.find((item) => item.name === name) ?? null;
}

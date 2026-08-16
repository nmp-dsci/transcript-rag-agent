/**
 * The one registry of metric labels and titles, for every metric either rubric
 * can report.
 *
 * The Scoreboard renders whatever metrics the selected run's rubric names, so
 * the label set has to cover both: three under `ragas-v1`, eight under
 * `depth-v2`. A metric with no entry here still renders — under its raw key —
 * because dropping an unknown column would hide a score rather than a label.
 *
 * Chat's score strip reads `RAGAS_METRIC_NAMES` out of this same map rather
 * than keeping its own copy: two registries that disagreed in scope is how a
 * metric ends up labelled one way on the Chat tab and another on the
 * Scoreboard.
 */

export interface MetricMeta {
  label: string;
  title: string;
}

export const METRIC_META: Record<string, MetricMeta> = {
  faithfulness: {
    label: 'Faithful',
    title: 'Faithfulness — is the answer supported by the retrieved chunks?',
  },
  answer_relevancy: {
    label: 'Relevant',
    title: 'Answer relevancy — does the answer address the question?',
  },
  context_precision: {
    label: 'Precision',
    title: 'Context precision — were the retrieved chunks useful for the answer?',
  },
  insight_depth: {
    label: 'Insight',
    title: 'Insight depth — does the answer synthesise across sources, or restate one?',
  },
  specificity: {
    label: 'Specific',
    title: 'Specificity — named schemes, figures and dates rather than generic advice.',
  },
  coverage: {
    label: 'Coverage',
    title: 'Coverage — how much of what the context offers on this question the answer uses.',
  },
  evidence_breadth: {
    label: 'Breadth',
    title: 'Evidence breadth — how many distinct sources or speakers the answer draws on.',
  },
  calibration: {
    label: 'Calibrated',
    title: 'Calibration — does the answer’s confidence match the evidence behind it?',
  },
};

/** The three RAGAS metrics, in the order the Chat strip has always shown them. */
export const RAGAS_METRIC_NAMES = [
  'faithfulness',
  'answer_relevancy',
  'context_precision',
] as const;

export function metricLabel(name: string): string {
  return METRIC_META[name]?.label ?? name;
}

export function metricTitle(name: string): string {
  return METRIC_META[name]?.title ?? name;
}

/** "20%" — a metric's share of the composite, or null when the rubric is flat. */
export function weightLabel(weight: number | undefined): string | null {
  if (weight == null) return null;
  return `${Math.round(weight * 100)}%`;
}

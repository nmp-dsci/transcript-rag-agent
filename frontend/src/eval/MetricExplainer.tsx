import { METRIC_EXPLAINERS, type MetricExplainerCopy } from './explainers';
import { useEvalStyles } from './styles';

/** One metric, described rather than derived — the fallback when no workings exist. */
export function MetricExplainer({ copy }: { copy: MetricExplainerCopy }) {
  useEvalStyles();
  return (
    <div className="explainer">
      <h3>{copy.label}</h3>
      <p>{copy.measures}</p>
      <code className="f">{copy.formula}</code>
      <div className="low">{copy.lowMeans}</div>
    </div>
  );
}

/**
 * Metric cards for the scoreboard and unexplained evaluations.
 *
 * `only` picks a single metric; `names` narrows to the rubric a run was judged
 * under (and orders the cards the way that rubric lists them), so a ragas-v1
 * run does not advertise depth metrics it never scored.
 */
export function MetricExplainers({ only, names }: { only?: string; names?: string[] }) {
  useEvalStyles();
  const byName = new Map(METRIC_EXPLAINERS.map((copy) => [copy.name, copy]));
  const cards = only
    ? METRIC_EXPLAINERS.filter((copy) => copy.name === only)
    : names && names.length > 0
      ? names.map((name) => byName.get(name)).filter((copy) => copy != null)
      : METRIC_EXPLAINERS;
  return (
    <div className="explainers">
      {cards.map((copy) => (
        <MetricExplainer copy={copy} key={copy.name} />
      ))}
    </div>
  );
}

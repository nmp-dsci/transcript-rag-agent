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

/** All three metrics as cards, for the scoreboard and unexplained evaluations. */
export function MetricExplainers({ only }: { only?: string }) {
  useEvalStyles();
  const cards = only
    ? METRIC_EXPLAINERS.filter((copy) => copy.name === only)
    : METRIC_EXPLAINERS;
  return (
    <div className="explainers">
      {cards.map((copy) => (
        <MetricExplainer copy={copy} key={copy.name} />
      ))}
    </div>
  );
}

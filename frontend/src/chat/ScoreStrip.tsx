import type { Evaluation } from '../api/types';

export const METRICS: [string, string, string][] = [
  ['faithfulness', 'Faithful', 'Faithfulness — is the answer supported by the retrieved chunks?'],
  ['answer_relevancy', 'Relevant', 'Answer relevancy — does the answer address the question?'],
  [
    'context_precision',
    'Precision',
    'Context precision — were the retrieved chunks useful for the answer?',
  ],
];

export function fmtScore(value: number | null | undefined): string {
  return value == null ? '—' : value.toFixed(2);
}

interface Props {
  evaluation: Evaluation | null;
  judging?: boolean;
}

export function ScoreStrip({ evaluation, judging }: Props) {
  if (judging) {
    return (
      <div className="eval">
        <div className="eval-head">
          <span className="microlabel">ragas eval</span>
        </div>
        <div className="waiting">
          <span className="pulse" />
          judging…
        </div>
      </div>
    );
  }
  if (!evaluation) {
    return (
      <div className="eval">
        <div className="eval-head">
          <span className="microlabel">ragas eval</span>
          <span className="composite na">not judged</span>
        </div>
      </div>
    );
  }

  const { composite, scores } = evaluation;
  return (
    <div className="eval">
      <div className="eval-head">
        <span className="microlabel">ragas eval</span>
        <span className={`composite${composite == null ? ' na' : ''}`}>
          {composite == null ? 'no score' : composite.toFixed(2)}
        </span>
      </div>
      {METRICS.map(([name, label, title]) => {
        const value = scores?.[name];
        return (
          <div className="metric" key={name} title={title}>
            <span className="mlab">{label}</span>
            <span className="mbar">
              <i style={{ width: `${value != null ? Math.round(value * 100) : 0}%` }} />
            </span>
            <span className="mval">{fmtScore(value)}</span>
          </div>
        );
      })}
      {evaluation.error ? (
        <div className="eval-note err">{evaluation.error}</div>
      ) : (
        <div className="eval-note">
          judged by {evaluation.judge_model || 'ragas'}
          {evaluation.ragas_version ? ` · ragas ${evaluation.ragas_version}` : ''}
          {evaluation.elapsed_seconds ? ` in ${evaluation.elapsed_seconds}s` : ''}
        </div>
      )}
    </div>
  );
}

import { useState } from 'react';

import type { Evaluation } from '../api/types';
import { BreakdownDrawer } from '../eval/BreakdownDrawer';
import { spreadRange } from '../eval/breakdown';
import { useEvalStyles } from '../eval/styles';

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

function pct(value: number): number {
  return Math.max(0, Math.min(100, Math.round(value * 100)));
}

interface Props {
  evaluation: Evaluation | null;
  judging?: boolean;
  /** The question that was asked, shown beside the judge's reconstruction. */
  question?: string;
}

export function ScoreStrip({ evaluation, judging, question }: Props) {
  useEvalStyles();
  const [openMetric, setOpenMetric] = useState<string | null>(null);

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
  const samples = evaluation.judge_samples ?? 0;
  const open = METRICS.find(([name]) => name === openMetric);

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
        const range = spreadRange(evaluation, name);
        const spreadText = range ? ` ±${range.spread.toFixed(2)}` : '';
        const readout = `${label} ${fmtScore(value)}${
          range
            ? `, spread plus or minus ${range.spread.toFixed(2)} across ${range.samples} judge samples`
            : ''
        }. Show how this was scored.`;
        return (
          <button
            type="button"
            className="metric metricbtn"
            key={name}
            title={`${title} — click for the judge's workings`}
            aria-label={readout}
            aria-expanded={openMetric === name}
            aria-haspopup="dialog"
            onClick={() => setOpenMetric(name)}
          >
            <span className="mlab">{label}</span>
            <span className={`mbar${range ? ' hasw' : ''}`}>
              <i style={{ width: `${value != null ? pct(value) : 0}%` }} />
              {range ? (
                <span
                  className="whisk"
                  style={{
                    left: `${pct(range.min)}%`,
                    width: `${Math.max(2, pct(range.max) - pct(range.min))}%`,
                  }}
                />
              ) : null}
            </span>
            <span className={`mval${range ? ' wide' : ''}`}>
              {fmtScore(value)}
              {spreadText}
            </span>
          </button>
        );
      })}

      {samples > 1 || evaluation.self_graded === true ? (
        <div className="eval-flags">
          {samples > 1 ? (
            <span
              className="badge plain"
              title={`Each metric is the mean of ${samples} judge runs; ± is the spread between the highest and lowest run.`}
            >
              {samples} judge samples
            </span>
          ) : null}
          {evaluation.self_graded === true ? (
            <span
              className="badge bad"
              title={`${evaluation.judge_model} both wrote and graded this answer, so these scores are self-assessment, not an independent check. Judge with a different model before trusting a comparison.`}
            >
              self-graded
            </span>
          ) : null}
        </div>
      ) : null}

      {evaluation.error ? (
        <div className="eval-note err">{evaluation.error}</div>
      ) : (
        <div className="eval-note">
          judged by {evaluation.judge_model || 'ragas'}
          {evaluation.ragas_version ? ` · ragas ${evaluation.ragas_version}` : ''}
          {evaluation.elapsed_seconds ? ` in ${evaluation.elapsed_seconds}s` : ''}
        </div>
      )}

      {open ? (
        <BreakdownDrawer
          metric={open[0]}
          label={open[1]}
          evaluation={evaluation}
          {...(question ? { question } : {})}
          onClose={() => setOpenMetric(null)}
        />
      ) : null}
    </div>
  );
}

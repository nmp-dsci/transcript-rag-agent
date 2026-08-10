import { Fragment } from 'react';

import type { MetricGroup, ScoreboardRow } from '../api/types';
import { fmtScore } from '../chat/ScoreStrip';
import { metricLabel, metricTitle, weightLabel } from '../eval/metrics';

interface Props {
  /** The metrics the run's rubric names, in rubric order. */
  metrics: string[];
  /** Each metric's share of the composite. */
  weights: Record<string, number>;
  /** Grouping to render; empty for a flat rubric like ragas-v1. */
  groups: MetricGroup[];
  /** Which rubric produced these numbers, labelled on the panel. */
  rubricVersion: string;
  /** One column per aggregated leaderboard row. */
  rows: ScoreboardRow[];
  /** How the composite is described, in the rubric's own words. */
  composite: string;
}

/**
 * The rubric, read the other way round: one row per metric, columns per setup.
 *
 * The leaderboard answers "which setup won"; this answers "what was it scored
 * on, and how much did each part count". Under `depth-v2` that is the whole
 * argument — a metric's weight is the claim being made about what a good
 * answer is, so the weight sits next to the number it produced rather than
 * living only in a design doc. A flat rubric renders the same table without
 * group bands.
 */
export function RubricPanel({
  metrics,
  weights,
  groups,
  rubricVersion,
  rows,
  composite,
}: Props) {
  if (metrics.length === 0 || rows.length === 0) return null;

  const metricRow = (name: string) => (
    <tr key={name} className="metricrow" data-metric={name}>
      <td title={metricTitle(name)}>{metricLabel(name)}</td>
      <td className="num">
        <span className="badge plain">{weightLabel(weights[name]) ?? '—'}</span>
      </td>
      {rows.map((row) => {
        const value = row.avg_scores[name];
        return (
          <td key={`${row.key}:${row.model ?? 'legacy'}`}>
            <span className="cellbar">
              <span className="mbar">
                <i style={{ width: `${value != null ? Math.round(value * 100) : 0}%` }} />
              </span>
              <span className="num">{fmtScore(value)}</span>
            </span>
          </td>
        );
      })}
    </tr>
  );

  const grouped = groups.filter((group) => group.metrics.length > 0);
  const ungrouped = metrics.filter(
    (name) => !grouped.some((group) => group.metrics.includes(name)),
  );

  return (
    <div className="panel">
      <h2>
        Rubric <span className="badge acc">{rubricVersion}</span>
      </h2>
      <p className="sub">
        {composite}. Every metric this run was scored on, with the weight it carries in the
        composite above.
      </p>
      <div className="tblwrap">
        <table aria-label="Rubric metrics">
          <thead>
            <tr>
              <th>Metric</th>
              <th>Weight</th>
              {rows.map((row) => (
                <th key={`${row.key}:${row.model ?? 'legacy'}`}>{row.title}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {grouped.map((group) => (
              <Fragment key={`group:${group.key}`}>
                <tr className="grouprow">
                  <td colSpan={2 + rows.length}>
                    {group.label} <span className="nchip">{weightLabel(group.weight)}</span>
                  </td>
                </tr>
                {group.metrics.filter((name) => metrics.includes(name)).map(metricRow)}
              </Fragment>
            ))}
            {ungrouped.map(metricRow)}
          </tbody>
        </table>
      </div>
    </div>
  );
}

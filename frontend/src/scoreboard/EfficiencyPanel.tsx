import type { ScoreboardRow } from '../api/types';
import { LOW_N, efficiency } from '../eval/breakdown';
import { useEvalStyles } from '../eval/styles';

interface Ranked {
  row: ScoreboardRow;
  value: number;
}

/** Rows that can be placed on the axis at all, best value first. */
export function rankByEfficiency(rows: readonly ScoreboardRow[]): Ranked[] {
  return rows
    .map((row) => ({ row, value: efficiency(row.avg_composite, row.avg_token_estimate) }))
    .filter((item): item is Ranked => item.value != null)
    .sort((a, b) => b.value - a.value);
}

function label(row: ScoreboardRow): string {
  return row.model ? `${row.title} · ${row.model}` : row.title;
}

/**
 * What each setup's composite costs in tokens.
 *
 * The scoreboard reports composite and token spend in separate columns, which
 * lets a setup burning 19k tokens for a lower score than a 3k-token setup look
 * merely "slower". Dividing one by the other makes that trade the headline.
 */
export function EfficiencyPanel({ rows }: { rows: readonly ScoreboardRow[] }) {
  useEvalStyles();
  const ranked = rankByEfficiency(rows);
  if (ranked.length === 0) return null;

  const best = ranked[0] as Ranked;
  const worst = ranked[ranked.length - 1] as Ranked;
  const scale = best.value || 1;
  const spendsMore =
    (worst.row.avg_token_estimate ?? 0) > (best.row.avg_token_estimate ?? 0) &&
    (worst.row.avg_composite ?? 0) < (best.row.avg_composite ?? 0);

  return (
    <div className="panel" style={{ marginTop: 16 }}>
      <h2>Efficiency — composite per 1k tokens</h2>
      <p className="sub">
        Quality bought per unit of spend: <code>avg composite ÷ (avg tokens ÷ 1000)</code>. Longer
        bar is better value.
      </p>
      {ranked.map(({ row, value }, index) => {
        const lowN = row.judged < LOW_N;
        return (
          <div className="effrow" key={`${row.key}:${row.model ?? 'legacy'}`}>
            <div className="effname" title={label(row)}>
              {label(row)}
              <span className="effsub">
                composite {(row.avg_composite ?? 0).toFixed(2)} · ~{row.avg_token_estimate} tok ·
                n={row.judged}
                {lowN ? ' (low n)' : ''}
              </span>
            </div>
            <span
              className={`effbar${index === 0 ? ' best' : ''}`}
              role="img"
              aria-label={`${label(row)}: ${value.toFixed(3)} composite per 1k tokens over ${row.judged} judged questions`}
            >
              <i style={{ width: `${Math.max(2, Math.round((value / scale) * 100))}%` }} />
            </span>
            <div className="effval">
              {value.toFixed(3)}
              <span className="u">per 1k tok</span>
            </div>
          </div>
        );
      })}
      {ranked.length > 1 && spendsMore ? (
        <p className="board-note" style={{ marginTop: 10 }}>
          <b>{label(worst.row)}</b> spends ~{worst.row.avg_token_estimate} tokens per answer
          against <b>{label(best.row)}</b>&apos;s ~{best.row.avg_token_estimate}, and still scores
          lower ({(worst.row.avg_composite ?? 0).toFixed(2)} vs{' '}
          {(best.row.avg_composite ?? 0).toFixed(2)}) — roughly{' '}
          {(best.value / worst.value).toFixed(1)}× less quality per token. Extra retrieval rounds
          are not paying for themselves here.
        </p>
      ) : null}
    </div>
  );
}

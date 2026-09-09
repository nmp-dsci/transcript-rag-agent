import type { ScoreboardRow } from '../api/types';
import { efficiency, isLowN } from '../eval/breakdown';

/**
 * The one-sentence answer to "so which setup won?".
 *
 * The board computes everything needed for this — composites, win rates,
 * tokens per answer — and then made the reader derive the verdict themselves
 * by comparing columns. Stating it costs nothing and is the first thing
 * anyone opening an eval board wants.
 *
 * It deliberately reports a *tie* as a tie. The interesting case in this
 * corpus is two setups landing on the same composite at different cost, and a
 * strip that always crowned a winner would hide exactly that.
 */

export interface Verdict {
  /** Highest composite, or the first of several rows sharing it. */
  best: ScoreboardRow;
  /** Rows tied with `best` on composite, excluding it. */
  tiedWith: ScoreboardRow[];
  /** Best value for money, when it can be computed. */
  cheapest: ScoreboardRow | null;
  /** True when the composite leader is not also the efficiency leader. */
  splitDecision: boolean;
}

function label(row: ScoreboardRow): string {
  return row.model ? `${row.title} · ${row.model}` : row.title;
}

/** Two composites count as tied when they round to the same two decimals —
 * the precision the table itself prints, so the strip cannot claim a lead the
 * reader is unable to see. */
function tied(a: number | null, b: number | null): boolean {
  if (a == null || b == null) return false;
  return a.toFixed(2) === b.toFixed(2);
}

export function computeVerdict(rows: readonly ScoreboardRow[]): Verdict | null {
  // Thin rows are excluded rather than ranked: an average over a handful of
  // questions moves several points on one bad answer, and a verdict is a
  // stronger claim than a table cell.
  const rankable = rows.filter((row) => row.avg_composite != null && !isLowN(row.judged));
  if (rankable.length < 2) return null;

  const best = rankable.reduce((top, row) =>
    (row.avg_composite ?? 0) > (top.avg_composite ?? 0) ? row : top,
  );
  const tiedWith = rankable.filter(
    (row) => row.key !== best.key && tied(row.avg_composite, best.avg_composite),
  );

  const byValue = rankable
    .map((row) => ({ row, value: efficiency(row.avg_composite, row.avg_token_estimate) }))
    .filter((item): item is { row: ScoreboardRow; value: number } => item.value != null)
    .sort((a, b) => b.value - a.value);
  const cheapest = byValue[0]?.row ?? null;

  return {
    best,
    tiedWith,
    cheapest,
    splitDecision: cheapest != null && cheapest.key !== best.key,
  };
}

function pct(rate: number | null): string {
  return rate == null ? '—' : `${Math.round(rate * 100)}%`;
}

export function VerdictStrip({ rows }: { rows: readonly ScoreboardRow[] }) {
  const verdict = computeVerdict(rows);
  if (!verdict) return null;
  const { best, tiedWith, cheapest, splitDecision } = verdict;

  // On a tie, the win rate is the one thing that separates the tied rows, so
  // it is reported as a comparison. Printing only the leader's rate right
  // after saying two setups tie would read as if it described both.
  // `contests` is tracked separately from `judged` and is independently
  // noisy when low, so it gets its own low-n gate before a win rate is
  // stated as fact.
  const tieBreaker =
    tiedWith.length === 1 &&
    best.win_rate != null &&
    !isLowN(best.contests) &&
    tiedWith[0]!.win_rate != null &&
    !isLowN(tiedWith[0]!.contests)
      ? [best, tiedWith[0]!].sort((a, b) => (b.win_rate ?? 0) - (a.win_rate ?? 0))
      : null;

  return (
    <div className="verdict" role="status">
      <span className="microlabel">this run</span>
      <p className="verdict-line">
        {tiedWith.length > 0 ? (
          <>
            <b>{label(best)}</b> and <b>{label(tiedWith[0]!)}</b>
            {tiedWith.length > 1 ? ` and ${tiedWith.length - 1} more` : ''} tie on composite at{' '}
            <b>{best.avg_composite!.toFixed(2)}</b>
            {tieBreaker ? (
              <>
                , but <b>{label(tieBreaker[0]!)}</b> wins more questions —{' '}
                {pct(tieBreaker[0]!.win_rate)} ({tieBreaker[0]!.wins}/{tieBreaker[0]!.contests}) vs{' '}
                {pct(tieBreaker[1]!.win_rate)} ({tieBreaker[1]!.wins}/{tieBreaker[1]!.contests})
              </>
            ) : null}
            .
          </>
        ) : (
          <>
            <b>{label(best)}</b> leads on composite at <b>{best.avg_composite!.toFixed(2)}</b>
            {best.win_rate != null && !isLowN(best.contests) ? (
              <>
                , winning {pct(best.win_rate)} of questions ({best.wins}/{best.contests})
              </>
            ) : null}
            .
          </>
        )}{' '}
        {cheapest && splitDecision ? (
          <>
            Best value per token is <b>{label(cheapest)}</b> — so the ranking and the bill
            disagree.
          </>
        ) : cheapest ? (
          <>
            <b>{label(cheapest)}</b> is also the best value per token.
          </>
        ) : null}
      </p>
    </div>
  );
}

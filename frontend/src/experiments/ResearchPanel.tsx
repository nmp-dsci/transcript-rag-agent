import { Fragment, useEffect, useMemo, useState } from 'react';

import { api } from '../api/client';
import { RediscoveryMark, VerdictBadge } from './PackPanel';
import type {
  ResearchDiffRow,
  ResearchFrontier,
  ResearchGap,
  ResearchGapClosure,
  ResearchReport,
  ResearchRound,
  ResearchScoreRow,
} from '../api/types';

function fmt(value: number | null | undefined, digits = 3): string {
  return typeof value === 'number' ? value.toFixed(digits) : '—';
}

function signed(value: number): string {
  return `${value > 0 ? '+' : ''}${value}`;
}

function pct(value: number | null | undefined): string {
  return typeof value === 'number' ? `${Math.round(value * 100)}%` : '—';
}

/**
 * The gap critic's findings, verbatim, each with what round two got from it.
 *
 * This is the section the slice stands or falls on. A second round is not a
 * result — a second round *the critic caused* is. So each gap carries the
 * retrieval query it produced and the rules that came back, traced through the
 * probe id rather than asserted: the rubric names its unit, the unit is the
 * probe, and the probe names this gap.
 */
function GapCritique({
  gaps,
  round,
  added,
  closure,
}: {
  gaps: ResearchGap[];
  round: ResearchRound | undefined;
  added: ResearchDiffRow[];
  closure: ResearchGapClosure[];
}) {
  const closureOf = useMemo(() => {
    const map: Record<string, ResearchGapClosure> = {};
    for (const row of closure) map[row.gap_id] = row;
    return map;
  }, [closure]);
  // unit_id -> the gap that caused it, read off the round's own probe records.
  const causeOf = useMemo(() => {
    const map: Record<string, string> = {};
    for (const outcome of round?.probes ?? []) {
      if (outcome.probe.origin.startsWith('gap:')) {
        map[outcome.unit_id] = outcome.probe.origin.slice(4);
      }
    }
    return map;
  }, [round]);

  const byGap = useMemo(() => {
    const map: Record<string, ResearchDiffRow[]> = {};
    for (const row of added) {
      const gapId = causeOf[row.unit_id];
      if (!gapId) continue;
      (map[gapId] ??= []).push(row);
    }
    return map;
  }, [added, causeOf]);

  if (gaps.length === 0) return null;
  return (
    <section className="pk-section">
      <h4>
        What the gap critic said was missing after round 1
        <span className="pk-hint">
          it read the round-1 criteria and nothing else — no evidence, no transcript,
          and nothing from the held-out expert
        </span>
      </h4>
      <ul className="rs-gaps">
        {gaps.map((gap) => {
          const rules = byGap[gap.gap_id] ?? [];
          const probe = (round?.probes ?? []).find(
            (outcome) => outcome.probe.origin === `gap:${gap.gap_id}`,
          );
          const closure = closureOf[gap.gap_id];
          return (
            <li key={gap.gap_id} className="rs-gap">
              <div className="rs-gaphead">
                <span className="rs-gapid">{gap.gap_id}</span>
                <span className="rs-gapmissing">{gap.missing}</span>
                <span className={`pk-badge ${rules.length > 0 ? 'ok' : 'thin'}`}>
                  {rules.length} rule{rules.length === 1 ? '' : 's'} added
                </span>
                {closure && (
                  <span
                    className={`pk-badge ${closure.closed ? 'ok' : 'contested'}`}
                    title={
                      `best new rule sits at ${fmt(closure.best_new_cosine, 2)} to this gap; ` +
                      `round 1's nearest rule was already at ${fmt(closure.round_one_best_cosine, 2)}`
                    }
                  >
                    {closure.closed ? 'closed' : 'not closed'}
                  </span>
                )}
              </div>
              {gap.why && <p className="rs-gapwhy">{gap.why}</p>}
              <p className="rs-probe">
                <span className="microlabel">probe it wrote</span>
                {gap.probe}
                {probe && (
                  <span className="rs-probemeta">
                    → <code>{probe.unit_id}</code> · {probe.chunks} chunks ·{' '}
                    {probe.creator_count} creators
                    {!probe.spent_call && <> · call not spent: {probe.reject_reason}</>}
                  </span>
                )}
              </p>
              {rules.length > 0 && (
                <ul className="rs-caused">
                  {rules.map((row) => (
                    <li key={row.rubric_id}>
                      <code>{row.rubric_id}</code>
                      <span className="rs-crit">{row.criterion}</span>
                      <span className="rs-critmeta">
                        {row.check ? `${row.check} · ` : ''}
                        {row.creators.length} creator
                        {row.creators.length === 1 ? '' : 's'} · {row.citations} quote
                        {row.citations === 1 ? '' : 's'}
                      </span>
                    </li>
                  ))}
                </ul>
              )}
            </li>
          );
        })}
      </ul>
      <p className="rs-note">
        “Closed” is a deliberately weak, deterministic test: a rule admitted from the
        gap's own probe sitting closer to the critic's words than every rule round 1
        already had. It cannot prove a rule is <em>about</em> the gap — nothing short of a
        second judgement could — but it separates “the loop found new ground here” from
        “the loop retrieved the same ground again under a new question”, and the second is
        what a reader is most likely to be sold as the first.
      </p>
    </section>
  );
}

/**
 * How close each new rule sits to a rule the pack already had.
 *
 * The dedupe threshold is a cliff, and a cliff has a shadow. This table is the
 * shadow: rules admitted at 0.8-something against a round-one rule that says
 * the same thing in different words. It is reported rather than filtered,
 * because a threshold tuned in this panel would make the loop's own numbers a
 * function of a knob the loop owns.
 */
function RestatementAudit({ report }: { report: ResearchReport }) {
  const [open, setOpen] = useState(false);
  const rows = report.restatements ?? [];
  if (rows.length === 0) return null;
  const near = rows.filter(
    (row) => (row.nearest_prior_cosine ?? 0) >= row.dedupe_threshold - 0.1,
  );
  return (
    <section className="pk-section">
      <h4>
        <button type="button" className="pk-linkbtn" onClick={() => setOpen(!open)}>
          {open ? '▾' : '▸'} {near.length} of {rows.length} added rules sit within 0.1 of
          the dedupe threshold
        </button>
        <span className="pk-hint">
          nearest surviving rule and its cosine, for every rule round 2 added — dedupe
          fires at {rows[0]?.dedupe_threshold ?? '—'}
        </span>
      </h4>
      {open && (
        <ul className="rs-diff">
          {rows.map((row) => {
            const cos = row.nearest_prior_cosine ?? 0;
            const hot = cos >= row.dedupe_threshold - 0.1;
            return (
              <li key={row.rubric_id} className={`rs-diffrow ${hot ? 'removed' : ''}`}>
                {/* Three decimals, not two: 0.859 rounds to 0.86 and would read
                    as equal to the threshold it actually sits under. */}
                <span className="rs-cos">{fmt(cos, 3)}</span>
                <span className="rs-diffbody">
                  <span className="rs-crit">{row.criterion}</span>
                  <span className="rs-critmeta">
                    <code>{row.rubric_id}</code> nearest prior{' '}
                    <code>{row.nearest_prior_id ?? '—'}</code>:{' '}
                    {row.nearest_prior_criterion ?? '—'}
                  </span>
                </span>
              </li>
            );
          })}
        </ul>
      )}
    </section>
  );
}

/**
 * The sub-questions one round actually asked, and what each came back with.
 *
 * Kept behind the round's own row rather than in a list of its own, because the
 * question a reader has at this table is "what did *this* round ask" — and the
 * two rounds asked their questions for different reasons, which is the whole
 * claim. The origin tag carries that: `plan` was written before anything
 * existed, `gap:gNN` names the critic finding that caused it.
 */
function RoundProbes({ round }: { round: ResearchRound }) {
  return (
    <tr className="rs-proberow">
      <td colSpan={9}>
        <ol className="rs-plan">
          {round.probes.map((outcome) => (
            <li key={outcome.unit_id}>
              <span className="rs-facet">{outcome.probe.facet}</span>
              <span className="rs-question">{outcome.probe.question}</span>
              <span className="rs-critmeta">
                <code>{outcome.probe.probe_id}</code> · {outcome.probe.origin} ·{' '}
                <code>{outcome.unit_id}</code> · {outcome.chunks} chunks ·{' '}
                {outcome.creator_count} creators ·{' '}
                {outcome.rubric_ids.length} rule
                {outcome.rubric_ids.length === 1 ? '' : 's'} extracted
                {!outcome.spent_call && <> · call not spent: {outcome.reject_reason}</>}
              </span>
            </li>
          ))}
        </ol>
      </td>
    </tr>
  );
}

/** Each round's delta, with the control that says whether iteration did it. */
function RoundDeltas({ report }: { report: ResearchReport }) {
  const [openArm, setOpenArm] = useState<string | null>(null);
  const first: ResearchRound | undefined = report.rounds[0];
  const second: ResearchRound | undefined = report.rounds[1];
  // The baseline every delta is read against is round 1, not the row above —
  // the control and the second round are two answers to the same question, and
  // chaining them would make the control look like a step of the loop.
  const rows: Array<{ row: ResearchRound; delta: ResearchRound | null; tone: string }> = [];
  if (first) rows.push({ row: first, delta: null, tone: '' });
  rows.push({ row: report.control, delta: first ?? null, tone: 'control' });
  if (second) rows.push({ row: second, delta: first ?? null, tone: 'loop' });
  // The frontier arm belongs in this table like any other round: it is the arm
  // with the strongest claim, and leaving it out put the claim somewhere the
  // spend-against-added comparison is not.
  if (report.frontier?.round) {
    rows.push({ row: report.frontier.round, delta: first ?? null, tone: 'loop' });
  }

  return (
    <section className="pk-section">
      <h4>
        Rounds
        <span className="pk-hint">
          the control spends the same executor budget as round 2 on the same opening
          probes — it differs only in where the last {report.settings.gap_probes} questions came from
        </span>
      </h4>
      <div className="exp-scroll">
        <table className="exp-table">
          <thead>
            <tr>
              <th>arm</th>
              <th>caused by</th>
              <th className="num" title="probes this round added, not the arm's total">
                probes added
              </th>
              <th className="num" title="of those probes, the ones that cleared the creator floor and were paid for">
                calls spent
              </th>
              <th className="num" title="rubrics in the arm's pack, cumulative">
                rubrics in pack
              </th>
              <th className="num" title="against round 1, not against the row above">
                Δ vs round 1
              </th>
              <th className="num">quotes</th>
              <th className="num">≥2 creators</th>
              <th className="num" title="candidates this round produced that restated a rule the pack already had">
                deduped
              </th>
            </tr>
          </thead>
          <tbody>
            {rows.map(({ row, delta, tone }) => (
              <Fragment key={row.arm}>
              <tr className={tone === 'loop' ? 'rs-loop' : undefined}>
                <td className="exp-cfg">
                  {row.arm}
                  {tone === 'control' && <span className="exp-basetag">control</span>}
                </td>
                <td className="pk-cellwrap">{row.caused_by}</td>
                <td className="num">
                  <button
                    type="button"
                    className="pk-linkbtn"
                    aria-expanded={openArm === row.arm}
                    onClick={() => setOpenArm(openArm === row.arm ? null : row.arm)}
                  >
                    {openArm === row.arm ? '▾' : '▸'} {row.probes.length}
                  </button>
                </td>
                <td className="num">{row.executor_calls}</td>
                <td className="num">{row.rubrics}</td>
                <td className="num">
                  {delta ? signed(row.rubrics - delta.rubrics) : '—'}
                </td>
                <td className="num">{row.citations}</td>
                <td className="num">{pct(row.multi_creator_share)}</td>
                <td className="num">{row.deduped_against_previous}</td>
              </tr>
              {openArm === row.arm && <RoundProbes round={row} />}
              </Fragment>
            ))}
          </tbody>
        </table>
      </div>
      <p className="rs-note">
        Open a round's <b>probes added</b> count to read the sub-questions it asked and
        where each one came from. “Deduped” is the number that says whether the critic
        found new ground: a gap
        answered with a restatement of a rule the pack already had is dropped at the
        same cosine threshold the shipped arms use, so round 2 cannot bank round 1's
        coverage as its own.
      </p>
    </section>
  );
}

/** Calls per arm, stated. A loop that won on spend has reported its budget. */
function BudgetTable({ report }: { report: ResearchReport }) {
  return (
    <section className="pk-section">
      <h4>
        Call budget
        <span className="pk-hint">
          one planner call serves every arm; round 2's only extra spend over the control
          is the single critic call
        </span>
      </h4>
      <div className="exp-scroll">
        <table className="exp-table">
          <thead>
            <tr>
              <th>arm</th>
              <th className="num">planner</th>
              <th className="num">executor</th>
              <th className="num">critic</th>
              <th className="num">total LLM calls</th>
              <th className="num">rubrics</th>
              <th className="num">quotes</th>
            </tr>
          </thead>
          <tbody>
            {report.budget.map((row) => (
              <tr key={row.arm}>
                <td className="exp-cfg">{row.arm}</td>
                <td className="num">{row.planner_calls}</td>
                <td className="num">
                  {row.executor_calls}
                  {row.executor_calls !== row.probes_budgeted && (
                    <span className="crit-spread">of {row.probes_budgeted} probes</span>
                  )}
                </td>
                <td className="num">{row.critic_calls}</td>
                <td className="num">{row.total_llm_calls}</td>
                <td className="num">{row.rubrics}</td>
                <td className="num">{row.citations}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

/**
 * The held-out scores, with the counts that let a reader rule out padding.
 *
 * `criteria_recall` rises with finding count — the corpus's own known attack
 * beat the honest baseline by reciting advice one citation at a time — so the
 * findings and quotes columns sit beside every score rather than in a footnote,
 * and the spread under each recall is the matcher disagreeing with itself.
 */
function ScoreTable({ report }: { report: ResearchReport }) {
  const scores = report.scores;
  if (!scores) {
    return (
      <p className="pk-nod2">
        This loop has not been scored yet. Run{' '}
        <code>uv run python -m src.cli deep-research --topic {report.topic} --score-only</code>{' '}
        to put every arm through the held-out critique harness.
      </p>
    );
  }
  const metrics =
    scores.metrics.length > 0
      ? scores.metrics
      : ['criteria_recall', 'evidence_precision', 'provenance', 'contested_coverage'];
  const best: Record<string, number> = {};
  for (const row of scores.rows) {
    for (const metric of metrics) {
      const value = row.scores[metric];
      if (typeof value === 'number' && (best[metric] === undefined || value > best[metric])) {
        best[metric] = value;
      }
    }
  }
  return (
    <section className="pk-section">
      <h4>
        Scored on the held-out expert
        <span className="pk-hint">
          {scores.held_out_title ? `${scores.held_out_title} · ` : ''}
          {scores.criteria_applicable ?? '—'}/{scores.criteria_total ?? '—'} criteria apply
          {scores.match_repeats ? ` · ${scores.match_repeats} matcher repeats` : ''}
        </span>
      </h4>
      {/* Ingestion does not stop for an experiment. The arms were all built at
          one corpus digest, so they stay comparable to each other — but the
          citations were resolved against the corpus as it stood at scoring
          time, and a reader must be told that rather than left to assume. The
          provenance column is where the drift would show. */}
      {scores.scored_on_build_corpus === false && (
        <p className="pk-notice">
          Built on corpus <code>{scores.build_corpus_digest ?? '—'}</code>, scored against{' '}
          <code>{scores.scoring_corpus_digest ?? '—'}</code> (
          {scores.scoring_chunk_count ?? '—'} chunks) — more was ingested after these arms
          were built. Every arm was built at the same digest, so the comparison between
          them is unaffected; the <b>provenance</b> column is what would show a stored
          quote whose chunk moved underneath it.
        </p>
      )}
      <div className="exp-scroll">
        <table className="exp-table">
          <thead>
            <tr>
              <th>arm</th>
              {metrics.map((metric) => (
                <th key={metric} className="num">
                  {metric}
                </th>
              ))}
              <th className="num" title="more findings mechanically raises recall — read this column beside it">
                findings
              </th>
              <th className="num">quotes</th>
              <th className="num">executor calls</th>
              <th className="num">leaks</th>
            </tr>
          </thead>
          <tbody>
            {scores.rows.map((row: ResearchScoreRow) => (
              <tr
                key={row.arm}
                className={
                  row.arm === 'deep-r2' || row.arm === report.frontier?.arm ? 'rs-loop' : undefined
                }
              >
                <td className="exp-cfg">
                  {row.arm}
                  {row.arm === scores.baseline && <span className="exp-basetag">base</span>}
                </td>
                {metrics.map((metric) => {
                  const value = row.scores[metric];
                  const low = row.score_spread[`${metric}_min`];
                  const high = row.score_spread[`${metric}_max`];
                  const isBest = typeof value === 'number' && value === best[metric];
                  return (
                    <td key={metric} className={`num${isBest ? ' best' : ''}`}>
                      {fmt(value)}
                      {typeof low === 'number' && typeof high === 'number' && low !== high && (
                        <span className="crit-spread">
                          {low.toFixed(3)}–{high.toFixed(3)}
                        </span>
                      )}
                    </td>
                  );
                })}
                <td className="num">
                  {row.findings_grounded ?? '—'}/{row.findings_total ?? '—'}
                </td>
                <td className="num">
                  {row.citations_resolved ?? '—'}/{row.citations_total ?? '—'}
                </td>
                <td className="num">{row.executor_calls ?? '—'}</td>
                <td className={`num${(row.held_out_leaks ?? 0) > 0 ? ' bad' : ''}`}>
                  {row.held_out_leaks ?? '—'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="pk-verdicts">
        {Object.entries(scores.verdicts).map(([metric, verdict]) => (
          <div key={metric} className="pk-verdict">
            <span className="microlabel">{metric}</span>
            <b>{verdict.leader ?? '—'}</b>
            <VerdictBadge verdict={verdict} />
            <span className="pk-verdictwhy">{verdict.reason}</span>
          </div>
        ))}
      </div>
      <p className="rs-note">
        A lead smaller than the range under it is not a lead. The findings column is
        there because recall rises with finding count on its own: an arm that gained by
        growing shows it here, and a reader should not have to take the number on trust.
      </p>
    </section>
  );
}

/** The v1 → v2 rubric diff: what survived, what the critic's round added. */
function RubricDiff({ report }: { report: ResearchReport }) {
  const [open, setOpen] = useState(true);
  const diff = report.diff;
  const rows: Array<{ kind: string; row: ResearchDiffRow }> = [
    ...diff.added.map((row) => ({ kind: 'added', row })),
    ...diff.removed.map((row) => ({ kind: 'removed', row })),
    ...diff.kept.map((row) => ({ kind: 'kept', row })),
  ];
  if (rows.length === 0) return null;
  return (
    <section className="pk-section">
      <h4>
        <button type="button" className="pk-linkbtn" onClick={() => setOpen(!open)}>
          {open ? '▾' : '▸'} {diff.before_arm} → {diff.after_arm}: {diff.added.length} added,{' '}
          {diff.kept.length} kept, {diff.removed.length} removed
        </button>
        <span className="pk-hint">
          keyed on rubric id — a rule that survived is literally the same object, carried
          forward untouched
        </span>
      </h4>
      {open && (
        <ul className="rs-diff">
          {rows.map(({ kind, row }) => (
            <li key={`${kind}-${row.rubric_id}`} className={`rs-diffrow ${kind}`}>
              <span className="rs-diffmark">
                {kind === 'added' ? '+' : kind === 'removed' ? '−' : ' '}
              </span>
              <span className="rs-diffbody">
                <span className="rs-crit">{row.criterion}</span>
                <span className="rs-critmeta">
                  <code>{row.rubric_id}</code> · <code>{row.unit_id}</code> ·{' '}
                  {row.creators.join(' · ') || 'no creator'} · {row.citations} quote
                  {row.citations === 1 ? '' : 's'}
                </span>
                {/* Same marker as the pack browser's diff. Both diffs on this
                    page carry it or neither does — a control whose additions
                    render unmarked reads as all-new beside a loop that admits
                    its rediscoveries, which would flatter the wrong arm. */}
                {kind === 'added' && <RediscoveryMark ids={row.already_in_shipped} />}
              </span>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

/** The plan, so a reader can see round 1 and the control are nested. */
function PlanList({ report }: { report: ResearchReport }) {
  const [open, setOpen] = useState(false);
  const cut = report.settings.round_one_probes ?? 0;
  return (
    <section className="pk-section">
      <h4>
        <button type="button" className="pk-linkbtn" onClick={() => setOpen(!open)}>
          {open ? '▾' : '▸'} The plan — {report.plan.length} sub-questions
        </button>
        <span className="pk-hint">
          one planner call, ordered most-important-first, so round 1 is a prefix of the
          control rather than a separate draw
        </span>
      </h4>
      {open && (
        <ol className="rs-plan">
          {report.plan.map((probe) => (
            <li key={probe.probe_id} className={probe.rank <= cut ? 'in-r1' : ''}>
              <span className="rs-facet">{probe.facet}</span>
              <span className="rs-question">{probe.question}</span>
              <span className="rs-critmeta">
                <code>{probe.probe_id}</code>
                {probe.rank <= cut ? ' · round 1 and control' : ' · control only'}
              </span>
            </li>
          ))}
        </ol>
      )}
    </section>
  );
}

const BASIS_LABEL: Record<string, string> = {
  'ranges-disjoint': 'ranges disjoint',
  'ranges-overlap': 'ranges overlap',
  deterministic: 'no model in its path',
  unrepeated: 'unrepeated',
  ungraded: 'not measured',
};

/**
 * The one comparison the verdict row cannot make: this arm against the baseline.
 *
 * `winner()` ranks every arm and reports the leader against the *runner-up*. The
 * gate this slice is held to asks something narrower — does the loop-built pack
 * reach the hand-built one — and when two loop arms tie for the lead the verdict
 * reads "tied" and says nothing about the pack they were built to beat.
 *
 * Read out of the report rather than computed here. It used to be derived in
 * this component, which meant the finding existed only while a browser was open
 * — absent from the run file, from `winner()`, and from anything a reader
 * outside the app could check. `against_baseline` in `src/evals/pack_ablation.py`
 * is now committed into the run and re-derived for older reports.
 */
function AgainstTheHandBuild({
  frontier,
  scores,
}: {
  frontier: ResearchFrontier;
  scores: ResearchReport['scores'];
}) {
  const table = scores?.against_baseline;
  if (!table) return null;
  const rows = ['criteria_recall', 'evidence_precision']
    .map((metric) => (table[metric] ?? []).find((row) => row.arm === frontier.arm))
    .filter((row): row is NonNullable<typeof row> => row !== undefined);
  if (rows.length === 0) return null;

  return (
    <div className="pk-verdicts">
      {rows.map((row) => (
        <div key={row.metric} className="pk-verdict">
          <span className="microlabel">
            {row.metric} vs {row.baseline}
          </span>
          <b>
            {fmt(row.value)} vs {fmt(row.baseline_value)}
          </b>
          <span className={`pk-badge ${row.beats_baseline ? 'ok' : 'contested'}`}>
            {BASIS_LABEL[row.basis] ?? row.basis}
          </span>
          <span className="pk-verdictwhy">{row.reason}</span>
        </div>
      ))}
      <p className="rs-note">
        The margin on <b>criteria_recall</b> is one criterion in{' '}
        {scores?.criteria_applicable ?? '—'} — 0.053 of a score that moves in steps of 0.053.
        One repeat either way and these ranges touch.
      </p>
    </div>
  );
}

/**
 * V8's frontier round: what it changed, and whether the change did anything.
 *
 * The gate this slice failed asks whether a loop-built pack beats the
 * hand-built one, and its evaluator found the mechanical reason it did not: of
 * the loop's ten added rules four cite chunks the shipped pack already cites,
 * and the one-shot control — no critic at all — rediscovers at exactly the same
 * rate. This section leads with that table rather than with the new arm's
 * scores, because a new arm that does not move it has not moved the finding.
 */
function FrontierRound({
  frontier,
  scores,
}: {
  frontier: ResearchFrontier;
  scores: ResearchReport['scores'];
}) {
  const [openCoverage, setOpenCoverage] = useState(false);
  const arms = Object.entries(frontier.rediscovery);
  const overlapOf = useMemo(() => {
    const map: Record<string, number> = {};
    for (const row of frontier.probe_overlap) map[row.unit_id] = row.from_round_one_ground;
    return map;
  }, [frontier.probe_overlap]);

  return (
    <section className="pk-section">
      <h4>
        The frontier round — <code>{frontier.arm}</code>
        <span className="pk-hint">
          corpus <code>{frontier.corpus_digest}</code> ({frontier.chunk_count} chunks,{' '}
          {frontier.video_count} videos) — the same corpus every other arm was built on, matched
          by digest rather than assumed
        </span>
      </h4>

      <p className="pk-intro">
        Round 1 is not rebuilt: this arm inherits <code>deep-r1</code>'s six probes, their
        retrievals and its membership, so it is nested inside the same plan as the loop and the
        control rather than being a fourth draw. Three things change, and each is a rule about
        what is <em>forbidden</em>. The critic is shown how much of each source its own probes
        have read — counts and the video titles the planner was already handed, never any
        transcript. A round-two probe may not retrieve into the{' '}
        {frontier.round_one_read} of {frontier.member_chunks} member passages round 1 already had
        in front of it. And a rule is admitted only if it rests on a passage no admitted rule
        rests on — the same refusal as the cosine dedupe, but on evidence identity, which has no
        cliff and so no shadow under it.
      </p>

      <div className="exp-scroll">
        <table className="exp-table">
          <thead>
            <tr>
              <th>arm</th>
              <th className="num" title="rules this arm added on top of round 1">
                added
              </th>
              <th
                className="num"
                title="of those, the ones citing a chunk the shipped pack already cites"
              >
                rediscovered
              </th>
              <th className="num">rate</th>
            </tr>
          </thead>
          <tbody>
            {arms.map(([arm, row]) => (
              <tr key={arm} className={arm === frontier.arm ? 'rs-loop' : undefined}>
                <td className="exp-cfg">{arm}</td>
                <td className="num">{row.added}</td>
                <td className="num">{row.rediscovered ?? '—'}</td>
                <td className="num">{pct(row.rate)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="rs-note">
        Measured on cited chunk ids against{' '}
        <code>{arms[0]?.[1]?.shipped_arm ?? 'the shipped pack'}</code>, not on wording: two rules
        phrased differently off the same transcript second are the same finding, and a similarity
        score would be a judgement where an identity check will do.
      </p>

      <AgainstTheHandBuild frontier={frontier} scores={scores} />

      <h4>
        What the coverage-aware critic said was missing
        <span className="pk-hint">
          it read the round-1 criteria, the plan's facets and the coverage counts — no evidence,
          no transcript, and nothing from the held-out expert
        </span>
      </h4>
      <ul className="rs-gaps">
        {frontier.gaps.map((gap) => {
          const probe = frontier.probes.find(
            (outcome) => outcome.probe.origin === `gap:${gap.gap_id}`,
          );
          return (
            <li key={gap.gap_id} className="rs-gap">
              <div className="rs-gaphead">
                <span className="rs-gapid">{gap.gap_id}</span>
                <span className="rs-gapmissing">{gap.missing}</span>
              </div>
              <p className="rs-probe">
                <span className="microlabel">probe it wrote</span>
                {gap.probe}
                {probe && (
                  <span className="rs-probemeta">
                    → <code>{probe.unit_id}</code> · {probe.chunks} chunks ·{' '}
                    {probe.creator_count} creators · {overlapOf[probe.unit_id] ?? 0} of them off
                    round 1's ground
                    {!probe.spent_call && <> · call not spent: {probe.reject_reason}</>}
                  </span>
                )}
              </p>
            </li>
          );
        })}
      </ul>

      {frontier.refused.length > 0 && (
        <>
          <h4>
            Refused for resting on no passage of their own
            <span className="pk-hint">
              an identity check on cited chunks, applied to this round's candidates only — the
              inherited round is passed first and is never the thing dropped
            </span>
          </h4>
          <ul className="rs-diff">
            {frontier.refused.map((row) => (
              <li key={row.rubric_id} className="rs-diffrow removed">
                <span className="rs-diffmark">−</span>
                <span className="rs-diffbody">
                  <span className="rs-crit">{row.criterion}</span>
                  <span className="rs-critmeta">
                    <code>{row.rubric_id}</code> rested only on {row.chunk_ids.length} passage
                    {row.chunk_ids.length === 1 ? '' : 's'} already held by{' '}
                    {row.already_rested_on_by.map((id) => (
                      <code key={id}>{id}</code>
                    ))}
                  </span>
                </span>
              </li>
            ))}
          </ul>
          <p className="rs-note">
            This rule resembles the scorer's exclusivity gate, so any{' '}
            <b>evidence_precision</b> it buys is partly bought by construction. That is what{' '}
            <code>{frontier.admission_arm}</code> is for — the same rule applied to{' '}
            <code>deep-r2</code> with nothing else changed and no call spent — so a reader can
            subtract it — and on this run the subtraction takes the whole of both point
            estimates: <code>{frontier.admission_arm}</code> reaches the{' '}
            <em>identical</em> set of applicable criteria for no LLM call at all. And it is not
            true, as this panel used to say, that <b>criteria_recall</b> cannot be bought by
            dropping rules: <code>{frontier.admission_arm}</code> is a strict subset of{' '}
            <code>deep-r2</code> and scores higher, because the matcher pairs each criterion
            with at most one finding and thinning the pool removes competitors. Dropping rules
            is not a <em>reliable</em> way to buy recall; it is not an impossible one. And the
            rule is not a handicap applied to one side only: run over{' '}
            <code>{frontier.admission_on_shipped.arm ?? 'the shipped pack'}</code>'s{' '}
            {frontier.admission_on_shipped.rubrics} rules it would refuse{' '}
            <b>{frontier.admission_on_shipped.would_refuse ?? '—'}</b>
            {frontier.admission_on_shipped.would_refuse === 0
              ? ' — every rule the hand build ships already rests on a passage of its own.'
              : `: ${frontier.admission_on_shipped.refused_ids.join(', ')}.`}
          </p>
        </>
      )}

      <h4>
        <button type="button" className="pk-linkbtn" onClick={() => setOpenCoverage(!openCoverage)}>
          {openCoverage ? '▾' : '▸'} The coverage table the critic was given
        </button>
        <span className="pk-hint">
          every member video, no cut-off, ordered most-unread first — a filter chosen here would
          be the build steering the critic
        </span>
      </h4>
      {openCoverage && (
        <div className="exp-scroll">
          <table className="exp-table">
            <thead>
              <tr>
                <th>video</th>
                <th className="num">read</th>
                <th className="num">passages</th>
                <th className="num">unread</th>
              </tr>
            </thead>
            <tbody>
              {frontier.coverage.map((row) => (
                <tr key={row.video_id}>
                  <td className="pk-cellwrap">{row.title || row.video_id}</td>
                  <td className="num">{row.read}</td>
                  <td className="num">{row.chunks}</td>
                  <td className="num">{row.chunks - row.read}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <h4>
        <span>
          {frontier.diff.before_arm} → {frontier.diff.after_arm}: {frontier.diff.added.length}{' '}
          added, {frontier.diff.kept.length} kept
        </span>
      </h4>
      <ul className="rs-diff">
        {frontier.diff.added.map((row) => (
          <li key={row.rubric_id} className="rs-diffrow added">
            <span className="rs-diffmark">+</span>
            <span className="rs-diffbody">
              <span className="rs-crit">{row.criterion}</span>
              <span className="rs-critmeta">
                <code>{row.rubric_id}</code> · <code>{row.unit_id}</code> ·{' '}
                {row.creators.join(' · ') || 'no creator'} · {row.citations} quote
                {row.citations === 1 ? '' : 's'}
              </span>
              <RediscoveryMark ids={row.already_in_shipped} />
            </span>
          </li>
        ))}
      </ul>
    </section>
  );
}

/**
 * The Build report for the deep-research loop.
 *
 * The claim this panel has to make legible is not "the pack got better" but
 * "the loop iterated, and the iteration is what did it". So the gap critic's
 * words come first, the rules its gaps produced are traced to it by probe id,
 * and the one-shot control sits in every table beside the second round on the
 * same executor budget.
 */
export function ResearchPanel() {
  const [reports, setReports] = useState<ResearchReport[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let live = true;
    void (async () => {
      try {
        const list = await api.packs();
        // Asked per declared topic rather than from a list endpoint: the loop is
        // expensive enough that only one topic has been run, and a topic with no
        // report answers null rather than 404 — so this is four cheap reads and
        // no new server surface.
        const found = await Promise.all(
          list.packs.map((pack) => api.packResearch(pack.topic).catch(() => null)),
        );
        if (!live) return;
        setReports(found.filter((row): row is ResearchReport => row !== null));
        setError(null);
      } catch (err) {
        if (live) setError((err as Error).message);
      }
    })();
    return () => {
      live = false;
    };
  }, []);

  if (error) {
    return (
      <section className="exp-card">
        <p className="exp-empty">Could not load the build report: {error}</p>
      </section>
    );
  }
  // No loop run yet is the ordinary case for three of the four packs, and an
  // empty card would read as a broken one.
  if (reports.length === 0) return null;
  return (
    <>
      {reports.map((report) => (
        <ResearchReportCard key={report.topic} report={report} />
      ))}
    </>
  );
}

function ResearchReportCard({ report }: { report: ResearchReport }) {
  const second = report.rounds[1];

  return (
    <section className="exp-card pk-card">
      <div className="exp-cardhead">
        <div>
          <h3>Build report: {report.topic}</h3>
          <span className="exp-sub">
            deep-research loop · corpus <code>{report.corpus_digest}</code> (
            {report.chunk_count} chunks) · {report.members} member videos · {report.model}
          </span>
        </div>
      </div>

      <p className="pk-intro">
        The shipped pack is built in one pass from the corpus's own clusters. This is the
        same pack built the slow way instead: a <b>planner</b> decomposes the topic into
        sub-questions, an <b>executor</b> answers each one out of the transcripts, a{' '}
        <b>gap critic</b> reads the result and names what a reviewer still could not check,
        and a <b>publisher</b> folds the answers to those gaps into a second version. Every
        rubric is written by the same extraction call the shipped pack uses — same prompt,
        same excerpt budget, same server-side citation reconciliation — so the arms differ
        in which passages the model saw and in nothing else.
      </p>

      <GapCritique
        gaps={report.gaps}
        round={second}
        added={report.diff.added}
        closure={report.gap_closure ?? []}
      />
      {report.frontier && (
        <FrontierRound frontier={report.frontier} scores={report.scores} />
      )}
      <RoundDeltas report={report} />
      <ScoreTable report={report} />
      <RestatementAudit report={report} />
      <RubricDiff report={report} />
      <BudgetTable report={report} />
      <PlanList report={report} />
    </section>
  );
}

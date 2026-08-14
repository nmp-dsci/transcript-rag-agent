import { Fragment, useCallback, useState } from 'react';

import { api } from '../api/client';
import type {
  CritiqueBallot,
  CritiqueCell,
  CritiqueRunDetail,
  CritiqueRunSummary,
} from '../api/types';

/** 3-decimal fixed, or an em dash for a metric the run could not measure. */
function fmt(value: number | null | undefined): string {
  return typeof value === 'number' ? value.toFixed(3) : '—';
}

/** A YouTube link that opens at the second a criterion was said. */
function watchUrl(videoId: string, seconds: number): string {
  return `https://www.youtube.com/watch?v=${videoId}&t=${Math.floor(seconds)}s`;
}

function clock(seconds: number): string {
  const total = Math.floor(seconds);
  return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, '0')}`;
}

/** Ballots for criteria a repeat paired and the majority vote then discarded. */
function outvotedBallots(cell: CritiqueCell): CritiqueBallot[] {
  // A ballot only exists when some repeat paired the criterion, so "consensus
  // chose nothing" is exactly "a pairing was made and then outvoted". The
  // verdict is read from the run, never re-decided here.
  return (cell.match_ballots ?? []).filter((b) => b.consensus_finding_id === null);
}

/** Repeats that backed no pairing at all, and the strongest one that lost. */
function splitOf(ballot: CritiqueBallot): { against: number; forTop: number } {
  const against = ballot.votes.find((v) => v.finding_id === null)?.count ?? 0;
  const forTop = ballot.votes.find((v) => v.finding_id !== null)?.count ?? 0;
  return { against, forTop };
}

/** How many repeats paired this criterion with the finding consensus chose. */
function votesFor(ballot: CritiqueBallot, findingId: string): number {
  return ballot.votes.find((v) => v.finding_id === findingId)?.count ?? 0;
}

/**
 * The matcher's own ballots, criterion by criterion and repeat by repeat.
 *
 * A recall of 0.000 has two entirely different causes — nothing was found, or
 * something was found and the vote threw it away — and the score cannot tell
 * them apart. On the run this section was written for, two of five repeats
 * paired a held-out criterion with a rubric finding and `None` took the other
 * three, so both pairings died at consensus and the arm published a zero. That
 * is a fact about the *scorer*, and until it was rendered it lived nowhere but
 * the JSON.
 *
 * Rendered open rather than behind a second disclosure: the reader has already
 * paid a click to be here, and the list is short by construction — only the
 * criteria some repeat paired appear, which is two to six of twenty-four, not a
 * wall of blank ballots.
 */
function BallotSection({
  ballots,
  criteriaTotal,
  setup,
  baselineSetup,
  baselineBallots,
}: {
  ballots: CritiqueBallot[];
  criteriaTotal: number;
  setup: string;
  baselineSetup: string;
  baselineBallots: CritiqueBallot[];
}) {
  if (ballots.length === 0) return null;
  const repeats = ballots[0]!.draws.length;
  const kept = ballots.filter((b) => b.consensus_finding_id !== null).length;
  const lost = ballots.length - kept;

  return (
    <section className="crit-ballots">
      <span className="microlabel">
        matcher ballots — {ballots.length} of {criteriaTotal} criteria were paired
        with a finding on at least one of {repeats} matcher repeats
        {kept === 0
          ? `, and the majority vote kept none of them. This row’s recall is what
             consensus discarded, not a rule the system never reached.`
          : `; the majority vote kept ${kept}${lost > 0 ? ` and outvoted ${lost}` : ''}.`}
      </span>
      <ul className="crit-list">
        {ballots.map((ballot) => {
          const { against, forTop } = splitOf(ballot);
          const loser = ballot.votes.find((v) => v.finding_id !== null);
          const base =
            setup === baselineSetup
              ? undefined
              : baselineBallots.find((b) => b.criterion_id === ballot.criterion_id);
          const baseId = base?.consensus_finding_id ?? null;
          return (
            <li
              key={ballot.criterion_id}
              className={`crit-item ${ballot.consensus_finding_id ? 'hit' : 'miss'}`}
            >
              <div className="crit-rule">
                <b>{ballot.criterion_id}</b> {ballot.criterion}
              </div>
              {/* One chip per repeat, in run order, so "the matcher disagreed
                  with itself" is something a reader sees rather than infers
                  from an agreement percentage. */}
              <div className="crit-draws">
                {ballot.draws.map((draw, index) => (
                  <span
                    key={`${ballot.criterion_id}-${index}`}
                    className={`crit-draw${draw ? ' on' : ''}`}
                    title={
                      draw
                        ? `repeat ${index + 1} paired this criterion with ${draw}`
                        : `repeat ${index + 1} paired this criterion with nothing`
                    }
                  >
                    <em>{index + 1}</em>
                    {draw ?? 'no match'}
                  </span>
                ))}
              </div>
              {ballot.consensus_finding_id ? (
                <div className="crit-found">
                  ↳ consensus paired it with <b>{ballot.consensus_finding_id}</b> on{' '}
                  {votesFor(ballot, ballot.consensus_finding_id)} of {repeats} repeats —{' '}
                  {ballot.consensus_finding_criterion}
                </div>
              ) : (
                <div className="crit-found">
                  ↳ consensus discarded this pairing: “no match” took {against} of{' '}
                  {repeats} repeats, <b>{loser?.finding_id}</b> took {forTop}. The
                  reading that lost: “{loser?.finding_criterion}”
                </div>
              )}
              {base && baseId && (
                // The comparison that explains the loss. Same criterion, same
                // matcher, same repeats — one arm's phrasing was read as the
                // expert's rule every time and the other's was not.
                <div className="crit-found crit-why">
                  ↳ the baseline <b>{baselineSetup}</b> paired the same criterion with{' '}
                  <b>{baseId}</b> on {votesFor(base, baseId)} of {base.draws.length}{' '}
                  repeats — “{base.consensus_finding_criterion}”
                </div>
              )}
            </li>
          );
        })}
      </ul>
    </section>
  );
}

/**
 * The expanded row: every held-out criterion, matched or not, with its timestamp.
 *
 * This is the part that makes the score falsifiable. A recall number on its own
 * is a claim; the list below is the evidence, criterion by criterion, each one
 * linked to the second of the held-out video it was extracted from — so a
 * reviewer can watch the expert say it and decide whether the system's finding
 * really is the same rule.
 */
function CritiqueDetail({
  run,
  setup,
  ballots,
  baselineSetup,
  baselineBallots,
}: {
  run: CritiqueRunDetail;
  setup: string;
  ballots: CritiqueBallot[];
  baselineSetup: string;
  baselineBallots: CritiqueBallot[];
}) {
  const cell = run.cells.find((c) => c.setup === setup);
  if (!cell) return <p className="crit-empty">No detail stored for {setup}.</p>;
  const ballotFor = (criterionId: string): CritiqueBallot | undefined =>
    ballots.find((b) => b.criterion_id === criterionId);
  // Findings some repeat did pair with a held-out criterion before the vote
  // discarded it. These are the ones the "outside the held-out list" gloss used
  // to sweep up, and they are not outside it.
  const drawnFindingIds = new Set(
    ballots.flatMap((b) => b.votes.map((v) => v.finding_id).filter((id) => id !== null)),
  );

  // `counted`, not `matched`: a criterion paired with a finding that rests on no
  // exclusive corpus evidence does not score, and showing it as reached would
  // print a fraction that disagrees with the row above it.
  const counted = cell.matches.filter((m) => m.counted);
  const missed = cell.matches.filter((m) => !m.counted);
  const applicable = cell.matches.filter((m) => m.applies_to.includes(run.artifact_kind));
  // Derived independently of `counted` so this stays right the day a
  // non-applicable criterion is matched — the earlier arithmetic assumed that
  // never happens and would have printed a negative count.
  const notApplicable = cell.matches.filter(
    (m) => !m.applies_to.includes(run.artifact_kind),
  ).length;
  const countedApplicable = counted.filter((m) =>
    m.applies_to.includes(run.artifact_kind),
  ).length;
  const ungrounded = cell.matches.filter((m) => m.ungrounded);
  const unmatchedFindings = cell.findings.filter(
    (f) => !cell.matches.some((m) => m.finding_id === f.id),
  );
  const conflicts = cell.conflicts ?? [];
  const fabrications = cell.fabricated_citations ?? [];
  const nearMisses = unmatchedFindings.filter((f) => drawnFindingIds.has(f.id));

  return (
    <div className="crit-detail">
      <div className="crit-cols">
        <section>
          <span className="microlabel">
            reached — {countedApplicable} of {applicable.length} criteria that apply to a{' '}
            {run.artifact_kind}
          </span>
          <ul className="crit-list">
            {counted.map((m) => (
              <li key={m.id} className="crit-item hit">
                <div className="crit-rule">
                  <b>{m.id}</b> {m.criterion}
                </div>
                <div className="crit-meta">
                  <a
                    href={watchUrl(m.video_id, m.start_seconds)}
                    target="_blank"
                    rel="noreferrer"
                    className="crit-ts"
                  >
                    {clock(m.start_seconds)}
                  </a>
                  <span className="crit-quote">“{m.quote}”</span>
                </div>
                <div className="crit-found">
                  ↳ {m.finding_criterion}
                  {m.agreement < 1 ? (
                    <span className="crit-agree">
                      {Math.round(m.agreement * 100)}% of matcher runs
                    </span>
                  ) : null}
                  {m.why ? <span className="crit-why"> — {m.why}</span> : null}
                </div>
              </li>
            ))}
            {counted.length === 0 && (
              <li className="crit-item">
                <div className="crit-rule">Nothing this expert said was reached.</div>
              </li>
            )}
          </ul>
        </section>

        <section>
          <span className="microlabel">
            missed — {missed.length} criteria, {notApplicable} of which a{' '}
            {run.artifact_kind} cannot be judged on
            {ungrounded.length > 0
              ? `, ${ungrounded.length} raised but resting on no evidence of its own`
              : ''}
          </span>
          <ul className="crit-list">
            {missed.map((m) => {
              // A criterion the matcher paired on a minority of its repeats is
              // not the same miss as one no repeat ever raised, and this column
              // read them identically before.
              const ballot = !m.matched ? ballotFor(m.id) : undefined;
              const split = ballot ? splitOf(ballot) : null;
              return (
              <li key={m.id} className="crit-item miss">
                <div className="crit-rule">
                  <b>{m.id}</b> {m.criterion}
                  {!m.applies_to.includes(run.artifact_kind) && (
                    <span className="crit-na">n/a to a {run.artifact_kind}</span>
                  )}
                  {ballot && split && split.forTop > 0 && (
                    <span
                      className="crit-agree lost"
                      title={`${split.forTop} of ${ballot.draws.length} matcher repeats paired this criterion with a finding; “no match” took ${split.against} and won. See the matcher ballots below.`}
                    >
                      outvoted {split.against}–{split.forTop}
                    </span>
                  )}
                </div>
                <div className="crit-meta">
                  <a
                    href={watchUrl(m.video_id, m.start_seconds)}
                    target="_blank"
                    rel="noreferrer"
                    className="crit-ts"
                  >
                    {clock(m.start_seconds)}
                  </a>
                  <span className="crit-quote">“{m.quote}”</span>
                </div>
                {m.ungrounded && (
                  <div className="crit-found">
                    ↳ raised as “{m.finding_criterion}” — but that finding rests on
                    no evidence another finding does not also claim, so it does not
                    count
                  </div>
                )}
                {!m.matched && m.why ? (
                  <div className="crit-found crit-why">↳ {m.why}</div>
                ) : null}
              </li>
              );
            })}
          </ul>
        </section>
      </div>

      <BallotSection
        ballots={ballots}
        criteriaTotal={run.criteria_total}
        setup={setup}
        baselineSetup={baselineSetup}
        baselineBallots={baselineBallots}
      />

      {fabrications.length > 0 && (
        <section className="crit-conflicts">
          <span className="microlabel">
            invented citations — words the model attributed to a speaker who did
            not say them
          </span>
          <ul className="crit-list">
            {fabrications.map((bad, index) => (
              <li key={`${bad.finding_id}-${index}`} className="crit-item">
                <div className="crit-rule">
                  <b>{bad.finding_id}</b> “{bad.claimed_quote}”
                </div>
                <div className="crit-meta">
                  <a
                    href={watchUrl(bad.video_id, bad.start_seconds)}
                    target="_blank"
                    rel="noreferrer"
                    className="crit-ts bad"
                  >
                    {bad.video_id} {clock(bad.start_seconds)}
                  </a>
                  <span className="crit-missnote">
                    matched {bad.ratio == null ? '—' : bad.ratio.toFixed(2)} against
                    the transcript there — {bad.reason}
                  </span>
                </div>
              </li>
            ))}
          </ul>
        </section>
      )}

      {conflicts.length > 0 && (
        <section className="crit-conflicts">
          <span className="microlabel">
            disagreements in this context — {conflicts.filter((c) => c.named_by.length > 0).length}{' '}
            of {conflicts.length} named rather than averaged away
          </span>
          <ul className="crit-list">
            {conflicts.map((conflict) => (
              <li
                key={conflict.conflict_id}
                className={`crit-item ${conflict.named_by.length > 0 ? 'hit' : ''}`}
              >
                {/* The axis, never a side. The panel shows whether the system
                    reported that the corpus splits here, not which creator it
                    should have agreed with. */}
                <div className="crit-rule">{conflict.axis}</div>
                <div className="crit-meta">
                  {conflict.video_ids.map((videoId) => (
                    <span key={videoId} className="crit-ts">
                      {videoId}
                    </span>
                  ))}
                  <span className={conflict.named_by.length > 0 ? '' : 'crit-missnote'}>
                    {conflict.named_by.length > 0
                      ? `named by ${conflict.named_by.join(', ')}`
                      : 'averaged away — no finding cited both sides'}
                  </span>
                </div>
              </li>
            ))}
          </ul>
        </section>
      )}

      {unmatchedFindings.length > 0 && (
        <section className="crit-extra">
          {/* "Outside the held-out list" was too generous to itself while any
              of these had been paired with a criterion on a minority of matcher
              repeats: those are inside the list on some repeats and were
              outvoted, not outside it. */}
          <span className="microlabel">
            {nearMisses.length > 0
              ? `${unmatchedFindings.length} findings this expert did not make — ${nearMisses.length} of
                 them were paired with a held-out criterion on a minority of matcher
                 repeats and outvoted; the other ${unmatchedFindings.length - nearMisses.length}
                 are outside the held-out list, not wrong`
              : `${unmatchedFindings.length} findings this expert did not make — not
                 wrong, just outside the held-out list`}
          </span>
          <ul className="crit-list">
            {unmatchedFindings.map((f) => {
              const paired = ballots.filter((b) =>
                b.votes.some((v) => v.finding_id === f.id),
              );
              return (
              <li key={f.id} className="crit-item">
                <div className="crit-rule">
                  <b>{f.id}</b> {f.criterion}
                  {paired.map((b) => (
                    <span
                      key={b.criterion_id}
                      className="crit-agree lost"
                      title="A minority of matcher repeats read this finding as the held-out criterion; the majority read it as a different rule."
                    >
                      paired with {b.criterion_id} on {votesFor(b, f.id)} of{' '}
                      {b.draws.length} repeats
                    </span>
                  ))}
                </div>
                <div className="crit-meta">
                  {f.citation_checks.map((c, index) => (
                    <a
                      key={`${c.video_id}-${c.start_seconds}-${index}`}
                      href={watchUrl(c.video_id, c.start_seconds)}
                      target="_blank"
                      rel="noreferrer"
                      className={`crit-ts ${c.resolved ? '' : 'bad'}`}
                      title={c.reason}
                    >
                      {c.video_id} {clock(c.start_seconds)}
                    </a>
                  ))}
                </div>
              </li>
              );
            })}
          </ul>
        </section>
      )}
    </div>
  );
}

function scoreOf(cell: CritiqueCell, metric: string): number | null {
  const value = cell.scores?.[metric];
  return typeof value === 'number' ? value : null;
}

/** The two metrics the grounding gate decides, and therefore can withhold. */
const GATED_METRICS = ['criteria_recall', 'evidence_precision'];

/**
 * An arm the gate could not grade — no score, rather than a bad one.
 *
 * The distinction the whole panel turns on. `criteria_recall` comes back null
 * for these, and null rendered as "—" is indistinguishable from a blank, which
 * a reader fills in with whatever the row said last week.
 */
function isUngraded(cell: CritiqueCell): boolean {
  return cell.gradable === false;
}

/** What the older, weaker rule scored for this metric — the published figure. */
function ungatedOf(cell: CritiqueCell, metric: string): number | null {
  const value =
    metric === 'criteria_recall'
      ? cell.criteria_recall_ungated
      : cell.evidence_precision_ungated;
  return typeof value === 'number' ? value : null;
}

/** "0.167–0.222" when the matcher's repeats disagreed, else nothing. */
function spreadOf(cell: CritiqueCell): string | null {
  const low = cell.score_spread?.criteria_recall_min;
  const high = cell.score_spread?.criteria_recall_max;
  if (typeof low !== 'number' || typeof high !== 'number' || low === high) return null;
  return `${low.toFixed(3)}–${high.toFixed(3)}`;
}

/**
 * This row's recall gap against the baseline, and whether it survives the noise.
 *
 * The whole point of putting a second arm in this table is the comparison, and
 * a comparison a reader has to do in their head is a comparison that gets done
 * wrong — particularly here, where the matcher disagrees with itself by more
 * than the gaps these arms produce. So the subtraction is printed, and it is
 * printed with the verdict attached: a lead whose repeat-range still overlaps
 * the baseline's has not shown anything, which is the same bar
 * `src/evals/pack_ablation.py:winner` applies to a pack ablation.
 *
 * Unless there is nothing to subtract from. When the gate cannot grade the
 * baseline, "−0.158 vs rag_llm_filtered" would republish the exact figure the
 * gate withdrew, dressed as a measured gap — the headline this panel used to
 * carry and the reason it stopped. That case returns `kind: 'none'` and says so
 * in words; it does not fall back to silence, because a missing delta reads as
 * "nobody got round to it" rather than "the comparison does not exist".
 */
type RecallDelta =
  | { kind: 'gap'; text: string; decisive: boolean }
  | { kind: 'none'; text: string; title: string };

function recallDelta(
  cell: CritiqueCell,
  base: CritiqueCell | undefined,
): RecallDelta | null {
  if (!base || base.setup === cell.setup) return null;
  const mine = scoreOf(cell, 'criteria_recall');
  const theirs = scoreOf(base, 'criteria_recall');
  if (theirs === null) {
    if (!isUngraded(base)) return null;
    return {
      kind: 'none',
      text: `no comparison — ${base.setup} is ungraded`,
      title:
        `The baseline ${base.setup} has no certified criteria_recall under this ` +
        `gate, so there is nothing to subtract from. Its ungated figure is a ` +
        `lower bound under the rule the gate replaced and is not a baseline.`,
    };
  }
  if (mine === null) return null;
  const gap = mine - theirs;
  const lo = base.score_spread?.criteria_recall_min;
  const hi = base.score_spread?.criteria_recall_max;
  const myLo = cell.score_spread?.criteria_recall_min;
  const myHi = cell.score_spread?.criteria_recall_max;
  const overlaps =
    typeof lo === 'number' &&
    typeof hi === 'number' &&
    typeof myLo === 'number' &&
    typeof myHi === 'number' &&
    myLo <= hi &&
    lo <= myHi;
  const sign = gap > 0 ? '+' : gap < 0 ? '−' : '±';
  return {
    kind: 'gap',
    text: `${sign}${Math.abs(gap).toFixed(3)} vs ${base.setup}`,
    decisive: gap !== 0 && !overlaps,
  };
}

/**
 * Why this arm has no score, directly under the row that has none.
 *
 * A full-width row rather than a chip. "Ungraded" as a badge invites the reader
 * to treat it as a footnote on a number; this arm has no number, and the reason
 * is a property of the *engine* — one shared retrieval pool for every finding —
 * which is a sentence, not a label. The published pair stays visible underneath
 * it, named as what it is: the figure the older rule produced, uncertified.
 */
function UngradedNote({ cell, gate }: { cell: CritiqueCell; gate: string | null }) {
  const total = cell.findings_total ?? 0;
  const withProvenance = cell.findings_with_provenance ?? 0;
  return (
    <div className="crit-ungraded">
      <b>{cell.setup} — not measured.</b> Every finding this engine produced came
      out of one shared retrieval pool, so per-finding provenance does not exist
      for it: {withProvenance} of {total} findings record what their own
      reasoning retrieved. The{' '}
      <code>{gate ?? 'retrieval_provenance'}</code> gate grounds a citation only
      when the chunk it resolves to is one that finding’s own retrieval returned,
      and grading this arm against the shared pool instead would pass the padding
      attack the pool was used to mount — so the scorer publishes no score here
      rather than one it cannot certify.{' '}
      <b>Ungated</b>, under the rule this run was published with, the row read
      criteria_recall <b>{fmt(cell.criteria_recall_ungated)}</b> ·
      evidence_precision <b>{fmt(cell.evidence_precision_ungated)}</b> — kept so
      nothing is lost, and a lower bound rather than a baseline: the same attack
      lifts it several-fold without reaching a single extra criterion.
    </div>
  );
}

export function CritiquePanel({ run }: { run: CritiqueRunSummary }) {
  const [open, setOpen] = useState<string | null>(null);
  const [detail, setDetail] = useState<CritiqueRunDetail | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Detail is fetched on first expand, not with the run list: /api/experiments
  // re-parses every committed run on every request, and the per-criterion rows
  // are the bulk of a critique file.
  const toggle = useCallback(
    async (setup: string) => {
      if (open === setup) {
        setOpen(null);
        return;
      }
      setOpen(setup);
      if (detail?.run_id === run.run_id) return;
      try {
        setDetail(await api.critiqueRun(run.run_id));
        setError(null);
      } catch (err) {
        setError((err as Error).message);
      }
    },
    [open, detail, run.run_id],
  );

  const leaks = run.held_out_leaks ?? 0;
  const fabricated = run.fabricated_citations ?? 0;
  const baseCell = run.cells.find((c) => c.setup === run.baseline);
  const gate = run.grounding_gate ?? null;
  const ungraded = run.cells.filter(isUngraded);
  // The baseline being ungraded changes what this table *is*: not a comparison
  // against a reference arm, but a set of arms each reported on its own.
  const baselineUngraded = baseCell !== undefined && isUngraded(baseCell);
  // setup + one per metric + recall_all + recall_grouped + findings + cited +
  // leaks + the expand button.
  const columns = run.metrics.length + 7;

  return (
    <section className="exp-card">
      <div className="exp-cardhead">
        <div>
          <h3>Critique eval — held-out expert</h3>
          <span className="exp-sub">
            {run.run_id} · reviewing <code>{run.artifact_url}</code> ·{' '}
            {run.criteria_applicable}/{run.criteria_total} criteria apply to a{' '}
            {run.artifact_kind}
          </span>
        </div>
        <div className="crit-chips">
          <span className={`exp-tag ${leaks === 0 ? 'ok' : 'bad'}`}>
            {leaks === 0 ? 'held-out absent · 0 leaks' : `${leaks} held-out leaks`}
          </span>
          {/* A model inventing a citation is the failure this harness exists to
              catch, so it gets a chip of its own rather than living inside a
              cell nobody expands. Silence here is the good state. */}
          {fabricated > 0 ? (
            <span className="exp-tag bad" title="Cited quotes that are not in the transcript at the timestamp cited.">
              {fabricated} invented citation{fabricated === 1 ? '' : 's'}
            </span>
          ) : null}
          {/* Which rule produced the numbers below. Two runs of this harness
              scored under different gates are not comparable, and a reader who
              has to open the JSON to find out which one applied will not. */}
          {gate ? (
            <span
              className="exp-tag"
              title="A citation grounds a finding only when the chunk it resolves to is one that finding’s own reasoning retrieved. Arms that cannot record that are reported as not measured rather than scored."
            >
              gate · {gate}
            </span>
          ) : null}
          {ungraded.length > 0 ? (
            <span
              className="exp-tag"
              title={`${ungraded.map((c) => c.setup).join(', ')} produced every finding from one shared retrieval pool, so this gate cannot certify a score for them.`}
            >
              {ungraded.length} arm{ungraded.length === 1 ? '' : 's'} ungraded
            </span>
          ) : null}
        </div>
      </div>

      <p className="crit-intro">
        <b>{run.held_out_title}</b> was excluded from every retrieval path. The
        criteria that expert applies were extracted by hand with the timestamp
        each was said at; the system then reviewed the document above using only
        the rest of the corpus. Recall is how many of those criteria it reached
        without ever seeing the video they came from — counting only findings
        that rest on corpus evidence no other finding also claims, so reciting
        advice the model already knew cannot score.
        {gate ? (
          <>
            {' '}
            Under the <code>{gate}</code> gate that evidence must also be a chunk
            that finding’s <i>own</i> reasoning retrieved, which is why an arm
            that emits every finding from one shared pool is reported as{' '}
            <b>not measured</b> rather than scored.
          </>
        ) : null}
        {run.match_repeats
          ? ` The pairing is a majority vote over ${run.match_repeats} matcher runs;
             any range shown beside recall is that vote disagreeing with itself, and
             a row that says “outvoted” had a pairing some of those runs made and the
             majority threw away — expand it for the ballots.`
          : ''}{' '}
        {baselineUngraded ? (
          <>
            The baseline <b>{run.baseline}</b> is <b>ungraded</b> under this run’s
            grounding gate, so no row carries a gap against it: subtracting from a
            figure the scorer does not certify would manufacture a comparison that
            does not exist. Each arm is reported on its own, and every ungraded row
            says below itself why it has no number and what it read before.
          </>
        ) : (
          <>
            Every row after the baseline carries its recall gap against it, marked
            “within noise” when that gap is smaller than the range the matcher
            produced on its own — the arms are reported side by side whichever way
            the subtraction falls.
          </>
        )}
      </p>

      <div className="exp-scroll">
        <table className="exp-table">
          <thead>
            <tr>
              <th>setup</th>
              {run.metrics.map((metric) => (
                <th key={metric} className="num">
                  {metric}
                </th>
              ))}
              {/* Both denominators, always. criteria_recall drops the criteria a
                  portfolio cannot be judged on, and every one of those
                  exclusions removed a criterion the baseline missed — so the
                  number that is immune to that choice is rendered beside it
                  rather than left in the JSON. */}
              <th className="num" title="recall over all 24 criteria, including the resume-only ones">
                recall_all
              </th>
              <th className="num" title="recall counted once per rule, merging near-duplicate criteria">
                recall_grouped
              </th>
              <th className="num">findings</th>
              <th className="num">cited</th>
              <th className="num">leaks</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {run.cells.map((cell) => {
              const isOpen = open === cell.setup;
              const delta = recallDelta(cell, baseCell);
              const rowUngraded = isUngraded(cell);
              return (
                <Fragment key={cell.setup}>
                  <tr>
                    <td className="exp-cfg">
                      {cell.setup}
                      {cell.setup === run.baseline && (
                        <span className="exp-basetag">base</span>
                      )}
                    </td>
                    {run.metrics.map((metric) => (
                      <td key={metric} className="num">
                        {/* Words, not a dash. A blank cell in a numeric column
                            is read as "the number is elsewhere", and the number
                            a reader supplies from memory is the one the gate
                            just withdrew. The published figure is right here,
                            underneath, labelled as ungated. */}
                        {rowUngraded && GATED_METRICS.includes(metric) ? (
                          <>
                            <span
                              className="crit-nomeasure"
                              title={
                                cell.ungradable_reason ??
                                'This arm cannot be graded under the run’s grounding gate.'
                              }
                            >
                              not measured
                            </span>
                            <span
                              className="crit-spread lost"
                              title="What the rule this gate replaced scored — the figure this run published. A lower bound, not a baseline."
                            >
                              ungated {fmt(ungatedOf(cell, metric))}
                            </span>
                          </>
                        ) : (
                          fmt(scoreOf(cell, metric))
                        )}
                        {/* Recall is a vote over repeats that disagree with each
                            other. Printing it bare invites a later slice to
                            claim a lift smaller than this range. */}
                        {metric === 'criteria_recall' && spreadOf(cell) ? (
                          <span className="crit-spread">{spreadOf(cell)}</span>
                        ) : null}
                        {/* The comparison itself, not the ingredients for one.
                            Published whichever way it falls — a later arm that
                            loses to the chunk dump says so here. */}
                        {/* Why the number is what it is, before anyone clicks.
                            A recall the vote pushed down and a recall nothing
                            reached print the same digits, and the reader who
                            walks away after reading "0.000" is the one this
                            line exists for. */}
                        {metric === 'criteria_recall' && outvotedBallots(cell).length > 0 ? (
                          <span
                            className="crit-spread lost"
                            title={`${outvotedBallots(cell).length} held-out criteria were paired with a finding on some of the matcher’s repeats, and “no match” won the majority vote on each. Expand this row for the ballots.`}
                          >
                            {outvotedBallots(cell).length} pairing
                            {outvotedBallots(cell).length === 1 ? '' : 's'} outvoted
                          </span>
                        ) : null}
                        {/* Not on a row that is itself ungraded: "not measured"
                            above already says there is no number here, and
                            "no comparison" underneath it is the same fact
                            twice. The line is for a *graded* arm that has lost
                            its reference point. */}
                        {metric === 'criteria_recall' && delta && !rowUngraded ? (
                          <span
                            className={`crit-spread${delta.kind === 'none' ? ' lost nocmp' : ''}`}
                            title={
                              delta.kind === 'none'
                                ? delta.title
                                : delta.decisive
                                  ? 'This gap is larger than the matcher’s own range across repeats.'
                                  : 'This gap sits inside the matcher’s own range across repeats — the two arms are not distinguishable here.'
                            }
                          >
                            {delta.text}
                            {delta.kind === 'gap' && !delta.decisive ? ' (within noise)' : ''}
                          </span>
                        ) : null}
                        {/* The denominator, always, and it is the whole point
                            of this column: "0.000" and "no disagreement was in
                            context" are different results and the number alone
                            cannot tell them apart. Retrieval fixes this
                            fraction before the answering call, so no amount of
                            extra findings can move it. */}
                        {metric === 'contested_coverage' ? (
                          <span
                            className="crit-spread"
                            title="Disagreements the corpus contains and this context held BOTH sides of, versus how many the findings named instead of averaging away."
                          >
                            {cell.conflicts_named ?? 0}/{cell.conflicts_in_context ?? 0} in
                            context
                          </span>
                        ) : null}
                      </td>
                    ))}
                    {/* These two are recall under different denominators, so the
                        gate withholds them as well — and they are exactly where
                        a withheld headline would leak back in if it did not. */}
                    <td
                      className="num"
                      title={rowUngraded ? 'not measured — see the note below this row' : undefined}
                    >
                      {fmt(cell.criteria_recall_all)}
                    </td>
                    <td
                      className="num"
                      title={rowUngraded ? 'not measured — see the note below this row' : undefined}
                    >
                      {fmt(cell.criteria_recall_grouped)}
                    </td>
                    <td className="num">
                      {cell.findings_grounded ?? '—'}/{cell.findings_total ?? '—'}
                    </td>
                    <td className="num">
                      {cell.citations_resolved ?? '—'}/{cell.citations_total ?? '—'}
                    </td>
                    <td className={`num${(cell.held_out_leaks ?? 0) > 0 ? ' bad' : ''}`}>
                      {cell.held_out_leaks ?? '—'}
                    </td>
                    <td>
                      <button
                        type="button"
                        className="btn sm"
                        onClick={() => void toggle(cell.setup)}
                      >
                        {isOpen ? 'Collapse' : 'Expand'}
                      </button>
                    </td>
                  </tr>
                  {rowUngraded && (
                    <tr className="crit-ungradedrow">
                      <td colSpan={columns}>
                        <UngradedNote cell={cell} gate={gate} />
                      </td>
                    </tr>
                  )}
                  {/* A second row rather than a block inside the last cell.
                      The precedent elsewhere avoids colspan, but a criterion is
                      a whole sentence and the metric columns are narrow — put
                      the detail in the last cell and it renders in a column a
                      few characters wide, pushed off to the right. */}
                  {isOpen && (
                    <tr className="crit-detailrow">
                      <td colSpan={columns}>
                        {error && <p className="errtext">{error}</p>}
                        {!error && !detail && (
                          <p className="crit-empty">Loading criteria…</p>
                        )}
                        {detail && (
                          // Ballots come from the run *list*, not the detail
                          // fetch: they are already on the row that drew this
                          // reader in, and reading them from one place keeps
                          // the chip above and the section below consistent.
                          <CritiqueDetail
                            run={detail}
                            setup={cell.setup}
                            ballots={cell.match_ballots ?? []}
                            baselineSetup={run.baseline}
                            baselineBallots={baseCell?.match_ballots ?? []}
                          />
                        )}
                      </td>
                    </tr>
                  )}
                </Fragment>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}

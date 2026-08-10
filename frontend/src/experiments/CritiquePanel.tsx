import { Fragment, useCallback, useState } from 'react';

import { api } from '../api/client';
import type {
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

/**
 * The expanded row: every held-out criterion, matched or not, with its timestamp.
 *
 * This is the part that makes the score falsifiable. A recall number on its own
 * is a claim; the list below is the evidence, criterion by criterion, each one
 * linked to the second of the held-out video it was extracted from — so a
 * reviewer can watch the expert say it and decide whether the system's finding
 * really is the same rule.
 */
function CritiqueDetail({ run, setup }: { run: CritiqueRunDetail; setup: string }) {
  const cell = run.cells.find((c) => c.setup === setup);
  if (!cell) return <p className="crit-empty">No detail stored for {setup}.</p>;

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
            {missed.map((m) => (
              <li key={m.id} className="crit-item miss">
                <div className="crit-rule">
                  <b>{m.id}</b> {m.criterion}
                  {!m.applies_to.includes(run.artifact_kind) && (
                    <span className="crit-na">n/a to a {run.artifact_kind}</span>
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
            ))}
          </ul>
        </section>
      </div>

      {unmatchedFindings.length > 0 && (
        <section className="crit-extra">
          <span className="microlabel">
            {unmatchedFindings.length} findings this expert did not make — not
            wrong, just outside the held-out list
          </span>
          <ul className="crit-list">
            {unmatchedFindings.map((f) => (
              <li key={f.id} className="crit-item">
                <div className="crit-rule">
                  <b>{f.id}</b> {f.criterion}
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
            ))}
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

/** "0.167–0.222" when the matcher's repeats disagreed, else nothing. */
function spreadOf(cell: CritiqueCell): string | null {
  const low = cell.score_spread?.criteria_recall_min;
  const high = cell.score_spread?.criteria_recall_max;
  if (typeof low !== 'number' || typeof high !== 'number' || low === high) return null;
  return `${low.toFixed(3)}–${high.toFixed(3)}`;
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
        <span className={`exp-tag ${leaks === 0 ? 'ok' : 'bad'}`}>
          {leaks === 0 ? 'held-out absent · 0 leaks' : `${leaks} held-out leaks`}
        </span>
      </div>

      <p className="crit-intro">
        <b>{run.held_out_title}</b> was excluded from every retrieval path. The
        criteria that expert applies were extracted by hand with the timestamp
        each was said at; the system then reviewed the document above using only
        the rest of the corpus. Recall is how many of those criteria it reached
        without ever seeing the video they came from — counting only findings
        that rest on corpus evidence no other finding also claims, so reciting
        advice the model already knew cannot score.
        {run.match_repeats
          ? ` The pairing is a majority vote over ${run.match_repeats} matcher runs;
             any range shown beside recall is that vote disagreeing with itself.`
          : ''}
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
                        {fmt(scoreOf(cell, metric))}
                        {/* Recall is a vote over repeats that disagree with each
                            other. Printing it bare invites a later slice to
                            claim a lift smaller than this range. */}
                        {metric === 'criteria_recall' && spreadOf(cell) ? (
                          <span className="crit-spread">{spreadOf(cell)}</span>
                        ) : null}
                      </td>
                    ))}
                    <td className="num">{fmt(cell.criteria_recall_all)}</td>
                    <td className="num">{fmt(cell.criteria_recall_grouped)}</td>
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
                        {detail && <CritiqueDetail run={detail} setup={cell.setup} />}
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

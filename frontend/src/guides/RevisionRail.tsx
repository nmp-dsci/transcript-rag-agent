import type { GuideDetail, GuideJob, GuideReceipt } from '../api/types';
import { ActivityLog, StageRail } from './ResearchMap';

interface Props {
  guide: GuideDetail;
  /** The current job, whatever guide it belongs to; the rail shows it only when it is this guide's revision. */
  job: GuideJob | null;
  viewing: number;
  onView: (version: number) => void;
}

export interface RevisionRow {
  version: number;
  date: string;
  origin: 'written' | 'imported' | 'revised';
  addressed: number;
  deferred: number;
  rejected: number;
  receipt: GuideReceipt | null;
}

/** One row per published version, newest first: v1 is the write or import,
 *  every later version is a revision with the receipt that produced it. */
export function revisionRows(guide: GuideDetail): RevisionRow[] {
  const imported = guide.provenance?.pipeline !== 'guides write';
  const receipts = Object.values(guide.receipts);
  return guide.versions
    .map((version): RevisionRow => {
      const receipt = receipts.find((r) => r.version === version) ?? null;
      const count = (outcome: string) => receipt?.items.filter((item) => item.outcome === outcome).length ?? 0;
      return {
        version,
        date: receipt?.created_at.slice(0, 10) ?? (version === 1 ? guide.compiled_at : ''),
        origin: receipt ? 'revised' : imported ? 'imported' : 'written',
        addressed: count('addressed'),
        deferred: count('deferred'),
        rejected: count('rejected'),
        receipt,
      };
    })
    .sort((a, b) => b.version - a.version);
}

/** True when the job is a revision of this guide — running, done or failed. */
export function isRevisionOf(job: GuideJob | null, slug: string): job is GuideJob {
  return !!job && job.kind === 'revise' && job.slug === slug;
}

function LiveRevision({ job }: { job: GuideJob }) {
  const n = job.comment_ids.length;
  return (
    <section className={`rv-live ${job.status}`} aria-label="Revision in progress">
      <div className="rv-live-h">
        <span className={`badge ${job.status === 'running' ? 'acc' : job.status === 'error' ? 'bad' : 'good'}`}>
          {job.status === 'running' ? `● ${job.stage ?? 'starting'}` : job.status === 'error' ? '✕ failed' : `✓ published v${job.version ?? '?'}`}
        </span>
        <span className="rmeta">
          {n} comment{n === 1 ? '' : 's'} as one batch · started {job.started_at.slice(11, 16)}
        </span>
      </div>
      <StageRail job={job} />
      <div className="gj-revise-ids">
        {job.comment_ids.map((id) => (
          <code key={id}>{id}</code>
        ))}
      </div>
      <div className="rv-kpis">
        <span>{job.counters.files_read ?? 0} files read</span>
        <span>{job.counters.retrieval_queries ?? 0} retrievals</span>
        <span>{job.counters.tool_calls ?? 0} tool calls</span>
        {job.cite_total != null && (
          <span>
            cites {job.cite_valid}/{job.cite_total}
          </span>
        )}
      </div>
      {job.status === 'error' && <p className="gj-err rv-msg">{job.error}</p>}
      {job.message && job.status === 'running' && <p className="rv-msg">{job.message}</p>}
      <ActivityLog activity={job.activity} limit={200} />
    </section>
  );
}

/** The revisions rail: the live revision log while the agent works, so the
 *  reader never leaves the page, and a table of every published version
 *  with the receipt that produced it. */
export function RevisionRail({ guide, job, viewing, onView }: Props) {
  const rows = revisionRows(guide);
  const live = isRevisionOf(job, guide.slug) ? job : null;
  const revisions = rows.filter((row) => row.receipt).length;
  return (
    <aside className="rail rv" aria-label="Revisions">
      <div className="rail-head">
        <strong>Revisions</strong>
        <span className="rmeta">
          v{guide.current_version} current · {revisions} revision{revisions === 1 ? '' : 's'}
        </span>
      </div>
      <div className="rail-list rv-list">
        {live && <LiveRevision job={live} />}
        {!live && guide.comments_open > 0 && (
          <div className="rail-empty">
            {guide.comments_open} open comment{guide.comments_open === 1 ? '' : 's'} — send them from Commentary and the
            revision runs here while you keep reading.
          </div>
        )}
        <table className="rv-table">
          <thead>
            <tr>
              <th>version</th>
              <th>date</th>
              <th>outcome</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr
                key={row.version}
                className={row.version === viewing ? 'on' : ''}
                onClick={() => onView(row.version)}
                title={`view v${row.version}`}
              >
                <td>
                  <code>v{row.version}</code>
                  {row.version === guide.current_version && <span className="rv-cur"> current</span>}
                </td>
                <td>{row.date || '—'}</td>
                <td>
                  {row.receipt ? (
                    <>
                      {row.addressed > 0 && <span className="badge good">{row.addressed} addressed</span>}
                      {row.deferred > 0 && <span className="badge plain">{row.deferred} deferred</span>}
                      {row.rejected > 0 && <span className="badge bad">{row.rejected} rejected</span>}
                    </>
                  ) : (
                    <span className="badge plain">{row.origin}</span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        {rows
          .filter((row) => row.receipt)
          .map((row) => (
            <details key={row.version} className="cr-receipt" open={row.version === viewing}>
              <summary>
                Receipt for v{row.version} · {row.receipt!.items.length} comment{row.receipt!.items.length === 1 ? '' : 's'}
              </summary>
              {row.receipt!.summary && <p>{row.receipt!.summary}</p>}
              <ul>
                {row.receipt!.items.map((item) => (
                  <li key={item.id}>
                    <span className={`badge ${item.outcome === 'addressed' ? 'good' : item.outcome === 'rejected' ? 'bad' : 'plain'}`}>
                      {item.outcome}
                    </span>{' '}
                    <code>{item.id}</code>
                    {item.reason && <span className="cr-reason"> — {item.reason}</span>}
                  </li>
                ))}
              </ul>
              {row.receipt!.changed_sections.length > 0 && <p className="cr-dim">changed: {row.receipt!.changed_sections.join(', ')}</p>}
            </details>
          ))}
      </div>
    </aside>
  );
}

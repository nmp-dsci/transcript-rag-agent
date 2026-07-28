import { useEffect, useRef, useState } from 'react';

import { api } from '../api/client';
import type { MatrixJob } from '../api/types';

interface Props {
  /** Called once each time a run finishes, so the tab can reload its runs. */
  onRunFinished: () => void;
}

const STATUS_BADGE: Record<MatrixJob['status'], string> = {
  running: 'acc',
  done: 'good',
  error: 'bad',
};

export function MatrixRunPanel({ onRunFinished }: Props) {
  const [job, setJob] = useState<MatrixJob | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);
  const notifiedRef = useRef<Set<string>>(new Set());

  // One live subscription for the panel's lifetime. The stream is seeded with
  // the current job, so a tab opened mid-run shows it immediately — and a run
  // started from another tab shows up here too.
  useEffect(() => {
    const controller = new AbortController();
    void api
      .subscribeMatrixRun(
        {
          snapshot: (data) => setJob(data.job),
          job: (data) => setJob(data.job),
        },
        controller.signal,
      )
      .catch((err) => {
        if (!controller.signal.aborted) console.error('Matrix run stream failed', err);
      });
    return () => controller.abort();
  }, []);

  // Reload the committed runs once per finished job, not on every progress tick.
  useEffect(() => {
    if (!job || job.status !== 'done' || notifiedRef.current.has(job.id)) return;
    notifiedRef.current.add(job.id);
    onRunFinished();
  }, [job, onRunFinished]);

  const running = job?.status === 'running';

  const start = async () => {
    setStarting(true);
    setError(null);
    try {
      setJob(await api.startMatrixRun());
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setStarting(false);
    }
  };

  const pct =
    job && job.cells_total > 0 ? Math.round((job.cells_done / job.cells_total) * 100) : 0;

  return (
    <div className="panel mrun">
      <div className="mrun-head">
        <div>
          <b style={{ fontSize: 13 }}>Run the eval matrix</b>
          <p className="sub" style={{ margin: '4px 0 0' }}>
            Every engine answers the same golden questions, judged by one judge — the run the
            Scoreboard ranks. Cells already scored under this exact configuration are reused,
            so re-running after adding a question only pays for what changed.
          </p>
        </div>
        <button
          type="button"
          className="btn pri"
          onClick={() => void start()}
          disabled={running || starting}
        >
          {running ? 'Running…' : starting ? 'Starting…' : '↻ Run eval matrix'}
        </button>
      </div>

      {error ? <p className="errtext">{error}</p> : null}

      {job ? (
        <div className="mrun-job">
          <div className="mrun-status">
            {running ? <span className="pulse" /> : null}
            <span className={`badge ${STATUS_BADGE[job.status]}`}>{job.status}</span>
            <span className="mrun-setups">{job.setups.join(' · ')}</span>
            {job.cells_total > 0 ? (
              <span className="badge plain">
                {job.cells_done}/{job.cells_total} cells
              </span>
            ) : null}
            <span className="nchip">
              {job.cache_hits} cached · {job.cache_misses} scored
            </span>
          </div>

          {job.cells_total > 0 ? (
            <div className="mbar mrun-bar">
              <i style={{ width: `${pct}%` }} />
            </div>
          ) : null}

          {job.status === 'running' && job.message ? (
            <p className="sub mrun-message">{job.message}</p>
          ) : null}

          {job.status === 'error' && job.error ? <p className="errtext">{job.error}</p> : null}

          {job.status === 'done' && job.run_id ? (
            <p className="sub mrun-message">
              Committed <code>{job.run_id}</code> — pick it in the Scoreboard&apos;s run
              selector to rank against it.
            </p>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}

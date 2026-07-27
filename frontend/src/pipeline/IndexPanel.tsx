import { useEffect, useRef, useState } from 'react';

import { api } from '../api/client';
import type { IngestionJob } from '../api/types';
import { insightBadgeClass } from './insights';

interface Props {
  onIndexed: () => void;
  /** Jump the corpus tree to a video the run just added. */
  onViewVideo: (videoId: string) => void;
}

type Mode = 'video' | 'channel';

const STATUS_LABEL: Record<IngestionJob['status'], string> = {
  queued: 'Queued',
  running: 'Indexing…',
  done: 'Done',
  error: 'Failed',
};

/** One job's live progress: status pill, stage message, and — once done —
 * the same added-videos/insights summary the single-job panel used to show. */
function JobRow({ job, onViewVideo }: { job: IngestionJob; onViewVideo: (videoId: string) => void }) {
  const result = job.result;
  return (
    <li className={`idxq-row ${job.status}`}>
      <div className="idxq-head">
        {job.status === 'running' ? <span className="pulse" /> : null}
        <span className={`badge ${STATUS_BADGE_CLASS[job.status]}`}>
          {STATUS_LABEL[job.status]}
        </span>
        <span className="idxq-target">
          {job.mode === 'channel' ? `channel · latest ${job.latest ?? ''} · ` : ''}
          {job.target}
        </span>
      </div>

      {job.status === 'running' && job.message ? (
        <p className="sub idxq-message">{job.message}</p>
      ) : null}

      {job.status === 'error' && job.error ? <p className="errtext">{job.error}</p> : null}

      {job.status === 'done' && result ? (
        <div className="idx-result">
          <div className="idx-result-head">
            <span className="badge good">+{result.added_video_count} videos</span>
            <span className="badge acc">+{result.added_chunk_count} chunks</span>
            <span className="badge plain">
              now {result.totals.videos} videos · {result.totals.chunks} chunks ·{' '}
              {result.totals.channels} channels
            </span>
          </div>

          {result.added_videos.length > 0 ? (
            <div className="idx-added">
              {result.added_videos.map((video) => (
                <button
                  type="button"
                  className="btn sm"
                  key={video.video_id}
                  onClick={() => onViewVideo(video.video_id)}
                >
                  view in tree · {(video.title || video.video_id).slice(0, 44)}
                </button>
              ))}
            </div>
          ) : (
            <p className="sub" style={{ margin: '8px 0 0' }}>
              Nothing new — every video was already in the index.
            </p>
          )}

          {result.graph ? (
            <p className="sub" style={{ margin: '8px 0 0' }}>
              {result.graph.ok ? (
                <span className="badge acc">
                  graph: +{result.graph.extracted ?? 0} chunk{result.graph.extracted === 1 ? '' : 's'} extracted
                </span>
              ) : (
                <span className="badge bad" title={result.graph.error}>
                  graph extraction failed — vector RAG only until index-graph runs again
                </span>
              )}
            </p>
          ) : null}

          {result.insights.length > 0 ? (
            <div className="idx-added">
              {result.insights.map((insight, index) => (
                <span
                  className={`badge ${insightBadgeClass(insight.level)}`}
                  key={`${insight.kind}-${index}`}
                >
                  {insight.message}
                </span>
              ))}
            </div>
          ) : null}
        </div>
      ) : null}
    </li>
  );
}

const STATUS_BADGE_CLASS: Record<IngestionJob['status'], string> = {
  queued: 'plain',
  running: 'acc',
  done: 'good',
  error: 'bad',
};

export function IndexPanel({ onIndexed, onViewVideo }: Props) {
  const [open, setOpen] = useState(false);
  const [mode, setMode] = useState<Mode>('video');
  const [url, setUrl] = useState('');
  const [channel, setChannel] = useState('');
  const [latest, setLatest] = useState(5);
  const [submitError, setSubmitError] = useState<string | null>(null);

  const [jobs, setJobs] = useState<IngestionJob[]>([]);
  const notifiedDoneRef = useRef<Set<string>>(new Set());

  // A single live subscription for the panel's lifetime — not per-job. Every
  // queued/running/done/error transition for every job (including ones
  // queued from another browser tab) arrives on this one connection, which
  // is what lets the form stay open and enabled while jobs run: submitting
  // never opens or blocks on a request of its own.
  useEffect(() => {
    const controller = new AbortController();
    void api
      .subscribeIndexQueue(
        {
          snapshot: (data) => setJobs(data.jobs),
          job: (data) =>
            setJobs((current) => {
              const index = current.findIndex((job) => job.id === data.job.id);
              if (index === -1) return [...current, data.job];
              const next = [...current];
              next[index] = data.job;
              return next;
            }),
        },
        controller.signal,
      )
      .catch((err) => {
        if (!controller.signal.aborted) console.error('Ingestion queue stream failed', err);
      });
    return () => controller.abort();
  }, []);

  // Refresh the corpus once per job that finishes, instead of on every
  // progress tick — a channel run can otherwise trigger this dozens of times.
  useEffect(() => {
    for (const job of jobs) {
      if (job.status !== 'done' || notifiedDoneRef.current.has(job.id)) continue;
      notifiedDoneRef.current.add(job.id);
      onIndexed();
    }
  }, [jobs, onIndexed]);

  const submit = async () => {
    const target = mode === 'video' ? url.trim() : channel.trim();
    if (!target) {
      setSubmitError(mode === 'video' ? 'Enter a video URL.' : 'Enter a channel URL or @handle.');
      return;
    }
    setSubmitError(null);
    const payload =
      mode === 'video'
        ? { mode: 'video' as const, url: target }
        : { mode: 'channel' as const, channel: target, latest };
    try {
      await api.enqueueIndex(payload);
      // Clear for the next entry immediately — enqueuing never blocks, so
      // the form is ready for another submission right away.
      if (mode === 'video') setUrl('');
      else setChannel('');
    } catch (err) {
      setSubmitError((err as Error).message);
    }
  };

  const running = jobs.filter((job) => job.status === 'running');
  const queued = jobs.filter((job) => job.status === 'queued');
  const headline =
    running.length > 0
      ? `Indexing ${running.length} · ${queued.length} queued`
      : queued.length > 0
        ? `${queued.length} queued`
        : '';

  return (
    <div className="pipe-index">
      <div className="formrow" style={{ margin: 0 }}>
        <button
          type="button"
          className={`pill${open ? ' on' : ''}`}
          onClick={() => setOpen(!open)}
          aria-expanded={open}
        >
          + Index new content
        </button>
        {!open && headline ? <span className="result acc">{headline}</span> : null}
        {!open && running.length > 0 ? <span className="pulse" /> : null}
      </div>

      {open ? (
        <div className="pipe-index-body">
          <p className="sub" style={{ margin: '10px 0 8px' }}>
            Fetch transcripts, chunk them on transcript timings, embed the chunks and write a
            per-video summary. Add as many videos or channels as you like — each one queues and
            runs in turn, so you never have to wait for one to finish before adding the next.
          </p>

          <div className="formrow">
            <button
              type="button"
              className={`pill${mode === 'video' ? ' on' : ''}`}
              onClick={() => setMode('video')}
            >
              Single video
            </button>
            <button
              type="button"
              className={`pill${mode === 'channel' ? ' on' : ''}`}
              onClick={() => setMode('channel')}
            >
              Channel · latest N
            </button>
          </div>

          {mode === 'video' ? (
            <div className="formrow">
              <input
                type="text"
                value={url}
                spellCheck={false}
                placeholder="https://www.youtube.com/watch?v=…"
                aria-label="Video URL"
                onChange={(event) => setUrl(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === 'Enter') void submit();
                }}
              />
            </div>
          ) : (
            <div className="formrow">
              <input
                type="text"
                value={channel}
                spellCheck={false}
                placeholder="Channel URL or @handle"
                aria-label="Channel"
                onChange={(event) => setChannel(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === 'Enter') void submit();
                }}
              />
              <input
                type="number"
                min={1}
                max={50}
                value={latest}
                title="How many latest videos"
                aria-label="How many latest videos"
                onChange={(event) => setLatest(Number(event.target.value) || 5)}
              />
            </div>
          )}

          <div className="formrow">
            <button type="button" className="btn pri" onClick={() => void submit()}>
              Add to queue
            </button>
            {submitError ? <span className="errtext">{submitError}</span> : null}
          </div>

          {jobs.length > 0 ? (
            <ul className="idxq-list" aria-label="Ingestion queue">
              {[...jobs].reverse().map((job) => (
                <JobRow key={job.id} job={job} onViewVideo={onViewVideo} />
              ))}
            </ul>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}

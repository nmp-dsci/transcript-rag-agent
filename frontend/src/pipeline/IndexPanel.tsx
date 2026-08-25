import { useEffect, useRef, useState } from 'react';

import { api } from '../api/client';
import { CORE_STAGES } from '../api/types';
import type { EnrichmentState, EnrichmentSummary, IngestionJob, Video } from '../api/types';
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

const ENRICHMENT_BADGE: Record<EnrichmentState, string> = {
  done: 'good',
  pending: 'warn',
  failed: 'bad',
};

/** Where a run has actually reached, stage by stage.
 *
 * Only rendered while a job is running or once it is done — a queued job has
 * not started any stage, and showing it at 0/4 would suggest otherwise. */
function StageSteps({ job }: { job: IngestionJob }) {
  if (job.status === 'queued') return null;
  const reached = job.status === 'done' ? CORE_STAGES.length : (job.stage_index ?? 0);
  return (
    <>
      <ol className="idxq-steps" aria-label="Indexing stages">
        {CORE_STAGES.map((stage, index) => {
          const position = index + 1;
          const state =
            job.status === 'error' && position === reached
              ? 'failed'
              : position < reached
                ? 'done'
                : position === reached
                  ? 'active'
                  : 'waiting';
          return (
            <li key={stage} className={`idxq-step ${state}`}>
              <span className="idxq-dot">{state === 'done' ? '✓' : position}</span>
              <span className="idxq-step-name">{stage}</span>
            </li>
          );
        })}
      </ol>
      <div className="idxq-steprow">
        <span className="idxq-stepcount">
          {reached} / {CORE_STAGES.length}
          {job.status === 'done' ? ' indexed' : ''}
        </span>
        {job.status === 'running' && job.message ? (
          <span className="sub">{job.message}</span>
        ) : null}
      </div>
    </>
  );
}

/** Enrichment status for one video, spelled out rather than left to the
 * absence of a summary to imply. `pending` and `failed` look identical in a
 * corpus that only stores the summary itself — which is how 14 videos went
 * unnoticed. */
function EnrichmentLine({ videos }: { videos: Video[] }) {
  if (videos.length === 0) return null;
  const worst = (pick: (video: Video) => EnrichmentState): EnrichmentState =>
    videos.some((v) => pick(v) === 'failed')
      ? 'failed'
      : videos.some((v) => pick(v) === 'pending')
        ? 'pending'
        : 'done';
  const summary = worst((v) => v.summary_status);
  const graph = worst((v) => v.graph_status);
  const source = videos.find((v) => v.summary_source)?.summary_source;
  return (
    <div className="idxq-enrich">
      <span className="idxq-enrich-label">ENRICHMENT</span>
      <span className={`badge ${ENRICHMENT_BADGE[summary]}`}>
        summary · {summary}
        {source && summary === 'done' ? ` · ${source}` : ''}
      </span>
      <span className={`badge ${ENRICHMENT_BADGE[graph]}`}>graph · {graph}</span>
      {graph !== 'done' ? (
        <span className="sub">Retrievable now; GraphRAG catches up on the next pass.</span>
      ) : null}
    </div>
  );
}

/** The corpus-wide enrichment backlog, and the button that clears it.
 *
 * Turns the health strip's count into the action that fixes it. Graph
 * extraction is the only step that still needs a paid provider, so it is the
 * only one that can sit pending indefinitely — summaries come from the
 * YouTube description and are written during indexing. */
function EnrichmentBanner({ jobs }: { jobs: IngestionJob[] }) {
  const [state, setState] = useState<EnrichmentSummary | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Re-read whenever a job finishes: that is exactly when the backlog moves.
  const doneCount = jobs.filter((job) => job.status === 'done').length;
  useEffect(() => {
    let live = true;
    api
      .enrichmentState()
      .then((next) => live && setState(next))
      .catch(() => undefined);
    return () => {
      live = false;
    };
  }, [doneCount]);

  if (!state) return null;
  const pending = state.graph_pending.length;
  const summaryPending = state.summary_pending.length;
  if (pending === 0 && summaryPending === 0) return null;

  const run = async () => {
    setBusy(true);
    setError(null);
    try {
      await api.runEnrichment({});
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="idxq-row enrich-banner">
      <div className="idxq-head">
        <span className="badge plain">Corpus</span>
        <span className="idxq-target">
          {pending} video(s) awaiting graph enrichment
          {summaryPending > 0 ? ` · ${summaryPending} without a summary` : ''}
        </span>
      </div>
      <div className="idxq-enrich">
        <span className="sub">
          Reads chunks already indexed — no transcript re-fetch, so no Supadata credits.
        </span>
        <button type="button" className="btn" disabled={busy || pending === 0} onClick={() => void run()}>
          {busy ? 'Queueing…' : 'Run enrichment pass'}
        </button>
        {error ? <span className="errtext">{error}</span> : null}
      </div>
    </div>
  );
}

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

      {job.mode === 'enrichment' ? null : <StageSteps job={job} />}

      {job.status === 'error' && job.error ? <p className="errtext">{job.error}</p> : null}

      {job.status === 'done' && result?.added_videos?.length ? (
        <EnrichmentLine videos={result.added_videos} />
      ) : null}

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

  // Refresh the corpus once per batch of newly finished jobs, instead of on
  // every progress tick — a channel run can otherwise trigger this dozens of
  // times. One batch is usually one job, but the snapshot that seeds a fresh
  // mount carries every job the server has ever run: those all count as new
  // to this instance, and they are worth exactly one refresh between them.
  useEffect(() => {
    let anyNew = false;
    for (const job of jobs) {
      if (job.status !== 'done' || notifiedDoneRef.current.has(job.id)) continue;
      notifiedDoneRef.current.add(job.id);
      anyNew = true;
    }
    if (anyNew) onIndexed();
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
            per-video summary from the creator's own description — no LLM, so indexing cannot
            fail on a provider. Add as many videos or channels as you like: three run at a time
            and the rest queue behind them.
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

          <EnrichmentBanner jobs={jobs} />

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

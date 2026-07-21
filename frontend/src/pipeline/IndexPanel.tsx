import { useEffect, useRef, useState } from 'react';

import { api } from '../api/client';
import type { IndexResult } from '../api/types';
import { insightBadgeClass } from './insights';
import { STAGES, type StageName, appendLog, stageStatuses } from './stages';

interface Props {
  onIndexed: () => void;
  /** Jump the corpus tree to a video the run just added. */
  onViewVideo: (videoId: string) => void;
}

type Mode = 'video' | 'channel';

export function IndexPanel({ onIndexed, onViewVideo }: Props) {
  const [open, setOpen] = useState(false);
  const [mode, setMode] = useState<Mode>('video');
  const [url, setUrl] = useState('');
  const [channel, setChannel] = useState('');
  const [latest, setLatest] = useState(5);

  const [running, setRunning] = useState(false);
  const [activeStage, setActiveStage] = useState<StageName | null>(null);
  const [log, setLog] = useState<string[]>([]);
  const [result, setResult] = useState<IndexResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const abortRef = useRef<AbortController | null>(null);
  const logRef = useRef<HTMLDivElement>(null);

  // Abandon an in-flight run if the view unmounts, so the stream does not keep
  // pushing into state nobody is rendering.
  useEffect(() => () => abortRef.current?.abort(), []);

  useEffect(() => {
    const box = logRef.current;
    if (box) box.scrollTop = box.scrollHeight;
  }, [log]);

  const start = async () => {
    const target = mode === 'video' ? url.trim() : channel.trim();
    if (!target) {
      setError(mode === 'video' ? 'Enter a video URL.' : 'Enter a channel URL or @handle.');
      return;
    }

    const controller = new AbortController();
    abortRef.current = controller;
    setRunning(true);
    setError(null);
    setResult(null);
    setLog([]);
    setActiveStage(null);

    const payload =
      mode === 'video'
        ? { mode: 'video' as const, url: target }
        : { mode: 'channel' as const, channel: target, latest };

    try {
      await api.indexStream(
        payload,
        {
          stage: (data) => {
            setActiveStage(data.stage);
            setLog((current) => appendLog(current, `${data.stage} · ${data.message}`));
          },
          done: (data) => {
            setResult(data);
            if (mode === 'video') setUrl('');
            onIndexed();
          },
          error: (data) => setError(data.message),
        },
        controller.signal,
      );
    } catch (err) {
      if (controller.signal.aborted) setError('Indexing cancelled.');
      else setError((err as Error).message);
    } finally {
      abortRef.current = null;
      setRunning(false);
    }
  };

  const cancel = () => abortRef.current?.abort();

  const statuses = stageStatuses(activeStage, Boolean(result));
  const showProgress = running || result !== null || log.length > 0;
  const startDisabled = running || !(mode === 'video' ? url.trim() : channel.trim());

  const headline = running
    ? `Indexing ${mode === 'video' ? 'video' : 'channel'}…`
    : result
      ? `Added ${result.added_video_count} video${result.added_video_count === 1 ? '' : 's'}`
      : (error ?? '');

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
        {!open && headline ? (
          <span className={`result ${error ? 'err' : result ? 'ok' : ''}`}>{headline}</span>
        ) : null}
        {!open && running ? <span className="pulse" /> : null}
      </div>

      {open ? (
        <div className="pipe-index-body">
          <p className="sub" style={{ margin: '10px 0 8px' }}>
            Fetch transcripts, chunk them on transcript timings, embed the chunks and write a
            per-video summary. A channel run repeats the last four stages once per video, so it
            can take several minutes.
          </p>

          <div className="formrow">
            <button
              type="button"
              className={`pill${mode === 'video' ? ' on' : ''}`}
              onClick={() => setMode('video')}
              disabled={running}
            >
              Single video
            </button>
            <button
              type="button"
              className={`pill${mode === 'channel' ? ' on' : ''}`}
              onClick={() => setMode('channel')}
              disabled={running}
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
                disabled={running}
                placeholder="https://www.youtube.com/watch?v=…"
                aria-label="Video URL"
                onChange={(event) => setUrl(event.target.value)}
              />
            </div>
          ) : (
            <div className="formrow">
              <input
                type="text"
                value={channel}
                spellCheck={false}
                disabled={running}
                placeholder="Channel URL or @handle"
                aria-label="Channel"
                onChange={(event) => setChannel(event.target.value)}
              />
              <input
                type="number"
                min={1}
                max={50}
                value={latest}
                disabled={running}
                title="How many latest videos"
                aria-label="How many latest videos"
                onChange={(event) => setLatest(Number(event.target.value) || 5)}
              />
            </div>
          )}

          <div className="formrow">
            <button
              type="button"
              className="btn pri"
              onClick={() => void start()}
              disabled={startDisabled}
            >
              {running ? 'Indexing…' : 'Start indexing'}
            </button>
            {running ? (
              <button type="button" className="btn danger" onClick={cancel}>
                Cancel
              </button>
            ) : null}
            {error ? <span className="errtext">{error}</span> : null}
          </div>

          {showProgress ? (
            <ol className="idx-stages" aria-label="Indexing stages">
              {STAGES.map((stage) => (
                <li className={`idx-stage ${statuses[stage.name]}`} key={stage.name}>
                  <span className="idx-dot" aria-hidden="true" />
                  <span className="idx-name">{stage.label}</span>
                  <span className="idx-hint">{stage.hint}</span>
                  <span className="sr-only">{statuses[stage.name]}</span>
                </li>
              ))}
            </ol>
          ) : null}

          {log.length > 0 ? (
            <div className="idx-log" ref={logRef} aria-label="Indexing log">
              {log.map((line, index) => (
                // Messages repeat verbatim across videos, so position is the
                // only stable key available here.
                <div key={`${index}-${line}`}>{line}</div>
              ))}
            </div>
          ) : null}

          {result ? (
            <div className="idx-result">
              <div className="idx-result-head">
                <b>Indexed {result.target}</b>
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
        </div>
      ) : null}
    </div>
  );
}

import { useEffect, useRef, useState } from 'react';

import type { Corpus, RetrievalMode, SetupSpec, Video } from '../api/types';

/**
 * What retrieval is allowed to look at.
 *
 * The two fields are linked, not independent: a pinned video always implies its
 * owning channel, so `channelId` is kept in sync for display. Only one of them
 * ever reaches the wire — see {@link scopePayload}.
 */
export interface ChatScope {
  /** Channel to scope retrieval to; null searches every channel. */
  channelId: string | null;
  /** `source_url` of a single pinned video; null searches the whole channel. */
  videoUrl: string | null;
}

export const WHOLE_CORPUS: ChatScope = { channelId: null, videoUrl: null };

/** Composer preferences that survive a reload. */
export interface AskPrefs {
  autoJudge: boolean;
  retrievalMode: RetrievalMode;
  filterTranscripts: boolean;
}

export interface AskOptions extends AskPrefs {
  question: string;
  setups: string[];
  /** Set only for a single-video scope. Mutually exclusive with channelId. */
  url: string | null;
  /** Set only for a channel-wide scope. Mutually exclusive with url. */
  channelId: string | null;
  topK: number | null;
}

const AUTOJUDGE_KEY = 'tlab.autojudge';
const RETRIEVAL_MODE_KEY = 'tlab.retrievalmode';
const FILTER_TRANSCRIPTS_KEY = 'tlab.filtertranscripts';

/**
 * Read the persisted advanced preferences.
 *
 * Used by callers that ask without going through the composer UI (the empty
 * state's suggestion chips), so those runs honour the same settings.
 */
export function readAskPrefs(): AskPrefs {
  return {
    autoJudge: localStorage.getItem(AUTOJUDGE_KEY) !== '0',
    retrievalMode: localStorage.getItem(RETRIEVAL_MODE_KEY) === 'hybrid' ? 'hybrid' : 'semantic',
    filterTranscripts: localStorage.getItem(FILTER_TRANSCRIPTS_KEY) === '1',
  };
}

/** Videos that can be pinned, narrowed to one channel when one is selected. */
export function videosInScope(corpus: Corpus | null, channelId: string | null): Video[] {
  const videos = (corpus?.videos ?? []).filter((video) => video.source_url);
  return channelId ? videos.filter((video) => video.channel_id === channelId) : videos;
}

function findVideo(corpus: Corpus | null, url: string | null): Video | null {
  if (!url) return null;
  return (corpus?.videos ?? []).find((video) => video.source_url === url) ?? null;
}

/**
 * The server ignores `channel_id` when `url` is set, so a video scope must send
 * the url alone — never both — or the request would misrepresent itself.
 */
export function scopePayload(scope: ChatScope): { url: string | null; channelId: string | null } {
  if (scope.videoUrl) return { url: scope.videoUrl, channelId: null };
  return { url: null, channelId: scope.channelId };
}

function count(n: number, word: string): string {
  return `${n} ${word}${n === 1 ? '' : 's'}`;
}

/** Video and chunk totals for a channel, derived if the corpus omits the record. */
function channelTotals(corpus: Corpus | null, channelId: string) {
  const record = (corpus?.channels ?? []).find((channel) => channel.channel_id === channelId);
  if (record) {
    return {
      name: record.channel_name || channelId,
      videos: record.video_count,
      chunks: record.chunk_count,
    };
  }
  const videos = (corpus?.videos ?? []).filter((video) => video.channel_id === channelId);
  return {
    name: videos[0]?.channel_name || channelId,
    videos: videos.length,
    chunks: videos.reduce((total, video) => total + video.chunk_count, 0),
  };
}

/** Plain-language statement of what this scope will actually search. */
export function describeScope(corpus: Corpus | null, scope: ChatScope): string {
  if (scope.videoUrl) {
    const video = findVideo(corpus, scope.videoUrl);
    const name = video?.title || video?.video_id || 'the selected video';
    return `searching 1 video · ${count(video?.chunk_count ?? 0, 'chunk')} in “${name}”`;
  }
  if (scope.channelId) {
    const totals = channelTotals(corpus, scope.channelId);
    return `searching ${count(totals.videos, 'video')} · ${count(totals.chunks, 'chunk')} in ${totals.name}`;
  }
  if (!corpus) return 'searching the whole corpus';
  return `searching the whole corpus (${count(corpus.totals.videos, 'video')}, ${count(
    corpus.totals.chunks,
    'chunk',
  )})`;
}

interface Props {
  setups: SetupSpec[];
  corpus: Corpus | null;
  busy: boolean;
  scope: ChatScope;
  onScopeChange: (scope: ChatScope) => void;
  defaultSetup: string;
  onDefaultSetupChange: (key: string) => void;
  onAsk: (options: AskOptions) => void;
  onCancel: () => void;
}

export function Composer({
  setups,
  corpus,
  busy,
  scope,
  onScopeChange,
  defaultSetup,
  onDefaultSetupChange,
  onAsk,
  onCancel,
}: Props) {
  const [question, setQuestion] = useState('');
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [topK, setTopK] = useState('');
  const [extraSetups, setExtraSetups] = useState<string[]>([]);
  const [autoJudge, setAutoJudge] = useState(() => readAskPrefs().autoJudge);
  const [retrievalMode, setRetrievalMode] = useState<RetrievalMode>(
    () => readAskPrefs().retrievalMode,
  );
  const [filterTranscripts, setFilterTranscripts] = useState(
    () => readAskPrefs().filterTranscripts,
  );
  const textarea = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    localStorage.setItem(AUTOJUDGE_KEY, autoJudge ? '1' : '0');
  }, [autoJudge]);

  useEffect(() => {
    localStorage.setItem(RETRIEVAL_MODE_KEY, retrievalMode);
  }, [retrievalMode]);

  useEffect(() => {
    localStorage.setItem(FILTER_TRANSCRIPTS_KEY, filterTranscripts ? '1' : '0');
  }, [filterTranscripts]);

  useEffect(() => {
    const node = textarea.current;
    if (!node) return;
    node.style.height = 'auto';
    node.style.height = `${Math.min(node.scrollHeight, 140)}px`;
  }, [question]);

  const channels = corpus?.channels ?? [];
  const selectableVideos = videosInScope(corpus, scope.channelId);

  /** Picking a channel drops a pinned video that does not belong to it. */
  const selectChannel = (value: string) => {
    const channelId = value || null;
    if (!channelId) {
      onScopeChange(WHOLE_CORPUS);
      return;
    }
    const pinned = findVideo(corpus, scope.videoUrl);
    onScopeChange({
      channelId,
      videoUrl: pinned && pinned.channel_id === channelId ? scope.videoUrl : null,
    });
  };

  /** Picking a video adopts its owning channel so both selects agree. */
  const selectVideo = (value: string) => {
    if (!value) {
      onScopeChange({ channelId: scope.channelId, videoUrl: null });
      return;
    }
    const video = findVideo(corpus, value);
    onScopeChange({ channelId: video?.channel_id ?? null, videoUrl: value });
  };

  const submit = () => {
    const trimmed = question.trim();
    if (!trimmed || busy) return;
    // The default setup always runs; advanced selections add to it.
    const keys = [defaultSetup, ...extraSetups.filter((key) => key !== defaultSetup)];
    const { url, channelId } = scopePayload(scope);
    onAsk({
      question: trimmed,
      setups: keys,
      url,
      channelId,
      topK: topK ? Number(topK) : null,
      autoJudge,
      retrievalMode,
      filterTranscripts,
    });
    setQuestion('');
  };

  return (
    <div className="composer">
      <div className="composer-inner">
        <textarea
          ref={textarea}
          rows={1}
          value={question}
          disabled={busy}
          placeholder="Ask the indexed transcripts anything…  (Enter to send, Shift+Enter for a newline)"
          onChange={(event) => setQuestion(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === 'Enter' && !event.shiftKey) {
              event.preventDefault();
              submit();
            }
          }}
          aria-label="Question"
        />
        <div className="crow">
          <label className="chipselect">
            Channel:{' '}
            <select
              value={scope.channelId ?? ''}
              onChange={(event) => selectChannel(event.target.value)}
              style={{ border: 'none', background: 'none', padding: 0 }}
              aria-label="Channel scope"
            >
              <option value="">All channels</option>
              {channels.map((channel) => (
                <option key={channel.channel_id} value={channel.channel_id}>
                  {`${channel.channel_name} (${channel.video_count})`}
                </option>
              ))}
            </select>
          </label>

          <label className="chipselect">
            Video:{' '}
            <select
              value={scope.videoUrl ?? ''}
              onChange={(event) => selectVideo(event.target.value)}
              style={{ border: 'none', background: 'none', padding: 0 }}
              aria-label="Video scope"
            >
              <option value="">All videos</option>
              {selectableVideos.map((video) => (
                <option key={video.video_id} value={video.source_url ?? ''}>
                  {(video.title || video.video_id).slice(0, 60)}
                </option>
              ))}
            </select>
          </label>

          <label className="chipselect">
            Agent:{' '}
            <select
              value={defaultSetup}
              onChange={(event) => onDefaultSetupChange(event.target.value)}
              style={{ border: 'none', background: 'none', padding: 0 }}
              aria-label="Answering agent"
            >
              {setups.map((setup) => (
                <option key={setup.key} value={setup.key}>
                  {setup.title}
                </option>
              ))}
            </select>
          </label>

          <button
            type="button"
            className={`pill${showAdvanced ? ' on' : ''}`}
            onClick={() => setShowAdvanced(!showAdvanced)}
            aria-expanded={showAdvanced}
          >
            ⚙ advanced
          </button>

          <span className="spacer" />

          {busy ? (
            <button type="button" className="btn danger" onClick={onCancel}>
              Cancel (Esc)
            </button>
          ) : null}
          <button
            type="button"
            className="btn pri"
            onClick={submit}
            disabled={busy || !question.trim()}
          >
            Send
          </button>
        </div>

        <div className="crow">
          <span className="microlabel" role="status" aria-live="polite">
            {describeScope(corpus, scope)}
          </span>
        </div>

        {showAdvanced ? (
          <div className="advanced">
            <label className="toggle">
              top_k
              <input
                type="number"
                min={1}
                max={50}
                value={topK}
                placeholder="10"
                onChange={(event) => setTopK(event.target.value)}
                style={{ width: 64 }}
              />
            </label>
            <label className="toggle">
              <input
                type="checkbox"
                checked={autoJudge}
                onChange={(event) => setAutoJudge(event.target.checked)}
              />
              auto-judge with RAGAS
            </label>
            <label
              className="toggle"
              title="Summary-first filtering: rank whole transcripts by their summary before retrieving chunks. Previously CLI-only."
            >
              <input
                type="checkbox"
                checked={filterTranscripts}
                onChange={(event) => setFilterTranscripts(event.target.checked)}
              />
              smart transcript filter
            </label>

            <span className="microlabel">retrieval</span>
            <button
              type="button"
              className={`pill${retrievalMode === 'semantic' ? ' on' : ''}`}
              aria-pressed={retrievalMode === 'semantic'}
              title="Embedding similarity only."
              onClick={() => setRetrievalMode('semantic')}
            >
              semantic
            </button>
            <button
              type="button"
              className={`pill${retrievalMode === 'hybrid' ? ' on' : ''}`}
              aria-pressed={retrievalMode === 'hybrid'}
              title="Semantic and BM25 rankings fused with reciprocal rank fusion."
              onClick={() => setRetrievalMode('hybrid')}
            >
              hybrid · semantic + BM25 (RRF)
            </button>

            <span className="microlabel">also run</span>
            {setups
              .filter((setup) => setup.key !== defaultSetup)
              .map((setup) => (
                <button
                  key={setup.key}
                  type="button"
                  title={setup.description}
                  className={`pill${extraSetups.includes(setup.key) ? ' on' : ''}`}
                  onClick={() =>
                    setExtraSetups((current) =>
                      current.includes(setup.key)
                        ? current.filter((key) => key !== setup.key)
                        : [...current, setup.key],
                    )
                  }
                >
                  {setup.title}
                </button>
              ))}
          </div>
        ) : null}
      </div>
    </div>
  );
}

import { useCallback, useEffect, useRef, useState } from 'react';

import { api } from '../api/client';
import type {
  AgentStep,
  Answer,
  AskRequest,
  Corpus,
  Entry,
  RetrievalMode,
  SetupSpec,
} from '../api/types';
import {
  type ChatScope,
  Composer,
  WHOLE_CORPUS,
  readAskPrefs,
  scopePayload,
} from './Composer';
import { HistoryRail } from './HistoryRail';
import { MessageBubble, type RunningSetup } from './MessageBubble';

/** Agentic answers are the best but slowest; D2 makes them the default. */
const DEFAULT_SETUP = 'rag_agent';

interface LiveRun {
  question: string;
  entryId: string | null;
  running: RunningSetup[];
  answers: Answer[];
}

interface Props {
  setups: SetupSpec[];
  history: Entry[];
  corpus: Corpus | null;
  onHistoryChange: (entries: Entry[]) => void;
  onActivity: () => void;
  pendingScope: string | null;
  onScopeConsumed: () => void;
  /** Optional channel hint for pendingScope; otherwise read off the corpus. */
  pendingChannel?: string | null;
}

function suggestionsFor(corpus: Corpus | null): string[] {
  const titles = (corpus?.videos ?? [])
    .map((video) => video.title)
    .filter((title): title is string => Boolean(title))
    .slice(0, 2);
  const base = [
    'What are the main themes across the indexed transcripts?',
    'Where do these videos disagree with each other?',
  ];
  return [...titles.map((title) => `Summarize the key claims in “${title}”`), ...base].slice(
    0,
    4,
  );
}

export function ChatView({
  setups,
  history,
  corpus,
  onHistoryChange,
  onActivity,
  pendingScope,
  onScopeConsumed,
  pendingChannel = null,
}: Props) {
  const [thread, setThread] = useState<Entry[]>([]);
  const [live, setLive] = useState<LiveRun | null>(null);
  // entry id -> setup key -> research steps, kept so a finished agentic answer
  // can still show its (collapsed) trace for the rest of the session.
  const [traces, setTraces] = useState<Record<string, Record<string, AgentStep[]>>>({});
  const [judgingId, setJudgingId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [scope, setScope] = useState<ChatScope>(WHOLE_CORPUS);
  const [defaultSetup, setDefaultSetup] = useState(DEFAULT_SETUP);
  const [status, setStatus] = useState('');
  const abort = useRef<AbortController | null>(null);
  const bottom = useRef<HTMLDivElement>(null);

  const busy = live !== null;
  const selectedId = thread.length ? (thread[thread.length - 1]?.id ?? null) : null;

  useEffect(() => {
    if (!pendingScope) return;
    const video = (corpus?.videos ?? []).find((item) => item.source_url === pendingScope);
    setScope({ channelId: video?.channel_id ?? pendingChannel, videoUrl: pendingScope });
    onScopeConsumed();
  }, [pendingScope, pendingChannel, corpus, onScopeConsumed]);

  // A scope can arrive before the corpus loads, leaving the owning channel
  // unknown; fill it in as soon as the corpus can answer for it.
  useEffect(() => {
    if (!scope.videoUrl || scope.channelId) return;
    const video = (corpus?.videos ?? []).find((item) => item.source_url === scope.videoUrl);
    if (video?.channel_id) {
      setScope((current) => ({ ...current, channelId: video.channel_id }));
    }
  }, [corpus, scope]);

  useEffect(() => {
    if (setups.length && !setups.some((setup) => setup.key === defaultSetup)) {
      setDefaultSetup(setups[0]!.key);
    }
  }, [setups, defaultSetup]);

  useEffect(() => {
    bottom.current?.scrollIntoView({ behavior: 'smooth', block: 'end' });
  }, [thread, live]);

  const upsertHistory = useCallback(
    (entry: Entry) => {
      const index = history.findIndex((item) => item.id === entry.id);
      onHistoryChange(
        index === -1
          ? [...history, entry]
          : history.map((item) => (item.id === entry.id ? entry : item)),
      );
    },
    [history, onHistoryChange],
  );

  const judge = useCallback(
    async (entryId: string) => {
      setJudgingId(entryId);
      try {
        await api.judge(entryId, {
          progress: (data) => setStatus(data.message),
          done: (entry) => {
            setThread((current) =>
              current.map((item) => (item.id === entry.id ? entry : item)),
            );
            upsertHistory(entry);
          },
          error: (data) => setError(`Judging failed: ${data.message}`),
        });
      } catch (err) {
        setError(`Judging failed: ${(err as Error).message}`);
      } finally {
        setJudgingId(null);
        setStatus('');
        onActivity();
      }
    },
    [upsertHistory, onActivity],
  );

  const run = useCallback(
    async (options: {
      question: string;
      setups: string[];
      /** Exactly one of url / channelId may be set; url wins if both are. */
      url: string | null;
      channelId: string | null;
      topK: number | null;
      autoJudge: boolean;
      retrievalMode: RetrievalMode;
      filterTranscripts: boolean;
      /** Prior turns, so a follow-up can be rewritten to stand alone. */
      history?: string[];
      entryId?: string;
    }) => {
      const titleOf = (key: string) =>
        setups.find((setup) => setup.key === key)?.title ?? key;

      setError(null);
      setLive({
        question: options.question,
        entryId: options.entryId ?? null,
        answers: [],
        running: options.setups.map((key) => ({
          key,
          title: titleOf(key),
          startedAt: Date.now(),
          steps: [],
        })),
      });

      const controller = new AbortController();
      abort.current = controller;
      let finished: Entry | null = null;
      const collected: Record<string, AgentStep[]> = {};

      // The server ignores channel_id whenever url pins a video, so send the
      // narrower of the two rather than both.
      const request: AskRequest = {
        question: options.question,
        setups: options.setups,
        url: options.url,
        top_k: options.topK,
        channel_id: options.url ? null : options.channelId,
        retrieval_mode: options.retrievalMode,
        filter_transcripts: options.filterTranscripts,
        ...(options.history?.length ? { history: options.history } : {}),
        ...(options.entryId ? { entry_id: options.entryId } : {}),
      };

      try {
        await api.ask(
          request,
          {
            progress: (data) => setStatus(data.message),
            agent_step: (step: AgentStep) => {
              collected[step.key] = [...(collected[step.key] ?? []), step];
              setLive((current) =>
                current
                  ? {
                      ...current,
                      running: current.running.map((setup) =>
                        setup.key === step.key
                          ? { ...setup, steps: [...setup.steps, step] }
                          : setup,
                      ),
                    }
                  : current,
              );
            },
            answer: (answer: Answer) =>
              setLive((current) =>
                current
                  ? {
                      ...current,
                      answers: [...current.answers, answer],
                      running: current.running.filter((setup) => setup.key !== answer.key),
                    }
                  : current,
              ),
            done: (entry) => {
              finished = entry;
            },
            error: (data) => setError(data.message),
          },
          controller.signal,
        );
      } catch (err) {
        if ((err as Error).name !== 'AbortError') {
          setError(`Request failed: ${(err as Error).message}`);
        } else {
          setStatus('Run cancelled');
        }
      } finally {
        abort.current = null;
        setLive(null);
        onActivity();
      }

      if (finished) {
        const entry: Entry = finished;
        if (Object.keys(collected).length) {
          setTraces((current) => ({
            ...current,
            [entry.id]: { ...(current[entry.id] ?? {}), ...collected },
          }));
        }
        setThread((current) => {
          const index = current.findIndex((item) => item.id === entry.id);
          return index === -1
            ? [...current, entry]
            : current.map((item) => (item.id === entry.id ? entry : item));
        });
        upsertHistory(entry);
        setStatus('');
        if (options.autoJudge) await judge(entry.id);
      }
    },
    [setups, upsertHistory, judge, onActivity],
  );

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape' && abort.current) abort.current.abort();
    };
    document.addEventListener('keydown', onKeyDown);
    return () => document.removeEventListener('keydown', onKeyDown);
  }, []);

  /** Re-run the setups that have not answered yet, under the original scope. */
  const compare = (entry: Entry) => {
    if (busy) return;
    const missing = setups
      .map((setup) => setup.key)
      .filter((key) => !entry.answers.some((answer) => answer.key === key));
    if (!missing.length) return;
    const prefs = readAskPrefs();
    const prior = entry.answers.find((answer) => answer.channel_id || answer.retrieval_mode);
    void run({
      question: entry.question,
      setups: missing,
      url: entry.url,
      channelId: entry.url ? null : (prior?.channel_id ?? null),
      topK: null,
      autoJudge: true,
      // Compare like with like: reuse the retrieval strategy of the answers
      // already in this entry rather than whatever the composer now shows.
      retrievalMode: prior?.retrieval_mode === 'hybrid' ? 'hybrid' : 'semantic',
      filterTranscripts: prefs.filterTranscripts,
      entryId: entry.id,
    });
  };

  /**
   * Ask a proposed follow-up as a new question, inheriting the scope, strategy
   * and conversation history of the answer that proposed it — a follow-up asked
   * against a different corpus slice would not be a follow-up.
   */
  const askFollowup = (query: string) => {
    if (busy) return;
    const source = thread[thread.length - 1];
    const prior = source?.answers.find(
      (answer) => answer.channel_id || answer.retrieval_mode,
    );
    const prefs = readAskPrefs();
    void run({
      question: query,
      setups: [defaultSetup],
      url: source?.url ?? null,
      channelId: source?.url ? null : (prior?.channel_id ?? null),
      topK: null,
      autoJudge: prefs.autoJudge,
      retrievalMode: prior?.retrieval_mode === 'hybrid' ? 'hybrid' : 'semantic',
      filterTranscripts: prefs.filterTranscripts,
      history: source ? [source.question, ...source.answers.map((a) => a.answer)] : [],
    });
  };

  const suggestions = suggestionsFor(corpus);
  const empty = thread.length === 0 && !live;

  return (
    <section className="view">
      <HistoryRail
        history={history}
        selectedId={selectedId}
        disabled={busy}
        onSelect={(id) => {
          const entry = history.find((item) => item.id === id);
          if (entry) setThread([entry]);
        }}
      />

      <div className="stage">
        <div className="thread">
          <div className="thread-inner">
            {empty ? (
              <div className="empty">
                <h2>Ask the transcripts anything</h2>
                <p>
                  {corpus?.totals.videos
                    ? `${corpus.totals.videos} videos · ${corpus.totals.chunks} chunks · ${corpus.totals.channels} channels indexed. Narrow the scope below, or ask across everything. Answers are cited back to the source timestamp and scored with RAGAS.`
                    : 'No transcripts indexed yet — add one from the RAG Pipeline tab first.'}
                </p>
                <div className="suggest">
                  {suggestions.map((suggestion) => (
                    <button
                      key={suggestion}
                      type="button"
                      onClick={() =>
                        void run({
                          question: suggestion,
                          setups: [defaultSetup],
                          ...scopePayload(scope),
                          topK: null,
                          ...readAskPrefs(),
                        })
                      }
                    >
                      {suggestion}
                    </button>
                  ))}
                </div>
              </div>
            ) : null}

            {thread.map((entry) => (
              <div key={entry.id} style={{ display: 'contents' }}>
                <div className="msg-user">{entry.question}</div>
                <MessageBubble
                  question={entry.question}
                  answers={entry.answers}
                  running={[]}
                  onAskFollowup={askFollowup}
                  judging={judgingId === entry.id}
                  busy={busy}
                  onJudge={() => void judge(entry.id)}
                  onCompare={() => compare(entry)}
                  traces={traces[entry.id]}
                  remainingSetups={
                    setups.filter(
                      (setup) => !entry.answers.some((answer) => answer.key === setup.key),
                    ).length
                  }
                />
              </div>
            ))}

            {live ? (
              <>
                {live.entryId ? null : <div className="msg-user">{live.question}</div>}
                <MessageBubble
                  question={live.question}
                  answers={live.answers}
                  running={live.running}
                  judging={false}
                  remainingSetups={0}
                />
              </>
            ) : null}

            {error ? <div className="errtext">{error}</div> : null}
            <div ref={bottom} />
          </div>
        </div>

        <Composer
          setups={setups}
          corpus={corpus}
          busy={busy}
          scope={scope}
          onScopeChange={setScope}
          defaultSetup={defaultSetup}
          onDefaultSetupChange={setDefaultSetup}
          onAsk={(options) => void run(options)}
          onCancel={() => abort.current?.abort()}
        />
        <div className="sr-only" role="status" aria-live="polite">
          {status}
        </div>
      </div>
    </section>
  );
}

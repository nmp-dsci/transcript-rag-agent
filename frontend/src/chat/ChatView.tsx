import { useCallback, useEffect, useRef, useState } from 'react';

import { api } from '../api/client';
import type {
  AgentStep,
  Answer,
  AskRequest,
  Corpus,
  Entry,
  RetrievalMode,
  ReviewedDocument,
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
import { DocumentCard } from './DocumentCard';
import { MessageBubble, type RunningSetup } from './MessageBubble';

/** Agentic answers are the best but slowest; D2 makes them the default. */
const DEFAULT_SETUP = 'rag_agent';

interface LiveRun {
  question: string;
  entryId: string | null;
  running: RunningSetup[];
  answers: Answer[];
  /** The page a URL in the message pointed at, once it has been fetched. */
  document?: ReviewedDocument | null;
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

/**
 * The card for one entry: the cached page, wearing that entry's own selection.
 *
 * The two halves come from different places and must be recombined per entry,
 * never per document. `GET /api/documents/:id` returns the page as stored — it
 * has no idea which question narrowed it — while `document_detail` and
 * `document_sections_selected` record what *this* answer was actually given.
 * Two conversations reviewing the same URL can select different sections, so
 * merging into the shared cache would show the first one's selection on both.
 */
export function documentForEntry(
  entry: Entry,
  documents: Record<string, ReviewedDocument>,
): ReviewedDocument | null {
  const page = entry.document_id ? documents[entry.document_id] : undefined;
  if (!page) return null;
  const selected = entry.document_sections_selected ?? null;
  if (entry.document_detail == null && selected == null) return page;
  return {
    ...page,
    detail: entry.document_detail ?? page.detail,
    sections_selected: selected ?? page.sections_selected,
    narrowed: selected ? selected.length < page.sections.length : page.narrowed,
  };
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
  // Reviewed documents by id, so a card survives the live run and can be
  // re-rendered for any history entry that references one.
  const [documents, setDocuments] = useState<Record<string, ReviewedDocument>>({});
  // Which section of a reviewed document is open, per entry. Held here rather
  // than inside DocumentCard because the thing that opens it is a rubric
  // verdict in the answer *below* the card.
  const [openSections, setOpenSections] = useState<Record<string, number>>({});
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

  // A reloaded conversation carries only document ids — the text lives
  // server-side, out of the committed history — so fetch what the thread
  // references and has not already got. A document that has been cleared from
  // the store simply has no card; the conversation still reads.
  //
  // This cache holds *pages*, keyed by document id, and nothing question-
  // specific: how much of a page a given answer read belongs to the entry, and
  // is applied at render by `documentForEntry`. Keeping it out of the cache is
  // not tidiness — the cache is filled once per id, so folding a selection into
  // it would freeze the first entry's selection onto every later conversation
  // that reviews the same URL.
  useEffect(() => {
    const missing = thread
      .map((entry) => entry.document_id)
      .filter((id): id is string => Boolean(id) && !(id! in documents));
    if (!missing.length) return;
    let live = true;
    void Promise.all(
      Array.from(new Set(missing)).map((id) =>
        api.document(id).then(
          (found): [string, ReviewedDocument] | null => [id, found],
          () => null,
        ),
      ),
    ).then((loaded) => {
      const found = loaded.filter((item): item is [string, ReviewedDocument] => item !== null);
      if (!live || !found.length) return;
      setDocuments((current) => ({ ...current, ...Object.fromEntries(found) }));
    });
    return () => {
      live = false;
    };
  }, [thread, documents]);

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
      documentEntryId?: string;
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
        ...(options.documentEntryId ? { document_entry_id: options.documentEntryId } : {}),
      };

      try {
        await api.ask(
          request,
          {
            progress: (data) => setStatus(data.message),
            document: (reviewed: ReviewedDocument) => {
              // Arrives before the answer, so the card renders while the
              // review is still being written. Only the document-capable
              // setups thread the page into their answer call, so the tabs
              // the server is about to skip stop waiting here too. Read off
              // the setup list rather than a spelled-in key: the server
              // derives the same list from the same flag, and two hand-kept
              // copies of it are how the next reviewing setup gets dropped.
              const capable = new Set(
                setups.filter((setup) => setup.document_capable).map((setup) => setup.key),
              );
              setDocuments((current) => ({ ...current, [reviewed.id]: reviewed }));
              setLive((current) =>
                current
                  ? {
                      ...current,
                      document: reviewed,
                      running: current.running.filter(
                        (setup) => capable.size === 0 || capable.has(setup.key),
                      ),
                    }
                  : current,
              );
            },
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

  /**
   * The setups that could still answer this entry.
   *
   * A document-grounded entry can only be filled in by setups that thread the
   * page into their answer call — the rest would answer from the corpus and the
   * tab would read as a second review of a document it never saw. Which is also
   * what makes the two that *can* worth offering: the rubric reviewer and the
   * chunk-dump review of the same page, side by side in one bubble, is the
   * comparison this whole slice is about.
   */
  const missingSetups = useCallback(
    (entry: Entry): string[] =>
      setups
        .filter((setup) => !entry.document_id || setup.document_capable)
        .map((setup) => setup.key)
        .filter((key) => !entry.answers.some((answer) => answer.key === key)),
    [setups],
  );

  /** Re-run the setups that have not answered yet, under the original scope. */
  const compare = (entry: Entry) => {
    if (busy) return;
    const missing = missingSetups(entry);
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
      documentEntryId: source?.id,
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
                {documentForEntry(entry, documents) ? (
                  <DocumentCard
                    // Remounted when the target section changes, so clicking a
                    // second verdict re-opens the card at the new section
                    // instead of leaving the first one showing: DocumentCard
                    // reads `openSection` once, as its initial state.
                    key={`${entry.id}-${openSections[entry.id] ?? 'none'}`}
                    document={documentForEntry(entry, documents)!}
                    openSection={openSections[entry.id] ?? null}
                  />
                ) : null}
                <MessageBubble
                  question={entry.question}
                  answers={entry.answers}
                  running={[]}
                  onAskFollowup={askFollowup}
                  onOpenSection={(index) =>
                    setOpenSections((current) => ({ ...current, [entry.id]: index }))
                  }
                  judging={judgingId === entry.id}
                  busy={busy}
                  onJudge={() => void judge(entry.id)}
                  onCompare={() => compare(entry)}
                  traces={traces[entry.id]}
                  remainingSetups={missingSetups(entry).length}
                />
              </div>
            ))}

            {live ? (
              <>
                {live.entryId ? null : <div className="msg-user">{live.question}</div>}
                {live.document ? <DocumentCard document={live.document} /> : null}
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

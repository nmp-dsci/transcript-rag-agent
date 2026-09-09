import { useCallback, useEffect, useState } from 'react';

import { startAnalytics } from './analytics';
import { api } from './api/client';
import type { Corpus, Entry, Health, SetupSpec } from './api/types';
import { ChatView } from './chat/ChatView';
import { DemoContext } from './demo';
import { Landing } from './landing/Landing';
import { SystemDesignView } from './design/SystemDesignView';
import { ExperimentsView } from './experiments/ExperimentsView';
import { Logo } from './Logo';
import { PipelineView } from './pipeline/PipelineView';
import { ScoreboardView } from './scoreboard/ScoreboardView';
import { type Theme, initialTheme, setTheme } from './theme';

export type Tab = 'chat' | 'pipeline' | 'board' | 'experiments' | 'design';

const TABS: { id: Tab; label: string }[] = [
  { id: 'chat', label: 'Chat' },
  { id: 'pipeline', label: 'RAG Pipeline' },
  { id: 'board', label: 'Scoreboard' },
  { id: 'experiments', label: 'Experiments' },
  { id: 'design', label: 'System Design' },
];

/** Old #library and #prompts links stay valid; #pipeline/#design are canonical. */
const HASH_ALIASES: Record<string, Tab> = { library: 'pipeline', prompts: 'design' };

function tabFromHash(): Tab {
  const hash = window.location.hash.replace('#', '');
  if (HASH_ALIASES[hash]) return HASH_ALIASES[hash];
  return TABS.some((tab) => tab.id === hash) ? (hash as Tab) : 'chat';
}

/** The landing shows on a plain visit (no deep link) once per session — in
 * both demo and dev. A shared #board / #pipeline link still lands exactly
 * where it points, and #landing brings the page back on purpose. */
export function shouldShowLanding(hash: string, entered: boolean): boolean {
  const clean = hash.replace('#', '');
  if (clean === 'landing') return true;
  return !entered && clean === '';
}

const ENTERED_KEY = 'tl-entered';

export function App() {
  const [tab, setTab] = useState<Tab>(tabFromHash);
  const [showLanding, setShowLanding] = useState(() =>
    shouldShowLanding(window.location.hash, sessionStorage.getItem(ENTERED_KEY) === '1'),
  );
  const [setups, setSetups] = useState<SetupSpec[]>([]);
  const [history, setHistory] = useState<Entry[]>([]);
  const [corpus, setCorpus] = useState<Corpus | null>(null);
  const [health, setHealth] = useState<Health | null>(null);
  const [offline, setOffline] = useState(false);
  /** Set by "Ask about this" in the pipeline view so Chat opens pre-scoped. */
  const [pendingScope, setPendingScope] = useState<string | null>(null);
  /** Set by a landing example question so Chat opens with it already typed. */
  const [pendingQuestion, setPendingQuestion] = useState<string | null>(null);
  // index.html applies the theme before first paint; this mirrors it so the
  // toggle can render the right label.
  const [theme, setThemeState] = useState<Theme>(initialTheme);

  const refreshHealth = useCallback(async () => {
    try {
      const next = await api.health();
      setHealth(next);
      setOffline(false);
      // No-op unless the server says demo AND a key was baked in at build.
      startAnalytics(next.mode);
    } catch {
      setOffline(true);
    }
  }, []);

  const refreshCorpus = useCallback(async () => {
    try {
      setCorpus(await api.corpus());
    } catch {
      setCorpus({
        videos: [],
        channels: [],
        totals: { videos: 0, chunks: 0, channels: 0 },
        insights: [],
      });
    }
  }, []);

  useEffect(() => {
    void (async () => {
      try {
        const [specs, entries] = await Promise.all([api.setups(), api.history()]);
        setSetups(specs);
        setHistory(entries);
      } catch {
        setOffline(true);
      }
      void refreshCorpus();
      void refreshHealth();
    })();
  }, [refreshCorpus, refreshHealth]);

  // The hash is the only router, so it has to drive *both* pieces of state.
  // Updating only the tab meant #landing silently resolved to Chat (it is not
  // a tab id), so the documented way back to the landing did nothing — and
  // neither did the browser's Back button once you had entered.
  useEffect(() => {
    const onHashChange = () => {
      setTab(tabFromHash());
      setShowLanding(
        shouldShowLanding(window.location.hash, sessionStorage.getItem(ENTERED_KEY) === '1'),
      );
    };
    window.addEventListener('hashchange', onHashChange);
    return () => window.removeEventListener('hashchange', onHashChange);
  }, []);

  const selectTab = (next: Tab) => {
    window.location.hash = next;
    setTab(next);
  };

  const enterApp = (question?: string) => {
    sessionStorage.setItem(ENTERED_KEY, '1');
    setPendingQuestion(question ?? null);
    setShowLanding(false);
    selectTab('chat');
  };

  const askAbout = (url: string) => {
    setPendingScope(url);
    selectTab('chat');
  };

  /** Back to the intro. Routed through the hash so Back/Forward keep working. */
  const showIntro = () => {
    window.location.hash = 'landing';
    setShowLanding(true);
  };

  const toggleTheme = () => {
    const next: Theme = theme === 'dark' ? 'light' : 'dark';
    setTheme(next);
    setThemeState(next);
  };

  const corpusBit = corpus
    ? `${corpus.totals.videos} videos · ${corpus.totals.chunks} chunks · `
    : '';

  // Server-decided; the server also enforces it (mutating routes 403), so
  // this only controls which chrome renders. System Design stays dev-only.
  const demo = health?.mode === 'demo';
  const visibleTabs = demo ? TABS.filter(({ id }) => id !== 'design') : TABS;
  const activeTab = demo && tab === 'design' ? 'chat' : tab;

  if (showLanding) {
    // The shell's data effects have already fired, so the corpus loads while
    // the visitor reads — entry lands on a warm app (Data Pilot's pre-warm).
    return <Landing corpus={corpus} demo={health ? demo : null} onEnter={enterApp} />;
  }

  return (
    <div className="app">
      <header className="topbar">
        {/* The brand is the way home, as it is on nearly every site. Before
            this there was no route back to the intro at all: #landing was
            documented but did not work, and nothing on the page offered it. */}
        <button
          type="button"
          className="brand"
          onClick={showIntro}
          title="Back to the intro"
          aria-label="Back to the intro"
        >
          <Logo />
          <span>
            transcript·<em>lab</em>
          </span>
        </button>
        <nav className="nav" aria-label="Views">
          {visibleTabs.map(({ id, label }) => (
            <button
              key={id}
              type="button"
              className={activeTab === id ? 'on' : ''}
              aria-current={activeTab === id ? 'page' : undefined}
              onClick={() => selectTab(id)}
            >
              {label}
            </button>
          ))}
        </nav>
        <div className="topstat">
          <span className={`hdot ${offline ? 'err' : health ? 'ok' : ''}`} />
          <span className="topstat-text">
            {offline
              ? 'server unreachable'
              : demo
                ? `${corpusBit}demo replay`
                : health
                  ? `${corpusBit}judge ${health.judge_model}${
                      health.runner_loaded ? ' · stack loaded' : ' · stack cold'
                    }`
                  : 'connecting…'}
          </span>
          <button
            type="button"
            className="themetoggle"
            onClick={toggleTheme}
            title={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
            aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
          >
            {theme === 'dark' ? '☀' : '☾'}
          </button>
        </div>
      </header>

      <main className="views">
        <DemoContext.Provider value={demo}>
        {activeTab === 'chat' && (
          <ChatView
            setups={setups}
            history={history}
            corpus={corpus}
            onHistoryChange={setHistory}
            onActivity={refreshHealth}
            pendingScope={pendingScope}
            onScopeConsumed={() => setPendingScope(null)}
            pendingQuestion={pendingQuestion}
            onQuestionConsumed={() => setPendingQuestion(null)}
            stt={health?.stt === true}
          />
        )}
        {activeTab === 'pipeline' && (
          <PipelineView
            corpus={corpus}
            onCorpusChange={refreshCorpus}
            onAskAbout={askAbout}
            embeddingModel={health?.embedding_model ?? null}
          />
        )}
        {activeTab === 'board' && <ScoreboardView />}
        {activeTab === 'experiments' && <ExperimentsView />}
        {activeTab === 'design' && !demo && <SystemDesignView />}
        </DemoContext.Provider>
      </main>
    </div>
  );
}

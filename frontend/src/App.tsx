import { useCallback, useEffect, useState } from 'react';

import { api } from './api/client';
import type { Corpus, Entry, Health, SetupSpec } from './api/types';
import { ChatView } from './chat/ChatView';
import { ExperimentsView } from './experiments/ExperimentsView';
import { Logo } from './Logo';
import { PipelineView } from './pipeline/PipelineView';
import { ScoreboardView } from './scoreboard/ScoreboardView';
import { type Theme, initialTheme, setTheme } from './theme';

export type Tab = 'chat' | 'pipeline' | 'board' | 'experiments';

const TABS: { id: Tab; label: string }[] = [
  { id: 'chat', label: 'Chat' },
  { id: 'pipeline', label: 'RAG Pipeline' },
  { id: 'board', label: 'Scoreboard' },
  { id: 'experiments', label: 'Experiments' },
];

/** Old #library links stay valid; #pipeline is canonical. */
const HASH_ALIASES: Record<string, Tab> = { library: 'pipeline' };

function tabFromHash(): Tab {
  const hash = window.location.hash.replace('#', '');
  if (HASH_ALIASES[hash]) return HASH_ALIASES[hash];
  return TABS.some((tab) => tab.id === hash) ? (hash as Tab) : 'chat';
}

export function App() {
  const [tab, setTab] = useState<Tab>(tabFromHash);
  const [setups, setSetups] = useState<SetupSpec[]>([]);
  const [history, setHistory] = useState<Entry[]>([]);
  const [corpus, setCorpus] = useState<Corpus | null>(null);
  const [health, setHealth] = useState<Health | null>(null);
  const [offline, setOffline] = useState(false);
  /** Set by "Ask about this" in the pipeline view so Chat opens pre-scoped. */
  const [pendingScope, setPendingScope] = useState<string | null>(null);
  // index.html applies the theme before first paint; this mirrors it so the
  // toggle can render the right label.
  const [theme, setThemeState] = useState<Theme>(initialTheme);

  const refreshHealth = useCallback(async () => {
    try {
      setHealth(await api.health());
      setOffline(false);
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

  useEffect(() => {
    const onHashChange = () => setTab(tabFromHash());
    window.addEventListener('hashchange', onHashChange);
    return () => window.removeEventListener('hashchange', onHashChange);
  }, []);

  const selectTab = (next: Tab) => {
    window.location.hash = next;
    setTab(next);
  };

  const askAbout = (url: string) => {
    setPendingScope(url);
    selectTab('chat');
  };

  const toggleTheme = () => {
    const next: Theme = theme === 'dark' ? 'light' : 'dark';
    setTheme(next);
    setThemeState(next);
  };

  const corpusBit = corpus
    ? `${corpus.totals.videos} videos · ${corpus.totals.chunks} chunks · `
    : '';

  return (
    <div className="app">
      <header className="topbar">
        <span className="brand">
          <Logo />
          <span>
            transcript<em>·lab</em>
          </span>
        </span>
        <nav className="nav" aria-label="Views">
          {TABS.map(({ id, label }) => (
            <button
              key={id}
              type="button"
              className={tab === id ? 'on' : ''}
              aria-current={tab === id ? 'page' : undefined}
              onClick={() => selectTab(id)}
            >
              {label}
            </button>
          ))}
        </nav>
        <div className="topstat">
          <span className={`hdot ${offline ? 'err' : health ? 'ok' : ''}`} />
          <span>
            {offline
              ? 'server unreachable'
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
        {tab === 'chat' && (
          <ChatView
            setups={setups}
            history={history}
            corpus={corpus}
            onHistoryChange={setHistory}
            onActivity={refreshHealth}
            pendingScope={pendingScope}
            onScopeConsumed={() => setPendingScope(null)}
          />
        )}
        {tab === 'pipeline' && (
          <PipelineView
            corpus={corpus}
            onCorpusChange={refreshCorpus}
            onAskAbout={askAbout}
            embeddingModel={health?.embedding_model ?? null}
          />
        )}
        {tab === 'board' && <ScoreboardView />}
        {tab === 'experiments' && <ExperimentsView />}
      </main>
    </div>
  );
}

import { useCallback, useEffect, useState } from 'react';

import { api } from './api/client';
import type { Corpus, Entry, Health, SetupSpec } from './api/types';
import { ChatView } from './chat/ChatView';
import { LibraryView } from './library/LibraryView';
import { ScoreboardView } from './scoreboard/ScoreboardView';

export type Tab = 'chat' | 'library' | 'board';

const TABS: { id: Tab; label: string }[] = [
  { id: 'chat', label: 'Chat' },
  { id: 'library', label: 'Library' },
  { id: 'board', label: 'Scoreboard' },
];

function tabFromHash(): Tab {
  const hash = window.location.hash.replace('#', '');
  return TABS.some((tab) => tab.id === hash) ? (hash as Tab) : 'chat';
}

export function App() {
  const [tab, setTab] = useState<Tab>(tabFromHash);
  const [setups, setSetups] = useState<SetupSpec[]>([]);
  const [history, setHistory] = useState<Entry[]>([]);
  const [corpus, setCorpus] = useState<Corpus | null>(null);
  const [health, setHealth] = useState<Health | null>(null);
  const [offline, setOffline] = useState(false);
  /** Set by "Ask about this" in the Library so Chat opens pre-scoped. */
  const [pendingScope, setPendingScope] = useState<string | null>(null);

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
      setCorpus({ videos: [], totals: { videos: 0, chunks: 0 } });
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

  const corpusBit = corpus
    ? `${corpus.totals.videos} videos · ${corpus.totals.chunks} chunks · `
    : '';

  return (
    <div className="app">
      <header className="topbar">
        <span className="brand">
          transcript<em>·lab</em>
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
        {tab === 'library' && (
          <LibraryView corpus={corpus} onCorpusChange={refreshCorpus} onAskAbout={askAbout} />
        )}
        {tab === 'board' && <ScoreboardView />}
      </main>
    </div>
  );
}

/** The demo chrome: what the public read-only deployment renders.
 *
 * The server refuses every mutating route in demo mode; these tests protect
 * the visible half — no composer, no judge/compare, replay banner up, and the
 * newest conversation opened on landing — plus the default: none of that
 * chrome appears when the context is (or defaults to) full mode.
 */

import { render, screen, waitFor } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import type { Corpus, Entry, SetupSpec } from './api/types';
import { ChatView } from './chat/ChatView';
import { DemoContext } from './demo';

Element.prototype.scrollIntoView = vi.fn();

vi.mock('./api/client', () => ({
  api: { ask: vi.fn(), judge: vi.fn(), document: vi.fn() },
}));

const SETUPS: SetupSpec[] = [
  {
    key: 'rag_agent',
    label: 'rag_agent (agentic)',
    description: '',
    document_capable: true,
  } as unknown as SetupSpec,
];

const CORPUS: Corpus = {
  videos: [],
  channels: [],
  totals: { videos: 2, chunks: 40, channels: 1 },
  insights: [],
};

const ENTRY: Entry = {
  id: 'e1',
  question: 'What makes a resume work?',
  url: null,
  asked_at: '2026-08-01T00:00:00+00:00',
  answers: [
    {
      key: 'rag_agent',
      label: 'rag_agent',
      answer: 'Lead with evidence.',
      error: null,
      evaluation: null,
    } as unknown as Entry['answers'][number],
  ],
};

function demoView(history: Entry[] = [ENTRY]) {
  return render(
    <DemoContext.Provider value={true}>
      <ChatView
        setups={SETUPS}
        history={history}
        corpus={CORPUS}
        onHistoryChange={vi.fn()}
        onActivity={vi.fn()}
        pendingScope={null}
        onScopeConsumed={vi.fn()}
      />
    </DemoContext.Provider>,
  );
}

describe('demo chrome in Chat', () => {
  it('replaces the composer with the replay banner', () => {
    demoView();
    expect(screen.getByText(/Demo replay/)).toBeInTheDocument();
    expect(screen.queryByLabelText('Question')).not.toBeInTheDocument();
  });

  it('opens the newest conversation on landing', async () => {
    demoView();
    // The question appears in both the history rail and the opened bubble.
    await waitFor(() =>
      expect(screen.getAllByText('What makes a resume work?').length).toBeGreaterThan(1),
    );
    expect(screen.getByText('Lead with evidence.')).toBeInTheDocument();
  });

  it('renders no judge or compare controls on a replayed entry', async () => {
    demoView();
    await waitFor(() =>
      expect(screen.getAllByText('What makes a resume work?').length).toBeGreaterThan(1),
    );
    expect(screen.queryByText(/Judge with RAGAS|Re-judge/)).not.toBeInTheDocument();
    expect(screen.queryByText(/compare/i)).not.toBeInTheDocument();
  });

  it('invites browsing rather than asking when the history is empty', () => {
    demoView([]);
    expect(screen.getByText('Browse the recorded Q&A')).toBeInTheDocument();
    expect(screen.queryByText('Ask the transcripts anything')).not.toBeInTheDocument();
  });

  it('defaults to full mode: composer renders without a provider', () => {
    render(
      <ChatView
        setups={SETUPS}
        history={[]}
        corpus={CORPUS}
        onHistoryChange={vi.fn()}
        onActivity={vi.fn()}
        pendingScope={null}
        onScopeConsumed={vi.fn()}
      />,
    );
    expect(screen.getByLabelText('Question')).toBeInTheDocument();
    expect(screen.queryByText(/Demo replay/)).not.toBeInTheDocument();
  });
});

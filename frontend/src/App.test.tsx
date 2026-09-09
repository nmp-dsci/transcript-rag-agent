import { act, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { App } from './App';

const health = vi.fn();
const corpus = vi.fn();
const setups = vi.fn();
const history = vi.fn();

vi.mock('./api/client', () => ({
  api: {
    health: () => health(),
    corpus: () => corpus(),
    setups: () => setups(),
    history: () => history(),
  },
}));

// The views are exercised by their own suites; this one is about routing.
vi.mock('./chat/ChatView', () => ({ ChatView: () => <div>chat view</div> }));
vi.mock('./pipeline/PipelineView', () => ({ PipelineView: () => <div>pipeline view</div> }));
vi.mock('./scoreboard/ScoreboardView', () => ({ ScoreboardView: () => <div>scoreboard view</div> }));
vi.mock('./experiments/ExperimentsView', () => ({ ExperimentsView: () => <div>experiments view</div> }));
vi.mock('./design/SystemDesignView', () => ({ SystemDesignView: () => <div>design view</div> }));

/** Drive the hash the way a click or the Back button would. */
async function goToHash(hash: string) {
  await act(async () => {
    window.location.hash = hash;
    window.dispatchEvent(new HashChangeEvent('hashchange'));
  });
}

beforeEach(() => {
  sessionStorage.clear();
  window.location.hash = '';
  health.mockResolvedValue({
    mode: 'dev',
    judge_model: 'deepseek-v4-flash',
    embedding_model: 'all-MiniLM-L6-v2',
    runner_loaded: true,
    stt: false,
  });
  corpus.mockResolvedValue({
    videos: [],
    channels: [],
    totals: { videos: 3, chunks: 30, channels: 1 },
    insights: [],
  });
  setups.mockResolvedValue([]);
  history.mockResolvedValue([]);
});

describe('landing routing', () => {
  it('shows the landing on a plain first visit', async () => {
    render(<App />);
    expect(await screen.findByText('RAG you can audit.')).toBeInTheDocument();
  });

  it('enters the workbench and remembers it for the session', async () => {
    render(<App />);
    await userEvent.click(await screen.findByRole('button', { name: /Enter workbench/ }));
    expect(screen.getByText('chat view')).toBeInTheDocument();
    expect(sessionStorage.getItem('tl-entered')).toBe('1');
  });

  it('goes back to the landing from the brand, and forward again', async () => {
    // The reported bug: once you entered, there was no way back. #landing was
    // documented as the route home but the hash listener only updated the tab,
    // so it resolved to Chat — and nothing on the page offered the route at all.
    render(<App />);
    await userEvent.click(await screen.findByRole('button', { name: /Enter workbench/ }));
    expect(screen.getByText('chat view')).toBeInTheDocument();

    await userEvent.click(screen.getByRole('button', { name: 'Back to the intro' }));
    expect(await screen.findByText('RAG you can audit.')).toBeInTheDocument();

    await userEvent.click(screen.getByRole('button', { name: /Enter workbench/ }));
    expect(screen.getByText('chat view')).toBeInTheDocument();
  });

  it('honours #landing after entering — the documented route home', async () => {
    render(<App />);
    await userEvent.click(await screen.findByRole('button', { name: /Enter workbench/ }));

    await goToHash('#landing');
    expect(await screen.findByText('RAG you can audit.')).toBeInTheDocument();
  });

  it('leaves the landing again when the hash moves to a tab (the Back button)', async () => {
    render(<App />);
    await userEvent.click(await screen.findByRole('button', { name: /Enter workbench/ }));
    await goToHash('#landing');
    expect(await screen.findByText('RAG you can audit.')).toBeInTheDocument();

    await goToHash('#board');
    await waitFor(() => expect(screen.getByText('scoreboard view')).toBeInTheDocument());
  });

  it('a deep link skips the landing entirely', async () => {
    window.location.hash = '#board';
    render(<App />);
    expect(await screen.findByText('scoreboard view')).toBeInTheDocument();
    expect(screen.queryByText('RAG you can audit.')).not.toBeInTheDocument();
  });
});

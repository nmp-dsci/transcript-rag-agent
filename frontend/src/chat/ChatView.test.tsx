import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import type { AskRequest, Corpus, Entry, SetupSpec, Video } from '../api/types';
import { ChatView } from './ChatView';

// jsdom has no layout engine, so the thread's scroll-to-bottom effect needs a stub.
Element.prototype.scrollIntoView = vi.fn();

const ask = vi.fn();
const judge = vi.fn();

vi.mock('../api/client', () => ({
  api: {
    ask: (...args: unknown[]) => ask(...args),
    judge: (...args: unknown[]) => judge(...args),
  },
}));

function video(id: string, title: string, channelId: string | null): Video {
  return {
    video_id: id,
    title,
    channel_name: channelId === 'c1' ? 'Smart Property Investment' : 'Aussie Firebug',
    channel_id: channelId,
    thumbnail_url: null,
    source_url: `https://youtu.be/${id}`,
    duration_seconds: 600,
    upload_date: '2026-01-01',
    view_count: 10,
    summary: null,
    fetched_at: null,
    chunk_count: 20,
  };
}

const CORPUS: Corpus = {
  videos: [
    video('v1', 'Negative gearing explained', 'c1'),
    video('v2', 'Index funds vs property', 'c2'),
  ],
  channels: [
    {
      channel_id: 'c1',
      channel_name: 'Smart Property Investment',
      video_count: 1,
      chunk_count: 20,
      video_ids: ['v1'],
    },
    {
      channel_id: 'c2',
      channel_name: 'Aussie Firebug',
      video_count: 1,
      chunk_count: 20,
      video_ids: ['v2'],
    },
  ],
  totals: { videos: 2, chunks: 40, channels: 2 },
  insights: [],
};

const SETUPS: SetupSpec[] = [
  { key: 'rag_agent', title: 'rag_agent (agentic)', description: 'multi hop' },
];

const ENTRY: Entry = {
  id: 'e1',
  question: 'q',
  url: null,
  asked_at: '2026-07-21T00:00:00+00:00',
  answers: [],
};

function view(props: Partial<Parameters<typeof ChatView>[0]> = {}) {
  return render(
    <ChatView
      setups={SETUPS}
      history={[]}
      corpus={CORPUS}
      onHistoryChange={vi.fn()}
      onActivity={vi.fn()}
      pendingScope={null}
      onScopeConsumed={vi.fn()}
      {...props}
    />,
  );
}

async function send(question: string) {
  await userEvent.type(screen.getByLabelText('Question'), question);
  await userEvent.click(screen.getByRole('button', { name: 'Send' }));
  await waitFor(() => expect(ask).toHaveBeenCalled());
  return ask.mock.calls[0]![0] as AskRequest;
}

beforeEach(() => {
  localStorage.clear();
  // Auto-judging would fire a second request that these tests do not exercise.
  localStorage.setItem('tlab.autojudge', '0');
  ask.mockReset();
  judge.mockReset();
  ask.mockImplementation(
    async (_request: AskRequest, handlers: { done?: (entry: Entry) => void }) => {
      handlers.done?.(ENTRY);
    },
  );
});

describe('ChatView request payload', () => {
  it('sends neither url nor channel_id for the whole corpus', async () => {
    view();
    const request = await send('what are the themes?');
    expect(request.url).toBeNull();
    expect(request.channel_id).toBeNull();
  });

  it('sends channel_id alone for a channel scope', async () => {
    view();
    await userEvent.selectOptions(screen.getByLabelText('Channel scope'), 'c1');
    const request = await send('what are the themes?');
    expect(request.channel_id).toBe('c1');
    expect(request.url).toBeNull();
  });

  it('sends url alone for a video scope, dropping the implied channel', async () => {
    view();
    await userEvent.selectOptions(
      screen.getByLabelText('Video scope'),
      'https://youtu.be/v1',
    );
    expect((screen.getByLabelText('Channel scope') as HTMLSelectElement).value).toBe('c1');
    const request = await send('what are the themes?');
    expect(request.url).toBe('https://youtu.be/v1');
    expect(request.channel_id).toBeNull();
  });

  it('sends the retrieval mode and transcript filter from the advanced panel', async () => {
    view();
    await userEvent.click(screen.getByRole('button', { name: /advanced/ }));
    await userEvent.click(screen.getByRole('button', { name: /hybrid/ }));
    await userEvent.click(screen.getByLabelText(/smart transcript filter/));
    const request = await send('what are the themes?');
    expect(request.retrieval_mode).toBe('hybrid');
    expect(request.filter_transcripts).toBe(true);
  });

  it('defaults to semantic retrieval with the filter off', async () => {
    view();
    const request = await send('what are the themes?');
    expect(request.retrieval_mode).toBe('semantic');
    expect(request.filter_transcripts).toBe(false);
  });
});

describe('ChatView pending scope', () => {
  it('adopts the pinned video and its owning channel', async () => {
    const onScopeConsumed = vi.fn();
    view({ pendingScope: 'https://youtu.be/v2', onScopeConsumed });
    await waitFor(() => expect(onScopeConsumed).toHaveBeenCalled());
    expect((screen.getByLabelText('Video scope') as HTMLSelectElement).value).toBe(
      'https://youtu.be/v2',
    );
    expect((screen.getByLabelText('Channel scope') as HTMLSelectElement).value).toBe('c2');
    expect(
      screen.getByText(/searching 1 video · 20 chunks in “Index funds vs property”/),
    ).toBeInTheDocument();
  });

  it('back-fills the channel once the corpus arrives', async () => {
    const rendered = view({ corpus: null, pendingScope: 'https://youtu.be/v1' });
    rendered.rerender(
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
    await waitFor(() =>
      expect((screen.getByLabelText('Channel scope') as HTMLSelectElement).value).toBe('c1'),
    );
  });
});

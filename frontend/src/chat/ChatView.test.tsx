import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import type {
  AskRequest,
  Corpus,
  Entry,
  ReviewedDocument,
  SetupSpec,
  Video,
} from '../api/types';
import { ChatView, documentForEntry } from './ChatView';

// jsdom has no layout engine, so the thread's scroll-to-bottom effect needs a stub.
Element.prototype.scrollIntoView = vi.fn();

const ask = vi.fn();
const judge = vi.fn();
const document = vi.fn();

vi.mock('../api/client', () => ({
  api: {
    ask: (...args: unknown[]) => ask(...args),
    judge: (...args: unknown[]) => judge(...args),
    document: (...args: unknown[]) => document(...args),
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
  document.mockReset();
  document.mockRejectedValue(new Error('not found'));
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

describe('ChatView document follow-up', () => {
  it('pins the reviewed document on a follow-up, keyed by entry id not question', async () => {
    const documentEntry: Entry = {
      id: 'doc-e1',
      question: 'review this article',
      url: null,
      asked_at: '2026-07-21T00:00:00+00:00',
      document_id: 'doc1',
      answers: [
        {
          key: 'rag_llm',
          title: 'rag_llm',
          command: 'rag_llm',
          answer: 'summary of the article',
          references: [],
          token_estimate: 10,
          chunk_count: 1,
          llm_calls: 1,
          iterations: null,
          terminated_reason: null,
          elapsed_seconds: 1,
          error: null,
          contexts: [],
          evaluation: null,
          model: null,
          embedding_model: null,
          top_k: null,
          followups: [
            {
              topic: 'related',
              rationale: 'digs deeper',
              followup_query: 'what does the article say about risk?',
              confidence: 0.9,
            },
          ],
        },
      ],
    };
    view({ history: [documentEntry] });
    await userEvent.click(screen.getByText('review this article').closest('button')!);
    await userEvent.click(screen.getByRole('button', { name: 'related' }));
    await waitFor(() => expect(ask).toHaveBeenCalled());
    const request = ask.mock.calls[0]![0] as AskRequest;
    expect(request.document_entry_id).toBe('doc-e1');
    expect(request.entry_id).toBeUndefined();
  });

  it('recombines the reloaded page with how much of it this question read', async () => {
    // The store knows the page; only the entry knows which sections the answer
    // was given, so a reloaded review must not claim more than it read.
    document.mockReset();
    document.mockResolvedValue({
      id: 'doc1',
      url: 'https://example.com/cv',
      requested_url: 'https://example.com/cv',
      title: 'A resume',
      truncated: false,
      fetched_at: '2026-07-21T00:00:00+00:00',
      sections: [
        { index: 0, heading: 'Summary', text: 'A summary.' },
        { index: 1, heading: 'Experience', text: 'Led a team of six.' },
        { index: 2, heading: 'Education', text: 'BSc computer science.' },
      ],
    });
    const entry: Entry = {
      id: 'doc-e2',
      question: 'review my cv',
      url: null,
      asked_at: '2026-07-21T00:00:00+00:00',
      document_id: 'doc1',
      document_detail: '2 of 3 sections selected by BM25',
      document_sections_selected: [0, 1],
      answers: [],
    };

    view({ history: [entry] });
    await userEvent.click(screen.getByText('review my cv').closest('button')!);

    await waitFor(() =>
      expect(screen.getByText('2 of 3 sections selected by BM25')).toBeInTheDocument(),
    );
    expect(screen.getByText('not reviewed')).toBeInTheDocument();
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

describe('ChatView document cache', () => {
  it('reuses the cached page across conversations without reusing its selection', async () => {
    // The real cache path. Selecting the first conversation fills `documents`
    // for doc1; selecting the second finds the id already cached and skips the
    // fetch. A per-document merge would therefore show the first entry's
    // selection line on the second entry's card.
    document.mockReset();
    document.mockResolvedValue({
      id: 'doc1',
      url: 'https://example.com/cv',
      requested_url: 'https://example.com/cv',
      title: 'A resume',
      truncated: false,
      fetched_at: '2026-08-09T00:00:00+00:00',
      sections: [
        { index: 0, heading: 'Summary', text: 'A summary.' },
        { index: 1, heading: 'Experience', text: 'Led a team of six.' },
        { index: 2, heading: 'Education', text: 'BSc computer science.' },
      ],
    });
    const base = {
      url: null,
      asked_at: '2026-08-09T00:00:00+00:00',
      answers: [],
      document_id: 'doc1',
    };
    const narrowed: Entry = {
      ...base,
      id: 'doc-e1',
      question: 'is my experience section strong?',
      document_detail: '2 of 3 sections selected by BM25',
      document_sections_selected: [0, 1],
    };
    const whole: Entry = {
      ...base,
      id: 'doc-e2',
      question: 'review my whole cv',
      document_detail: 'whole document — all 3 sections in context',
      document_sections_selected: [0, 1, 2],
    };

    view({ history: [narrowed, whole] });

    await userEvent.click(
      screen.getByText('is my experience section strong?').closest('button')!,
    );
    await waitFor(() =>
      expect(screen.getByText('2 of 3 sections selected by BM25')).toBeInTheDocument(),
    );
    expect(screen.getByText('not reviewed')).toBeInTheDocument();
    expect(document).toHaveBeenCalledTimes(1);

    await userEvent.click(screen.getByText('review my whole cv').closest('button')!);

    await waitFor(() =>
      expect(
        screen.getByText('whole document — all 3 sections in context'),
      ).toBeInTheDocument(),
    );
    expect(screen.queryByText('2 of 3 sections selected by BM25')).not.toBeInTheDocument();
    expect(screen.queryByText('not reviewed')).not.toBeInTheDocument();
    // The page itself was fetched once and reused, which is what made the bug
    // possible in the first place.
    expect(document).toHaveBeenCalledTimes(1);
  });
});

describe('documentForEntry', () => {
  const page: ReviewedDocument = {
    id: 'doc1',
    url: 'https://example.com/cv',
    requested_url: 'https://example.com/cv',
    title: 'A resume',
    truncated: false,
    fetched_at: '2026-08-09T00:00:00+00:00',
    sections: [
      { index: 0, heading: 'Summary', text: 'A summary.' },
      { index: 1, heading: 'Experience', text: 'Led a team of six.' },
      { index: 2, heading: 'Education', text: 'BSc computer science.' },
    ],
  };

  function entry(overrides: Partial<Entry>): Entry {
    return {
      id: 'e',
      question: 'review my cv',
      url: null,
      asked_at: '2026-08-09T00:00:00+00:00',
      answers: [],
      document_id: 'doc1',
      ...overrides,
    };
  }

  it('wears the entry own selection, not the cache one', () => {
    const narrowed = documentForEntry(
      entry({ id: 'e1', document_detail: '2 of 3 by BM25', document_sections_selected: [0, 1] }),
      { doc1: page },
    );

    expect(narrowed?.detail).toBe('2 of 3 by BM25');
    expect(narrowed?.sections_selected).toEqual([0, 1]);
    expect(narrowed?.narrowed).toBe(true);
  });

  it('gives two entries on the same document their own selection lines', () => {
    // The bug this replaces: the cache was filled once per document id, so the
    // first conversation's selection was frozen onto every later one.
    const documents = { doc1: page };
    const first = documentForEntry(
      entry({ id: 'e1', document_detail: '2 of 3 by BM25', document_sections_selected: [0, 1] }),
      documents,
    );
    const second = documentForEntry(
      entry({
        id: 'e2',
        document_detail: 'whole document — all 3 sections in context',
        document_sections_selected: [0, 1, 2],
      }),
      documents,
    );

    expect(first?.detail).toBe('2 of 3 by BM25');
    expect(second?.detail).toBe('whole document — all 3 sections in context');
    expect(first?.narrowed).toBe(true);
    expect(second?.narrowed).toBe(false);
  });

  it('leaves the cached page alone for entries that recorded no selection', () => {
    const older = documentForEntry(entry({ id: 'e3' }), { doc1: page });

    expect(older).toBe(page);
  });

  it('has no card when the document is not in the cache', () => {
    expect(documentForEntry(entry({ id: 'e4' }), {})).toBeNull();
    expect(documentForEntry(entry({ id: 'e5', document_id: null }), { doc1: page })).toBeNull();
  });
});

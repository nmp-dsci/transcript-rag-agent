import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import type { GuideDetail, GuideSummary } from '../api/types';
import { DemoContext } from '../demo';
import { GuidesView, appendActivity, citeRate, guideFromLocation, shortLabel } from './GuidesView';
import { job } from './ResearchMap.test';

const guides = vi.fn();
const guide = vi.fn();
const guideJob = vi.fn();
const subscribeGuideJob = vi.fn();
const askGuide = vi.fn();
vi.mock('../api/client', () => ({
  api: {
    guides: () => guides(),
    guide: (slug: string) => guide(slug),
    guideJob: () => guideJob(),
    // The stream never resolves on its own, exactly as the endpoint behaves.
    subscribeGuideJob: (handlers: unknown, signal: AbortSignal) => subscribeGuideJob(handlers, signal),
    askGuide: (payload: unknown) => askGuide(payload),
    addGuideComment: () => Promise.resolve(null),
    reviseGuide: () => Promise.resolve(null),
  },
}));
vi.mock('../speech/useSpeechToText', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../speech/useSpeechToText')>();
  return {
    ...actual,
    useSpeechToText: () => ({ supported: true, status: 'idle', transcript: { committed: '', interim: '' }, error: null, start: vi.fn(), stop: vi.fn(), reset: vi.fn() }),
  };
});

function summary(overrides: Partial<GuideSummary> = {}): GuideSummary {
  return {
    slug: 'ship-like-a-studio',
    title: 'Ship Like a Studio',
    topic: 'Ship Like a Studio',
    subtitle: 'How to build pages that read like a studio shipped them.',
    status: 'published',
    current_version: 1,
    compiled_at: '2026-08-31',
    model: { composer: 'hand-run (imported)' },
    video_ids: ['a', 'b'],
    sources: [],
    chunk_count: 838,
    cluster_count: 0,
    cite_total: 25,
    cite_valid: 25,
    gaps: [],
    web_allowed: false,
    web_urls: [],
    provenance: {},
    videos: 23,
    versions: [1],
    comments_open: 0,
    comments_total: 0,
    html_url: '/guides/ship-like-a-studio/guide.html',
    markdown_url: null,
    ...overrides,
  };
}

function detail(overrides: Partial<GuideDetail> = {}): GuideDetail {
  return {
    ...summary(),
    comments: [],
    claims: null,
    version_urls: { '1': '/guides/ship-like-a-studio/versions/v1.html' },
    receipts: {},
    ...overrides,
  };
}

beforeEach(() => {
  guides.mockReset();
  guide.mockReset();
  guideJob.mockReset();
  subscribeGuideJob.mockReset();
  askGuide.mockReset();
  guideJob.mockResolvedValue({ job: null, sdk: null });
  subscribeGuideJob.mockImplementation(() => new Promise<void>(() => undefined));
  window.history.replaceState(null, '', '/');
  guides.mockResolvedValue({
    guides: [summary(), summary({ slug: 'llm-as-a-judge', title: 'LLM-as-a-Judge', cite_total: 40, cite_valid: 35 })],
    write_command: 'python -m src.cli guides write --topic ...',
  });
  guide.mockImplementation((slug: string) =>
    Promise.resolve(
      slug === 'llm-as-a-judge'
        ? detail({ slug, title: 'LLM-as-a-Judge', html_url: '/guides/llm-as-a-judge/guide.html', markdown_url: '/guides/llm-as-a-judge/guide.md' })
        : detail(),
    ),
  );
});

describe('helpers', () => {
  it('reads only a well-formed slug from the query string', () => {
    expect(guideFromLocation('?guide=ship-like-a-studio')).toBe('ship-like-a-studio');
    expect(guideFromLocation('?guide=../etc')).toBeNull();
    expect(guideFromLocation('')).toBeNull();
  });

  it('folds a live activity event into the job', () => {
    const next = appendActivity(job({ activity: [], counters: {} }), { at: 't', label: 'compose', name: 'mcp__corpus__retrieve_chunks', message: 'q', question: 'why?' });
    expect(next?.activity).toHaveLength(1);
    expect(next?.counters).toEqual({ tool_calls: 1, retrieval_queries: 1 });
    expect(appendActivity(null, { at: 't', label: 'x', name: 'Read', message: 'm' })).toBeNull();
  });

  it('shortens section labels for the nav', () => {
    expect(shortLabel('The thesis: narrow it, calibrate it')).toBe('The thesis');
    expect(shortLabel('Scoring — what you ask the judge')).toBe('Scoring');
    expect(shortLabel('What this corpus does not tell you at all')).toBe('What this corpus does not');
    expect(shortLabel('Sources')).toBe('Sources');
  });

  it('formats the cite rate', () => {
    expect(citeRate({ cite_total: 25, cite_valid: 25 })).toBe('cites 25/25');
    expect(citeRate({ cite_total: 0, cite_valid: 0 })).toBe('no cites');
  });
});

describe('GuidesView', () => {
  it('lists the guides, opens the first, and loads its page in a sandboxed frame', async () => {
    render(<GuidesView />);
    expect(await screen.findByText('Ship Like a Studio', { selector: '.rq' })).toBeInTheDocument();
    expect(screen.getByText('LLM-as-a-Judge', { selector: '.rq' })).toBeInTheDocument();
    expect(await screen.findByRole('heading', { level: 2, name: 'Ship Like a Studio' })).toBeInTheDocument();
    const frame = screen.getByTitle('Ship Like a Studio') as HTMLIFrameElement;
    expect(frame.getAttribute('src')).toBe('/guides/ship-like-a-studio/guide.html?v=1');
    expect(frame.getAttribute('sandbox')).toContain('allow-scripts');
    expect(frame.getAttribute('sandbox')).not.toContain('allow-same-origin');
    // Provenance chips carry the numbers the page itself claims.
    expect(screen.getByText('838 chunks read')).toBeInTheDocument();
    expect(screen.getByText('corpus only')).toBeInTheDocument();
    expect(window.location.search).toBe('?guide=ship-like-a-studio');
  });

  it('switches guides from the rail and remembers the choice in the URL', async () => {
    render(<GuidesView />);
    await screen.findByRole('heading', { level: 2, name: 'Ship Like a Studio' });
    await userEvent.click(screen.getByText('LLM-as-a-Judge', { selector: '.rq' }));
    expect(await screen.findByRole('heading', { level: 2, name: 'LLM-as-a-Judge' })).toBeInTheDocument();
    expect(window.location.search).toBe('?guide=llm-as-a-judge');
    expect(screen.getByRole('link', { name: /for agents/ })).toHaveAttribute('href', '/guides/llm-as-a-judge/guide.md');
    // A cite rate below the bar is flagged in words, not only in colour.
    expect(screen.getByText('cites 35/40')).toHaveClass('warn');
  });

  it('honours a deep link over the default first guide', async () => {
    window.history.replaceState(null, '', '/?guide=llm-as-a-judge');
    render(<GuidesView />);
    expect(await screen.findByRole('heading', { level: 2, name: 'LLM-as-a-Judge' })).toBeInTheDocument();
  });

  it('builds a section nav from what the page reports, and scrolls through the bridge', async () => {
    render(<GuidesView />);
    const frame = (await screen.findByTitle('Ship Like a Studio')) as HTMLIFrameElement;
    const posted: unknown[] = [];
    const fakeWindow = { postMessage: (message: unknown) => posted.push(message) } as unknown as Window;
    Object.defineProperty(frame, 'contentWindow', { value: fakeWindow, configurable: true });
    fireEvent.load(frame);
    await act(async () => {
      window.dispatchEvent(
        new MessageEvent('message', {
          source: fakeWindow as unknown as MessageEventSource,
          data: { type: 'guide:ready', title: 'x', sections: [{ id: 'thesis', label: 'The operating principle' }] },
        }),
      );
    });
    await userEvent.click(await screen.findByRole('button', { name: 'The operating principle' }));
    expect(posted).toContainEqual({ type: 'guide:scroll', id: 'thesis' });
    expect(posted).toContainEqual({ type: 'guide:theme', theme: 'dark' });
  });

  it('ignores messages that did not come from the frame', async () => {
    render(<GuidesView />);
    await screen.findByTitle('Ship Like a Studio');
    await act(async () => {
      window.dispatchEvent(
        new MessageEvent('message', { data: { type: 'guide:ready', sections: [{ id: 'evil', label: 'Evil' }] } }),
      );
    });
    expect(screen.queryByRole('button', { name: 'Evil' })).not.toBeInTheDocument();
  });

  it('hides the compose panel and never opens the job stream in demo mode', async () => {
    render(
      <DemoContext.Provider value={true}>
        <GuidesView />
      </DemoContext.Provider>,
    );
    await screen.findByRole('heading', { level: 2, name: 'Ship Like a Studio' });
    expect(screen.queryByLabelText('New guide')).not.toBeInTheDocument();
    expect(subscribeGuideJob).not.toHaveBeenCalled();
    expect(screen.getByText(/the demo is read-only/)).toBeInTheDocument();
  });

  it('shows the running job from the stream, then opens the guide it published', async () => {
    let handlers: Record<string, (data: unknown) => void> = {};
    subscribeGuideJob.mockImplementation((h: Record<string, (data: unknown) => void>) => {
      handlers = h;
      return new Promise<void>(() => undefined);
    });
    render(<GuidesView />);
    await screen.findByRole('heading', { level: 2, name: 'Ship Like a Studio' });
    expect(screen.getByRole('button', { name: '+ New guide' })).toBeInTheDocument();
    const running = job({ slug: 'production-genai-systems', title: 'Production GenAI Systems' });
    await act(async () => handlers.snapshot?.({ job: running }));
    // The rail lists the run; clicking it shows the research map.
    await userEvent.click(screen.getByText('Production GenAI Systems', { selector: '.rq' }));
    expect(screen.getByText('chunks read in full')).toBeInTheDocument();
    await act(async () =>
      handlers.activity?.({ job_id: 'j1', event: { at: '2026-09-14T11:19:00+00:00', label: 'extract:cluster-2', name: 'Read', message: 'Read corpus/c.md' } }),
    );
    expect(screen.getByText('30 / 30')).toBeInTheDocument();
    // Done: the catalog reloads and the new guide opens in the reader.
    guides.mockResolvedValue({
      guides: [summary(), summary({ slug: 'production-genai-systems', title: 'Production GenAI Systems' })],
      write_command: 'x',
    });
    guide.mockImplementation((slug: string) => Promise.resolve(detail({ slug, title: slug === 'production-genai-systems' ? 'Production GenAI Systems' : 'Ship Like a Studio' })));
    await act(async () => handlers.job?.({ job: { ...running, status: 'done', version: 1 } }));
    expect(await screen.findByRole('heading', { level: 2, name: 'Production GenAI Systems' })).toBeInTheDocument();
    expect(window.location.search).toBe('?guide=production-genai-systems');
  });

  it('keeps the reader on the page during a revision and logs it in the Revisions tab', async () => {
    let handlers: Record<string, (data: unknown) => void> = {};
    subscribeGuideJob.mockImplementation((h: Record<string, (data: unknown) => void>) => {
      handlers = h;
      return new Promise<void>(() => undefined);
    });
    render(<GuidesView />);
    await screen.findByRole('heading', { level: 2, name: 'Ship Like a Studio' });
    const running = job({ kind: 'revise', slug: 'ship-like-a-studio', title: 'Ship Like a Studio', comment_ids: ['c-1', 'c-2'], stage: 'revise' });
    await act(async () => handlers.snapshot?.({ job: running }));
    // The page is still open — no research map took the window — and the
    // rail marks the guide, not a separate job entry.
    expect(screen.getByTitle('Ship Like a Studio')).toBeInTheDocument();
    expect(screen.queryByText('chunks read in full')).not.toBeInTheDocument();
    expect(screen.getByText('● revising')).toBeInTheDocument();
    await userEvent.click(screen.getByRole('tab', { name: /Revisions/ }));
    expect(screen.getByLabelText('Revision in progress')).toHaveTextContent('2 comments as one batch');
    expect(screen.getByText('c-2')).toBeInTheDocument();
    await act(async () =>
      handlers.activity?.({ job_id: 'j1', event: { at: '2026-09-14T11:19:00+00:00', label: 'revise', name: 'mcp__corpus__retrieve_chunks', message: 'q', question: 'position bias?' } }),
    );
    expect(screen.getByText(/retrieve_chunks "position bias\?"/)).toBeInTheDocument();
    // Published: the reader reloads the new version and the table lists it.
    guide.mockImplementation(() =>
      Promise.resolve(
        detail({
          current_version: 2,
          versions: [1, 2],
          version_urls: { '1': '/guides/ship-like-a-studio/versions/v1.html', '2': '/guides/ship-like-a-studio/versions/v2.html' },
          receipts: { v2: { version: 2, created_at: '2026-09-15T10:00:00+00:00', summary: 'Both done.', changed_sections: ['scoring'], items: [{ id: 'c-1', outcome: 'addressed', reason: '', sections: ['scoring'] }, { id: 'c-2', outcome: 'deferred', reason: 'no chunk', sections: [] }] } },
        }),
      ),
    );
    await act(async () => handlers.job?.({ job: { ...running, status: 'done', version: 2 } }));
    expect(await screen.findByText('1 addressed')).toBeInTheDocument();
    expect(screen.getByText('1 deferred')).toBeInTheDocument();
    expect((screen.getByTitle('Ship Like a Studio') as HTMLIFrameElement).getAttribute('src')).toBe('/guides/ship-like-a-studio/guide.html?v=2');
  });

  it('opens the ask surface when nothing is committed, and the read-only empty state in demo', async () => {
    guides.mockResolvedValue({ guides: [], write_command: 'x' });
    render(<GuidesView />);
    expect(await screen.findByRole('heading', { level: 2, name: 'What do you want a guide on?' })).toBeInTheDocument();
    expect(window.location.search).toBe('?guide=new');
    await waitFor(() => expect(guide).not.toHaveBeenCalled());
  });

  it('asks for a guide in plain language and opens the research map', async () => {
    let handlers: Record<string, (data: unknown) => void> = {};
    subscribeGuideJob.mockImplementation((h: Record<string, (data: unknown) => void>) => {
      handlers = h;
      return new Promise<void>(() => undefined);
    });
    const question = 'How do teams calibrate an LLM judge against human labels?';
    const started = job({ slug: 'calibrate-an-llm-judge', title: question, question, video_ids: ['a', 'b'], stage: 'export' });
    askGuide.mockResolvedValue(started);
    render(<GuidesView stt />);
    await screen.findByRole('heading', { level: 2, name: 'Ship Like a Studio' });
    await userEvent.click(screen.getByRole('button', { name: '+ New guide' }));
    expect(await screen.findByRole('heading', { level: 2, name: 'What do you want a guide on?' })).toBeInTheDocument();
    // The mic renders because the server has a speech relay.
    expect(screen.getByRole('button', { name: 'Start voice input' })).toBeInTheDocument();
    const box = screen.getByLabelText('Question');
    await userEvent.type(box, question);
    await userEvent.keyboard('{Enter}');
    await waitFor(() => expect(askGuide).toHaveBeenCalledWith({ question, allow_web: false }));
    // Straight to the research map, headed by the question, no checklist.
    expect(await screen.findByText('answering')).toBeInTheDocument();
    expect(screen.getByRole('heading', { level: 2, name: question })).toBeInTheDocument();
    expect(screen.getByText('2 videos chosen from the corpus')).toBeInTheDocument();
    expect(handlers.snapshot).toBeDefined();
  });

  it('hides the ask surface in demo mode', async () => {
    window.history.replaceState(null, '', '/?guide=new');
    render(
      <DemoContext.Provider value={true}>
        <GuidesView />
      </DemoContext.Provider>,
    );
    await screen.findByText('Ship Like a Studio', { selector: '.rq' });
    expect(screen.queryByRole('button', { name: '+ New guide' })).not.toBeInTheDocument();
    expect(screen.queryByRole('heading', { level: 2, name: 'What do you want a guide on?' })).not.toBeInTheDocument();
  });
});

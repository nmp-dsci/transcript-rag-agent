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
vi.mock('../api/client', () => ({
  api: {
    guides: () => guides(),
    guide: (slug: string) => guide(slug),
    guideJob: () => guideJob(),
    // The stream never resolves on its own, exactly as the endpoint behaves.
    subscribeGuideJob: (handlers: unknown, signal: AbortSignal) => subscribeGuideJob(handlers, signal),
    guideScope: () => Promise.resolve({ topic: '', probes: [], candidates: [], total_videos: 0 }),
    startGuide: () => Promise.resolve(null),
  },
}));

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
    expect(screen.getByLabelText('New guide')).toBeInTheDocument();
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

  it('shows the empty state when nothing is committed', async () => {
    guides.mockResolvedValue({ guides: [], write_command: 'x' });
    render(<GuidesView />);
    expect(await screen.findByText('No field guides yet')).toBeInTheDocument();
    await waitFor(() => expect(guide).not.toHaveBeenCalled());
  });
});

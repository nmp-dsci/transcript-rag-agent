import { act, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import type { EnrichmentSummary, IngestionJob } from '../api/types';
import { IndexPanel } from './IndexPanel';
import { ingestionJob, video } from './fixtures';

interface SubscribeHandlers {
  snapshot?: (data: { jobs: IngestionJob[] }) => void;
  job?: (data: { job: IngestionJob }) => void;
}

/** Captured per test so a case can push queue events by hand. */
let handlers: SubscribeHandlers;

const enqueueIndex = vi.fn(async (_payload: unknown) => ingestionJob());
const subscribeIndexQueue = vi.fn(
  (streamHandlers: SubscribeHandlers, _signal?: AbortSignal) => {
    handlers = streamHandlers;
    return new Promise<void>(() => {
      // Never resolves — the real endpoint is a persistent, unending feed;
      // the panel disconnects it by aborting the signal, not by the promise
      // settling.
    });
  },
);

vi.mock('../api/client', () => ({
  api: {
    enqueueIndex: (payload: unknown) => enqueueIndex(payload),
    subscribeIndexQueue: (handlers: SubscribeHandlers, signal?: AbortSignal) =>
      subscribeIndexQueue(handlers, signal),
    enrichmentState: () => enrichmentState(),
    runEnrichment: (payload: unknown) => runEnrichment(payload),
  },
}));

/** Nothing pending by default, so the enrichment banner stays out of the way
 * of the tests that are about the queue. Individual tests override it. */
const enrichmentState = vi.fn(
  async (): Promise<EnrichmentSummary> => ({
    summary_pending: [],
    graph_pending: [],
    needs_llm: true,
    total_videos: 0,
  }),
);
const runEnrichment = vi.fn(async (_payload?: unknown) => ({
  ok: true,
  started: 0,
  video_ids: [] as string[],
}));

/** Stream events arrive outside React's event loop, as they do in the browser. */
async function emit(fn: () => void) {
  await act(async () => {
    fn();
  });
}

async function openPanel() {
  const onIndexed = vi.fn();
  const onViewVideo = vi.fn();
  render(<IndexPanel onIndexed={onIndexed} onViewVideo={onViewVideo} />);
  await userEvent.click(screen.getByRole('button', { name: /Index new content/ }));
  await waitFor(() => expect(subscribeIndexQueue).toHaveBeenCalled());
  await emit(() => handlers.snapshot?.({ jobs: [] }));
  return { onIndexed, onViewVideo };
}

describe('IndexPanel', () => {
  beforeEach(() => {
    enqueueIndex.mockClear();
    subscribeIndexQueue.mockClear();
  });

  it('will not submit without a target', async () => {
    await openPanel();
    await userEvent.click(screen.getByRole('button', { name: 'Add to queue' }));
    expect(screen.getByText('Enter a video URL.')).toBeInTheDocument();
    expect(enqueueIndex).not.toHaveBeenCalled();
  });

  it('sends the channel payload in channel mode', async () => {
    await openPanel();
    await userEvent.click(screen.getByRole('button', { name: /Channel · latest N/ }));
    await userEvent.type(screen.getByLabelText('Channel'), '@alpha');
    await userEvent.click(screen.getByRole('button', { name: 'Add to queue' }));
    await waitFor(() => expect(enqueueIndex).toHaveBeenCalled());
    expect(enqueueIndex.mock.calls[0]?.[0]).toEqual({
      mode: 'channel',
      channel: '@alpha',
      latest: 5,
    });
  });

  it('the form stays enabled and can enqueue a second job right after the first', async () => {
    await openPanel();

    await userEvent.type(screen.getByLabelText('Video URL'), 'https://youtu.be/first');
    await userEvent.click(screen.getByRole('button', { name: 'Add to queue' }));
    await waitFor(() => expect(enqueueIndex).toHaveBeenCalledTimes(1));

    // No lock, no disabled state: the field cleared and is ready immediately.
    expect(screen.getByLabelText('Video URL')).toBeEnabled();
    expect(screen.getByRole('button', { name: 'Add to queue' })).toBeEnabled();

    // The first job is still running when the second is submitted.
    await emit(() =>
      handlers.job?.({
        job: ingestionJob({ id: 'a', target: 'https://youtu.be/first', status: 'running' }),
      }),
    );
    await userEvent.type(screen.getByLabelText('Video URL'), 'https://youtu.be/second');
    await userEvent.click(screen.getByRole('button', { name: 'Add to queue' }));
    await waitFor(() => expect(enqueueIndex).toHaveBeenCalledTimes(2));
  });

  it('renders queued, running, and errored jobs from queue events', async () => {
    await openPanel();

    await emit(() =>
      handlers.job?.({
        job: ingestionJob({ id: 'a', target: 'https://youtu.be/a', status: 'queued' }),
      }),
    );
    await emit(() =>
      handlers.job?.({
        job: ingestionJob({
          id: 'b',
          target: 'https://youtu.be/b',
          status: 'running',
          message: 'Chunking, embedding, and summarizing ...',
        }),
      }),
    );
    await emit(() =>
      handlers.job?.({
        job: ingestionJob({
          id: 'c',
          target: 'https://youtu.be/c',
          status: 'error',
          error: 'no transcript available for this video',
        }),
      }),
    );

    expect(screen.getByText('https://youtu.be/a')).toBeInTheDocument();
    expect(screen.getByText('Queued')).toBeInTheDocument();
    expect(screen.getByText('https://youtu.be/b')).toBeInTheDocument();
    expect(screen.getByText('Chunking, embedding, and summarizing ...')).toBeInTheDocument();
    expect(screen.getByText('no transcript available for this video')).toBeInTheDocument();
  });

  it('shows the result card and refreshes the corpus once a job completes', async () => {
    const { onIndexed, onViewVideo } = await openPanel();

    await emit(() =>
      handlers.job?.({
        job: ingestionJob({
          id: 'a',
          target: 'https://youtu.be/new',
          status: 'done',
          result: {
            ok: true,
            target: 'https://youtu.be/new',
            added_videos: [video({ video_id: 'new', title: 'A new talk' })],
            added_video_count: 1,
            added_chunk_count: 12,
            totals: { videos: 4, chunks: 293, channels: 2 },
            insights: [{ kind: 'size_spread', level: 'info', message: 'chunk counts vary widely' }],
            channels: [],
          },
        }),
      }),
    );

    expect(screen.getByText('+1 videos')).toBeInTheDocument();
    expect(screen.getByText('+12 chunks')).toBeInTheDocument();
    expect(screen.getByText(/now 4 videos · 293 chunks · 2 channels/)).toBeInTheDocument();
    expect(screen.getByText(/chunk counts vary widely/)).toBeInTheDocument();
    expect(onIndexed).toHaveBeenCalled();

    await userEvent.click(screen.getByRole('button', { name: /view in tree · A new talk/ }));
    expect(onViewVideo).toHaveBeenCalledWith('new');
  });

  it('shows the graph extraction badge once the automatic catch-up finishes', async () => {
    await openPanel();
    await emit(() =>
      handlers.job?.({
        job: ingestionJob({
          id: 'a',
          status: 'done',
          result: {
            ok: true,
            target: 'https://youtu.be/new',
            added_videos: [video({ video_id: 'new', title: 'A new talk' })],
            added_video_count: 1,
            added_chunk_count: 12,
            totals: { videos: 4, chunks: 293, channels: 2 },
            insights: [],
            channels: [],
            graph: { ok: true, extracted: 12, failed: 0 },
          },
        }),
      }),
    );
    expect(screen.getByText(/graph: \+12 chunks extracted/)).toBeInTheDocument();
  });

  it('flags a failed automatic graph extraction without hiding the vector-index success', async () => {
    await openPanel();
    await emit(() =>
      handlers.job?.({
        job: ingestionJob({
          id: 'a',
          status: 'done',
          result: {
            ok: true,
            target: 'https://youtu.be/new',
            added_videos: [video({ video_id: 'new', title: 'A new talk' })],
            added_video_count: 1,
            added_chunk_count: 12,
            totals: { videos: 4, chunks: 293, channels: 2 },
            insights: [],
            channels: [],
            graph: { ok: false, error: 'neo4j is down' },
          },
        }),
      }),
    );
    expect(screen.getByText('+1 videos')).toBeInTheDocument();
    expect(screen.getByText(/graph extraction failed/)).toBeInTheDocument();
  });

  it('says so when a completed job added nothing', async () => {
    await openPanel();
    await emit(() =>
      handlers.job?.({
        job: ingestionJob({
          id: 'a',
          status: 'done',
          result: {
            ok: true,
            target: 'https://youtu.be/new',
            added_videos: [],
            added_video_count: 0,
            added_chunk_count: 0,
            totals: { videos: 1, chunks: 10, channels: 1 },
            insights: [],
            channels: [],
          },
        }),
      }),
    );
    expect(screen.getByText(/every video was already in the index/i)).toBeInTheDocument();
  });

  it('surfaces a rejected enqueue call without disabling the form', async () => {
    await openPanel();
    enqueueIndex.mockRejectedValueOnce(new Error('HTTP 500'));
    await userEvent.type(screen.getByLabelText('Video URL'), 'https://youtu.be/x');
    await userEvent.click(screen.getByRole('button', { name: 'Add to queue' }));
    await waitFor(() => expect(screen.getByText('HTTP 500')).toBeInTheDocument());
    expect(screen.getByRole('button', { name: 'Add to queue' })).toBeEnabled();
  });

  it('refreshes the corpus once for a snapshot full of already-finished jobs', async () => {
    // The server never evicts jobs, so remounting the panel (leaving and
    // returning to the tab) replays every completed run — one refresh, not one
    // per historical job.
    const onIndexed = vi.fn();
    render(<IndexPanel onIndexed={onIndexed} onViewVideo={vi.fn()} />);
    await waitFor(() => expect(subscribeIndexQueue).toHaveBeenCalled());

    await emit(() =>
      handlers.snapshot?.({
        jobs: [
          ingestionJob({ id: 'a', status: 'done' }),
          ingestionJob({ id: 'b', status: 'done' }),
          ingestionJob({ id: 'c', status: 'done' }),
        ],
      }),
    );
    expect(onIndexed).toHaveBeenCalledTimes(1);

    // A job finishing later is still its own refresh.
    await emit(() => handlers.job?.({ job: ingestionJob({ id: 'd', status: 'done' }) }));
    expect(onIndexed).toHaveBeenCalledTimes(2);
  });

  it('seeds from the initial snapshot on connect', async () => {
    const onIndexed = vi.fn();
    render(<IndexPanel onIndexed={onIndexed} onViewVideo={vi.fn()} />);
    await userEvent.click(screen.getByRole('button', { name: /Index new content/ }));
    await waitFor(() => expect(subscribeIndexQueue).toHaveBeenCalled());

    await emit(() =>
      handlers.snapshot?.({
        jobs: [ingestionJob({ id: 'a', target: 'https://youtu.be/pre-existing', status: 'running' })],
      }),
    );
    expect(screen.getByText('https://youtu.be/pre-existing')).toBeInTheDocument();
  });
});

describe('enrichment state', () => {
  /** A job whose videos landed with the graph still pending. */
  const jobWithPendingGraph = {
    id: 'j1',
    mode: 'video' as const,
    target: 'https://www.youtube.com/watch?v=abc',
    latest: null,
    status: 'done' as const,
    stage: 'done',
    stage_index: 4,
    stage_total: 4,
    message: null,
    error: null,
    result: {
      ok: true,
      target: 'https://www.youtube.com/watch?v=abc',
      added_video_count: 1,
      added_chunk_count: 47,
      added_videos: [
        video({
          video_id: 'abc',
          summary: 'From the creator description.',
          summary_status: 'done',
          summary_source: 'description',
          graph_status: 'pending',
        }),
      ],
      totals: { videos: 1, chunks: 47, channels: 1 },
      insights: [],
      channels: [],
    },
  };

  it('says a video is indexed but not yet graphed', async () => {
    render(<IndexPanel onIndexed={vi.fn()} onViewVideo={vi.fn()} />);
    await userEvent.click(screen.getByRole('button', { name: /Index new content/ }));
    await emit(() => handlers.snapshot?.({ jobs: [jobWithPendingGraph] }));

    expect(screen.getByText(/summary · done · description/)).toBeInTheDocument();
    expect(screen.getByText(/graph · pending/)).toBeInTheDocument();
    // The reassuring half: the video is usable right now.
    expect(screen.getByText(/Retrievable now/)).toBeInTheDocument();
  });

  it('does not claim a source when the summary has not been written', async () => {
    const pending = {
      ...jobWithPendingGraph,
      result: {
        ...jobWithPendingGraph.result,
        added_videos: [
          video({ video_id: 'abc', summary: null, summary_status: 'pending', graph_status: 'pending' }),
        ],
      },
    };
    render(<IndexPanel onIndexed={vi.fn()} onViewVideo={vi.fn()} />);
    await userEvent.click(screen.getByRole('button', { name: /Index new content/ }));
    await emit(() => handlers.snapshot?.({ jobs: [pending] }));

    expect(screen.getByText('summary · pending')).toBeInTheDocument();
  });

  it('offers to clear a backlog, and says it costs no credits', async () => {
    enrichmentState.mockResolvedValueOnce({
      summary_pending: [],
      graph_pending: ['a', 'b', 'c'],
      needs_llm: true,
      total_videos: 3,
    });
    render(<IndexPanel onIndexed={vi.fn()} onViewVideo={vi.fn()} />);
    await userEvent.click(screen.getByRole('button', { name: /Index new content/ }));

    expect(await screen.findByText(/3 video\(s\) awaiting graph enrichment/)).toBeInTheDocument();
    expect(screen.getByText(/no Supadata credits/)).toBeInTheDocument();

    await userEvent.click(screen.getByRole('button', { name: /Run enrichment pass/ }));
    expect(runEnrichment).toHaveBeenCalled();
  });

  it('stays out of the way when there is no backlog', async () => {
    render(<IndexPanel onIndexed={vi.fn()} onViewVideo={vi.fn()} />);
    await userEvent.click(screen.getByRole('button', { name: /Index new content/ }));

    expect(screen.queryByRole('button', { name: /Run enrichment pass/ })).not.toBeInTheDocument();
  });
});

describe('stage progress', () => {
  const running = (stage: string, stageIndex: number) =>
    ingestionJob({
      id: 'run1',
      status: 'running',
      stage,
      stage_index: stageIndex,
      stage_total: 4,
      message: 'Chunking on transcript timings …',
    });

  it('shows how far a run has actually got', async () => {
    render(<IndexPanel onIndexed={vi.fn()} onViewVideo={vi.fn()} />);
    await userEvent.click(screen.getByRole('button', { name: /Index new content/ }));
    await emit(() => handlers.snapshot?.({ jobs: [running('chunk', 3)] }));

    expect(screen.getByText('3 / 4')).toBeInTheDocument();
    const steps = screen.getByRole('list', { name: 'Indexing stages' });
    const items = within(steps).getAllByRole('listitem');
    expect(items.map((li) => li.className.split(' ')[1])).toEqual([
      'done',
      'done',
      'active',
      'waiting',
    ]);
  });

  it('does not pretend a queued job has started a stage', async () => {
    render(<IndexPanel onIndexed={vi.fn()} onViewVideo={vi.fn()} />);
    await userEvent.click(screen.getByRole('button', { name: /Index new content/ }));
    await emit(() => handlers.snapshot?.({ jobs: [ingestionJob({ status: 'queued' })] }));

    expect(screen.queryByRole('list', { name: 'Indexing stages' })).not.toBeInTheDocument();
  });

  it('marks the stage a failed run died on', async () => {
    render(<IndexPanel onIndexed={vi.fn()} onViewVideo={vi.fn()} />);
    await userEvent.click(screen.getByRole('button', { name: /Index new content/ }));
    await emit(() =>
      handlers.snapshot?.({
        jobs: [
          ingestionJob({
            status: 'error',
            stage: 'fetch',
            stage_index: 2,
            error: 'Supadata request failed',
          }),
        ],
      }),
    );

    const items = within(screen.getByRole('list', { name: 'Indexing stages' })).getAllByRole(
      'listitem',
    );
    expect(items[1]?.className).toContain('failed');
    expect(screen.getByText('Supadata request failed')).toBeInTheDocument();
  });

  it('reads 4 / 4 once done, whatever stage_index the last event carried', async () => {
    render(<IndexPanel onIndexed={vi.fn()} onViewVideo={vi.fn()} />);
    await userEvent.click(screen.getByRole('button', { name: /Index new content/ }));
    await emit(() =>
      handlers.snapshot?.({
        jobs: [ingestionJob({ status: 'done', stage: 'done', stage_index: null })],
      }),
    );

    expect(screen.getByText('4 / 4 indexed')).toBeInTheDocument();
  });

  it('gives an enrichment job no stepper — it runs no indexing stages', async () => {
    render(<IndexPanel onIndexed={vi.fn()} onViewVideo={vi.fn()} />);
    await userEvent.click(screen.getByRole('button', { name: /Index new content/ }));
    await emit(() =>
      handlers.snapshot?.({
        jobs: [ingestionJob({ mode: 'enrichment', status: 'running', stage: 'graph' })],
      }),
    );

    expect(screen.queryByRole('list', { name: 'Indexing stages' })).not.toBeInTheDocument();
  });
});

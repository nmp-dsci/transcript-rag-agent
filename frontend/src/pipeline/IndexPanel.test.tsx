import { act, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import type { IngestionJob } from '../api/types';
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
  },
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

import { act, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import type { IndexResult, IndexStage } from '../api/types';
import { IndexPanel } from './IndexPanel';
import { indexResult, video } from './fixtures';

interface StreamHandlers {
  stage?: (data: IndexStage) => void;
  done?: (data: IndexResult) => void;
  error?: (data: { message: string }) => void;
}

/** Captured per test so each case can drive the stream by hand. */
let handlers: StreamHandlers;
let signal: AbortSignal | undefined;
let settle: () => void;

const indexStream = vi.fn(
  (_payload: unknown, streamHandlers: StreamHandlers, streamSignal?: AbortSignal) => {
    handlers = streamHandlers;
    signal = streamSignal;
    return new Promise<void>((resolve, reject) => {
      settle = resolve;
      // fetch() rejects an aborted request, and the panel relies on that to
      // tell a cancellation apart from a clean finish.
      streamSignal?.addEventListener('abort', () =>
        reject(new DOMException('Aborted', 'AbortError')),
      );
    });
  },
);

vi.mock('../api/client', () => ({
  api: {
    indexStream: (payload: unknown, streamHandlers: StreamHandlers, streamSignal?: AbortSignal) =>
      indexStream(payload, streamHandlers, streamSignal),
  },
}));

/** Stream events arrive outside React's event loop, as they do in the browser. */
async function emit(fn: () => void) {
  await act(async () => {
    fn();
  });
}

function stageRow(label: string): HTMLElement {
  const row = screen.getByText(label).closest('.idx-stage');
  if (!row) throw new Error(`no stage row for ${label}`);
  return row as HTMLElement;
}

async function startVideoRun() {
  const onIndexed = vi.fn();
  const onViewVideo = vi.fn();
  render(<IndexPanel onIndexed={onIndexed} onViewVideo={onViewVideo} />);
  await userEvent.click(screen.getByRole('button', { name: /Index new content/ }));
  await userEvent.type(screen.getByLabelText('Video URL'), 'https://youtu.be/new');
  await userEvent.click(screen.getByRole('button', { name: 'Start indexing' }));
  await waitFor(() => expect(indexStream).toHaveBeenCalled());
  return { onIndexed, onViewVideo };
}

describe('IndexPanel', () => {
  beforeEach(() => {
    indexStream.mockClear();
    signal = undefined;
  });

  it('will not start without a target', async () => {
    render(<IndexPanel onIndexed={vi.fn()} onViewVideo={vi.fn()} />);
    await userEvent.click(screen.getByRole('button', { name: /Index new content/ }));
    expect(screen.getByRole('button', { name: 'Start indexing' })).toBeDisabled();
    expect(indexStream).not.toHaveBeenCalled();
  });

  it('sends the channel payload in channel mode', async () => {
    render(<IndexPanel onIndexed={vi.fn()} onViewVideo={vi.fn()} />);
    await userEvent.click(screen.getByRole('button', { name: /Index new content/ }));
    await userEvent.click(screen.getByRole('button', { name: /Channel · latest N/ }));
    await userEvent.type(screen.getByLabelText('Channel'), '@alpha');
    await userEvent.click(screen.getByRole('button', { name: 'Start indexing' }));
    await waitFor(() => expect(indexStream).toHaveBeenCalled());
    expect(indexStream.mock.calls[0]?.[0]).toEqual({
      mode: 'channel',
      channel: '@alpha',
      latest: 5,
    });
  });

  it('advances the stage sequence as events arrive', async () => {
    await startVideoRun();

    expect(stageRow('Discover')).toHaveClass('pending');

    await emit(() => handlers.stage?.({ stage: 'discover', message: 'resolving 1 video' }));
    expect(stageRow('Discover')).toHaveClass('active');
    expect(stageRow('Embed')).toHaveClass('pending');

    await emit(() => handlers.stage?.({ stage: 'embed', message: 'embedding 12 chunks' }));
    expect(stageRow('Embed')).toHaveClass('active');
    expect(stageRow('Discover')).toHaveClass('done');
    expect(stageRow('Fetch')).toHaveClass('done');
    expect(stageRow('Summarize')).toHaveClass('pending');
  });

  it('streams each stage message into the log', async () => {
    await startVideoRun();
    await emit(() => handlers.stage?.({ stage: 'fetch', message: 'pulling transcript' }));
    await emit(() => handlers.stage?.({ stage: 'chunk', message: 'split into 12' }));
    expect(screen.getByText(/pulling transcript/)).toBeInTheDocument();
    expect(screen.getByText(/split into 12/)).toBeInTheDocument();
  });

  it('shows the result card and refreshes the corpus when the run completes', async () => {
    const { onIndexed, onViewVideo } = await startVideoRun();

    await emit(() =>
      handlers.done?.(
        indexResult({
          target: 'https://youtu.be/new',
          added_videos: [video({ video_id: 'new', title: 'A new talk' })],
          added_video_count: 1,
          added_chunk_count: 12,
          totals: { videos: 4, chunks: 293, channels: 2 },
          insights: [{ kind: 'size_spread', level: 'info', message: 'chunk counts vary widely' }],
        }),
      ),
    );
    await emit(() => settle());

    expect(screen.getByText('+1 videos')).toBeInTheDocument();
    expect(screen.getByText('+12 chunks')).toBeInTheDocument();
    expect(screen.getByText(/now 4 videos · 293 chunks · 2 channels/)).toBeInTheDocument();
    expect(screen.getByText(/chunk counts vary widely/)).toBeInTheDocument();
    expect(onIndexed).toHaveBeenCalled();

    // Every stage reads as done once the summary arrives.
    expect(stageRow('Summarize')).toHaveClass('done');

    await userEvent.click(screen.getByRole('button', { name: /view in tree · A new talk/ }));
    expect(onViewVideo).toHaveBeenCalledWith('new');
  });

  it('says so when a completed run added nothing', async () => {
    await startVideoRun();
    await emit(() => handlers.done?.(indexResult({ added_video_count: 0, added_videos: [] })));
    await emit(() => settle());
    expect(screen.getByText(/every video was already in the index/i)).toBeInTheDocument();
  });

  it('surfaces an error event from the stream', async () => {
    await startVideoRun();
    await emit(() => handlers.error?.({ message: 'no transcript available for this video' }));
    await emit(() => settle());
    expect(screen.getByText('no transcript available for this video')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Start indexing' })).toBeEnabled();
  });

  it('surfaces a rejected request', async () => {
    render(<IndexPanel onIndexed={vi.fn()} onViewVideo={vi.fn()} />);
    indexStream.mockImplementationOnce(() => Promise.reject(new Error('HTTP 500')));
    await userEvent.click(screen.getByRole('button', { name: /Index new content/ }));
    await userEvent.type(screen.getByLabelText('Video URL'), 'https://youtu.be/x');
    await userEvent.click(screen.getByRole('button', { name: 'Start indexing' }));
    await waitFor(() => expect(screen.getByText('HTTP 500')).toBeInTheDocument());
  });

  it('cancels an in-flight run through the abort signal', async () => {
    await startVideoRun();
    expect(signal?.aborted).toBe(false);
    await userEvent.click(screen.getByRole('button', { name: 'Cancel' }));
    expect(signal?.aborted).toBe(true);
    await waitFor(() => expect(screen.getByText('Indexing cancelled.')).toBeInTheDocument());
  });
});

import { act, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import type { MatrixJob } from '../api/types';
import { MatrixRunPanel } from './MatrixRunPanel';

interface StreamHandlers {
  snapshot?: (data: { job: MatrixJob | null }) => void;
  job?: (data: { job: MatrixJob }) => void;
}

let handlers: StreamHandlers;

const startMatrixRun = vi.fn();
const subscribeMatrixRun = vi.fn((streamHandlers: StreamHandlers, _signal?: AbortSignal) => {
  handlers = streamHandlers;
  // Never resolves — the real endpoint is a persistent feed the caller aborts.
  return new Promise<void>(() => undefined);
});

vi.mock('../api/client', () => ({
  api: {
    startMatrixRun: (...args: unknown[]) => startMatrixRun(...args),
    subscribeMatrixRun: (h: StreamHandlers, signal?: AbortSignal) =>
      subscribeMatrixRun(h, signal),
  },
}));

function job(overrides: Partial<MatrixJob> = {}): MatrixJob {
  return {
    id: 'j1',
    setups: ['rag_llm', 'graph_rag'],
    status: 'running',
    message: 'rag_llm × g001 — cached',
    cells_done: 5,
    cells_total: 40,
    cache_hits: 5,
    cache_misses: 0,
    run_id: null,
    error: null,
    ...overrides,
  };
}

/** Stream events arrive outside React's event loop, as they do in the browser. */
async function emit(fn: () => void) {
  await act(async () => {
    fn();
  });
}

async function mount(onRunFinished = vi.fn()) {
  render(<MatrixRunPanel onRunFinished={onRunFinished} />);
  await waitFor(() => expect(subscribeMatrixRun).toHaveBeenCalled());
  await emit(() => handlers.snapshot?.({ job: null }));
  return { onRunFinished };
}

describe('MatrixRunPanel', () => {
  beforeEach(() => {
    startMatrixRun.mockReset();
    subscribeMatrixRun.mockClear();
    startMatrixRun.mockResolvedValue(job({ cells_done: 0, cells_total: 0 }));
  });

  it('starts a run when pressed', async () => {
    await mount();
    await userEvent.click(screen.getByRole('button', { name: /Run eval matrix/ }));
    await waitFor(() => expect(startMatrixRun).toHaveBeenCalledTimes(1));
  });

  it('shows live progress from the stream', async () => {
    await mount();
    await emit(() => handlers.job?.({ job: job() }));

    expect(screen.getByText('5/40 cells')).toBeInTheDocument();
    expect(screen.getByText('5 cached · 0 scored')).toBeInTheDocument();
    expect(screen.getByText('rag_llm × g001 — cached')).toBeInTheDocument();
    expect(screen.getByText('rag_llm · graph_rag')).toBeInTheDocument();
  });

  it('disables the button while a run is in flight', async () => {
    await mount();
    await emit(() => handlers.job?.({ job: job() }));
    expect(screen.getByRole('button', { name: 'Running…' })).toBeDisabled();
  });

  it('reloads the committed runs once when a run finishes', async () => {
    const { onRunFinished } = await mount();
    await emit(() => handlers.job?.({ job: job() }));
    expect(onRunFinished).not.toHaveBeenCalled();

    await emit(() =>
      handlers.job?.({ job: job({ status: 'done', run_id: 'matrix-abc', message: null }) }),
    );
    expect(onRunFinished).toHaveBeenCalledTimes(1);
    expect(screen.getByText('matrix-abc')).toBeInTheDocument();

    // A late duplicate event for the same job must not refetch again.
    await emit(() =>
      handlers.job?.({ job: job({ status: 'done', run_id: 'matrix-abc', message: null }) }),
    );
    expect(onRunFinished).toHaveBeenCalledTimes(1);
  });

  it('surfaces a failed run without blocking a retry', async () => {
    const { onRunFinished } = await mount();
    await emit(() =>
      handlers.job?.({ job: job({ status: 'error', error: 'deepseek 402', message: null }) }),
    );
    expect(screen.getByText('deepseek 402')).toBeInTheDocument();
    expect(onRunFinished).not.toHaveBeenCalled();
    expect(screen.getByRole('button', { name: /Run eval matrix/ })).toBeEnabled();
  });

  it('picks up a run already in flight when the tab opens', async () => {
    render(<MatrixRunPanel onRunFinished={vi.fn()} />);
    await waitFor(() => expect(subscribeMatrixRun).toHaveBeenCalled());
    // The stream is seeded, so a run started from another tab is visible here.
    await emit(() => handlers.snapshot?.({ job: job({ cells_done: 12 }) }));
    expect(screen.getByText('12/40 cells')).toBeInTheDocument();
  });

  it('reports a rejected start without leaving the button stuck', async () => {
    await mount();
    startMatrixRun.mockRejectedValueOnce(new Error('HTTP 500'));
    await userEvent.click(screen.getByRole('button', { name: /Run eval matrix/ }));
    await waitFor(() => expect(screen.getByText('HTTP 500')).toBeInTheDocument());
    expect(screen.getByRole('button', { name: /Run eval matrix/ })).toBeEnabled();
  });
});

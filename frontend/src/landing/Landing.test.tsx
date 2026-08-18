/** The front door: renders the pitch, counts the corpus honestly, fires the
 * funnel events, and the gate helper decides who sees it. */

import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';

import posthog from 'posthog-js';
import { resetAnalyticsForTest, startAnalytics } from '../analytics';
import type { Corpus } from '../api/types';
import { shouldShowLanding } from '../App';
import { Landing } from './Landing';

vi.mock('posthog-js', () => ({
  default: { init: vi.fn(), capture: vi.fn() },
}));

const CORPUS: Corpus = {
  videos: [],
  channels: [],
  totals: { videos: 102, chunks: 2827, channels: 61 },
  insights: [],
};

describe('Landing', () => {
  afterEach(() => {
    resetAnalyticsForTest();
    vi.clearAllMocks();
    vi.unstubAllEnvs();
  });

  it('renders the pitch with live corpus counts', () => {
    render(<Landing corpus={CORPUS} demo={true} onEnter={vi.fn()} />);
    expect(screen.getByText('RAG you can audit.')).toBeInTheDocument();
    expect(screen.getByText('102 videos · 2,827 chunks indexed')).toBeInTheDocument();
    expect(screen.getByText('Q&A with receipts')).toBeInTheDocument();
    expect(screen.getByText('Scored, not vibes')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Enter demo/ })).toBeInTheDocument();
  });

  it('labels the door for the workbench on a dev serve', () => {
    render(<Landing corpus={CORPUS} demo={false} onEnter={vi.fn()} />);
    expect(screen.getByRole('button', { name: /Enter workbench/ })).toBeInTheDocument();
    expect(screen.getByText(/Dev build — everything live/)).toBeInTheDocument();
  });

  it('waits for health before printing the fine print', () => {
    render(<Landing corpus={CORPUS} demo={null} onEnter={vi.fn()} />);
    expect(screen.queryByText(/Read-only walkthrough/)).not.toBeInTheDocument();
    expect(screen.queryByText(/Dev build/)).not.toBeInTheDocument();
  });

  it('calls onEnter from the door', async () => {
    const onEnter = vi.fn();
    render(<Landing corpus={CORPUS} demo={true} onEnter={onEnter} />);
    await userEvent.click(screen.getByRole('button', { name: /Enter demo/ }));
    expect(onEnter).toHaveBeenCalledTimes(1);
  });

  it('fires the funnel events when analytics is live', async () => {
    vi.stubEnv('VITE_POSTHOG_KEY', 'phc_test');
    startAnalytics('demo');
    vi.mocked(posthog.capture).mockClear();
    render(<Landing corpus={CORPUS} demo={true} onEnter={vi.fn()} />);
    expect(posthog.capture).toHaveBeenCalledWith('demo_landing_view', undefined);
    await userEvent.click(screen.getByRole('button', { name: /Enter demo/ }));
    expect(posthog.capture).toHaveBeenCalledWith('demo_enter_click', undefined);
  });

  it('renders fine without analytics — events are silent no-ops', async () => {
    render(<Landing corpus={CORPUS} demo={true} onEnter={vi.fn()} />);
    await userEvent.click(screen.getByRole('button', { name: /Enter demo/ }));
    expect(posthog.capture).not.toHaveBeenCalled();
  });
});

describe('shouldShowLanding', () => {
  it('shows on a plain first visit', () => {
    expect(shouldShowLanding('', false)).toBe(true);
  });

  it('is bypassed by any deep link', () => {
    for (const hash of ['#board', '#pipeline', '#chat', '#experiments']) {
      expect(shouldShowLanding(hash, false)).toBe(false);
    }
  });

  it('does not return within a session once entered', () => {
    expect(shouldShowLanding('', true)).toBe(false);
  });

  it('#landing brings it back on purpose, even after entering', () => {
    expect(shouldShowLanding('#landing', true)).toBe(true);
  });
});

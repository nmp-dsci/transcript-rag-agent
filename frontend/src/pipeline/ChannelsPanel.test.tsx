import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { api } from '../api/client';
import type { ChannelList, WatchedChannel } from '../api/types';
import { ChannelsPanel } from './ChannelsPanel';

function channel(overrides: Partial<WatchedChannel> = {}): WatchedChannel {
  return {
    id: 'hamel-dev',
    kind: 'rss',
    label: 'Hamel Husain',
    url: 'https://hamel.dev/index.xml',
    enabled: true,
    sources: 3,
    last_polled_at: new Date(Date.now() - 2 * 3_600_000).toISOString(),
    next_due_at: null,
    interval_hours: 24,
    consecutive_failures: 0,
    disabled_reason: null,
    last_error: null,
    seen: 20,
    ...overrides,
  };
}

function list(channels: WatchedChannel[], error?: string): ChannelList {
  return {
    channels,
    totals: {
      channels: channels.length,
      enabled: channels.filter((item) => item.enabled).length,
      sources: channels.reduce((sum, item) => sum + item.sources, 0),
    },
    ...(error ? { error } : {}),
  };
}

function mount(body: ChannelList) {
  const channels = vi.spyOn(api, 'channels').mockResolvedValue(body);
  const poll = vi.spyOn(api, 'pollChannels').mockResolvedValue({
    id: 'job1',
    mode: 'channels',
    target: 'Hamel Husain',
    latest: null,
    status: 'queued',
    stage: null,
    message: null,
    result: null,
    error: null,
    stage_index: null,
    stage_total: 4,
    channel_ids: ['hamel-dev'],
  });
  const onPolled = vi.fn();
  render(<ChannelsPanel onPolled={onPolled} />);
  return { channels, poll, onPolled };
}

/** The panel ships collapsed, so anything below the headline needs a click. */
async function expand() {
  await userEvent.click(
    await screen.findByRole('button', { name: '+ Watched text sources' }),
  );
}

beforeEach(() => {
  vi.restoreAllMocks();
});

describe('ChannelsPanel', () => {
  it('lists each watched source with its kind and document count', async () => {
    mount(list([channel(), channel({ id: 'anthropic', label: 'Anthropic Engineering', kind: 'sitemap', sources: 25 })]));
    await expand();
    expect(await screen.findByText('Hamel Husain')).toBeTruthy();
    expect(screen.getByText('Anthropic Engineering')).toBeTruthy();
    expect(screen.getByText('RSS')).toBeTruthy();
    expect(screen.getByText('sitemap')).toBeTruthy();
    expect(screen.getByText('25')).toBeTruthy();
  });

  it('ships collapsed, so a long register cannot squeeze out the corpus pane', async () => {
    // The panel sits in a column that does not scroll. Rendering all 18 rows
    // on mount pushed the tree below the fold with no way to reach it.
    mount(list([channel(), channel({ id: 'two', label: 'Two' })]));
    expect(await screen.findByText('2 of 2 on \u00b7 6 documents')).toBeTruthy();
    expect(screen.queryByText('Hamel Husain')).toBeNull();
    await expand();
    expect(await screen.findByText('Hamel Husain')).toBeTruthy();
  });

  it('reports how many channels are on and how many documents they hold', async () => {
    mount(list([channel(), channel({ id: 'off', label: 'Off', enabled: false, sources: 1 })]));
    expect(await screen.findByText('1 of 2 on · 4 documents')).toBeTruthy();
  });

  it('marks a source whose feed carries the whole article', async () => {
    // The distinction is operational: those sources are never page-fetched.
    mount(list([channel({ body_in_feed: true })]));
    await expand();
    expect(await screen.findByText('full text')).toBeTruthy();
  });

  it('says a channel is off rather than merely un-polled', async () => {
    mount(list([channel({ enabled: false, last_polled_at: null })]));
    await expand();
    const badge = await screen.findByTitle(/set enabled: true in channels.yaml/);
    expect(badge).toBeTruthy();
  });

  it('surfaces the reason a channel was disabled by repeated failures', async () => {
    mount(list([channel({ disabled_reason: '5 consecutive failed polls; last error: boom' })]));
    await expand();
    expect(await screen.findByTitle(/5 consecutive failed polls/)).toBeTruthy();
  });

  it('queues a poll for one channel and tells the reader where to watch it', async () => {
    const { poll, onPolled } = mount(list([channel()]));
    await expand();
    await userEvent.click(await screen.findByRole('button', { name: 'Poll' }));
    await waitFor(() => expect(poll).toHaveBeenCalledWith(['hamel-dev']));
    expect(onPolled).toHaveBeenCalled();
    expect(await screen.findByText(/Watch it in the queue above/)).toBeTruthy();
  });

  it('polls every enabled channel when asked for all', async () => {
    const { poll } = mount(list([channel(), channel({ id: 'two', label: 'Two' })]));
    await expand();
    await userEvent.click(await screen.findByRole('button', { name: 'Poll all' }));
    await waitFor(() => expect(poll).toHaveBeenCalledWith([]));
  });

  it('cannot poll when nothing is enabled', async () => {
    mount(list([channel({ enabled: false })]));
    await expand();
    const button = await screen.findByRole('button', { name: 'Poll all' });
    expect((button as HTMLButtonElement).disabled).toBe(true);
  });

  it('explains an empty register instead of rendering a bare list', async () => {
    mount(list([]));
    await expand();
    expect(await screen.findByText(/No channels configured/)).toBeTruthy();
  });

  it('says why the register could not be read', async () => {
    // A bad channels.yaml must read as a reason, not as "nothing configured".
    mount(list([], "channel id 'Bad Id' must be lowercase kebab-case"));
    expect(await screen.findByText(/must be lowercase kebab-case/)).toBeTruthy();
  });

  it('reports a failed poll without losing the list', async () => {
    mount(list([channel()]));
    await expand();
    vi.spyOn(api, 'pollChannels').mockRejectedValue(new Error('server said no'));
    await userEvent.click(await screen.findByRole('button', { name: 'Poll' }));
    expect(await screen.findByText('server said no')).toBeTruthy();
    expect(screen.getByText('Hamel Husain')).toBeTruthy();
  });

  it('states that videos are not part of this and stay manual', async () => {
    // The scope rule is deliberate and easy to forget, so the panel says it.
    mount(list([channel()]));
    await expand();
    expect(await screen.findByText(/videos stay a manual paste/)).toBeTruthy();
  });
});

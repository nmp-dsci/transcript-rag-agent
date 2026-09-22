import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';

import type { WebChannelGroup, WebChunk, WebSourceSummary } from '../api/types';
import { CorpusTree, sortWebSources } from './CorpusTree';
import { WebSourceDetail } from './WebSourceDetail';

function source(overrides: Partial<WebSourceSummary> = {}): WebSourceSummary {
  return {
    key: 'web:aaa',
    external_id: 'a',
    channel_id: 'seeds',
    title: 'Evals for AI engineers',
    url: 'https://example.com/evals',
    state: 'live',
    state_reason: null,
    revision: 1,
    chunk_count: 3,
    word_count: 900,
    section_count: 4,
    truncated: false,
    from_feed: false,
    verifiable: true,
    published_at: '2026-02-01T00:00:00Z',
    last_fetched_at: '2026-03-01T00:00:00Z',
    last_changed_at: '2026-03-01T00:00:00Z',
    ...overrides,
  };
}

function group(overrides: Partial<WebChannelGroup> = {}): WebChannelGroup {
  return {
    channel_id: 'seeds',
    label: 'Seed articles',
    sources: [source()],
    chunk_count: 3,
    ...overrides,
  };
}

function webChunk(overrides: Partial<WebChunk> = {}): WebChunk {
  return {
    chunk_index: 0,
    text: 'How to measure an agent.',
    heading: 'Measuring agents',
    section_index: 0,
    part_index: 0,
    anchor: 'measuring-agents',
    url: 'https://example.com/evals',
    citation: 'Evals for AI engineers · § Measuring agents',
    revision: 1,
    ...overrides,
  };
}

function tree(props: Partial<Parameters<typeof CorpusTree>[0]> = {}) {
  const onSelectWebSource = vi.fn();
  const onSelectWebChunk = vi.fn();
  render(
    <CorpusTree
      videos={[]}
      sort="title"
      onSortChange={vi.fn()}
      selectedVideo={null}
      selectedChunk={null}
      chunks={{}}
      onSelectVideo={vi.fn()}
      onSelectChunk={vi.fn()}
      webChannels={[group()]}
      selectedWebSource={null}
      selectedWebChunk={null}
      webChunks={{}}
      onSelectWebSource={onSelectWebSource}
      onSelectWebChunk={onSelectWebChunk}
      {...props}
    />,
  );
  return { onSelectWebSource, onSelectWebChunk };
}

describe('the tree’s web half', () => {
  it('offers websites as a second root beside the videos', () => {
    // The corpus has two source types; one root for both would have to
    // pretend an article is a video.
    tree();
    expect(screen.getByText('All videos')).toBeTruthy();
    expect(screen.getByText('Websites & docs')).toBeTruthy();
  });

  it('nests channel then document, the way the video half nests channel then video', async () => {
    tree();
    await userEvent.click(screen.getByText('Websites & docs'));
    expect(screen.getByText('Seed articles')).toBeTruthy();
    await userEvent.click(screen.getByText('Seed articles'));
    expect(screen.getByText('Evals for AI engineers')).toBeTruthy();
  });

  it('selects the document when it is expanded, so the detail pane follows', async () => {
    const { onSelectWebSource } = tree();
    await userEvent.click(screen.getByText('Websites & docs'));
    await userEvent.click(screen.getByText('Seed articles'));
    await userEvent.click(screen.getByText('Evals for AI engineers'));
    expect(onSelectWebSource).toHaveBeenCalledWith('web:aaa');
  });

  it('names a chunk by its section heading, not by the first words of its text', async () => {
    // The heading is this half's citation unit, the way mm:ss is the other's.
    tree({ webChunks: { 'web:aaa': [webChunk()] }, selectedWebSource: 'web:aaa' });
    await userEvent.click(screen.getByText('Websites & docs'));
    await userEvent.click(screen.getByText('Seed articles'));
    await userEvent.click(screen.getByText('Evals for AI engineers'));
    expect(screen.getByText('§ Measuring agents')).toBeTruthy();
  });

  it('falls back to the URL when a page offered no title', async () => {
    tree({ webChannels: [group({ sources: [source({ title: null })] })] });
    await userEvent.click(screen.getByText('Websites & docs'));
    await userEvent.click(screen.getByText('Seed articles'));
    expect(screen.getByText('https://example.com/evals')).toBeTruthy();
  });

  it('badges a document whose state needs explaining, and leaves live ones bare', async () => {
    tree({
      webChannels: [
        group({ sources: [source({ state: 'gone', state_reason: '404 twice', verifiable: false })] }),
      ],
    });
    await userEvent.click(screen.getByText('Websites & docs'));
    await userEvent.click(screen.getByText('Seed articles'));
    expect(screen.getByTitle('404 twice').textContent).toBe('gone');
  });

  it('says the web half is empty rather than rendering a bare root', async () => {
    tree({ webChannels: [] });
    await userEvent.click(screen.getByText('Websites & docs'));
    expect(screen.getByText(/no watched sources yet/)).toBeTruthy();
  });

  it('leaves the video root open when there is no web half to compete with it', () => {
    // Collapsing both is only worth it once there are two roots to see.
    tree({ webChannels: [] });
    const root = screen.getByText('All videos').closest('details');
    expect((root as HTMLDetailsElement).open).toBe(true);
  });
});

describe('sortWebSources', () => {
  const small = source({ key: 'web:s', title: 'Zebra', chunk_count: 1, last_changed_at: '2026-01-01T00:00:00Z' });
  const big = source({ key: 'web:b', title: 'Apple', chunk_count: 50, last_changed_at: '2026-06-01T00:00:00Z' });

  it('falls back to size for "top views", which a document does not have', () => {
    expect(sortWebSources([small, big], 'views').map((item) => item.key)).toEqual(['web:b', 'web:s']);
  });

  it('reads "most recent" as when this copy last changed, not when it was published', () => {
    // The tree is a view of the store, so what matters is the stored copy.
    const stale = source({ key: 'web:x', published_at: '2030-01-01T00:00:00Z', last_changed_at: '2020-01-01T00:00:00Z' });
    expect(sortWebSources([stale, big], 'recent').map((item) => item.key)).toEqual(['web:b', 'web:x']);
  });

  it('orders by title, using the URL for documents that have none', () => {
    expect(sortWebSources([small, big], 'title').map((item) => item.key)).toEqual(['web:b', 'web:s']);
  });
});

describe('WebSourceDetail', () => {
  it('links a chunk to its own section anchor', () => {
    render(
      <WebSourceDetail source={source()} chunks={[webChunk()]} selectedChunk={null} />,
    );
    const link = screen.getByText(/open at this section/) as HTMLAnchorElement;
    expect(link.getAttribute('href')).toBe('https://example.com/evals#measuring-agents');
  });

  it('offers the page itself when a chunk has no anchor', () => {
    render(
      <WebSourceDetail
        source={source()}
        chunks={[webChunk({ anchor: null, heading: null })]}
        selectedChunk={null}
      />,
    );
    expect((screen.getByText(/open the page/) as HTMLAnchorElement).getAttribute('href')).toBe(
      'https://example.com/evals',
    );
  });

  it('refuses to render a dead source as a live link', () => {
    // The chunks are kept — an answer built on them is still real — but the
    // citation has to say it cannot be re-checked.
    render(
      <WebSourceDetail
        source={source({ state: 'gone', state_reason: '404 twice', verifiable: false })}
        chunks={[webChunk()]}
        selectedChunk={null}
      />,
    );
    expect(screen.queryByRole('link')).toBeNull();
    expect(screen.getByText(/no longer resolves/)).toBeTruthy();
  });

  it('does not offer to ask about a source that can no longer be verified', () => {
    render(
      <WebSourceDetail
        source={source({ state: 'blocked', verifiable: false })}
        chunks={[]}
        selectedChunk={null}
        onAskAbout={vi.fn()}
      />,
    );
    expect(screen.queryByRole('button', { name: /Ask about this/ })).toBeNull();
  });

  it('says a truncated document is only part of one', () => {
    render(
      <WebSourceDetail
        source={source({ state: 'truncated', truncated: true })}
        chunks={[]}
        selectedChunk={null}
      />,
    );
    expect(screen.getByText(/part of a document/)).toBeTruthy();
  });
});

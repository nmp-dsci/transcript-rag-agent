import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { useState } from 'react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import type { Corpus, SetupSpec, Video } from '../api/types';
import {
  type AskOptions,
  type ChatScope,
  Composer,
  WHOLE_CORPUS,
  describeScope,
  readAskPrefs,
  scopePayload,
  videosInScope,
} from './Composer';

function video(
  id: string,
  title: string,
  channelId: string | null,
  channelName: string | null,
  chunks: number,
): Video {
  return {
    video_id: id,
    title,
    channel_name: channelName,
    channel_id: channelId,
    thumbnail_url: null,
    source_url: `https://youtu.be/${id}`,
    duration_seconds: 600,
    upload_date: '2026-01-01',
    view_count: 10,
    summary: null,
    fetched_at: null,
    chunk_count: chunks,
  };
}

const CORPUS: Corpus = {
  videos: [
    video('v1', 'Negative gearing explained', 'c1', 'Smart Property Investment', 100),
    video('v2', 'Rentvesting basics', 'c1', 'Smart Property Investment', 56),
    video('v3', 'Index funds vs property', 'c2', 'Aussie Firebug', 40),
    // No channel_id: only reachable from the "All channels" video list.
    video('v4', 'Orphan upload', null, null, 5),
  ],
  channels: [
    {
      channel_id: 'c1',
      channel_name: 'Smart Property Investment',
      video_count: 2,
      chunk_count: 156,
      video_ids: ['v1', 'v2'],
    },
    {
      channel_id: 'c2',
      channel_name: 'Aussie Firebug',
      video_count: 1,
      chunk_count: 40,
      video_ids: ['v3'],
    },
  ],
  totals: { videos: 4, chunks: 201, channels: 2 },
  insights: [],
};

const SETUPS: SetupSpec[] = [
  { key: 'rag_llm', title: 'rag_llm (fast)', description: 'single hop' },
  { key: 'rag_agent', title: 'rag_agent (agentic)', description: 'multi hop' },
];

/** The composer is controlled, so tests drive it through a stateful host. */
function Harness({
  onAsk = vi.fn(),
  initial = WHOLE_CORPUS,
  corpus = CORPUS,
}: {
  onAsk?: (options: AskOptions) => void;
  initial?: ChatScope;
  corpus?: Corpus | null;
}) {
  const [scope, setScope] = useState<ChatScope>(initial);
  return (
    <Composer
      setups={SETUPS}
      corpus={corpus}
      busy={false}
      scope={scope}
      onScopeChange={setScope}
      defaultSetup="rag_agent"
      onDefaultSetupChange={vi.fn()}
      onAsk={onAsk}
      onCancel={vi.fn()}
    />
  );
}

const channelSelect = () => screen.getByLabelText('Channel scope') as HTMLSelectElement;
const videoSelect = () => screen.getByLabelText('Video scope') as HTMLSelectElement;
const videoOptionLabels = () =>
  within(videoSelect())
    .getAllByRole('option')
    .map((option) => option.textContent);

beforeEach(() => {
  localStorage.clear();
});

describe('Composer scope selects', () => {
  it('lists every video when no channel is selected', () => {
    render(<Harness />);
    expect(videoOptionLabels()).toEqual([
      'All videos',
      'Negative gearing explained',
      'Rentvesting basics',
      'Index funds vs property',
      'Orphan upload',
    ]);
  });

  it('narrows the video list to the selected channel', async () => {
    render(<Harness />);
    await userEvent.selectOptions(channelSelect(), 'c1');
    expect(videoOptionLabels()).toEqual([
      'All videos',
      'Negative gearing explained',
      'Rentvesting basics',
    ]);
  });

  it('shows each channel with its video count', () => {
    render(<Harness />);
    expect(
      within(channelSelect()).getByRole('option', { name: 'Smart Property Investment (2)' }),
    ).toBeInTheDocument();
  });

  it('auto-selects the owning channel when a video is picked', async () => {
    render(<Harness />);
    await userEvent.selectOptions(videoSelect(), 'https://youtu.be/v3');
    expect(channelSelect().value).toBe('c2');
    expect(videoSelect().value).toBe('https://youtu.be/v3');
  });

  it('clears a pinned video that does not belong to the newly chosen channel', async () => {
    render(<Harness />);
    await userEvent.selectOptions(videoSelect(), 'https://youtu.be/v1');
    expect(channelSelect().value).toBe('c1');
    await userEvent.selectOptions(channelSelect(), 'c2');
    expect(videoSelect().value).toBe('');
  });

  it('keeps a pinned video when its own channel is re-selected', async () => {
    render(<Harness />);
    await userEvent.selectOptions(videoSelect(), 'https://youtu.be/v1');
    await userEvent.selectOptions(channelSelect(), 'c1');
    expect(videoSelect().value).toBe('https://youtu.be/v1');
  });

  it('clears both selects when the channel is set back to all channels', async () => {
    render(<Harness />);
    await userEvent.selectOptions(videoSelect(), 'https://youtu.be/v1');
    await userEvent.selectOptions(channelSelect(), '');
    expect(channelSelect().value).toBe('');
    expect(videoSelect().value).toBe('');
    expect(screen.getByRole('status')).toHaveTextContent(/whole corpus/);
  });

  it('falls back to the channel scope when the video is cleared', async () => {
    render(<Harness />);
    await userEvent.selectOptions(videoSelect(), 'https://youtu.be/v1');
    await userEvent.selectOptions(videoSelect(), '');
    expect(channelSelect().value).toBe('c1');
    expect(screen.getByRole('status')).toHaveTextContent('in Smart Property Investment');
  });

  it('drops the channel when a video with no channel is pinned', async () => {
    render(<Harness />);
    await userEvent.selectOptions(channelSelect(), 'c1');
    await userEvent.selectOptions(channelSelect(), '');
    await userEvent.selectOptions(videoSelect(), 'https://youtu.be/v4');
    expect(channelSelect().value).toBe('');
  });
});

describe('scope summary', () => {
  it('reports the whole-corpus totals', () => {
    expect(describeScope(CORPUS, WHOLE_CORPUS)).toBe(
      'searching the whole corpus (4 videos, 201 chunks)',
    );
  });

  it('reports the selected channel’s counts', () => {
    expect(describeScope(CORPUS, { channelId: 'c1', videoUrl: null })).toBe(
      'searching 2 videos · 156 chunks in Smart Property Investment',
    );
  });

  it('singularises a one-video channel', () => {
    expect(describeScope(CORPUS, { channelId: 'c2', videoUrl: null })).toBe(
      'searching 1 video · 40 chunks in Aussie Firebug',
    );
  });

  it('reports the pinned video’s own chunk count', () => {
    expect(
      describeScope(CORPUS, { channelId: 'c1', videoUrl: 'https://youtu.be/v1' }),
    ).toBe('searching 1 video · 100 chunks in “Negative gearing explained”');
  });

  it('derives counts for a channel missing from the channels list', () => {
    const partial: Corpus = { ...CORPUS, channels: [] };
    expect(describeScope(partial, { channelId: 'c1', videoUrl: null })).toBe(
      'searching 2 videos · 156 chunks in Smart Property Investment',
    );
  });

  it('degrades gracefully before the corpus loads', () => {
    expect(describeScope(null, WHOLE_CORPUS)).toBe('searching the whole corpus');
  });

  it('updates live as the scope changes', async () => {
    render(<Harness />);
    expect(screen.getByRole('status')).toHaveTextContent(
      'searching the whole corpus (4 videos, 201 chunks)',
    );
    await userEvent.selectOptions(channelSelect(), 'c1');
    expect(screen.getByRole('status')).toHaveTextContent(
      'searching 2 videos · 156 chunks in Smart Property Investment',
    );
    await userEvent.selectOptions(videoSelect(), 'https://youtu.be/v2');
    expect(screen.getByRole('status')).toHaveTextContent(
      'searching 1 video · 56 chunks in “Rentvesting basics”',
    );
  });
});

describe('scopePayload', () => {
  it('sends nothing for the whole corpus', () => {
    expect(scopePayload(WHOLE_CORPUS)).toEqual({ url: null, channelId: null });
  });

  it('sends channel_id alone for a channel scope', () => {
    expect(scopePayload({ channelId: 'c1', videoUrl: null })).toEqual({
      url: null,
      channelId: 'c1',
    });
  });

  it('sends url alone for a video scope, never both', () => {
    expect(scopePayload({ channelId: 'c1', videoUrl: 'https://youtu.be/v1' })).toEqual({
      url: 'https://youtu.be/v1',
      channelId: null,
    });
  });
});

describe('videosInScope', () => {
  it('keeps only videos that can be pinned', () => {
    expect(videosInScope(CORPUS, null)).toHaveLength(4);
    expect(videosInScope(CORPUS, 'c1').map((item) => item.video_id)).toEqual(['v1', 'v2']);
    expect(videosInScope(null, null)).toEqual([]);
  });
});

describe('submitting', () => {
  const ask = async (onAsk: (options: AskOptions) => void, initial: ChatScope) => {
    render(<Harness onAsk={onAsk} initial={initial} />);
    await userEvent.type(screen.getByLabelText('Question'), 'what about yields?');
    await userEvent.click(screen.getByRole('button', { name: 'Send' }));
  };

  it('submits a channel scope as channel_id with no url', async () => {
    const onAsk = vi.fn();
    await ask(onAsk, { channelId: 'c1', videoUrl: null });
    expect(onAsk).toHaveBeenCalledWith(
      expect.objectContaining({
        question: 'what about yields?',
        url: null,
        channelId: 'c1',
        setups: ['rag_agent'],
      }),
    );
  });

  it('submits a video scope as url with no channel_id', async () => {
    const onAsk = vi.fn();
    await ask(onAsk, { channelId: 'c1', videoUrl: 'https://youtu.be/v1' });
    expect(onAsk).toHaveBeenCalledWith(
      expect.objectContaining({ url: 'https://youtu.be/v1', channelId: null }),
    );
  });

  it('sends the current advanced preferences with the question', async () => {
    const onAsk = vi.fn();
    await ask(onAsk, WHOLE_CORPUS);
    expect(onAsk).toHaveBeenCalledWith(
      expect.objectContaining({
        url: null,
        channelId: null,
        retrievalMode: 'semantic',
        filterTranscripts: false,
        autoJudge: true,
        topK: null,
      }),
    );
  });

  it('still sends on Enter and keeps Shift+Enter for newlines', async () => {
    const onAsk = vi.fn();
    render(<Harness onAsk={onAsk} />);
    const box = screen.getByLabelText('Question');
    await userEvent.type(box, 'first{Shift>}{Enter}{/Shift}second');
    expect(onAsk).not.toHaveBeenCalled();
    await userEvent.type(box, '{Enter}');
    expect(onAsk).toHaveBeenCalledWith(
      expect.objectContaining({ question: 'first\nsecond' }),
    );
  });
});

describe('advanced preferences', () => {
  const openAdvanced = async () =>
    userEvent.click(screen.getByRole('button', { name: /advanced/ }));

  it('defaults to semantic retrieval with the filter off', async () => {
    render(<Harness />);
    await openAdvanced();
    expect(screen.getByRole('button', { name: 'semantic' })).toHaveAttribute(
      'aria-pressed',
      'true',
    );
    expect(screen.getByLabelText(/smart transcript filter/)).not.toBeChecked();
  });

  it('persists the retrieval mode and restores it on remount', async () => {
    const first = render(<Harness />);
    await openAdvanced();
    await userEvent.click(screen.getByRole('button', { name: /hybrid/ }));
    expect(localStorage.getItem('tlab.retrievalmode')).toBe('hybrid');
    first.unmount();

    render(<Harness />);
    await openAdvanced();
    expect(screen.getByRole('button', { name: /hybrid/ })).toHaveAttribute(
      'aria-pressed',
      'true',
    );
  });

  it('persists the smart transcript filter and restores it on remount', async () => {
    const first = render(<Harness />);
    await openAdvanced();
    await userEvent.click(screen.getByLabelText(/smart transcript filter/));
    expect(localStorage.getItem('tlab.filtertranscripts')).toBe('1');
    first.unmount();

    render(<Harness />);
    await openAdvanced();
    expect(screen.getByLabelText(/smart transcript filter/)).toBeChecked();
  });

  it('keeps persisting auto-judge under its existing key', async () => {
    render(<Harness />);
    await openAdvanced();
    await userEvent.click(screen.getByLabelText(/auto-judge/));
    expect(localStorage.getItem('tlab.autojudge')).toBe('0');
    expect(readAskPrefs().autoJudge).toBe(false);
  });

  it('submits the persisted preferences once they are toggled', async () => {
    const onAsk = vi.fn();
    render(<Harness onAsk={onAsk} />);
    await openAdvanced();
    await userEvent.click(screen.getByRole('button', { name: /hybrid/ }));
    await userEvent.click(screen.getByLabelText(/smart transcript filter/));
    await userEvent.type(screen.getByLabelText('Question'), 'yields?');
    await userEvent.click(screen.getByRole('button', { name: 'Send' }));
    expect(onAsk).toHaveBeenCalledWith(
      expect.objectContaining({ retrievalMode: 'hybrid', filterTranscripts: true }),
    );
  });

  it('reads defaults back out of localStorage', () => {
    localStorage.setItem('tlab.retrievalmode', 'hybrid');
    localStorage.setItem('tlab.filtertranscripts', '1');
    localStorage.setItem('tlab.autojudge', '0');
    expect(readAskPrefs()).toEqual({
      autoJudge: false,
      retrievalMode: 'hybrid',
      filterTranscripts: true,
    });
  });
});

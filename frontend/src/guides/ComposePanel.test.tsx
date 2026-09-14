import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import type { GuideScope } from '../api/types';
import { ComposePanel, estimateMinutes, slugFromTitle } from './ComposePanel';
import { job } from './ResearchMap.test';

const guideScope = vi.fn();
const startGuide = vi.fn();
vi.mock('../api/client', () => ({
  api: {
    guideScope: (topic: string, limit: number) => guideScope(topic, limit),
    startGuide: (payload: unknown) => startGuide(payload),
  },
}));

const SCOPE: GuideScope = {
  topic: 'LLM-as-a-judge',
  probes: ['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8'],
  total_videos: 161,
  candidates: [
    { video_id: 'a', title: 'LLM-as-a-Judge 101', channel_name: 'Arize', chunk_count: 32, probe_hits: 7, probes: ['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7'], title_match: true, score: 9 },
    { video_id: 'b', title: 'Pilots to production', channel_name: 'X', chunk_count: 55, probe_hits: 3, probes: ['p1', 'p2', 'p3'], title_match: false, score: 3 },
    { video_id: 'c', title: 'Barely related', channel_name: 'Y', chunk_count: 20, probe_hits: 1, probes: ['p1'], title_match: false, score: 1 },
  ],
};

beforeEach(() => {
  guideScope.mockReset();
  startGuide.mockReset();
  guideScope.mockResolvedValue(SCOPE);
});

describe('helpers', () => {
  it('slugs and estimates', () => {
    expect(slugFromTitle('LLM-as-a-Judge: a field guide!')).toBe('llm-as-a-judge-a-field-guide');
    expect(estimateMinutes(700)).toBe(15);
    expect(estimateMinutes(0)).toBe(4);
  });
});

describe('ComposePanel', () => {
  it('scopes a topic, preselects strong candidates, and starts with the confirmed set', async () => {
    const onStarted = vi.fn();
    startGuide.mockResolvedValue(job({ id: 'j9', slug: 'llm-as-a-judge' }));
    render(<ComposePanel running={null} sdkProblem={null} onStarted={onStarted} />);
    await userEvent.type(screen.getByLabelText('Topic'), 'LLM-as-a-judge');
    await userEvent.click(screen.getByRole('button', { name: 'Find sources' }));
    expect(await screen.findByText('LLM-as-a-Judge 101')).toBeInTheDocument();
    expect(guideScope).toHaveBeenCalledWith('LLM-as-a-judge', 30);
    // Score ≥ 2 is ticked; the weak candidate is not.
    // Three candidate boxes, then the allow-web checkbox.
    const boxes = (screen.getAllByRole('checkbox') as HTMLInputElement[]).slice(0, 3);
    expect(boxes.map((box) => box.checked)).toEqual([true, true, false]);
    expect(screen.getByText(/2 videos · 87 chunks · 1 extraction pass/)).toBeInTheDocument();
    // Untick the production talk, tick the weak one, allow web.
    await userEvent.click(boxes[1]!);
    await userEvent.click(boxes[2]!);
    await userEvent.click(screen.getByRole('checkbox', { name: /allow web search/ }));
    await userEvent.click(screen.getByRole('button', { name: 'Write guide' }));
    await waitFor(() => expect(onStarted).toHaveBeenCalled());
    expect(startGuide).toHaveBeenCalledWith({
      topic: 'LLM-as-a-judge',
      title: 'LLM-As-A-Judge',
      slug: 'llm-as-a-judge',
      video_ids: ['a', 'c'],
      allow_web: true,
    });
    // The form resets for the next guide.
    expect((screen.getByLabelText('Topic') as HTMLInputElement).value).toBe('');
  });

  it('adds a video by id', async () => {
    render(<ComposePanel running={null} sdkProblem={null} onStarted={vi.fn()} />);
    await userEvent.type(screen.getByLabelText('Topic'), 'x');
    await userEvent.click(screen.getByRole('button', { name: 'Find sources' }));
    await screen.findByText('LLM-as-a-Judge 101');
    await userEvent.type(screen.getByLabelText('Add a video id'), 'zzz9');
    await userEvent.click(screen.getByRole('button', { name: 'add' }));
    expect(screen.getByText('zzz9')).toBeInTheDocument();
    expect(screen.getByText(/3 videos/)).toBeInTheDocument();
  });

  it('surfaces the server refusal instead of a dead button', async () => {
    startGuide.mockRejectedValue(new Error('/api/guides → HTTP 503: No CLAUDE_CODE_OAUTH_TOKEN'));
    render(<ComposePanel running={null} sdkProblem={null} onStarted={vi.fn()} />);
    await userEvent.type(screen.getByLabelText('Topic'), 'x');
    await userEvent.click(screen.getByRole('button', { name: 'Find sources' }));
    await screen.findByText('LLM-as-a-Judge 101');
    await userEvent.click(screen.getByRole('button', { name: 'Write guide' }));
    expect(await screen.findByText(/No CLAUDE_CODE_OAUTH_TOKEN/)).toBeInTheDocument();
  });

  it('blocks while a job runs and when the SDK is missing', async () => {
    render(<ComposePanel running={job()} sdkProblem="claude-agent-sdk is not installed" onStarted={vi.fn()} />);
    expect(screen.getByText('claude-agent-sdk is not installed')).toBeInTheDocument();
    expect(screen.getByText(/A guide job is running/)).toBeInTheDocument();
  });
});

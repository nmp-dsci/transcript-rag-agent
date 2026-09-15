import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import type { GuideComment, GuideDetail } from '../api/types';
import { CommentRail, commentCounts } from './CommentRail';
import { EvidenceMap, sectionEvidence } from './EvidenceMap';
import { job } from './ResearchMap.test';

const addGuideComment = vi.fn();
const reviseGuide = vi.fn();
vi.mock('../api/client', () => ({
  api: {
    addGuideComment: (slug: string, payload: unknown) => addGuideComment(slug, payload),
    reviseGuide: (slug: string, payload: unknown) => reviseGuide(slug, payload),
  },
}));

function comment(overrides: Partial<GuideComment> = {}): GuideComment {
  return {
    id: 'c-1',
    created_at: '2026-09-14T11:00:00+00:00',
    anchor: 'thesis-p2',
    section_id: 'thesis',
    quote: 'iteration budgets',
    body: 'Cite the numbers per creator',
    status: 'open',
    resolved_in_version: null,
    reason: null,
    ...overrides,
  };
}

function guide(overrides: Partial<GuideDetail> = {}): GuideDetail {
  return {
    slug: 'ship-like-a-studio',
    title: 'Ship Like a Studio',
    topic: 't',
    subtitle: '',
    status: 'published',
    current_version: 2,
    compiled_at: '2026-08-31',
    model: {},
    video_ids: ['v1', 'v2'],
    sources: [
      { video_id: 'v1', title: 'Video One', contributed: '', url: '' },
      { video_id: 'v2', title: 'Video Two', contributed: '', url: '' },
    ],
    chunk_count: 10,
    cluster_count: 1,
    cite_total: 3,
    cite_valid: 2,
    gaps: ['cost per judged answer'],
    web_allowed: false,
    web_urls: [],
    provenance: {},
    videos: 2,
    versions: [1, 2],
    comments_open: 1,
    comments_total: 2,
    html_url: '/guides/ship-like-a-studio/guide.html',
    markdown_url: null,
    comments: [comment(), comment({ id: 'c-2', status: 'addressed', resolved_in_version: 2, quote: '', anchor: null })],
    claims: {
      total: 3,
      valid: 2,
      pass_rate: 0.67,
      sections: ['thesis', 'pipeline', 'sources'],
      structure_errors: [],
      claims: [
        { cite_index: 0, section_id: 'thesis', video_id: 'v1', chunk_index: 1, chunk_id: 'chunk:v1:1', quote: null, text: 'One', video_ok: true, chunk_ok: true, quote_ok: null, valid: true },
        { cite_index: 1, section_id: 'thesis', video_id: 'v2', chunk_index: 4, chunk_id: 'chunk:v2:4', quote: null, text: 'Two', video_ok: true, chunk_ok: true, quote_ok: null, valid: true },
        { cite_index: 2, section_id: 'pipeline', video_id: 'v2', chunk_index: 9, chunk_id: 'chunk:v2:9', quote: null, text: 'Two', video_ok: true, chunk_ok: false, quote_ok: null, valid: false },
      ],
    },
    version_urls: { '1': '/guides/ship-like-a-studio/versions/v1.html', '2': '/guides/ship-like-a-studio/versions/v2.html' },
    receipts: {
      v2: { version: 2, created_at: 't', summary: 'One edit.', changed_sections: ['thesis-p2'], items: [{ id: 'c-2', outcome: 'addressed', reason: 'added figures', sections: ['thesis-p2'] }] },
    },
    ...overrides,
  };
}

beforeEach(() => {
  addGuideComment.mockReset();
  reviseGuide.mockReset();
});

describe('CommentRail', () => {
  it('counts statuses', () => {
    expect(commentCounts(guide().comments)).toEqual({ open: 1, addressed: 1, deferred: 0, rejected: 0 });
  });

  it('lists comments with status words and jumps on click', async () => {
    const onJump = vi.fn();
    render(<CommentRail guide={guide()} selection={null} running={null} sdkProblem={null} demo={false} onAdded={vi.fn()} onRevisionStarted={vi.fn()} onJump={onJump} />);
    expect(screen.getByText('1 open · 1 addressed')).toBeInTheDocument();
    expect(screen.getByText('open')).toHaveClass('warn');
    expect(screen.getByText('addressed in v2')).toHaveClass('good');
    await userEvent.click(screen.getAllByText('Cite the numbers per creator')[0]!);
    expect(onJump).toHaveBeenCalledWith('thesis-p2');
  });

  it('turns a page selection into an anchored comment', async () => {
    const onAdded = vi.fn();
    addGuideComment.mockResolvedValue(comment({ id: 'c-3', body: 'Add pairwise', anchor: 'pipeline-p4', quote: 'pairwise beats' }));
    render(
      <CommentRail
        guide={guide()}
        selection={{ quote: 'pairwise beats', anchor: 'pipeline-p4', sectionId: 'pipeline', rect: null }}
        running={null}
        sdkProblem={null}
        demo={false}
        onAdded={onAdded}
        onRevisionStarted={vi.fn()}
        onJump={vi.fn()}
      />,
    );
    expect(screen.getByText('“pairwise beats”')).toBeInTheDocument();
    expect(screen.getByPlaceholderText('comment on pipeline-p4')).toBeInTheDocument();
    await userEvent.type(screen.getByLabelText('Comment'), 'Add pairwise');
    await userEvent.click(screen.getByRole('button', { name: 'Add comment' }));
    await waitFor(() => expect(onAdded).toHaveBeenCalled());
    expect(addGuideComment).toHaveBeenCalledWith('ship-like-a-studio', {
      body: 'Add pairwise',
      anchor: 'pipeline-p4',
      section_id: 'pipeline',
      quote: 'pairwise beats',
    });
    expect((screen.getByLabelText('Comment') as HTMLTextAreaElement).value).toBe('');
  });

  it('sends the open comments to the agent as one batch', async () => {
    const onRevisionStarted = vi.fn();
    reviseGuide.mockResolvedValue(job({ kind: 'revise', slug: 'ship-like-a-studio' }));
    render(<CommentRail guide={guide()} selection={null} running={null} sdkProblem={null} demo={false} onAdded={vi.fn()} onRevisionStarted={onRevisionStarted} onJump={vi.fn()} />);
    await userEvent.click(screen.getByRole('button', { name: 'Send 1 to the agent' }));
    await waitFor(() => expect(onRevisionStarted).toHaveBeenCalled());
    expect(reviseGuide).toHaveBeenCalledWith('ship-like-a-studio', {});
  });

  it('disables sending while a job runs, when the SDK is absent, and hides composing in demo', () => {
    const { rerender } = render(<CommentRail guide={guide()} selection={null} running={job()} sdkProblem={null} demo={false} onAdded={vi.fn()} onRevisionStarted={vi.fn()} onJump={vi.fn()} />);
    expect(screen.getByRole('button', { name: 'Send 1 to the agent' })).toBeDisabled();
    rerender(<CommentRail guide={guide()} selection={null} running={null} sdkProblem="no token" demo={false} onAdded={vi.fn()} onRevisionStarted={vi.fn()} onJump={vi.fn()} />);
    expect(screen.getByRole('button', { name: 'Send 1 to the agent' })).toBeDisabled();
    expect(screen.getByText('no token')).toBeInTheDocument();
    rerender(<CommentRail guide={guide()} selection={null} running={null} sdkProblem={null} demo={true} onAdded={vi.fn()} onRevisionStarted={vi.fn()} onJump={vi.fn()} />);
    expect(screen.queryByLabelText('Comment')).not.toBeInTheDocument();
  });
});

describe('EvidenceMap', () => {
  it('derives per-section evidence from claims.json', () => {
    const rows = sectionEvidence(guide().claims);
    expect(rows.map((r) => [r.id, r.claims, r.valid, r.videos.length])).toEqual([
      ['thesis', 2, 2, 2],
      ['pipeline', 1, 0, 1],
      ['sources', 0, 0, 0],
    ]);
    expect(sectionEvidence(null)).toEqual([]);
  });

  it('renders the consolidated view and jumps to a section', async () => {
    const onJump = vi.fn();
    render(<EvidenceMap guide={guide()} onJump={onJump} />);
    expect(screen.getByText('claims cited').previousSibling).toHaveTextContent('3');
    expect(screen.getByText('cites resolve').previousSibling).toHaveTextContent('67%');
    expect(screen.getByText('2 claims · 2 videos')).toBeInTheDocument();
    expect(screen.getByText('· 1 unresolved')).toBeInTheDocument();
    expect(screen.getAllByTitle('Video Two')).toHaveLength(2);
    await userEvent.click(screen.getByText('pipeline'));
    expect(onJump).toHaveBeenCalledWith('pipeline');
    expect(screen.getByText(/What the corpus does not cover \(1\)/)).toBeInTheDocument();
  });
});

import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import type { CritiqueRunDetail, CritiqueRunSummary } from '../api/types';
import { CritiquePanel } from './CritiquePanel';

const critiqueRun = vi.fn();

vi.mock('../api/client', () => ({
  api: { critiqueRun: (...args: unknown[]) => critiqueRun(...args) },
}));

function cell(): CritiqueRunSummary['cells'][number] {
  return {
    setup: 'rag_llm_filtered',
    scores: {
      criteria_recall: 0.278,
      evidence_precision: 1,
      provenance: 1,
      contested_rate: 0,
    },
    score_spread: {
      criteria_recall_min: 0.167,
      criteria_recall_median: 0.278,
      criteria_recall_max: 0.333,
    },
    match_repeats: 5,
    criteria_recall_all: 0.208,
    criteria_recall_grouped: 0.333,
    criteria_groups: 15,
    criteria_applicable: 18,
    criteria_matched: 5,
    criteria_matched_ungrounded: 1,
    findings_total: 9,
    findings_grounded: 9,
    findings_sharing_evidence: 0,
    citations_total: 10,
    citations_resolved: 10,
    contested_findings: 0,
    held_out_leaks: 0,
    elapsed_seconds: 61,
    token_estimate: 4000,
    error: null,
    retrieved_video_ids: ['vidA'],
  };
}

function summary(overrides: Partial<CritiqueRunSummary> = {}): CritiqueRunSummary {
  return {
    run_id: 'critique-HELD-20260810-120000',
    created_at: '2026-08-10T12:00:00+00:00',
    held_out_video_id: 'HELD',
    held_out_title: 'You asked me to roast your AI resumes.. so I did.',
    artifact_url: 'https://nmp-dsci.github.io/',
    artifact_kind: 'portfolio',
    criteria_total: 24,
    criteria_applicable: 18,
    criteria_groups: 15,
    match_repeats: 5,
    metrics: ['criteria_recall', 'evidence_precision', 'provenance', 'contested_rate'],
    baseline: 'rag_llm_filtered',
    held_out_leaks: 0,
    exclusion_version: 'v1',
    config: {},
    cells: [cell()],
    ...overrides,
  };
}

function detail(): CritiqueRunDetail {
  return {
    ...summary(),
    cells: [
      {
        ...cell(),
        matches: [
          {
            id: 'c05',
            criterion: 'Put explicit numbers on claims of scale or reliability.',
            quote: 'be a little bit more explicit about how many releases there actually were',
            video_id: 'HELD',
            start_seconds: 179.24,
            applies_to: ['resume', 'portfolio'],
            note: null,
            group: 'c05',
            matched: true,
            counted: true,
            ungrounded: false,
            agreement: 1,
            finding_id: 'f01',
            finding_criterion: 'Quantify achievements with metrics and data to show impact.',
            score: 1,
            why: 'both require numbers on the claim',
          },
          {
            id: 'c01',
            criterion: 'A resume should fit on a single page.',
            quote: 'you want to keep your resume to just a single page',
            video_id: 'HELD',
            start_seconds: 57,
            applies_to: ['resume'],
            note: null,
            group: 'c01',
            matched: false,
            counted: false,
            ungrounded: false,
            agreement: 1,
            finding_id: null,
            finding_criterion: null,
            score: 0,
            why: null,
          },
        ],
        findings: [
          {
            id: 'f01',
            criterion: 'Quantify achievements with metrics and data to show impact.',
            detail: 'the hero numbers do this well',
            contested: false,
            grounded: true,
            exclusive_chunk_ids: ['chunk:vidA:1'],
            citation_checks: [
              {
                video_id: 'vidA',
                start_seconds: 12,
                quote: 'numbers',
                resolved: true,
                reason: 'ok',
                ratio: 1,
                chunk_id: 'chunk:vidA:1',
                shared: false,
              },
            ],
          },
          {
            id: 'f02',
            criterion: 'Include your location so recruiters can screen for geography.',
            detail: 'no location anywhere',
            contested: false,
            grounded: true,
            exclusive_chunk_ids: ['chunk:vidA:3'],
            citation_checks: [
              {
                video_id: 'vidA',
                start_seconds: 40,
                quote: 'location',
                resolved: true,
                reason: 'ok',
                ratio: 1,
                chunk_id: 'chunk:vidA:3',
                shared: false,
              },
            ],
          },
        ],
      },
    ],
  };
}

describe('CritiquePanel', () => {
  beforeEach(() => critiqueRun.mockReset());

  it('renders the baseline row with a score for every metric', () => {
    render(<CritiquePanel run={summary()} />);

    const row = screen.getByText('rag_llm_filtered').closest('tr');
    expect(row).not.toBeNull();
    expect(row?.textContent).toContain('0.278');
    expect(row?.textContent).toContain('1.000');
    // The "base" chip is what marks this as the row later slices must beat.
    expect(row?.textContent).toContain('base');
  });

  it('states that the held-out video was absent', () => {
    render(<CritiquePanel run={summary()} />);

    expect(screen.getByText(/held-out absent · 0 leaks/)).toBeTruthy();
  });

  it('calls out a leak rather than showing a clean badge', () => {
    render(<CritiquePanel run={summary({ held_out_leaks: 3 })} />);

    expect(screen.getByText('3 held-out leaks')).toBeTruthy();
  });

  it('fetches the criteria only when a row is expanded', async () => {
    critiqueRun.mockResolvedValue(detail());
    render(<CritiquePanel run={summary()} />);

    expect(critiqueRun).not.toHaveBeenCalled();

    await userEvent.click(screen.getByRole('button', { name: 'Expand' }));

    await waitFor(() => expect(critiqueRun).toHaveBeenCalledWith('critique-HELD-20260810-120000'));
  });

  it('shows matched and unmatched criteria, each with its timestamp', async () => {
    critiqueRun.mockResolvedValue(detail());
    render(<CritiquePanel run={summary()} />);

    await userEvent.click(screen.getByRole('button', { name: 'Expand' }));

    expect(await screen.findByText(/reached — 1 of 1 criteria that apply to a portfolio/)).toBeTruthy();
    expect(screen.getByText(/missed — 1 criteria, 1 of which a portfolio cannot be judged on/)).toBeTruthy();
    // A criterion is only checkable if you can watch the expert say it.
    const stamp = screen.getByText('2:59');
    expect(stamp.getAttribute('href')).toBe('https://www.youtube.com/watch?v=HELD&t=179s');
    expect(screen.getByText('0:57').getAttribute('href')).toBe(
      'https://www.youtube.com/watch?v=HELD&t=57s',
    );
  });

  it('marks a criterion that cannot apply to this kind of artifact', async () => {
    critiqueRun.mockResolvedValue(detail());
    render(<CritiquePanel run={summary()} />);

    await userEvent.click(screen.getByRole('button', { name: 'Expand' }));

    // Recall is measured over the applicable subset, so a resume-only rule in
    // the missed column must say why it could never have been reached.
    expect(await screen.findByText('n/a to a portfolio')).toBeTruthy();
  });

  it('separates findings the held-out expert never made', async () => {
    critiqueRun.mockResolvedValue(detail());
    render(<CritiquePanel run={summary()} />);

    await userEvent.click(screen.getByRole('button', { name: 'Expand' }));

    expect(await screen.findByText(/1 findings this expert did not make/)).toBeTruthy();
    expect(screen.getByText(/Include your location/)).toBeTruthy();
  });

  it('renders the spread beside recall so a lift can be judged against noise', () => {
    render(<CritiquePanel run={summary()} />);

    // 0.278 that came out of 0.167-0.333 is not a measurement a later slice can
    // claim to have beaten by 0.05.
    expect(screen.getByText('0.167–0.222'.replace('0.222', '0.333'))).toBeTruthy();
  });

  it('renders both recall denominators, not just the flattering one', () => {
    render(<CritiquePanel run={summary()} />);

    const row = screen.getByText('rag_llm_filtered').closest('tr');
    // criteria_recall drops the criteria a portfolio cannot be judged on, and
    // every one of those exclusions removed a criterion the baseline missed.
    expect(row?.textContent).toContain('0.208');
    expect(row?.textContent).toContain('0.333');
  });

  it('does not credit a criterion whose finding rests on no evidence of its own', async () => {
    const withUngrounded = detail();
    const match = withUngrounded.cells[0]!.matches[0]!;
    match.counted = false;
    match.ungrounded = true;
    critiqueRun.mockResolvedValue(withUngrounded);
    render(<CritiquePanel run={summary()} />);

    await userEvent.click(screen.getByRole('button', { name: 'Expand' }));

    expect(await screen.findByText(/reached — 0 of 1 criteria/)).toBeTruthy();
    expect(screen.getByText(/rests on\s+no evidence another finding does not also claim/)).toBeTruthy();
  });

  it('flags a pairing the matcher runs did not agree on', async () => {
    const shaky = detail();
    shaky.cells[0]!.matches[0]!.agreement = 0.6;
    critiqueRun.mockResolvedValue(shaky);
    render(<CritiquePanel run={summary()} />);

    await userEvent.click(screen.getByRole('button', { name: 'Expand' }));

    expect(await screen.findByText('60% of matcher runs')).toBeTruthy();
  });

  it('collapses without refetching', async () => {
    critiqueRun.mockResolvedValue(detail());
    render(<CritiquePanel run={summary()} />);

    await userEvent.click(screen.getByRole('button', { name: 'Expand' }));
    await screen.findByText(/reached — 1 of 1 criteria that apply to a portfolio/);
    await userEvent.click(screen.getByRole('button', { name: 'Collapse' }));

    expect(screen.queryByText(/reached — 1 of 1 criteria that apply to a portfolio/)).toBeNull();
    expect(critiqueRun).toHaveBeenCalledTimes(1);
  });
});

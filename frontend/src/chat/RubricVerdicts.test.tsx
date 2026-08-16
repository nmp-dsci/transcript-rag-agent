import { fireEvent, render, screen, within } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import type { RubricReview, RubricVerdict } from '../api/types';
import { RubricVerdicts } from './RubricVerdicts';

function verdict(overrides: Partial<RubricVerdict> = {}): RubricVerdict {
  return {
    rubric_id: 'r0302',
    topic: 'resume-design',
    pack_name: 'Resume design',
    criterion: 'Quantify your contributions with facts and figures instead of listing duties.',
    check: 'If a bullet has no number, fail.',
    why: 'Recruiters skim for numbers.',
    unit_title: 'Quantified impact',
    creators: ['A Recruiter'],
    verdict: 'fail',
    severity: 'major',
    finding: 'The Data Pilot card in §4 contains no number.',
    sections: [3],
    evidence: [
      {
        video_id: 'QPUmFKboiqY',
        chunk_id: 'chunk:QPUmFKboiqY:0',
        quote: 'facts and figures',
        channel_name: 'A Recruiter',
        title: 'Resume secrets',
        start_seconds: 125,
        url: 'https://www.youtube.com/watch?v=QPUmFKboiqY&t=58s',
      },
    ],
    note: '',
    ...overrides,
  };
}

function review(verdicts: RubricVerdict[]): RubricReview {
  const counts: Record<string, number> = { pass: 0, fail: 0, 'n-a': 0, unjudged: 0 };
  const severities: Record<string, number> = { blocker: 0, major: 0, minor: 0, none: 0 };
  for (const row of verdicts) {
    counts[row.verdict] = (counts[row.verdict] ?? 0) + 1;
    if (row.verdict === 'fail') severities[row.severity] = (severities[row.severity] ?? 0) + 1;
  }
  return {
    document_id: 'doc:f52b88d70b35b908',
    document_url: 'https://nmp-dsci.github.io/',
    document_kind: 'portfolio',
    verdicts,
    packs: [
      {
        topic: 'resume-design',
        name: 'Resume design',
        artifact: 'resume',
        rubrics: verdicts.length,
        elapsed_seconds: 90,
        error: null,
        unknown_rubric_ids: [],
        duplicate_rubric_ids: [],
        missing_rubric_ids: [],
        unanchored_failures: [],
      },
    ],
    stats: {
      rubrics_total: verdicts.length,
      verdicts: counts,
      severities,
      packs_used: 1,
      packs_failed: 0,
      with_rubric_id: verdicts.length,
      with_timestamp: verdicts.filter((row) => row.evidence.length > 0).length,
      with_id_and_timestamp: verdicts.filter((row) => row.evidence.length > 0).length,
      id_and_timestamp_share: 1,
      evidence_links: verdicts.reduce((total, row) => total + row.evidence.length, 0),
      sections_cited: [3],
    },
  };
}

describe('RubricVerdicts', () => {
  it('shows the rubric id and the timestamp on the collapsed row', () => {
    // The two halves of the claim this setup makes. Behind an expander they are
    // a claim most readers never check.
    render(<RubricVerdicts review={review([verdict()])} />);

    expect(screen.getByRole('button', { name: 'r0302' })).toBeInTheDocument();
    expect(screen.getByRole('link', { name: '2:05' })).toHaveAttribute(
      'href',
      'https://www.youtube.com/watch?v=QPUmFKboiqY&t=58s',
    );
  });

  it('states how many verdicts carry both, rather than leaving it to be counted', () => {
    render(<RubricVerdicts review={review([verdict()])} />);

    expect(screen.getByText(/1\/1 carry rubric id \+ timestamp \(100%\)/)).toBeInTheDocument();
  });

  it('opens the rubric behind a verdict — its check, its rationale, its creators', () => {
    render(<RubricVerdicts review={review([verdict()])} />);

    expect(screen.queryByText('If a bullet has no number, fail.')).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: 'r0302' }));

    expect(screen.getByText('If a bullet has no number, fail.')).toBeInTheDocument();
    expect(screen.getByText('Recruiters skim for numbers.')).toBeInTheDocument();
    expect(screen.getByText('“facts and figures”')).toHaveClass('rvquote');
  });

  it('clicks a finding through to the document section it lands on', () => {
    const onOpenSection = vi.fn();
    render(<RubricVerdicts review={review([verdict()])} onOpenSection={onOpenSection} />);

    fireEvent.click(screen.getByRole('button', { name: '§4' }));

    // Zero-based on the wire, §N+1 on screen — the card indexes the same way.
    expect(onOpenSection).toHaveBeenCalledWith(3);
  });

  it('shows failures first and hides passes until they are asked for', () => {
    const rows = [
      verdict(),
      verdict({ rubric_id: 'r0102', verdict: 'pass', severity: 'none', sections: [] }),
      verdict({ rubric_id: 'r0203', verdict: 'n-a', severity: 'none', sections: [] }),
    ];
    render(<RubricVerdicts review={review(rows)} />);

    expect(screen.getByRole('button', { name: 'r0302' })).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'r0102' })).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: /^n\/a 1$/ }));

    expect(screen.getByRole('button', { name: 'r0203' })).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'r0302' })).not.toBeInTheDocument();
  });

  it('filters failures by severity', () => {
    const rows = [
      verdict(),
      verdict({ rubric_id: 'r0401', severity: 'minor' }),
    ];
    render(<RubricVerdicts review={review(rows)} />);

    expect(screen.getByRole('button', { name: 'r0302' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'r0401' })).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: 'minor 1' }));

    expect(screen.queryByRole('button', { name: 'r0302' })).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'r0401' })).toBeInTheDocument();
  });

  it('renders the whole criterion rather than clipping it', () => {
    // jsdom has no layout, so this pins the markup rather than the pixels: the
    // criterion is its own wrapping block, never a nowrap metadata cell.
    const long = 'A'.repeat(273);
    render(<RubricVerdicts review={review([verdict({ criterion: long })])} />);

    const node = screen.getByText(long);
    expect(node).toHaveClass('rvcrit');
    expect(node.textContent).toHaveLength(273);
  });

  it('says a rubric was not decided instead of dropping it', () => {
    const rows = [
      verdict({
        rubric_id: 'r0505',
        verdict: 'unjudged',
        severity: 'none',
        sections: [],
        finding: '',
        note: 'failed, but named no section of the document — not counted as a finding',
      }),
    ];
    render(<RubricVerdicts review={review(rows)} />);

    fireEvent.click(screen.getByRole('button', { name: /^not decided 1$/ }));

    const row = screen.getByRole('button', { name: 'r0505' }).closest('li')!;
    expect(within(row).getByText(/named no section of the document/)).toBeInTheDocument();
  });

  it('reports a pack whose call failed rather than showing a shorter review', () => {
    const data = review([verdict()]);
    data.packs[0]!.error = 'ReadTimeout: no reply in 120s';
    render(<RubricVerdicts review={data} />);

    expect(screen.getByText(/could not be applied/)).toBeInTheDocument();
    expect(screen.getByText(/ReadTimeout/)).toBeInTheDocument();
  });
});

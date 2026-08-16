import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import type { CritiqueBallot, CritiqueRunDetail, CritiqueRunSummary } from '../api/types';
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
      contested_coverage: 0.5,
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
    conflicts_in_context: 2,
    conflicts_named: 1,
    self_declared_contested: 0,
    fabricated_citations: [],
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
    metrics: ['criteria_recall', 'evidence_precision', 'provenance', 'contested_coverage'],
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
        conflicts: [
          {
            conflict_id: 'conflict:2',
            axis: 'What font size should the body text of a resume be?',
            video_ids: ['vidA', 'vidB'],
            chunk_ids: ['chunk:vidA:1', 'chunk:vidB:4'],
            named_by: ['f01'],
          },
          {
            conflict_id: 'conflict:0',
            axis: 'Do coding agents need a human in the loop?',
            video_ids: ['vidC', 'vidD'],
            chunk_ids: ['chunk:vidC:9', 'chunk:vidD:0'],
            named_by: [],
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

  it('shows the contested column with its denominator, not a bare rate', () => {
    render(<CritiquePanel run={summary()} />);

    const row = screen.getByText('rag_llm_filtered').closest('tr');
    // "0.500" alone cannot be told apart from a run that was never shown a
    // disagreement, which is exactly how the metric this replaced read 0.000
    // forever. The fraction is what makes it a measurement.
    expect(row?.textContent).toContain('0.500');
    expect(row?.textContent).toContain('1/2 in context');
  });

  it('names the axis of every disagreement in context, and takes no side', async () => {
    critiqueRun.mockResolvedValue(detail());
    render(<CritiquePanel run={summary()} />);

    await userEvent.click(screen.getByRole('button', { name: 'Expand' }));
    await waitFor(() =>
      expect(
        screen.getByText('What font size should the body text of a resume be?'),
      ).toBeTruthy(),
    );
    // The one it averaged away is shown too — a conflict the system did not
    // report is the finding, not an omission.
    expect(
      screen.getByText(/averaged away — no finding cited both sides/),
    ).toBeTruthy();
    expect(screen.getByText(/named by f01/)).toBeTruthy();
  });

  it('publishes the recall gap against the baseline for every other arm', () => {
    // V6's gate: the rubric-driven reviewer's recall has to be readable
    // *against* the chunk dump's without the reader doing the subtraction.
    const rubric = {
      ...cell(),
      setup: 'rubric_packs',
      scores: { ...cell().scores, criteria_recall: 0.111 },
      score_spread: {
        criteria_recall_min: 0.056,
        criteria_recall_median: 0.111,
        criteria_recall_max: 0.111,
      },
    };
    render(<CritiquePanel run={summary({ cells: [cell(), rubric] })} />);

    const row = screen.getByText('rubric_packs').closest('tr');
    // 0.111 - 0.278, printed as a loss rather than quietly omitted.
    expect(row?.textContent).toContain('−0.167 vs rag_llm_filtered');
    // 0.056–0.111 does not overlap 0.167–0.333, so the gap is real.
    expect(row?.textContent).not.toContain('within noise');
    // The baseline row itself has nothing to compare against.
    expect(screen.getByText('rag_llm_filtered').closest('tr')?.textContent).not.toContain(
      'vs rag_llm_filtered',
    );
  });

  it('calls a gap smaller than the matcher’s own disagreement what it is', () => {
    const rubric = {
      ...cell(),
      setup: 'rubric_packs',
      scores: { ...cell().scores, criteria_recall: 0.333 },
      score_spread: {
        criteria_recall_min: 0.222,
        criteria_recall_median: 0.333,
        criteria_recall_max: 0.389,
      },
    };
    render(<CritiquePanel run={summary({ cells: [cell(), rubric] })} />);

    const row = screen.getByText('rubric_packs').closest('tr');
    expect(row?.textContent).toContain('+0.055 vs rag_llm_filtered');
    // 0.222–0.389 overlaps the baseline's 0.167–0.333: nothing was shown.
    expect(row?.textContent).toContain('within noise');
  });

  // ── the matcher's own ballots ──────────────────────────────────────────
  //
  // The committed rubric_packs arm scores criteria_recall 0.000, and two of its
  // five matcher repeats *did* pair a held-out criterion with a finding — the
  // vote discarded both. A reader who reads the zero as "nothing was reached"
  // has been misled, so these pin the ballots down at every level they render.

  /** The baseline's c05: paired unanimously, five repeats out of five. */
  function baseBallot(): CritiqueBallot {
    return {
      criterion_id: 'c05',
      criterion: 'Put explicit numbers on claims of scale or reliability.',
      applies_to: ['resume', 'portfolio'],
      draws: ['f01', 'f01', 'f01', 'f01', 'f01'],
      votes: [
        {
          finding_id: 'f01',
          finding_criterion: 'Quantify accomplishments with metrics and data.',
          count: 5,
        },
      ],
      consensus_finding_id: 'f01',
      consensus_finding_criterion: 'Quantify accomplishments with metrics and data.',
      agreement: 1,
    };
  }

  /** The rubric arm's c05: paired on two repeats, outvoted 3–2 by "no match". */
  function lostBallot(): CritiqueBallot {
    return {
      criterion_id: 'c05',
      criterion: 'Put explicit numbers on claims of scale or reliability.',
      applies_to: ['resume', 'portfolio'],
      draws: [null, null, 'job-search:r0601', 'job-search:r0601', null],
      votes: [
        { finding_id: null, finding_criterion: null, count: 3 },
        {
          finding_id: 'job-search:r0601',
          finding_criterion: 'Quantify every claimed result with a numeric metric.',
          count: 2,
        },
      ],
      consensus_finding_id: null,
      consensus_finding_criterion: null,
      agreement: 0.6,
    };
  }

  function rubricRun(): CritiqueRunSummary {
    return summary({
      cells: [
        { ...cell(), match_ballots: [baseBallot()] },
        {
          ...cell(),
          setup: 'rubric_packs',
          scores: { ...cell().scores, criteria_recall: 0 },
          criteria_matched: 0,
          match_ballots: [lostBallot()],
        },
      ],
    });
  }

  /** The rubric arm's detail: c05 missed, and r0601 among the extra findings. */
  function rubricDetail(): CritiqueRunDetail {
    const base = detail();
    const rubric = {
      ...base.cells[0]!,
      setup: 'rubric_packs',
      matches: [
        {
          ...base.cells[0]!.matches[0]!,
          matched: false,
          counted: false,
          agreement: 0.6,
          finding_id: null,
          finding_criterion: null,
          score: 0,
          why: null,
        },
      ],
      findings: [
        {
          id: 'job-search:r0601',
          criterion: 'Quantify every claimed result with a numeric metric.',
          detail: 'no numbers anywhere',
          contested: false,
          grounded: true,
          exclusive_chunk_ids: ['chunk:vidA:1'],
          citation_checks: [],
        },
        ...base.cells[0]!.findings.slice(1),
      ],
    };
    return { ...base, cells: [base.cells[0]!, rubric] };
  }

  it('says a pairing was outvoted on the row itself, before any click', () => {
    render(<CritiquePanel run={rubricRun()} />);

    const row = screen.getByText('rubric_packs').closest('tr');
    expect(row?.textContent).toContain('0.000');
    // "0.000 because the vote threw a pairing away" and "0.000 because nothing
    // was reached" print the same digits. The reader who walks away after
    // reading the number is the one this line exists for.
    expect(row?.textContent).toContain('1 pairing outvoted');
    // The baseline kept its only pairing, so it claims nothing here.
    expect(screen.getByText('rag_llm_filtered').closest('tr')?.textContent).not.toContain(
      'outvoted',
    );
  });

  it('shows which repeats paired what, and what the vote did with it', async () => {
    critiqueRun.mockResolvedValue(rubricDetail());
    render(<CritiquePanel run={rubricRun()} />);

    await userEvent.click(screen.getAllByRole('button', { name: 'Expand' })[1]!);

    expect(
      await screen.findByText(
        /matcher ballots — 1 of 24 criteria were paired with a finding on at least one of 5 matcher repeats/,
      ),
    ).toBeTruthy();
    // The zero named as the vote's doing rather than left to be misread.
    expect(
      screen.getByText(/the majority vote kept none of them/),
    ).toBeTruthy();
    // Every repeat, in order — the matcher disagreeing with itself is the point.
    const chips = document.querySelectorAll('.crit-draw');
    expect([...chips].map((c) => c.textContent)).toEqual([
      '1no match',
      '2no match',
      '3job-search:r0601',
      '4job-search:r0601',
      '5no match',
    ]);
    const ballots = document.querySelector('.crit-ballots');
    expect(ballots?.textContent).toContain('“no match” took 3 of 5 repeats');
    // The reading that lost is named, not just counted.
    expect(ballots?.textContent).toContain('job-search:r0601');
    expect(ballots?.textContent).toContain(
      'Quantify every claimed result with a numeric metric.',
    );
  });

  it('puts the baseline’s verdict on the same criterion beside the loss', async () => {
    critiqueRun.mockResolvedValue(rubricDetail());
    render(<CritiquePanel run={rubricRun()} />);

    await userEvent.click(screen.getAllByRole('button', { name: 'Expand' })[1]!);

    // The illuminating contrast: the same criterion, the same matcher, five
    // repeats each — one arm's phrasing was read as the expert's rule every
    // time and the other's was read as a different rule three times in five.
    await waitFor(() => expect(document.querySelector('.crit-ballots')).toBeTruthy());
    const text = document.querySelector('.crit-ballots')?.textContent ?? '';
    expect(text).toContain('the baseline rag_llm_filtered paired the same criterion with');
    expect(text).toContain('f01 on 5 of 5 repeats');
    expect(text).toContain('Quantify accomplishments with metrics and data.');
  });

  it('marks a missed criterion that was outvoted rather than never raised', async () => {
    critiqueRun.mockResolvedValue(rubricDetail());
    render(<CritiquePanel run={rubricRun()} />);

    await userEvent.click(screen.getAllByRole('button', { name: 'Expand' })[1]!);

    expect(await screen.findByText('outvoted 3–2')).toBeTruthy();
  });

  it('does not call an outvoted finding “outside the held-out list”', async () => {
    critiqueRun.mockResolvedValue(rubricDetail());
    render(<CritiquePanel run={rubricRun()} />);

    await userEvent.click(screen.getAllByRole('button', { name: 'Expand' })[1]!);

    // The old gloss swept these up as "not wrong, just outside the held-out
    // list". They were inside it on two of five repeats.
    expect(
      await screen.findByText(
        /2 findings this expert did not make — 1 of\s+them were paired with a held-out criterion on a minority of matcher\s+repeats and outvoted; the other 1\s+are outside the held-out list, not wrong/,
      ),
    ).toBeTruthy();
    expect(screen.getByText('paired with c05 on 2 of 5 repeats')).toBeTruthy();
  });

  it('keeps the plain gloss when no finding was outvoted', async () => {
    critiqueRun.mockResolvedValue(detail());
    render(<CritiquePanel run={summary()} />);

    await userEvent.click(screen.getByRole('button', { name: 'Expand' }));

    expect(
      await screen.findByText(/1 findings this expert did not make — not\s+wrong, just outside the held-out list/),
    ).toBeTruthy();
  });

  it('flags an invented citation on the card and shows what was claimed', async () => {
    const bad = {
      finding_id: 'f02',
      video_id: 'vidA',
      start_seconds: 70.1,
      claimed_quote: "skill is not equal to subject, don't write machine learning under skills",
      ratio: 0.412,
      chunk_id: 'chunk:vidA:1',
      reason: 'quote not found in the transcript at this timestamp',
    };
    const run = summary({ fabricated_citations: 1 });
    const full = detail();
    full.cells[0]!.fabricated_citations = [bad];
    critiqueRun.mockResolvedValue(full);
    render(<CritiquePanel run={run} />);

    // A model inventing evidence is the failure this harness exists to catch,
    // so it is on the card rather than inside a cell nobody expands.
    expect(screen.getByText('1 invented citation')).toBeTruthy();

    await userEvent.click(screen.getByRole('button', { name: 'Expand' }));
    await waitFor(() => expect(screen.getByText(/invented citations/)).toBeTruthy());
    expect(screen.getByText(/skill is not equal to subject/)).toBeTruthy();
    expect(screen.getByText(/matched 0.41 against/)).toBeTruthy();
  });

  // ── an arm the grounding gate could not grade ───────────────────────────
  //
  // The committed baseline `rag_llm_filtered` emits every finding from one
  // shared retrieval pool, so the retrieval-provenance gate cannot certify a
  // score for it: criteria_recall and evidence_precision come back null. The
  // failure these guard against is the panel rendering that as a blank — or
  // worse, still rendering the 0.158 the gate withdrew as if it were measured.

  /** The chunk-dump baseline as the gate leaves it: no score, published pair kept. */
  function ungradedCell(): CritiqueRunSummary['cells'][number] {
    return {
      ...cell(),
      scores: {
        criteria_recall: null,
        evidence_precision: null,
        provenance: 1,
        contested_coverage: null,
      },
      score_spread: {
        criteria_recall_min: null,
        criteria_recall_median: null,
        criteria_recall_max: null,
      },
      criteria_recall_all: null,
      criteria_recall_grouped: null,
      gradable: false,
      ungradable_reason:
        '11 of 11 findings record no per-finding retrieval, so this arm cannot be ' +
        'graded under the retrieval-provenance gate',
      grounding_gate: 'retrieval_provenance',
      criteria_recall_ungated: 0.1579,
      evidence_precision_ungated: 0.3636,
      findings_with_provenance: 0,
      provenance_distinct_sets: 0,
      citations_off_retrieval: 0,
      findings_total: 11,
    };
  }

  /** The rubric arm, which does have per-finding provenance and is graded. */
  function gradedCell(): CritiqueRunSummary['cells'][number] {
    return {
      ...cell(),
      setup: 'rubric_packs',
      scores: { ...cell().scores, criteria_recall: 0, evidence_precision: 1 },
      score_spread: {
        criteria_recall_min: 0,
        criteria_recall_median: 0,
        criteria_recall_max: 0.105,
      },
      gradable: true,
      grounding_gate: 'retrieval_provenance',
      criteria_recall_ungated: 0,
      evidence_precision_ungated: 1,
      findings_with_provenance: 17,
      provenance_distinct_sets: 6,
      citations_off_retrieval: 0,
    };
  }

  function gatedRun(): CritiqueRunSummary {
    return summary({
      grounding_gate: 'retrieval_provenance',
      ungraded_cells: ['rag_llm_filtered'],
      cells: [ungradedCell(), gradedCell()],
    });
  }

  /** The table row for a setup — the intro also names it, so text alone is ambiguous. */
  function rowFor(setup: string): HTMLElement | null {
    const td = [...document.querySelectorAll('td.exp-cfg')].find((cell) =>
      cell.textContent?.startsWith(setup),
    );
    return (td?.closest('tr') as HTMLElement) ?? null;
  }

  it('renders an ungraded arm as not measured, never as a number', () => {
    render(<CritiquePanel run={gatedRun()} />);

    const row = rowFor('rag_llm_filtered');
    expect(row?.textContent).toContain('not measured');
    // The published figure is present but labelled, and nowhere does it stand
    // in the position a score would occupy.
    expect(row?.textContent).toContain('ungated 0.158');
    expect(row?.textContent).not.toContain('0.158 ');
    // Every other recall column withholds too — this is where a withdrawn
    // headline would otherwise leak back in.
    expect(row?.textContent).not.toContain('0.125');
    expect(row?.textContent).not.toContain('0.208');
  });

  it('says in words why an arm is ungraded, and names the gate', () => {
    render(<CritiquePanel run={gatedRun()} />);

    const note = document.querySelector('.crit-ungraded');
    expect(note?.textContent).toContain('rag_llm_filtered — not measured.');
    // The true reason, not a badge: it is a property of the engine.
    expect(note?.textContent).toContain('one shared retrieval pool');
    expect(note?.textContent).toContain('0 of 11 findings record what their own');
    expect(note?.textContent).toContain('retrieval_provenance');
    // And the published pair survives, named as uncertified rather than as a
    // baseline anything may be measured against.
    expect(note?.textContent).toContain('0.158');
    expect(note?.textContent).toContain('0.364');
    expect(note?.textContent).toContain('lower bound rather than a baseline');
    // A graded arm gets no such note.
    expect(document.querySelectorAll('.crit-ungraded')).toHaveLength(1);
  });

  it('refuses to subtract from a baseline the gate did not certify', () => {
    render(<CritiquePanel run={gatedRun()} />);

    const row = screen.getByText('rubric_packs').closest('tr');
    // The V6 headline. "−0.158 vs rag_llm_filtered" would republish the exact
    // figure the gate withdrew, dressed as a measured gap.
    expect(row?.textContent).not.toContain('−0.158');
    expect(row?.textContent).not.toContain('vs rag_llm_filtered');
    // And it says the comparison is gone rather than quietly omitting it.
    expect(row?.textContent).toContain('no comparison — rag_llm_filtered is ungraded');
    // Only on a graded row: the ungraded baseline's own cell already reads
    // "not measured", and saying it twice is noise, not candour.
    expect(rowFor('rag_llm_filtered')?.textContent).not.toContain('no comparison');
  });

  it('does not promise a gap against the baseline when there cannot be one', () => {
    render(<CritiquePanel run={gatedRun()} />);

    const intro = document.querySelector('.crit-intro');
    expect(intro?.textContent).toContain('is ungraded');
    expect(intro?.textContent).not.toContain('Every row after the baseline carries its recall gap');
    // The gate is named where the scores are read, not only in the JSON.
    expect(screen.getByText('gate · retrieval_provenance')).toBeTruthy();
    expect(screen.getByText('1 arm ungraded')).toBeTruthy();
  });

  it('still keeps the subtraction when the baseline is graded', () => {
    // The delta is withheld because the baseline has no score, not because the
    // panel stopped comparing arms.
    render(<CritiquePanel run={summary({ cells: [cell(), gradedCell()] })} />);

    const row = screen.getByText('rubric_packs').closest('tr');
    expect(row?.textContent).toContain('−0.278 vs rag_llm_filtered');
  });

  it('keeps the matcher ballots on an ungraded row', () => {
    // The ballots explain why an arm scored what it scored; an arm with no
    // score still has ballots, and they are the only account of what the
    // matcher did with it.
    const run = gatedRun();
    run.cells[0]!.match_ballots = [lostBallot()];
    render(<CritiquePanel run={run} />);

    const row = rowFor('rag_llm_filtered');
    expect(row?.textContent).toContain('1 pairing outvoted');
  });
});

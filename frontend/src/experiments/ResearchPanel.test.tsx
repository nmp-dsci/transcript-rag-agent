import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import type { PackList, ResearchReport, ResearchScores } from '../api/types';
import { ResearchPanel } from './ResearchPanel';

const packs = vi.fn();
const packResearch = vi.fn();

vi.mock('../api/client', () => ({
  api: {
    packs: (...args: unknown[]) => packs(...args),
    packResearch: (...args: unknown[]) => packResearch(...args),
  },
}));

function list(): PackList {
  return {
    packs: [
      {
        topic: 'resume-design',
        name: 'Resume design',
        blurb: 'What a technical resume has to do on the page.',
        artifact: 'resume',
        built: true,
      },
      {
        topic: 'job-search',
        name: 'Job search',
        blurb: 'Everything around the resume.',
        artifact: 'job search',
        built: true,
      },
    ],
    build_command: 'uv run python -m src.cli build-packs',
  };
}

function report(overrides: Partial<ResearchReport> = {}): ResearchReport {
  return {
    kind: 'deep-research',
    topic: 'resume-design',
    name: 'Resume design',
    artifact: 'resume',
    generated_at: '2026-08-10T16:07:14Z',
    corpus_digest: 'e1b17b35cb7190cf',
    chunk_count: 1372,
    model: 'deepseek-v4-flash',
    members: 14,
    excluded_video_ids: [],
    held_out_video_ids: ['15rTnqKBlO8'],
    settings: {
      round_one_probes: 6,
      gap_probes: 4,
      retrieval_pool: 18,
      excerpts_per_unit: 12,
      max_rubrics_per_unit: 3,
    },
    plan: [
      {
        probe_id: 'p01',
        facet: 'Recruiter skimming behavior',
        question: 'What do recruiters look for in the first few seconds?',
        why: 'first filter',
        origin: 'plan',
        rank: 1,
      },
      {
        probe_id: 'p07',
        facet: 'Formatting and layout',
        question: 'What layout makes a tech resume easy to read?',
        why: 'readability',
        origin: 'plan',
        rank: 7,
      },
    ],
    gaps: [
      {
        gap_id: 'g01',
        missing:
          'No criterion requires the resume to include name and contact information.',
        why: 'A recruiter cannot reach an applicant without contact info.',
        probe: 'What contact information should be on a resume?',
      },
      {
        gap_id: 'g04',
        missing: 'No criterion checks for a projects or portfolio section.',
        why: 'Evidence of real projects matters for technical roles.',
        probe: 'Should an engineer include GitHub links on their resume?',
      },
    ],
    gap_closure: [
      {
        gap_id: 'g01',
        missing: 'No criterion requires contact information.',
        probe: 'What contact information should be on a resume?',
        rules_from_this_probe: 2,
        best_new_rubric_id: 'r5102',
        best_new_criterion: 'Use ATS-safe, plain formatting.',
        best_new_cosine: 0.21,
        round_one_best_cosine: 0.34,
        closed: false,
      },
      {
        gap_id: 'g04',
        missing: 'No criterion checks for a projects or portfolio section.',
        probe: 'Should an engineer include GitHub links on their resume?',
        rules_from_this_probe: 3,
        best_new_rubric_id: 'r5403',
        best_new_criterion: 'Link only to GitHub pages that are live and organised.',
        best_new_cosine: 0.62,
        round_one_best_cosine: 0.28,
        closed: true,
      },
    ],
    restatements: [
      {
        rubric_id: 'r5301',
        criterion: 'Keep the resume formatting ATS-safe: one column, standard font.',
        unit_id: 'probe:p53',
        nearest_prior_id: 'r0601',
        nearest_prior_criterion:
          'Format the resume for ATS parsing: use a single column, no tables.',
        nearest_prior_cosine: 0.83,
        dedupe_threshold: 0.86,
      },
      {
        rubric_id: 'r5403',
        criterion: 'Link only to GitHub pages that are live and organised.',
        unit_id: 'probe:p54',
        nearest_prior_id: 'r0102',
        nearest_prior_criterion: 'Include only content relevant to the target role.',
        nearest_prior_cosine: 0.31,
        dedupe_threshold: 0.86,
      },
    ],
    rounds: [
      {
        round: 1,
        arm: 'deep-r1',
        label: "round 1 — the plan's opening probes",
        caused_by: 'planner',
        probes: [],
        executor_calls: 6,
        planner_calls: 1,
        critic_calls: 0,
        rubrics: 16,
        citations: 32,
        multi_creator_share: 0.5,
        deduped_against_previous: 2,
        added_rubric_ids: [],
      },
      {
        round: 2,
        arm: 'deep-r2',
        label: "round 2 — the gap critic's probes",
        caused_by: 'gap critic reading deep-r1',
        probes: [
          {
            probe: {
              probe_id: 'p51',
              facet: 'contact info',
              question: 'What contact information should be on a resume?',
              why: '',
              origin: 'gap:g01',
              rank: 51,
            },
            unit_id: 'probe:p51',
            chunks: 18,
            videos: 5,
            creator_count: 5,
            top_creator: 'Alpha',
            top_creator_share: 0.3,
            eligible: true,
            reject_reason: '',
            rubric_ids: ['r5102'],
            reason: '',
            spent_call: true,
          },
          {
            probe: {
              probe_id: 'p54',
              facet: 'projects',
              question: 'Should an engineer include GitHub links?',
              why: '',
              origin: 'gap:g04',
              rank: 54,
            },
            unit_id: 'probe:p54',
            chunks: 18,
            videos: 6,
            creator_count: 6,
            top_creator: 'Beta',
            top_creator_share: 0.28,
            eligible: true,
            reject_reason: '',
            rubric_ids: ['r5403'],
            reason: '',
            spent_call: true,
          },
        ],
        executor_calls: 4,
        planner_calls: 0,
        critic_calls: 1,
        rubrics: 26,
        citations: 52,
        multi_creator_share: 0.42,
        deduped_against_previous: 2,
        added_rubric_ids: ['r5102', 'r5403'],
      },
    ],
    control: {
      round: 2,
      arm: 'deep-oneshot',
      label: 'control — the same plan continued, no critic',
      caused_by: 'planner',
      probes: [],
      executor_calls: 4,
      planner_calls: 0,
      critic_calls: 0,
      rubrics: 26,
      citations: 51,
      multi_creator_share: 0.46,
      deduped_against_previous: 1,
      added_rubric_ids: [],
    },
    diff: {
      before_arm: 'deep-r1',
      after_arm: 'deep-r2',
      kept: [
        {
          rubric_id: 'r0101',
          criterion: 'Keep the resume to one page.',
          check: 'count the pages',
          unit_id: 'probe:p01',
          creators: ['Alpha', 'Beta'],
          citations: 2,
        },
      ],
      added: [
        {
          rubric_id: 'r5102',
          criterion: 'Use ATS-safe, plain formatting.',
          check: 'look for a single column',
          unit_id: 'probe:p51',
          creators: ['Alpha'],
          citations: 2,
        },
        {
          rubric_id: 'r5403',
          criterion: 'Link only to GitHub pages that are live and organised.',
          check: 'open every link',
          unit_id: 'probe:p54',
          creators: ['Beta', 'Gamma'],
          citations: 2,
        },
      ],
      removed: [],
    },
    control_diff: {
      before_arm: 'deep-r1',
      after_arm: 'deep-oneshot',
      kept: [],
      added: [],
      removed: [],
    },
    budget: [
      {
        arm: 'deep-r1',
        label: 'round 1',
        planner_calls: 1,
        critic_calls: 0,
        executor_calls: 6,
        probes_budgeted: 6,
        total_llm_calls: 7,
        rubrics: 16,
        citations: 32,
      },
      {
        arm: 'deep-oneshot',
        label: 'control',
        planner_calls: 1,
        critic_calls: 0,
        executor_calls: 10,
        probes_budgeted: 10,
        total_llm_calls: 11,
        rubrics: 26,
        citations: 51,
      },
      {
        arm: 'deep-r2',
        label: 'round 2',
        planner_calls: 1,
        critic_calls: 1,
        executor_calls: 10,
        probes_budgeted: 10,
        total_llm_calls: 12,
        rubrics: 26,
        citations: 52,
      },
    ],
    arms: {},
    scores: {
      generated_at: '2026-08-10T16:30:00Z',
      run_id: 'critique-15rTnqKBlO8-20260810-093357',
      metrics: ['criteria_recall', 'evidence_precision', 'provenance'],
      baseline: 'merged',
      held_out_title: 'Resume teardown',
      held_out_video_id: '15rTnqKBlO8',
      criteria_total: 24,
      criteria_applicable: 19,
      match_repeats: 5,
      // The arms were built at one digest and scored against a later one —
      // the real state of the committed run, so the panel is tested on it.
      build_corpus_digest: 'e1b17b35cb7190cf',
      scoring_corpus_digest: '03f0763b5a511f57',
      scoring_chunk_count: 1460,
      scored_on_build_corpus: false,
      rows: [
        {
          arm: 'merged',
          scores: { criteria_recall: 0.2632, evidence_precision: 0.8235, provenance: 1 },
          score_spread: { criteria_recall_min: 0.2105, criteria_recall_max: 0.3684 },
          findings_total: 17,
          findings_grounded: 14,
          citations_total: 30,
          citations_resolved: 30,
          criteria_matched: 5,
          criteria_applicable: 19,
          criteria_recall_grouped: 0.3,
          executor_calls: null,
          unit_budget: 6,
          held_out_leaks: 0,
        },
        {
          arm: 'deep-r2',
          scores: { criteria_recall: 0.4211, evidence_precision: 0.7, provenance: 1 },
          score_spread: { criteria_recall_min: 0.3684, criteria_recall_max: 0.4737 },
          findings_total: 26,
          findings_grounded: 18,
          citations_total: 52,
          citations_resolved: 52,
          criteria_matched: 8,
          criteria_applicable: 19,
          criteria_recall_grouped: 0.4,
          executor_calls: 10,
          unit_budget: 10,
          held_out_leaks: 0,
        },
      ],
      verdicts: {
        criteria_recall: {
          metric: 'criteria_recall',
          leader: 'deep-r2',
          decisive: false,
          reason: "the lead sits inside the scorer's own range across repeats",
        },
      },
    },
    ...overrides,
  };
}

/** V8's frontier round, as the report stores it. */
function frontier(): NonNullable<ResearchReport['frontier']> {
  return {
    generated_at: '2026-08-11T12:00:00Z',
    arm: 'deep-frontier',
    admission_arm: 'deep-r2-admit',
    model: 'deepseek-v4-flash',
    corpus_digest: 'e1b17b35cb7190cf',
    chunk_count: 1372,
    video_count: 56,
    member_videos: 14,
    member_chunks: 182,
    round_one_read: 72,
    settings: { gap_probes: 4, retrieval_pool: 18, excluded_chunks: 72, prompt_version: 'v1' },
    coverage: [
      {
        video_id: 'by8wrrXW3So',
        title: 'What AI Engineer JOBS Need on a Resume',
        channel_name: 'Headless Headhunter',
        chunks: 25,
        read: 4,
      },
      {
        video_id: 'QPUmFKboiqY',
        title: 'Write the Perfect Professional Summary',
        channel_name: 'Greg Langstaff',
        chunks: 5,
        read: 5,
      },
    ],
    gaps: [
      {
        gap_id: 'g04',
        missing: "No criterion checks the resume's file format or filename.",
        why: 'A resume nobody can open is not reviewable.',
        probe: 'What file format should I submit my resume in?',
      },
    ],
    probes: [
      {
        probe: {
          probe_id: 'p64',
          facet: 'file format',
          question: 'What file format should I submit my resume in?',
          why: '',
          origin: 'gap:g04',
          rank: 64,
        },
        unit_id: 'probe:p64',
        chunks: 18,
        videos: 7,
        creator_count: 7,
        top_creator: 'Alpha',
        top_creator_share: 0.22,
        eligible: true,
        reject_reason: '',
        rubric_ids: ['r6402', 'r6403'],
        reason: '',
        spent_call: true,
      },
    ],
    probe_overlap: [{ unit_id: 'probe:p64', chunks: 18, from_round_one_ground: 0 }],
    refused: [
      {
        rubric_id: 'r6401',
        criterion: 'Use a single-column, ATS-friendly layout.',
        unit_id: 'probe:p64',
        chunk_ids: ['chunk:_MT4SgfQ8QY:4', 'chunk:fD0E57QYSPk:3'],
        already_rested_on_by: ['r0601'],
      },
    ],
    admission_on_shipped: { arm: 'merged', rubrics: 17, would_refuse: 0, refused_ids: [] },
    deduped_against_previous: ['r6202'],
    diff: {
      before_arm: 'deep-r1',
      after_arm: 'deep-frontier',
      kept: [],
      added: [
        {
          rubric_id: 'r6402',
          criterion: 'Lead with a complete contact block.',
          check: 'look at the header',
          unit_id: 'probe:p64',
          creators: ['Alpha'],
          citations: 2,
          already_in_shipped: [],
        },
        {
          rubric_id: 'r6403',
          criterion: 'Write every bullet as an impact statement.',
          check: 'read the bullets',
          unit_id: 'probe:p64',
          creators: ['Beta'],
          citations: 2,
          already_in_shipped: ['r0403'],
        },
      ],
      removed: [],
    },
    rediscovery: {
      'deep-frontier': { shipped_arm: 'merged', added: 8, rediscovered: 2, rate: 0.25, rows: [] },
      'deep-r2': { shipped_arm: 'merged', added: 10, rediscovered: 4, rate: 0.4, rows: [] },
      'deep-oneshot': { shipped_arm: 'merged', added: 10, rediscovered: 4, rate: 0.4, rows: [] },
    },
    gap_closure: [],
    restatements: [],
    budget: [],
  };
}

beforeEach(() => {
  vi.clearAllMocks();
  packs.mockResolvedValue(list());
  packResearch.mockImplementation(async (topic: string) =>
    topic === 'resume-design' ? report() : null,
  );
});

describe('ResearchPanel', () => {
  it('renders nothing at all when no loop has been run', async () => {
    packResearch.mockResolvedValue(null);
    const { container } = render(<ResearchPanel />);
    await waitFor(() => expect(packResearch).toHaveBeenCalled());
    expect(container.querySelector('.pk-card')).toBeNull();
  });

  it('quotes the gap critic in full rather than summarising it', async () => {
    render(<ResearchPanel />);
    // The verbatim finding is the evidence this slice rests on, so it must not
    // be truncated or paraphrased into a count.
    expect(
      await screen.findByText(
        'No criterion requires the resume to include name and contact information.',
      ),
    ).toBeInTheDocument();
    expect(
      screen.getByText('What contact information should be on a resume?'),
    ).toBeInTheDocument();
  });

  it('traces each added rule back to the gap whose probe produced it', async () => {
    render(<ResearchPanel />);
    const gap = (
      await screen.findByText('No criterion checks for a projects or portfolio section.')
    ).closest('li') as HTMLElement;
    // r5403 came from probe:p54, and p54's origin is gap:g04 — the causal chain
    // is read off the data, not asserted by the panel.
    expect(
      within(gap).getByText('Link only to GitHub pages that are live and organised.'),
    ).toBeInTheDocument();
    expect(within(gap).getByText('r5403')).toBeInTheDocument();
    // r5102 belongs to g01, so it must not appear under this gap.
    expect(within(gap).queryByText('Use ATS-safe, plain formatting.')).toBeNull();
  });

  it('marks a gap the second round did not actually get nearer to', async () => {
    render(<ResearchPanel />);
    const gap = (
      await screen.findByText(
        'No criterion requires the resume to include name and contact information.',
      )
    ).closest('li') as HTMLElement;
    expect(within(gap).getByText('not closed')).toBeInTheDocument();
    const closed = (
      await screen.findByText('No criterion checks for a projects or portfolio section.')
    ).closest('li') as HTMLElement;
    expect(within(closed).getByText('closed')).toBeInTheDocument();
  });

  it('puts the one-shot control in the rounds table on the same budget', async () => {
    render(<ResearchPanel />);
    const rounds = (await screen.findByText('Rounds')).closest('section') as HTMLElement;
    const control = within(rounds).getByText('deep-oneshot').closest('tr') as HTMLElement;
    const loop = within(rounds).getByText('deep-r2').closest('tr') as HTMLElement;
    // Same executor spend, same delta over round 1 — the difference the reader
    // is being asked to attribute is only where the questions came from.
    expect(within(control).getByText('control')).toBeInTheDocument();
    expect(within(loop).getByText('gap critic reading deep-r1')).toBeInTheDocument();
  });

  it('opens a round to the sub-questions it actually asked and where they came from', async () => {
    render(<ResearchPanel />);
    const rounds = (await screen.findByText('Rounds')).closest('section') as HTMLElement;
    const loop = within(rounds).getByText('deep-r2').closest('tr') as HTMLElement;
    await userEvent.click(within(loop).getByRole('button', { name: /2$/ }));
    const probes = within(rounds).getByText('Should an engineer include GitHub links?');
    const row = probes.closest('li') as HTMLElement;
    // The origin is the point of the section: this question exists because a
    // critic named g04, not because a planner listed it up front.
    expect(row.textContent).toContain('gap:g04');
    expect(row.textContent).toContain('1 rule extracted');
    expect(
      within(rounds).getByText('What contact information should be on a resume?'),
    ).toBeInTheDocument();
  });

  it('shows each score with the spread it came out of', async () => {
    render(<ResearchPanel />);
    await screen.findByText('0.421');
    expect(screen.getByText('0.368–0.474')).toBeInTheDocument();
    expect(screen.getByText('0.210–0.368')).toBeInTheDocument();
  });

  it('puts the finding and quote counts beside the scores so padding is visible', async () => {
    render(<ResearchPanel />);
    // 26 findings against merged's 17 is exactly the shape the known recall
    // attack has, so the columns that show it are not optional.
    await screen.findByText('18/26');
    expect(screen.getByText('14/17')).toBeInTheDocument();
    expect(screen.getByText('52/52')).toBeInTheDocument();
  });

  it('says which corpus the scores were measured on when it is not the one built from', async () => {
    render(<ResearchPanel />);
    const notice = await screen.findByText(/more was ingested after these arms were built/);
    expect(notice.textContent).toContain('e1b17b35cb7190cf');
    expect(notice.textContent).toContain('03f0763b5a511f57');
    expect(notice.textContent).toContain('1460 chunks');
  });

  it('says plainly when a lead sits inside the scorer noise', async () => {
    render(<ResearchPanel />);
    expect(await screen.findByText('inside scorer noise')).toBeInTheDocument();
  });

  it('lists how near each added rule sits to a rule the pack already had', async () => {
    render(<ResearchPanel />);
    const toggle = await screen.findByRole('button', {
      name: /1 of 2 added rules sit within 0.1 of the dedupe threshold/,
    });
    await userEvent.click(toggle);
    // Three decimals: 0.859 would round to 0.86 and read as equal to the
    // threshold it actually sits under.
    expect(screen.getByText('0.830')).toBeInTheDocument();
    expect(
      screen.getByText(/Format the resume for ATS parsing: use a single column, no tables\./),
    ).toBeInTheDocument();
  });

  it('renders the v1 → v2 diff with what was added and what survived', async () => {
    render(<ResearchPanel />);
    const toggle = await screen.findByRole('button', {
      name: /deep-r1 → deep-r2: 2 added, 1 kept, 0 removed/,
    });
    expect(toggle).toBeInTheDocument();
    expect(screen.getByText('Keep the resume to one page.')).toBeInTheDocument();
  });

  it('states the call budget per arm rather than leaving it implied', async () => {
    render(<ResearchPanel />);
    await screen.findByText('Call budget');
    const budget = (await screen.findByText('Call budget')).closest('section') as HTMLElement;
    const row = within(budget).getByText('deep-r2').closest('tr') as HTMLElement;
    // one planner, ten executors, one critic — twelve, one more than the control.
    expect(within(row).getByText('12')).toBeInTheDocument();
  });

  it('marks which plan probes round 1 spent and which only the control did', async () => {
    render(<ResearchPanel />);
    const toggle = await screen.findByRole('button', { name: /The plan — 2 sub-questions/ });
    await userEvent.click(toggle);
    expect(screen.getByText(/round 1 and control/)).toBeInTheDocument();
    expect(screen.getByText(/control only/)).toBeInTheDocument();
  });

  // ─── V8: the frontier round ───────────────────────────────────────────────

  it('renders the original three-arm report unchanged when no frontier round exists', async () => {
    render(<ResearchPanel />);
    await screen.findByText('Rounds');
    expect(screen.queryByText(/The frontier round/)).toBeNull();
  });

  it('leads the frontier section with rediscovery beside the arms it must beat', async () => {
    // The gate turns on this number, not on the new arm's scores: deep-r2 and
    // the no-critic control both rediscovered 4 of 10, which is what said the
    // loop was searching more rather than better.
    packResearch.mockImplementation(async (topic: string) =>
      topic === 'resume-design' ? report({ frontier: frontier() }) : null,
    );
    render(<ResearchPanel />);
    const heading = await screen.findByText(/The frontier round/);
    const section = heading.closest('section') as HTMLElement;
    const table = within(section).getAllByRole('table')[0] as HTMLElement;
    const rowFor = (arm: string) =>
      within(table).getByText(arm).closest('tr') as HTMLElement;
    expect(within(rowFor('deep-frontier')).getByText('25%')).toBeInTheDocument();
    for (const arm of ['deep-r2', 'deep-oneshot']) {
      expect(within(rowFor(arm)).getByText('40%')).toBeInTheDocument();
    }
  });

  it('says how much of each frontier probe landed on ground round 1 had read', async () => {
    packResearch.mockImplementation(async (topic: string) =>
      topic === 'resume-design' ? report({ frontier: frontier() }) : null,
    );
    render(<ResearchPanel />);
    expect(await screen.findByText(/0 of them off round 1's ground/)).toBeInTheDocument();
  });

  it('shows the refused rules and admits the precision they buy is by construction', async () => {
    packResearch.mockImplementation(async (topic: string) =>
      topic === 'resume-design' ? report({ frontier: frontier() }) : null,
    );
    render(<ResearchPanel />);
    expect(
      await screen.findByText('Use a single-column, ATS-friendly layout.'),
    ).toBeInTheDocument();
    // The caveat is not optional: the rule resembles the scorer's own gate.
    expect(screen.getByText(/partly bought by construction/)).toBeInTheDocument();
    // And the fairness check beside it — the rule is not a handicap applied to
    // one side only.
    expect(
      screen.getByText(/every rule the hand build ships already rests on a passage of its own/),
    ).toBeInTheDocument();
  });

  it("marks the frontier round's own additions for rediscovery, like every other diff", async () => {
    // A diff whose rows render unmarked reads as all-new, which would flatter
    // exactly the arm claiming it rediscovers less.
    packResearch.mockImplementation(async (topic: string) =>
      topic === 'resume-design' ? report({ frontier: frontier() }) : null,
    );
    render(<ResearchPanel />);
    const row = (await screen.findByText('Write every bullet as an impact statement.')).closest(
      'li',
    ) as HTMLElement;
    expect(within(row).getByText(/rediscovered/)).toBeInTheDocument();
    const fresh = (
      await screen.findByText('Lead with a complete contact block.')
    ).closest('li') as HTMLElement;
    expect(within(fresh).getByText(/new ground/)).toBeInTheDocument();
  });

  /** The comparison as `src/evals/pack_ablation.py` commits it into the run. */
  function againstBaseline(basis: string): NonNullable<ResearchScores['against_baseline']> {
    return {
      criteria_recall: [
        {
          arm: 'deep-frontier',
          baseline: 'merged',
          metric: 'criteria_recall',
          value: 0.4211,
          baseline_value: 0.2632,
          arm_min: 0.4211,
          baseline_max: 0.3684,
          delta: 0.1579,
          beats_baseline: basis === 'ranges-disjoint',
          basis,
          reason:
            basis === 'ranges-disjoint'
              ? "this arm's worst of the run's repeats (0.4211) sits above the baseline's best (0.3684)"
              : 'the two repeat ranges cross, so neither arm is credited with a lead',
        },
      ],
      evidence_precision: [
        {
          arm: 'deep-frontier',
          baseline: 'merged',
          metric: 'evidence_precision',
          value: 0.875,
          baseline_value: 0.8235,
          delta: 0.0515,
          beats_baseline: true,
          basis: 'deterministic',
          reason:
            "scored once per arm because there is no model in this metric's path — the gap is the whole of the comparison",
        },
      ],
    };
  }

  it('reads the baseline comparison out of the report rather than deriving it', async () => {
    // It used to be computed in this component, which meant the finding existed
    // only while a browser was open — absent from the run file, from `winner()`,
    // and from anything a reader outside the app could check.
    packResearch.mockImplementation(async (topic: string) =>
      topic === 'resume-design'
        ? report({
            frontier: frontier(),
            scores: { ...report().scores!, against_baseline: againstBaseline('ranges-disjoint') },
          })
        : null,
    );
    render(<ResearchPanel />);
    expect(await screen.findByText('criteria_recall vs merged')).toBeInTheDocument();
    expect(screen.getByText('ranges disjoint')).toBeInTheDocument();
    expect(screen.getByText(/sits above the baseline's best/)).toBeInTheDocument();
    // Precision is a single draw because nothing in it can disagree, and that
    // must not be filed under the same doubt as a judged number.
    expect(screen.getByText('no model in its path')).toBeInTheDocument();
    // And the margin is stated, because it is one criterion in nineteen.
    expect(screen.getByText(/one criterion in 19/)).toBeInTheDocument();
  });

  it('says the ranges overlap when the committed comparison says they do', async () => {
    packResearch.mockImplementation(async (topic: string) =>
      topic === 'resume-design'
        ? report({
            frontier: frontier(),
            scores: { ...report().scores!, against_baseline: againstBaseline('ranges-overlap') },
          })
        : null,
    );
    render(<ResearchPanel />);
    expect(await screen.findByText('ranges overlap')).toBeInTheDocument();
    expect(screen.getByText(/neither arm is credited with a lead/)).toBeInTheDocument();
  });

  it('puts the frontier round in the Rounds table with the others', async () => {
    // The arm with the strongest claim was the one arm missing from the table
    // where spend is read against what a round added.
    packResearch.mockImplementation(async (topic: string) =>
      topic === 'resume-design'
        ? report({
            frontier: {
              ...frontier(),
              round: {
                round: 2,
                arm: 'deep-frontier',
                label: 'frontier round',
                caused_by: 'coverage-aware gap critic reading deep-r1',
                probes: frontier().probes,
                executor_calls: 4,
                planner_calls: 0,
                critic_calls: 1,
                rubrics: 24,
                citations: 48,
                multi_creator_share: 0.875,
                deduped_against_previous: 4,
                added_rubric_ids: ['r6402', 'r6403'],
              },
            },
          })
        : null,
    );
    render(<ResearchPanel />);
    const rounds = (await screen.findByText('Rounds')).closest('section') as HTMLElement;
    const row = within(rounds).getByText('deep-frontier').closest('tr') as HTMLElement;
    // 24 rubrics against round 1's 16 — the delta the table exists to show.
    expect(within(row).getByText('+8')).toBeInTheDocument();
    expect(within(row).getByText('48')).toBeInTheDocument();
  });

  it('opens the coverage table the critic was given, with no video filtered out', async () => {
    packResearch.mockImplementation(async (topic: string) =>
      topic === 'resume-design' ? report({ frontier: frontier() }) : null,
    );
    render(<ResearchPanel />);
    const toggle = await screen.findByRole('button', {
      name: /The coverage table the critic was given/,
    });
    await userEvent.click(toggle);
    const fully = screen
      .getByText('Write the Perfect Professional Summary')
      .closest('tr') as HTMLElement;
    // A video the probes read end to end stays on the menu rather than being
    // filtered off it — the critic is handed the whole table.
    expect(within(fully).getAllByText('5').length).toBeGreaterThan(0);
  });
});

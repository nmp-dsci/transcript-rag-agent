import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import type { PackDetail, PackList, PackRubric, ResearchReport } from '../api/types';
import { PackPanel } from './PackPanel';

const packs = vi.fn();
const pack = vi.fn();
const setPackMember = vi.fn();
const packResearch = vi.fn();

vi.mock('../api/client', () => ({
  api: {
    packs: (...args: unknown[]) => packs(...args),
    pack: (...args: unknown[]) => pack(...args),
    setPackMember: (...args: unknown[]) => setPackMember(...args),
    packResearch: (...args: unknown[]) => packResearch(...args),
  },
}));

function rubric(overrides: Partial<PackRubric> = {}): PackRubric {
  return {
    rubric_id: 'r0101',
    criterion:
      'Write every experience bullet as a full qualification: what you did, how you did it, the result, and the context it happened in.',
    check: 'For each bullet, identify all four parts: action, method, outcome and context.',
    why: 'Recruiters hunt for qualifications, not bare tasks.',
    contested: false,
    unit_id: 'raptor:theme:4',
    unit_kind: 'raptor',
    unit_title: 'Tailor resumes for ATS with quantified impact',
    creators: ['Farah Sharghi', 'Headless Headhunter'],
    videos: ['eGmZZFJ-8PY', 'by8wrrXW3So'],
    evidence: [
      {
        video_id: 'by8wrrXW3So',
        chunk_id: 'chunk:by8wrrXW3So:1',
        quote: 'We are looking for these in full sentences that say what you did',
        model_quote: 'We are looking for these in full sentences that say what you did.',
        channel_name: 'Headless Headhunter',
        title: 'What AI Engineer JOBS Need on a Resume',
        start_seconds: 73.36,
        quote_start_seconds: 83.31,
        ratio: 1,
        resolved: true,
        url: 'https://www.youtube.com/watch?v=by8wrrXW3So&t=81s',
      },
    ],
    ...overrides,
  };
}

function list(): PackList {
  return {
    packs: [
      {
        topic: 'resume-design',
        name: 'Resume design',
        blurb: 'What a technical resume has to do on the page.',
        artifact: 'resume',
        built: true,
        arm: 'merged',
        corpus_digest: 'e1b17b35cb7190cf',
      },
      {
        topic: 'app-architecture',
        name: 'App architecture',
        blurb: 'How to cut an application into modules.',
        artifact: 'codebase',
        built: false,
      },
    ],
    excluded_video_ids: ['7m27Go3K1d0'],
    held_out_video_ids: ['15rTnqKBlO8'],
    build_command: 'uv run python -m src.cli build-packs',
  };
}

function detail(overrides: Partial<PackDetail> = {}): PackDetail {
  return {
    topic: 'resume-design',
    name: 'Resume design',
    blurb: 'What a technical resume has to do on the page.',
    artifact: 'resume',
    routing_text: 'How should a software engineer write their resume?',
    built: true,
    overrides: {},
    arms: {
      raptor: {
        arm: 'raptor',
        checks: {
          rubrics: 18,
          evidence_total: 35,
          evidence_resolved: 35,
          quote_resolution: 1,
          multi_creator_rubrics: 16,
          multi_creator_share: 0.889,
          contested_rubrics: 0,
          creators: 11,
          excluded_video_citations: 0,
        },
        units_by_kind: { raptor: 6 },
        unit_budget: 6,
        gaps: 13,
      },
      merged: {
        arm: 'merged',
        checks: {
          rubrics: 17,
          evidence_total: 30,
          evidence_resolved: 30,
          quote_resolution: 1,
          multi_creator_rubrics: 13,
          multi_creator_share: 0.765,
          contested_rubrics: 1,
          creators: 11,
          excluded_video_citations: 0,
        },
        units_by_kind: { communities: 3, raptor: 3 },
        unit_budget: 6,
        gaps: 13,
      },
    },
    ablation: null,
    excluded_video_ids: ['7m27Go3K1d0'],
    held_out_video_ids: ['15rTnqKBlO8'],
    build_command: 'uv run python -m src.cli build-packs',
    arm: 'merged',
    version: 1,
    generated_at: '2026-08-10T14:50:00+00:00',
    provenance: {
      corpus_digest: 'e1b17b35cb7190cf',
      chunk_count: 1372,
      video_count: 56,
      theme_count: 30,
      graph_extractions: 1329,
      excluded_video_ids: ['7m27Go3K1d0'],
      held_out_video_ids: ['15rTnqKBlO8'],
      embedding_model: 'sentence-transformers/all-MiniLM-L6-v2',
      rubric_model: 'deepseek-v4-flash',
      units_budget: 6,
      routing_top_k: 14,
      routing_min_score: 0.25,
      community_min_entities: 5,
    },
    checks: {
      rubrics: 17,
      evidence_total: 30,
      evidence_resolved: 30,
      quote_resolution: 1,
      multi_creator_rubrics: 13,
      multi_creator_share: 0.765,
      contested_rubrics: 1,
      creators: 11,
      excluded_video_citations: 0,
    },
    rubrics: [rubric(), rubric({ rubric_id: 'r0102', contested: true, criterion: 'Contested rule.' })],
    members: [
      {
        video_id: '5gLVxMKeSGM',
        title: 'How to Write a Winning Tech Resume',
        channel_name: 'Anthony D. Mays',
        score: 0.7331,
        routed: true,
        override: null,
        chunk_count: 27,
        in_units: true,
        cited: true,
      },
      {
        video_id: 'late1',
        title: 'Ingested after the theme layer was built',
        channel_name: 'Later Channel',
        score: 0.512,
        routed: true,
        override: null,
        chunk_count: 16,
        in_units: false,
        cited: false,
      },
    ],
    units: [],
    gaps: [
      {
        unit_id: 'raptor:theme:17',
        unit_kind: 'raptor',
        unit_title: 'Tailor resumes to one role with proof',
        videos: 5,
        chunks: 16,
        reason: 'relevant but outside the 6-unit budget for this arm',
      },
    ],
    ...overrides,
  };
}

/**
 * The slice of a build report the pack browser reads: the v1 → v2 diff, the
 * gap that caused each added rule, and the control's diff for comparison.
 */
function report(): ResearchReport {
  return {
    kind: 'deep-research',
    topic: 'resume-design',
    name: 'Resume design',
    artifact: 'resume',
    generated_at: '2026-08-10T00:00:00+00:00',
    corpus_digest: 'e1b17b35cb7190cf',
    chunk_count: 1372,
    model: 'deepseek-v4-flash',
    members: 22,
    excluded_video_ids: [],
    held_out_video_ids: ['15rTnqKBlO8'],
    settings: { round_one_probes: 6, gap_probes: 4 },
    plan: [],
    gaps: [
      {
        gap_id: 'g04',
        missing: 'No criterion requires or checks for a projects/portfolio section or links to code.',
        why: '',
        probe: 'Should an engineer include projects or GitHub links on a resume?',
      },
    ],
    gap_closure: [],
    restatements: [],
    rounds: [
      { round: 1, arm: 'deep-r1', label: 'round 1', caused_by: 'planner', probes: [],
        executor_calls: 6, planner_calls: 1, critic_calls: 0, rubrics: 16, citations: 32,
        multi_creator_share: 0.81, deduped_against_previous: 0, added_rubric_ids: [] },
      {
        round: 2,
        arm: 'deep-r2',
        label: 'round 2',
        caused_by: 'gap critic reading deep-r1',
        probes: [
          {
            probe: {
              probe_id: 'p54',
              facet: 'projects and links',
              question: 'Should an engineer include projects or GitHub links on a resume?',
              why: '',
              origin: 'gap:g04',
              rank: 54,
            },
            unit_id: 'probe:p54',
            chunks: 18,
            videos: 6,
            creator_count: 5,
            top_creator: 'a',
            top_creator_share: 0.3,
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
        multi_creator_share: 0.85,
        deduped_against_previous: 2,
        added_rubric_ids: ['r5403'],
      },
    ],
    control: { round: 2, arm: 'deep-oneshot', label: 'control', caused_by: 'planner', probes: [],
      executor_calls: 4, planner_calls: 0, critic_calls: 0, rubrics: 26, citations: 51,
      multi_creator_share: 0.85, deduped_against_previous: 2, added_rubric_ids: [] },
    diff: {
      before_arm: 'deep-r1',
      after_arm: 'deep-r2',
      kept: [],
      added: [
        {
          rubric_id: 'r5403',
          criterion: 'Link only to GitHub, portfolio or blog pages that are live and updated.',
          check: '',
          unit_id: 'probe:p54',
          creators: ['Farah Sharghi'],
          citations: 2,
        },
      ],
      removed: [],
    },
    control_diff: { before_arm: 'deep-r1', after_arm: 'deep-oneshot', kept: [], added: [], removed: [] },
    budget: [],
    arms: {},
    scores: null,
  };
}

beforeEach(() => {
  packs.mockReset();
  pack.mockReset();
  setPackMember.mockReset();
  packResearch.mockReset();
  packs.mockResolvedValue(list());
  pack.mockResolvedValue(detail());
  packResearch.mockResolvedValue(report());
});

describe('PackPanel', () => {
  it('shows the pack version and the corpus snapshot hash in the header', async () => {
    render(<PackPanel />);
    const digest = await screen.findByText(/e1b17b35cb7190cf/);
    const header = digest.closest('.exp-sub') as HTMLElement;
    expect(header.textContent).toContain('v1');
    expect(header.textContent).toContain('1372 chunks');
    expect(header.textContent).toContain('merged');
    expect(header.textContent).toContain('deepseek-v4-flash');
  });

  it('states the two gate numbers with their fractions', async () => {
    render(<PackPanel />);
    await screen.findByText('quotes resolving');
    expect(screen.getByText('30/30')).toBeTruthy();
    expect(screen.getByText('rules citing ≥2 creators')).toBeTruthy();
    expect(screen.getByText('13/17')).toBeTruthy();
  });

  it('reports zero excluded-video citations as the leakage check', async () => {
    render(<PackPanel />);
    await screen.findByText('excluded-video citations');
    expect(screen.getByText(/1 videos blocked/)).toBeTruthy();
  });

  it('renders a rubric criterion in full, not truncated', async () => {
    render(<PackPanel />);
    const criterion = await screen.findByText(/Write every experience bullet as a full qualification/);
    // The whole sentence, including its tail — this tab has clipped long text
    // before and jsdom cannot see layout, so the guard is on the text node.
    expect(criterion.textContent).toContain('the context it happened in.');
  });

  it('badges a contested rubric and counts creators per rule', async () => {
    render(<PackPanel />);
    const contested = await screen.findByText('Contested rule.');
    const row = contested.closest('.pk-rubric') as HTMLElement;
    expect(within(row).getByText('contested')).toBeTruthy();
    expect(within(row).getByText('2 creators')).toBeTruthy();
    // The uncontested rule carries the creator count but no contested badge.
    const plain = screen
      .getByText(/Write every experience bullet/)
      .closest('.pk-rubric') as HTMLElement;
    expect(within(plain).queryByText('contested')).toBeNull();
    expect(within(plain).getByText('2 creators')).toBeTruthy();
  });

  it('opens a rubric to its evidence quote, creator and timestamp link', async () => {
    const user = userEvent.setup();
    render(<PackPanel />);
    await user.click(await screen.findByText(/Write every experience bullet/));
    expect(
      screen.getByText(/We are looking for these in full sentences/),
    ).toBeTruthy();
    expect(screen.getByText('Headless Headhunter')).toBeTruthy();
    const link = screen.getByRole('link', { name: '1:23' }) as HTMLAnchorElement;
    expect(link.href).toBe('https://www.youtube.com/watch?v=by8wrrXW3So&t=81s');
    // One `t=` and one only: eight corpus urls already carry one, and YouTube
    // honours the first — a second lands the viewer ~20s early.
    expect(link.href.match(/t=/g)?.length).toBe(1);
  });

  it('shows the three arms side by side even without a D2 run, and says why', async () => {
    render(<PackPanel />);
    await screen.findByText('Arms as built');
    expect(screen.getByText(/No D2 score for this pack/)).toBeTruthy();
    expect(screen.getByText('3 communities · 3 raptor')).toBeTruthy();
  });

  it('renders the D2 rows with the scorer spread beside the point estimate', async () => {
    pack.mockResolvedValue(
      detail({
        ablation: {
          generated_at: '2026-08-10T15:00:00+00:00',
          metrics: ['criteria_recall', 'evidence_precision'],
          baseline: 'raptor',
          held_out_title: 'You asked me to roast your AI resumes',
          artifact_url: null,
          criteria_total: 24,
          criteria_applicable: 19,
          match_repeats: 5,
          cells: [
            {
              arm: 'merged',
              scores: { criteria_recall: 0.263, evidence_precision: 0.6 },
              score_spread: { criteria_recall_min: 0.211, criteria_recall_max: 0.316 },
              rubrics: 17,
              criteria_recall_all: 0.208,
              criteria_recall_grouped: 0.267,
              findings_total: 17,
              findings_grounded: 12,
              citations_total: 30,
              citations_resolved: 30,
              held_out_leaks: 0,
              units_by_kind: { raptor: 3, communities: 3 },
            },
          ],
          verdicts: {
            criteria_recall: {
              metric: 'criteria_recall',
              leader: 'merged',
              decisive: false,
              reason: 'the lead sits inside the scorer’s own range across repeats',
            },
          },
        },
      }),
    );
    render(<PackPanel />);
    await screen.findByText(/raptor vs communities vs merged/);
    expect(screen.getByText('0.211–0.316')).toBeTruthy();
    expect(screen.getByText('inside scorer noise')).toBeTruthy();
  });

  it('records a membership override and says it applies at the next build', async () => {
    const user = userEvent.setup();
    setPackMember.mockResolvedValue({
      topic: 'resume-design',
      overrides: { '5gLVxMKeSGM': false },
      applies: 'at the next build',
    });
    render(<PackPanel />);
    const row = (await screen.findByText('Anthony D. Mays')).closest('tr');
    await user.click(within(row as HTMLElement).getByRole('button', { name: 'out' }));
    await waitFor(() =>
      expect(screen.getByText(/applies at the next build/)).toBeTruthy(),
    );
    expect(setPackMember).toHaveBeenCalledWith('resume-design', '5gLVxMKeSGM', false);
  });

  it('marks a member that routed in but reached no source unit', async () => {
    render(<PackPanel />);
    const row = (await screen.findByText('Later Channel')).closest('tr') as HTMLElement;
    expect(within(row).getByText('no units')).toBeTruthy();
    // Routed-in-but-silent is called out in the section heading too, so the
    // membership count cannot be read as a coverage claim.
    expect(
      screen.getByText(/1 routed in but reached no source unit/),
    ).toBeTruthy();
    const contributing = (await screen.findByText('Anthony D. Mays')).closest(
      'tr',
    ) as HTMLElement;
    expect(within(contributing).queryByText('no units')).toBeNull();
  });

  it('lists a declared pack that was never built rather than hiding it', async () => {
    const user = userEvent.setup();
    pack.mockResolvedValueOnce(detail()).mockResolvedValueOnce(
      detail({
        topic: 'app-architecture',
        built: false,
        rubrics: undefined,
        checks: undefined,
        provenance: undefined,
        arms: {},
      }),
    );
    render(<PackPanel />);
    await screen.findByRole('tab', { name: /app-architecture/ });
    await user.click(screen.getByRole('tab', { name: /app-architecture/ }));
    await screen.findByText(/is declared but not built/);
  });

  it('logs the source units that produced no rule', async () => {
    const user = userEvent.setup();
    render(<PackPanel />);
    await user.click(await screen.findByText(/1 source units produced no rule/));
    expect(screen.getByText(/outside the 6-unit budget/)).toBeTruthy();
  });

  it('shows the loop-built v1 → v2 diff with the gap that asked for each added rule', async () => {
    const user = userEvent.setup();
    render(<PackPanel />);
    const toggle = await screen.findByText(/deep-r1 → deep-r2/);
    // The control's own additions are stated on the closed row: growth on the
    // same budget without a critic must not be readable as iteration.
    expect(
      screen.getByText(/the one-shot control, same budget and no critic, added 0/),
    ).toBeTruthy();
    await user.click(toggle);
    const added = screen.getByText(/Link only to GitHub, portfolio or blog pages/);
    const row = added.closest('.rs-diffrow') as HTMLElement;
    expect(row.className).toContain('added');
    expect(row.textContent).toContain('g04');
    expect(row.textContent).toContain('No criterion requires or checks for a projects');
  });

  it('renders no diff section for a pack no research loop has been run for', async () => {
    packResearch.mockResolvedValue(null);
    render(<PackPanel />);
    await screen.findByText(/1 source units produced no rule/);
    expect(screen.queryByText(/deep-r1 → deep-r2/)).toBeNull();
  });
});

import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import type {
  MatrixRunOption,
  Scoreboard,
  ScoreboardQuestion,
  ScoreboardQuestionSetup,
  ScoreboardRow,
} from '../api/types';
import { ScoreboardView } from './ScoreboardView';

const scoreboard = vi.fn();
vi.mock('../api/client', () => ({ api: { scoreboard: (...args: unknown[]) => scoreboard(...args) } }));

function row(overrides: Partial<ScoreboardRow> = {}): ScoreboardRow {
  return {
    key: 'rag_llm',
    title: 'rag_llm (single-hop)',
    model: 'deepseek-v4',
    legacy: false,
    answers: 8,
    judged: 8,
    avg_scores: { faithfulness: 0.7, answer_relevancy: 0.6, context_precision: 0.55 },
    avg_composite: 0.62,
    wins: 5,
    contests: 8,
    win_rate: 0.625,
    avg_latency_seconds: 4,
    avg_token_estimate: 3000,
    ...overrides,
  };
}

/** A picker entry. Corpus fields absent by default — a pre-corpus-identity run. */
function matrixRun(overrides: Partial<MatrixRunOption> = {}): MatrixRunOption {
  return {
    run_id: 'matrix-20260811-023531',
    created_at: '2026-08-11T02:35:31+00:00',
    setups: ['rag_llm', 'rag_llm_filtered'],
    entry_count: 20,
    judged: true,
    rubric_version: 'ragas-v1',
    ...overrides,
  };
}

function board(setups: ScoreboardRow[], overrides: Partial<Scoreboard> = {}): Scoreboard {
  return {
    setups,
    entries_total: 10,
    entries_judged: 9,
    group_by: 'setup_model',
    judge_model: 'deepseek-v4',
    run_id: 'matrix-20260727-015519',
    runs: [
      {
        run_id: 'matrix-20260727-015519',
        created_at: '2026-07-27T01:55:19+00:00',
        setups: ['rag_llm', 'rag_agent'],
        entry_count: 10,
        judged: true,
      },
    ],
    provenance: {
      judge_models: ['deepseek-v4'],
      ragas_versions: ['0.4.3'],
      embedding_models: ['all-MiniLM-L6-v2'],
      last_judged: '2026-07-21T00:00:00+00:00',
      metrics: ['faithfulness', 'answer_relevancy', 'context_precision'],
      composite: 'mean of the three metrics',
    },
    questions: [],
    ...overrides,
  };
}

function qsetup(overrides: Partial<ScoreboardQuestionSetup> = {}): ScoreboardQuestionSetup {
  return {
    key: 'rag_llm',
    title: 'rag_llm (single-hop)',
    composite: 0.71,
    judged: true,
    error: null,
    answer: 'answer for g001',
    model: 'deepseek-v4',
    elapsed_seconds: 10,
    token_estimate: 100,
    chunk_count: 1,
    ...overrides,
  };
}

function question(overrides: Partial<ScoreboardQuestion> = {}): ScoreboardQuestion {
  return {
    id: 'g001',
    question: 'What are the key tax changes affecting property investors right now?',
    domain: 'property',
    question_type: 'local',
    setups: [
      qsetup(),
      qsetup({ key: 'rag_agent', title: 'rag_agent (agentic)', composite: 0.84 }),
    ],
    ...overrides,
  };
}

/** The Questions panel alone. The Answers panel below it shows the same
 * questions and the same unjudged/error badges, so an unscoped query would
 * match both — these assertions are about the per-question score breakdown. */
function questionsPanel(): HTMLElement {
  const heading = screen.getByText(/^Questions \(/);
  const panel = heading.closest('details');
  if (!panel) throw new Error('Questions panel not found');
  return panel as HTMLElement;
}

describe('ScoreboardView', () => {
  beforeEach(() => {
    scoreboard.mockReset();
  });

  it('shows n for every aggregated row', async () => {
    scoreboard.mockResolvedValue(board([row()]));
    render(<ScoreboardView />);
    const table = await screen.findByRole('table', { name: 'Leaderboard' });
    expect(within(table).getByText('n=8')).toBeInTheDocument();
    expect(within(table).getByText('of 8 answers')).toBeInTheDocument();
  });

  it('de-emphasises a row averaged over too few judged questions', async () => {
    scoreboard.mockResolvedValue(board([row({ judged: 3, answers: 3 })]));
    const { container } = render(<ScoreboardView />);
    const table = await screen.findByRole('table', { name: 'Leaderboard' });
    expect(within(table).getByText('n=3')).toBeInTheDocument();
    expect(container.querySelectorAll('tr.lown')).toHaveLength(1);
    expect(within(table).getByText('thin')).toBeInTheDocument();
  });

  it('leaves a well-evidenced row at full strength', async () => {
    scoreboard.mockResolvedValue(board([row({ judged: 12, answers: 12 })]));
    const { container } = render(<ScoreboardView />);
    const table = await screen.findByRole('table', { name: 'Leaderboard' });
    expect(within(table).getByText('n=12')).toBeInTheDocument();
    expect(container.querySelectorAll('tr.lown')).toHaveLength(0);
    expect(within(table).queryByText('thin')).not.toBeInTheDocument();
  });

  it('marks a win rate decided by only a handful of contests', async () => {
    scoreboard.mockResolvedValue(board([row({ judged: 9, contests: 2, wins: 2, win_rate: 1 })]));
    render(<ScoreboardView />);
    await screen.findByText('n=9');
    expect(screen.getByText('n=2')).toBeInTheDocument();
  });

  it('surfaces the token-efficiency comparison beneath the table', async () => {
    scoreboard.mockResolvedValue(
      board([
        row(),
        row({
          key: 'rag_agent',
          title: 'rag_agent (agentic)',
          avg_composite: 0.55,
          avg_token_estimate: 19000,
        }),
      ]),
    );
    render(<ScoreboardView />);
    expect(await screen.findByText('Efficiency — composite per 1k tokens')).toBeInTheDocument();
    expect(screen.getByText('0.207')).toBeInTheDocument();
    expect(screen.getByText('0.029')).toBeInTheDocument();
  });

  it('explains the metrics even with nothing to derive from', async () => {
    scoreboard.mockResolvedValue(board([row()]));
    render(<ScoreboardView />);
    expect(await screen.findByText('What the metrics mean')).toBeInTheDocument();
    expect(screen.getByText('supported claims ÷ total claims')).toBeInTheDocument();
    expect(
      screen.getByText('average precision — mean of precision@k over the ranks judged useful'),
    ).toBeInTheDocument();
  });

  it('keeps the provenance bar and the methodology note', async () => {
    scoreboard.mockResolvedValue(board([row()]));
    const { container } = render(<ScoreboardView />);
    await waitFor(() => expect(container.querySelector('.provbar')).not.toBeNull());
    expect(container.querySelector('.board-note')).not.toBeNull();
  });

  it('lists every committed matrix run in the picker', async () => {
    scoreboard.mockResolvedValue(
      board([row()], {
        runs: [
          {
            run_id: 'matrix-newer',
            created_at: '2026-07-27T00:00:00+00:00',
            setups: ['rag_llm', 'rag_agent'],
            entry_count: 20,
            judged: true,
          },
          {
            run_id: 'matrix-older',
            created_at: '2026-07-01T00:00:00+00:00',
            setups: ['rag_llm'],
            entry_count: 14,
            judged: false,
          },
        ],
      }),
    );
    render(<ScoreboardView />);
    const picker = await screen.findByLabelText('Matrix run');
    expect(picker).toBeEnabled();
    expect(screen.getByRole('option', { name: 'newest run' })).toBeInTheDocument();
    expect(
      screen.getByRole('option', {
        name: 'matrix-newer · 20 questions · 2 setups · no corpus recorded',
      }),
    ).toBeInTheDocument();
    // An unjudged run is labelled as such, so it is not mistaken for a ranking.
    expect(
      screen.getByRole('option', {
        name: 'matrix-older · 14 questions · 1 setup · unjudged · no corpus recorded',
      }),
    ).toBeInTheDocument();
  });

  it('distinguishes two otherwise identical runs by the corpus each was scored on', async () => {
    // The defect this fixes: both of these labelled "20 questions · 2 setups ·
    // ragas-v1", so flipping between two runs scored on different corpora read
    // as one number moving rather than as two incomparable measurements.
    scoreboard.mockResolvedValue(
      board([row()], {
        runs: [
          matrixRun({
            run_id: 'matrix-20260811-023531',
            corpus: 'dceb228df01f7418',
            corpus_videos: 71,
            corpus_chunks: 1792,
          }),
          matrixRun({ run_id: 'matrix-20260810-110656', corpus: '1bdb1971fc49bb77' }),
          matrixRun({ run_id: 'matrix-20260809-071818-depth-v2' }),
        ],
      }),
    );
    render(<ScoreboardView />);
    await screen.findByLabelText('Matrix run');
    expect(
      screen.getByRole('option', {
        name: 'matrix-20260811-023531 · 20 questions · 2 setups · ragas-v1 · corpus dceb228d (71 videos · 1792 chunks)',
      }),
    ).toBeInTheDocument();
    // Digest recorded, counts not — the label says so rather than borrowing
    // the other run's counts or implying the size is unknown-because-equal.
    expect(
      screen.getByRole('option', {
        name: 'matrix-20260810-110656 · 20 questions · 2 setups · ragas-v1 · corpus 1bdb1971 (size not recorded)',
      }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole('option', {
        name: 'matrix-20260809-071818-depth-v2 · 20 questions · 2 setups · ragas-v1 · no corpus recorded',
      }),
    ).toBeInTheDocument();
  });

  it('states the corpus a run was scored on beside its numbers', async () => {
    scoreboard.mockResolvedValue(
      board([row()], {
        run: matrixRun({
          run_id: 'matrix-20260811-023531',
          corpus: 'dceb228df01f7418',
          corpus_videos: 71,
          corpus_chunks: 1792,
        }),
      }),
    );
    render(<ScoreboardView />);
    expect(await screen.findByText('corpus scored on')).toBeInTheDocument();
    expect(screen.getByText('dceb228d')).toBeInTheDocument();
    expect(screen.getAllByText('71 videos · 1792 chunks').length).toBeGreaterThan(0);
  });

  it('says a run recorded no corpus rather than leaving the header to speak for it', async () => {
    scoreboard.mockResolvedValue(board([row()], { run: matrixRun() }));
    render(<ScoreboardView />);
    expect(await screen.findByText('not recorded')).toBeInTheDocument();
    expect(screen.getByText('run predates corpus identity')).toBeInTheDocument();
  });

  it('refetches the board for the run the user picks', async () => {
    scoreboard.mockResolvedValue(board([row()]));
    render(<ScoreboardView />);
    const picker = await screen.findByLabelText('Matrix run');
    await waitFor(() =>
      expect(scoreboard).toHaveBeenCalledWith('setup_model', null, null),
    );

    await userEvent.selectOptions(picker, 'matrix-20260727-015519');
    await waitFor(() =>
      expect(scoreboard).toHaveBeenCalledWith(
        'setup_model',
        null,
        'matrix-20260727-015519',
      ),
    );
  });

  it('lists the judged questions for the selected run, with each setup\'s score', async () => {
    scoreboard.mockResolvedValue(
      board([row()], { questions: [question()] }),
    );
    render(<ScoreboardView />);
    expect(await screen.findByText('Questions (1, 1 judged)')).toBeInTheDocument();
    const panel = within(questionsPanel());
    expect(
      panel.getByText('What are the key tax changes affecting property investors right now?'),
    ).toBeInTheDocument();
    expect(panel.getByText('property')).toBeInTheDocument();
    expect(panel.getByText('local')).toBeInTheDocument();
    expect(panel.getByText('0.71')).toBeInTheDocument();
    expect(panel.getByText('0.84')).toBeInTheDocument();
  });

  it('marks an unjudged or errored setup for a question instead of showing a score', async () => {
    scoreboard.mockResolvedValue(
      board([row()], {
        questions: [
          question({
            setups: [
              qsetup({ composite: null, judged: false }),
              qsetup({
                key: 'rag_agent',
                title: 'rag_agent (agentic)',
                composite: null,
                judged: false,
                error: 'timeout',
              }),
            ],
          }),
        ],
      }),
    );
    render(<ScoreboardView />);
    // A run committed with --no-judge has questions but no judged cell, so the
    // header must not claim it judged them.
    await screen.findByText('Questions (1, 0 judged)');
    const panel = within(questionsPanel());
    expect(panel.getByText('unjudged')).toBeInTheDocument();
    expect(panel.getByText('error')).toBeInTheDocument();
  });

  it('counts a question as judged when any one of its setups was', async () => {
    scoreboard.mockResolvedValue(
      board([row()], {
        questions: [
          question({
            setups: [
              qsetup(),
              qsetup({
                key: 'rag_agent',
                title: 'rag_agent (agentic)',
                composite: null,
                judged: false,
              }),
            ],
          }),
          question({ id: 'g002', setups: [] }),
        ],
      }),
    );
    render(<ScoreboardView />);
    expect(await screen.findByText('Questions (2, 1 judged)')).toBeInTheDocument();
  });

  it('shows no questions panel when the run has none', async () => {
    scoreboard.mockResolvedValue(board([row()], { questions: [] }));
    render(<ScoreboardView />);
    await screen.findByRole('table', { name: 'Leaderboard' });
    expect(screen.queryByText(/Questions \(\d+/)).not.toBeInTheDocument();
  });

  it('points at the Experiments tab when no run has been committed', async () => {
    scoreboard.mockResolvedValue(
      board([], { runs: [], run_id: null, entries_judged: 0, entries_total: 0 }),
    );
    render(<ScoreboardView />);
    expect(await screen.findByText(/No committed matrix run yet/)).toBeInTheDocument();
    expect(screen.getByLabelText('Matrix run')).toBeDisabled();
  });
});

/** A depth-v2 board: eight weighted metrics in two groups, with a capped row. */
function depthBoard(): Scoreboard {
  const metrics = [
    'faithfulness',
    'context_precision',
    'answer_relevancy',
    'insight_depth',
    'specificity',
    'coverage',
    'evidence_breadth',
    'calibration',
  ];
  return board(
    [
      row({
        avg_scores: Object.fromEntries(metrics.map((name) => [name, 0.6])),
        avg_composite: 0.73,
        capped: 2,
      }),
    ],
    {
      runs: [
        {
          run_id: 'matrix-20260809-051901-depth-v2',
          created_at: '2026-08-09T05:19:01+00:00',
          setups: ['rag_llm'],
          entry_count: 5,
          judged: true,
          rubric_version: 'depth-v2',
        },
        {
          run_id: 'matrix-20260729-061607',
          created_at: '2026-07-29T06:16:07+00:00',
          setups: ['rag_llm'],
          entry_count: 5,
          judged: true,
          rubric_version: 'ragas-v1',
        },
      ],
      provenance: {
        judge_models: ['deepseek-v4-flash'],
        ragas_versions: ['0.4.3'],
        embedding_models: ['all-MiniLM-L6-v2'],
        last_judged: '2026-08-09T05:19:01+00:00',
        rubric_version: 'depth-v2',
        rubric_versions: ['depth-v2'],
        metrics,
        metric_weights: {
          faithfulness: 0.2,
          context_precision: 0.1,
          answer_relevancy: 0.1,
          insight_depth: 0.2,
          specificity: 0.15,
          coverage: 0.1,
          evidence_breadth: 0.1,
          calibration: 0.05,
        },
        metric_groups: [
          {
            key: 'grounding',
            label: 'grounding',
            weight: 0.4,
            metrics: ['faithfulness', 'context_precision', 'answer_relevancy'],
          },
          {
            key: 'depth',
            label: 'depth',
            weight: 0.6,
            metrics: [
              'insight_depth',
              'specificity',
              'coverage',
              'evidence_breadth',
              'calibration',
            ],
          },
        ],
        composite:
          'weighted sum — grounding 40%, depth 60%; capped at 0.5 when faithfulness < 0.6',
      },
    },
  );
}

describe('ScoreboardView under depth-v2', () => {
  beforeEach(() => {
    scoreboard.mockReset();
  });

  it('renders the rubric as eight grouped metric rows', async () => {
    scoreboard.mockResolvedValue(depthBoard());
    const { container } = render(<ScoreboardView />);
    await screen.findByRole('table', { name: 'Rubric metrics' });
    expect(container.querySelectorAll('tr.metricrow')).toHaveLength(8);
    const groups = container.querySelectorAll('tr.grouprow');
    expect(groups).toHaveLength(2);
    expect(groups[0]).toHaveTextContent('grounding');
    expect(groups[0]).toHaveTextContent('40%');
    expect(groups[1]).toHaveTextContent('depth');
    expect(groups[1]).toHaveTextContent('60%');
  });

  it('keeps a ragas-v1 run at three ungrouped metric rows', async () => {
    scoreboard.mockResolvedValue(board([row()]));
    const { container } = render(<ScoreboardView />);
    await screen.findByRole('table', { name: 'Rubric metrics' });
    expect(container.querySelectorAll('tr.metricrow')).toHaveLength(3);
    expect(container.querySelectorAll('tr.grouprow')).toHaveLength(0);
  });

  it('widens the leaderboard to the rubric it was judged under', async () => {
    scoreboard.mockResolvedValue(depthBoard());
    render(<ScoreboardView />);
    const table = await screen.findByRole('table', { name: 'Leaderboard' });
    for (const label of ['Insight', 'Specific', 'Coverage', 'Breadth', 'Calibrated']) {
      expect(within(table).getByText(label)).toBeInTheDocument();
    }
  });

  it('marks a leaderboard row whose answers hit the cap', async () => {
    scoreboard.mockResolvedValue(depthBoard());
    render(<ScoreboardView />);
    const table = await screen.findByRole('table', { name: 'Leaderboard' });
    expect(within(table).getByText('capped 2')).toBeInTheDocument();
  });

  it('names the rubric on every run in the picker, so rankings can be compared', async () => {
    scoreboard.mockResolvedValue(depthBoard());
    render(<ScoreboardView />);
    await screen.findByLabelText('Matrix run');
    expect(
      screen.getByRole('option', {
        name: 'matrix-20260809-051901-depth-v2 · 5 questions · 1 setup · depth-v2 · no corpus recorded',
      }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole('option', {
        name: 'matrix-20260729-061607 · 5 questions · 1 setup · ragas-v1 · no corpus recorded',
      }),
    ).toBeInTheDocument();
  });

  it('explains the five new metrics alongside the original three', async () => {
    scoreboard.mockResolvedValue(depthBoard());
    render(<ScoreboardView />);
    expect(await screen.findByText('Insight depth')).toBeInTheDocument();
    expect(screen.getByText('Evidence breadth')).toBeInTheDocument();
    expect(screen.getByText('Calibration')).toBeInTheDocument();
  });
});

describe('ScoreboardView honesty signals', () => {
  beforeEach(() => {
    scoreboard.mockReset();
  });

  it('badges a row judged on exactly LOW_N questions', async () => {
    // The boundary the shipped depth-v2 run sat on: a strict `<` let n=5
    // escape its own warning, which is the size the warning is for.
    scoreboard.mockResolvedValue(board([row({ judged: 5, answers: 5 })]));
    const { container } = render(<ScoreboardView />);
    const table = await screen.findByRole('table', { name: 'Leaderboard' });
    expect(within(table).getByText('n=5')).toBeInTheDocument();
    expect(container.querySelectorAll('tr.lown')).toHaveLength(1);
    expect(within(table).getByText('thin')).toBeInTheDocument();
  });

  it('leaves a row above the boundary at full strength', async () => {
    scoreboard.mockResolvedValue(board([row({ judged: 6, answers: 6 })]));
    const { container } = render(<ScoreboardView />);
    await screen.findByRole('table', { name: 'Leaderboard' });
    expect(container.querySelectorAll('tr.lown')).toHaveLength(0);
  });

  it('marks a row whose answers breached the grounding floor without being capped', async () => {
    scoreboard.mockResolvedValue(board([row({ capped: 0, ungrounded: 3 })]));
    render(<ScoreboardView />);
    const table = await screen.findByRole('table', { name: 'Leaderboard' });
    expect(within(table).getByText('ungrounded 3')).toBeInTheDocument();
    expect(within(table).queryByText(/^capped/)).not.toBeInTheDocument();
  });

  it('warns when one run mixes rubrics instead of labelling it as one', async () => {
    scoreboard.mockResolvedValue(
      board([row()], {
        provenance: {
          ...board([]).provenance,
          rubric_versions: ['depth-v2', 'ragas-v1'],
        },
      }),
    );
    render(<ScoreboardView />);
    expect(await screen.findByText('mixed rubrics')).toBeInTheDocument();
    expect(screen.getByText(/not on one scale/)).toBeInTheDocument();
  });

  it('says nothing about rubrics when the run has only one', async () => {
    scoreboard.mockResolvedValue(board([row()]));
    render(<ScoreboardView />);
    await screen.findByRole('table', { name: 'Leaderboard' });
    expect(screen.queryByText('mixed rubrics')).not.toBeInTheDocument();
  });

  it('declares a self-graded ranking above the table and in the provenance bar', async () => {
    scoreboard.mockResolvedValue(
      board([row()], {
        provenance: {
          ...board([]).provenance,
          self_graded: true,
          self_graded_answers: 30,
          depth_judge_models: ['deepseek-v4-flash'],
        },
      }),
    );
    render(<ScoreboardView />);
    expect(await screen.findAllByText('self-graded')).toHaveLength(2);
    expect(screen.getByText(/self-assessment, not an independent verdict/)).toBeInTheDocument();
    expect(screen.getByText('depth judge')).toBeInTheDocument();
  });

  it('stays quiet about self-grading when the judge is independent', async () => {
    scoreboard.mockResolvedValue(board([row()]));
    render(<ScoreboardView />);
    await screen.findByRole('table', { name: 'Leaderboard' });
    expect(screen.queryByText('self-graded')).not.toBeInTheDocument();
  });

  it('states how many cells the rubric could not cover', async () => {
    scoreboard.mockResolvedValue(
      board([row()], {
        run: {
          run_id: 'matrix-20260727-015519',
          created_at: '2026-07-27T01:55:19+00:00',
          setups: ['rag_llm'],
          entry_count: 20,
          judged: true,
          rubric_version: 'depth-v2',
          rejudged_cells: 79,
          skipped_cells: 1,
        },
      }),
    );
    render(<ScoreboardView />);
    expect(await screen.findByText(/1 of 80 cells in this run/)).toBeInTheDocument();
    expect(screen.getByText(/excluded from every average/)).toBeInTheDocument();
  });

  it('says nothing about coverage when the rubric scored every cell', async () => {
    scoreboard.mockResolvedValue(board([row()]));
    render(<ScoreboardView />);
    await screen.findByRole('table', { name: 'Leaderboard' });
    expect(screen.queryByText(/cells in this run could not/)).not.toBeInTheDocument();
  });
});

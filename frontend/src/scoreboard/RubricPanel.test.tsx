import { render, screen, within } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import type { MetricGroup, ScoreboardRow } from '../api/types';
import { RubricPanel } from './RubricPanel';

const DEPTH_METRICS = [
  'faithfulness',
  'context_precision',
  'answer_relevancy',
  'insight_depth',
  'specificity',
  'coverage',
  'evidence_breadth',
  'calibration',
];

const DEPTH_WEIGHTS: Record<string, number> = {
  faithfulness: 0.2,
  context_precision: 0.1,
  answer_relevancy: 0.1,
  insight_depth: 0.2,
  specificity: 0.15,
  coverage: 0.1,
  evidence_breadth: 0.1,
  calibration: 0.05,
};

const DEPTH_GROUPS: MetricGroup[] = [
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
    metrics: ['insight_depth', 'specificity', 'coverage', 'evidence_breadth', 'calibration'],
  },
];

const LEGACY_METRICS = ['faithfulness', 'answer_relevancy', 'context_precision'];

function row(overrides: Partial<ScoreboardRow> = {}): ScoreboardRow {
  return {
    key: 'rag_llm',
    title: 'rag_llm (single-hop)',
    model: 'deepseek-v4-flash',
    legacy: false,
    answers: 5,
    judged: 5,
    avg_scores: Object.fromEntries(DEPTH_METRICS.map((name) => [name, 0.6])),
    avg_composite: 0.6,
    wins: 2,
    contests: 5,
    win_rate: 0.4,
    avg_latency_seconds: 31.8,
    avg_token_estimate: 3108,
    ...overrides,
  };
}

function metricRows(): HTMLElement[] {
  return Array.from(document.querySelectorAll('tr.metricrow'));
}

describe('RubricPanel under depth-v2', () => {
  function renderDepth() {
    return render(
      <RubricPanel
        metrics={DEPTH_METRICS}
        weights={DEPTH_WEIGHTS}
        groups={DEPTH_GROUPS}
        rubricVersion="depth-v2"
        rows={[row(), row({ key: 'graph_rag', title: 'graph_rag (knowledge graph)' })]}
        composite="weighted sum — grounding 40%, depth 60%; capped at 0.5 when faithfulness < 0.6"
      />,
    );
  }

  it('renders one row per metric — eight of them', () => {
    renderDepth();
    expect(metricRows()).toHaveLength(8);
    expect(
      metricRows().map((tr) => tr.getAttribute('data-metric')),
    ).toEqual(DEPTH_METRICS);
  });

  it('bands the metrics into grounding (40%) and depth (60%)', () => {
    renderDepth();
    const groupRows = Array.from(document.querySelectorAll('tr.grouprow'));
    expect(groupRows).toHaveLength(2);
    expect(groupRows[0]).toHaveTextContent('grounding');
    expect(groupRows[0]).toHaveTextContent('40%');
    expect(groupRows[1]).toHaveTextContent('depth');
    expect(groupRows[1]).toHaveTextContent('60%');
  });

  it('shows each metric its own weight', () => {
    renderDepth();
    const table = screen.getByRole('table', { name: 'Rubric metrics' });
    const insight = within(table).getByText('Insight').closest('tr');
    expect(insight).toHaveTextContent('20%');
    const calibration = within(table).getByText('Calibrated').closest('tr');
    expect(calibration).toHaveTextContent('5%');
  });

  it('names the rubric and how its composite is formed', () => {
    renderDepth();
    expect(screen.getByText('depth-v2')).toBeInTheDocument();
    expect(screen.getByText(/capped at 0\.5 when faithfulness < 0\.6/)).toBeInTheDocument();
  });

  it('gives every setup a column', () => {
    renderDepth();
    const table = screen.getByRole('table', { name: 'Rubric metrics' });
    expect(within(table).getByText('rag_llm (single-hop)')).toBeInTheDocument();
    expect(within(table).getByText('graph_rag (knowledge graph)')).toBeInTheDocument();
  });
});

describe('RubricPanel under ragas-v1', () => {
  function renderLegacy() {
    return render(
      <RubricPanel
        metrics={LEGACY_METRICS}
        weights={{
          faithfulness: 1 / 3,
          answer_relevancy: 1 / 3,
          context_precision: 1 / 3,
        }}
        groups={[]}
        rubricVersion="ragas-v1"
        rows={[row({ avg_scores: { faithfulness: 0.9 } })]}
        composite="mean of the metric scores"
      />,
    );
  }

  it('still renders its three rows, ungrouped', () => {
    renderLegacy();
    expect(metricRows()).toHaveLength(3);
    expect(document.querySelectorAll('tr.grouprow')).toHaveLength(0);
  });

  it('labels itself ragas-v1', () => {
    renderLegacy();
    expect(screen.getByText('ragas-v1')).toBeInTheDocument();
    expect(screen.getByText(/mean of the metric scores/)).toBeInTheDocument();
  });

  it('shows an em dash where a setup has no score for a metric', () => {
    renderLegacy();
    const table = screen.getByRole('table', { name: 'Rubric metrics' });
    expect(within(table).getByText('Faithful').closest('tr')).toHaveTextContent('0.90');
    expect(within(table).getByText('Relevant').closest('tr')).toHaveTextContent('—');
  });

  it('renders nothing at all when there is no run to describe', () => {
    const { container } = render(
      <RubricPanel
        metrics={[]}
        weights={{}}
        groups={[]}
        rubricVersion="ragas-v1"
        rows={[]}
        composite=""
      />,
    );
    expect(container).toBeEmptyDOMElement();
  });
});

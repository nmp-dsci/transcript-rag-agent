import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import type { Provenance } from '../api/types';
import { ProvenanceBar } from './ProvenanceBar';

function provenance(overrides: Partial<Provenance> = {}): Provenance {
  return {
    judge_models: ['deepseek-v4-flash'],
    ragas_versions: ['0.4.3'],
    embedding_models: ['all-MiniLM-L6-v2'],
    last_judged: '2026-08-09T05:19:01+00:00',
    metrics: ['faithfulness', 'answer_relevancy', 'context_precision'],
    composite: 'mean of the metric scores',
    ...overrides,
  };
}

describe('ProvenanceBar', () => {
  it('names the depth judge alongside the grounding judge', () => {
    // It produces 60% of a depth-v2 composite; naming only the grounding judge
    // credits an independent verdict to the smaller half.
    render(
      <ProvenanceBar
        provenance={provenance({ depth_judge_models: ['deepseek-v4-flash'] })}
      />,
    );
    expect(screen.getByText('judge')).toBeInTheDocument();
    expect(screen.getByText('depth judge')).toBeInTheDocument();
  });

  it('omits the depth judge under a rubric that has none', () => {
    render(<ProvenanceBar provenance={provenance()} />);
    expect(screen.queryByText('depth judge')).not.toBeInTheDocument();
  });

  it('declares a self-graded ranking', () => {
    render(<ProvenanceBar provenance={provenance({ self_graded: true })} />);
    expect(screen.getByText('self-graded')).toBeInTheDocument();
    expect(
      screen.getByText(/the answering model graded its own answers/),
    ).toBeInTheDocument();
  });

  it('says nothing when the judge is independent of the answering model', () => {
    render(<ProvenanceBar provenance={provenance({ self_graded: false })} />);
    expect(screen.queryByText('self-graded')).not.toBeInTheDocument();
  });
});

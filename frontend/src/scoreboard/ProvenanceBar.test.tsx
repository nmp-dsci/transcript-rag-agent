import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import type { MatrixRunOption, Provenance } from '../api/types';
import { ProvenanceBar } from './ProvenanceBar';

function run(overrides: Partial<MatrixRunOption> = {}): MatrixRunOption {
  return {
    run_id: 'matrix-20260811-023531',
    created_at: '2026-08-11T02:35:31+00:00',
    setups: ['rag_llm', 'rag_llm_filtered'],
    entry_count: 20,
    judged: true,
    ...overrides,
  };
}

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

  it('carries the run\'s corpus in the methodology line', () => {
    render(
      <ProvenanceBar
        provenance={provenance()}
        run={run({ corpus: 'dceb228df01f7418', corpus_videos: 71, corpus_chunks: 1792 })}
      />,
    );
    expect(screen.getByText('corpus')).toBeInTheDocument();
    expect(
      screen.getByText('dceb228df01f7418 · 71 videos · 1792 chunks'),
    ).toBeInTheDocument();
  });

  it('reports an unrecorded corpus rather than dropping the row', () => {
    // Dropping it would leave one fewer thing on the bar to check, which reads
    // as one fewer thing to worry about.
    render(<ProvenanceBar provenance={provenance()} run={run()} />);
    expect(
      screen.getByText('not recorded · run predates corpus identity'),
    ).toBeInTheDocument();
  });
});

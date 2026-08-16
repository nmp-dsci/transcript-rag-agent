import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import type { MatrixRunOption } from '../api/types';
import { corpusPeers, corpusState, hasCorpusHazard } from './corpus';
import { CorpusNote } from './CorpusNote';

function run(overrides: Partial<MatrixRunOption> = {}): MatrixRunOption {
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

const SIZED = run({
  run_id: 'matrix-20260811-023531',
  corpus: 'dceb228df01f7418',
  corpus_videos: 71,
  corpus_chunks: 1792,
});
const DIGEST_ONLY = run({ run_id: 'matrix-20260810-110656', corpus: '1bdb1971fc49bb77' });
const UNRECORDED = run({ run_id: 'matrix-20260809-071818-depth-v2' });

describe('corpus state', () => {
  it('separates a sized corpus from a digest with no counts and from none at all', () => {
    expect(corpusState(SIZED)).toBe('sized');
    expect(corpusState(DIGEST_ONLY)).toBe('digest');
    expect(corpusState(UNRECORDED)).toBe('unrecorded');
    expect(corpusState(null)).toBe('unrecorded');
  });

  it('never reads an unrecorded corpus as agreement with a recorded one', () => {
    // The distinction this project has had to restore twice: absent is not
    // equal. A run that records nothing has an *unknown* relationship even to
    // runs that record a digest, so `same` stays empty in both directions.
    expect(corpusPeers(UNRECORDED, [SIZED]).same).toHaveLength(0);
    expect(corpusPeers(UNRECORDED, [SIZED]).unknown).toHaveLength(1);
    expect(corpusPeers(SIZED, [UNRECORDED]).same).toHaveLength(0);
    expect(corpusPeers(SIZED, [UNRECORDED]).unknown).toHaveLength(1);
  });

  it('pairs runs that name the same digest, and separates ones that do not', () => {
    const twin = run({ run_id: 'matrix-twin', corpus: 'dceb228df01f7418' });
    const peers = corpusPeers(SIZED, [SIZED, twin, DIGEST_ONLY]);
    expect(peers.same.map((r) => r.run_id)).toEqual(['matrix-twin']);
    expect(peers.different.map((r) => r.run_id)).toEqual(['matrix-20260810-110656']);
    expect(peers.unknown).toHaveLength(0);
    expect(hasCorpusHazard(peers, 'sized')).toBe(true);
    // Only one corpus in the picker: nothing to mistake for a comparison.
    const clean = corpusPeers(SIZED, [SIZED, twin]);
    expect(hasCorpusHazard(clean, 'sized')).toBe(false);
  });
});

describe('CorpusNote', () => {
  it('states the corpus a sized run was scored on, and de-asserts the page header', () => {
    render(<CorpusNote run={SIZED} runs={[SIZED, DIGEST_ONLY]} />);
    expect(screen.getByText('corpus dceb228df01f7418')).toBeInTheDocument();
    expect(screen.getByText('71 videos · 1792 chunks')).toBeInTheDocument();
    expect(screen.getByText(/the corpus indexed/)).toBeInTheDocument();
    expect(screen.getByText(/right now/)).toBeInTheDocument();
  });

  it('reports a digest whose counts the run never recorded, without inventing them', () => {
    render(<CorpusNote run={DIGEST_ONLY} runs={[DIGEST_ONLY, SIZED]} />);
    expect(screen.getByText('corpus 1bdb1971fc49bb77')).toBeInTheDocument();
    expect(screen.getByText(/video and chunk counts this run did not record/)).toBeInTheDocument();
    expect(screen.queryByText(/1792/)).not.toBeInTheDocument();
  });

  it('calls a run with no corpus unknown rather than equal', () => {
    render(<CorpusNote run={UNRECORDED} runs={[UNRECORDED, SIZED]} />);
    expect(screen.getByText('no corpus recorded')).toBeInTheDocument();
    expect(screen.getByText(/predates corpus identity/)).toBeInTheDocument();
    expect(screen.getByText(/is not the same claim as/)).toBeInTheDocument();
  });

  it('names the runs scored on a different corpus as not like-for-like', () => {
    const { container } = render(<CorpusNote run={SIZED} runs={[SIZED, DIGEST_ONLY]} />);
    expect(container.textContent).toContain('1 other committed run here was scored on a different');
    expect(container.textContent).toContain('1bdb1971');
    expect(container.textContent).toContain('not a like-for-like comparison');
    // The hazard styling, so it does not read as a caption.
    expect(container.querySelector('.rubricwarn')).not.toBeNull();
  });

  it('says the engine version is unrecorded instead of naming one', () => {
    const { container } = render(<CorpusNote run={SIZED} runs={[SIZED]} />);
    expect(container.textContent).toContain('No run records the engine version');
    expect(container.textContent).toContain('evals/runs/README.md');
    // Nothing to warn about between runs here, so it renders quietly.
    expect(container.querySelector('.rubricwarn')).toBeNull();
    expect(container.querySelector('.board-note')).not.toBeNull();
  });

  it('renders nothing when no run is selected', () => {
    const { container } = render(<CorpusNote run={null} runs={[]} />);
    expect(container.textContent).toBe('');
  });
});
